"""
Violation Classification Module (Task 4)
==========================================
Two-stage pipeline:
  Stage 1 — YOLOv8 detects ROI bounding boxes (see detector.py)
  Stage 2 — EfficientNet-B3 classifies each cropped ROI into a violation class

Why EfficientNet-B3:
  - Compound scaling (depth/width/resolution) → best accuracy per FLOP
  - 12M params; ~85% top-1 on ImageNet; 640-input inference ~10ms on T4
  - Supports timm pretrained weights (ImageNet-21k)

Innovations:
  - MixUp + CutMix training
  - Label smoothing to prevent overconfident predictions
  - Temperature scaling for calibrated confidence
  - Per-class focal loss weighting
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Violation class metadata
# ---------------------------------------------------------------------------

VIOLATION_CLASSES = {
    0: {"name": "vehicle",            "severity": "info",   "fine_inr": 0},
    1: {"name": "helmet_violation",   "severity": "high",   "fine_inr": 1000},
    2: {"name": "seatbelt_violation", "severity": "medium", "fine_inr": 1000},
    3: {"name": "triple_riding",      "severity": "high",   "fine_inr": 1000},
    4: {"name": "wrong_side_driving", "severity": "critical","fine_inr": 5000},
    5: {"name": "stop_line_violation","severity": "medium", "fine_inr": 500},
    6: {"name": "red_light_violation","severity": "critical","fine_inr": 5000},
    7: {"name": "illegal_parking",    "severity": "low",    "fine_inr": 500},
}


# ---------------------------------------------------------------------------
# EfficientNet backbone (via timm)
# ---------------------------------------------------------------------------

def build_efficientnet(num_classes: int = 8, pretrained: bool = True,
                        variant: str = "efficientnet_b3",
                        dropout_rate: float = 0.3) -> nn.Module:
    """
    Load EfficientNet from timm with a custom classification head.
    Falls back to a lightweight CNN if timm is unavailable.
    """
    try:
        import timm
        model = timm.create_model(variant, pretrained=pretrained, num_classes=0)
        in_features = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        print(f"[Classifier] Loaded {variant} via timm (in_features={in_features})")
        return model
    except ImportError:
        print("[Classifier] timm not found – using lightweight CNN fallback.")
        return _LightweightCNN(num_classes)


class _LightweightCNN(nn.Module):
    """Fallback CNN for environments without timm."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# MixUp / CutMix augmentation
# ---------------------------------------------------------------------------

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    cut_rat = np.sqrt(1 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Scales logits by a learned temperature T > 1 to soften overconfident predictions.
    Calibrated via NLL minimisation on the validation set.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model: nn.Module, val_loader, device: str = "cpu"):
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                logits_list.append(model(imgs).cpu())
                labels_list.append(lbls)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def _eval():
            optimizer.zero_grad()
            loss = F.nll_loss(F.log_softmax(self(logits), dim=1), labels)
            loss.backward()
            return loss
        optimizer.step(_eval)
        print(f"[Calibration] Temperature: {self.temperature.item():.4f}")


# ---------------------------------------------------------------------------
# Violation Classifier
# ---------------------------------------------------------------------------

class ViolationClassifier:
    """
    Classifies cropped detection ROIs into violation categories.

    Usage
    -----
    clf = ViolationClassifier.from_config(cfg)
    result = clf.classify_roi(image_bgr, bbox=[x1,y1,x2,y2])
    """

    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    INPUT_SIZE = 224

    def __init__(self, num_classes: int = 8, variant: str = "efficientnet_b3",
                 pretrained: bool = True, device: str = "cpu",
                 checkpoint: Optional[str] = None):
        self.device   = device
        self.nc       = num_classes
        self.model    = build_efficientnet(num_classes, pretrained, variant).to(device)
        self.scaler   = TemperatureScaler().to(device)
        if checkpoint and Path(checkpoint).exists():
            self.load(checkpoint)

    @classmethod
    def from_config(cls, cfg: dict) -> "ViolationClassifier":
        cl  = cfg.get("classification", {})
        hw  = cfg.get("hardware", {})
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            num_classes=cfg.get("dataset", {}).get("num_classes", 8),
            variant=cl.get("backbone", "efficientnet_b3"),
            pretrained=cl.get("pretrained", True),
            device=dev,
        )

    def _preprocess_roi(self, roi: np.ndarray) -> torch.Tensor:
        roi = cv2.resize(roi, (self.INPUT_SIZE, self.INPUT_SIZE))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(roi.transpose(2, 0, 1)).float() / 255.0
        t   = (t - self.IMG_MEAN) / self.IMG_STD
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify_roi(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Classify a single detection region.
        bbox: [x1, y1, x2, y2] in pixel coords.
        Returns dict with class_id, class_name, confidence, severity, fine_inr.
        """
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {"class_id": -1, "class_name": "invalid", "confidence": 0.0}
        self.model.eval()
        tensor = self._preprocess_roi(roi)
        logits = self.model(tensor)
        logits = self.scaler(logits)
        probs  = torch.softmax(logits, dim=1)[0]
        cls_id = int(probs.argmax())
        conf   = float(probs[cls_id])
        meta   = VIOLATION_CLASSES.get(cls_id, {})
        return {
            "class_id":   cls_id,
            "class_name": meta.get("name", str(cls_id)),
            "confidence": round(conf, 4),
            "severity":   meta.get("severity", "unknown"),
            "fine_inr":   meta.get("fine_inr", 0),
            "all_probs":  {VIOLATION_CLASSES[i]["name"]: round(float(probs[i]), 4)
                           for i in range(self.nc)}
        }

    def classify_batch(self, images: List[np.ndarray],
                        bboxes: List[List[int]]) -> List[Dict]:
        return [self.classify_roi(img, bb) for img, bb in zip(images, bboxes)]

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def build_optimizer(self, lr: float = 1e-4, wd: float = 1e-4) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def build_scheduler(self, optimizer, epochs: int = 30):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    def train_epoch(self, loader, optimizer, criterion, use_mixup: bool = True,
                    use_cutmix: bool = True, mixup_alpha: float = 0.2,
                    cutmix_alpha: float = 1.0) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            # Randomly choose MixUp or CutMix per batch
            r = np.random.rand()
            if use_cutmix and r < 0.5:
                imgs, lbl_a, lbl_b, lam = cutmix_data(imgs, labels, cutmix_alpha)
            elif use_mixup:
                imgs, lbl_a, lbl_b, lam = mixup_data(imgs, labels, mixup_alpha)
            else:
                lbl_a, lbl_b, lam = labels, labels, 1.0

            logits = self.model(imgs)
            loss = mixup_criterion(criterion, logits, lbl_a, lbl_b, lam)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            n += len(imgs)
        return total_loss / n

    @torch.no_grad()
    def eval_epoch(self, loader) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []
        for imgs, labels in loader:
            logits = self.model(imgs.to(self.device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        return {"accuracy": acc, "report": report}

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }, path)
        print(f"[Classifier] Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        print(f"[Classifier] Loaded ← {path}")


if __name__ == "__main__":
    clf = ViolationClassifier(num_classes=8, pretrained=False)
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = clf.classify_roi(img, [100, 100, 400, 400])
    print("Classification result:", result)
