"""
Training Driver (Task 2 & 4)
==============================
Orchestrates:
  - YOLOv8 detection model training
  - EfficientNet classification model training
  - EMA, mixed-precision, early stopping
  - Checkpoint management
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path


def load_config(cfg_path: str) -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def train_detector(cfg: dict):
    """Train YOLOv8 detection model."""
    from src.models.detector import TrafficDetector, generate_data_yaml
    from src.data.dataset import CLASS_NAMES

    print("\n" + "="*60)
    print("TRAINING: Detection Model (YOLOv8)")
    print("="*60)

    det_cfg = cfg["detection"]
    tr_cfg  = cfg["training"]
    paths   = cfg["paths"]

    # Generate data YAML
    data_yaml = os.path.join(paths["output_dir"], "data.yaml")
    os.makedirs(paths["output_dir"], exist_ok=True)
    generate_data_yaml(
        data_root=paths["data_root"],
        class_names=cfg["dataset"]["class_names"],
        out_path=data_yaml
    )

    detector = TrafficDetector.from_config(cfg)
    if detector._backend == "ultralytics":
        detector.train(
            data_yaml=data_yaml,
            epochs=tr_cfg["epochs"],
            batch=tr_cfg["batch_size"],
            img_size=cfg["dataset"]["image_size"][0],
            project=paths["model_dir"],
            name="detect_run",
            workers=cfg["dataset"]["num_workers"],
            amp=tr_cfg["amp"],
        )
    else:
        print("[Train] ultralytics backend unavailable – skipping detection training.")
    return detector


def train_classifier(cfg: dict):
    """Train EfficientNet classification model with MixUp/CutMix."""
    from src.models.classifier import ViolationClassifier, ViolationFocalLoss
    from src.data.dataset import TrafficViolationDataset, collate_fn
    from src.models.detector import ModelEMA
    from torch.utils.data import DataLoader

    print("\n" + "="*60)
    print("TRAINING: Violation Classifier (EfficientNet-B3)")
    print("="*60)

    cl_cfg   = cfg["classification"]
    tr_cfg   = cfg["training"]
    paths    = cfg["paths"]
    nc       = cfg["dataset"]["num_classes"]
    dev      = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = ViolationClassifier.from_config(cfg)

    # Build simple classification datasets (images + integer class labels)
    # For YOLO datasets, use the class_id of the dominant box as label
    # This is a simplified classification loader; a dedicated cls dataset is ideal
    class FlatClassDataset(torch.utils.data.Dataset):
        """Wraps YOLO dataset for classification: takes dominant class per image."""
        def __init__(self, img_dir, lbl_dir, img_size=224, augment=True):
            from src.data.dataset import TrafficViolationDataset
            self.base = TrafficViolationDataset(img_dir, lbl_dir, img_size=img_size, augment=augment)
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, boxes, _ = self.base[idx]
            if len(boxes) > 0:
                label = int(boxes[boxes[:, 0].argmax(), 0]) if boxes.shape[1] > 0 else 0
            else:
                label = 0
            return img, torch.tensor(label, dtype=torch.long)

    try:
        train_ds = FlatClassDataset(
            paths["train_images"], paths["train_labels"], augment=True)
        val_ds   = FlatClassDataset(
            paths["val_images"], paths["val_labels"], augment=False)
        train_loader = DataLoader(train_ds, batch_size=tr_cfg["batch_size"],
                                   shuffle=True, num_workers=cfg["dataset"]["num_workers"])
        val_loader   = DataLoader(val_ds, batch_size=tr_cfg["batch_size"],
                                   shuffle=False, num_workers=cfg["dataset"]["num_workers"])
    except Exception as e:
        print(f"[Train] Dataset load failed ({e}) – classification training skipped.")
        return classifier

    # Loss: focal + label smoothing
    alpha = torch.ones(nc).to(dev)
    # Up-weight rare violations
    alpha[3] = 3.0  # triple riding
    alpha[4] = 3.0  # wrong side
    alpha[6] = 2.5  # red light
    from src.models.classifier import ViolationFocalLoss
    criterion  = ViolationFocalLoss(alpha=alpha, gamma=cl_cfg.get("focal_loss_gamma", 2.0))
    optimizer  = classifier.build_optimizer(cl_cfg["learning_rate"], cl_cfg["weight_decay"])
    scheduler  = classifier.build_scheduler(optimizer, tr_cfg["epochs"])

    best_f1, patience = 0.0, 0
    max_patience = tr_cfg.get("early_stopping_patience", 10)

    for epoch in range(1, tr_cfg["epochs"] + 1):
        train_loss = classifier.train_epoch(
            train_loader, optimizer, criterion,
            use_mixup=True, use_cutmix=True,
            mixup_alpha=cl_cfg.get("mixup_alpha", 0.2),
            cutmix_alpha=cl_cfg.get("cutmix_alpha", 1.0),
        )
        val_metrics = classifier.eval_epoch(val_loader)
        f1 = val_metrics["report"].get("macro avg", {}).get("f1-score", 0.0)
        scheduler.step()

        print(f"[Epoch {epoch:3d}/{tr_cfg['epochs']}] loss={train_loss:.4f} "
              f"acc={val_metrics['accuracy']:.4f} f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            save_path = os.path.join(paths["model_dir"], "best_classifier.pth")
            os.makedirs(paths["model_dir"], exist_ok=True)
            classifier.save(save_path)
        else:
            patience += 1
            if patience >= max_patience:
                print(f"[Train] Early stopping at epoch {epoch} (best F1={best_f1:.4f})")
                break

    print(f"[Train] Classification training complete. Best F1: {best_f1:.4f}")
    return classifier


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train AutoViolate-CV models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--task", choices=["detect", "classify", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["model_dir"],  exist_ok=True)

    if args.task in ("detect", "all"):
        train_detector(cfg)
    if args.task in ("classify", "all"):
        train_classifier(cfg)


if __name__ == "__main__":
    main()
