"""
Data Loading, Augmentation, and Dataset Management
====================================================
Supports YOLO-format annotations (standard for Roboflow datasets).
Includes heavy augmentation for violation rarity class balancing.
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# YOLO annotation parser
# ---------------------------------------------------------------------------

def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Parse YOLO-format label file.
    Returns Nx5 array: [class_id, x1, y1, x2, y2] in pixel coords.
    """
    boxes = []
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append([cls, x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


def to_yolo_format(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert [cls,x1,y1,x2,y2] → [cls,cx,cy,bw,bh] normalised."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / img_w
    out[:, 2] = ((boxes[:, 2] + boxes[:, 4]) / 2) / img_h
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / img_w
    out[:, 4] = (boxes[:, 4] - boxes[:, 2]) / img_h
    return out


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Heavy augmentation designed for traffic violation scenarios:
    - Mosaic (mix 4 images) for density variety
    - MixUp for class boundary smoothing
    - Random HSV, flip, scale, translate
    - Cutout for occlusion robustness
    - Random Rain/Noise simulation for adverse conditions
    """

    def __init__(self, img_size=640, mosaic_prob=0.5, mixup_prob=0.15,
                 flip_prob=0.5, hsv_prob=0.8, cutout_prob=0.3,
                 rain_prob=0.1, noise_prob=0.2):
        self.img_size   = img_size
        self.mosaic_p   = mosaic_prob
        self.mixup_p    = mixup_prob
        self.flip_p     = flip_prob
        self.hsv_p      = hsv_prob
        self.cutout_p   = cutout_prob
        self.rain_p     = rain_prob
        self.noise_p    = noise_prob

    # ---- individual transforms ----

    def random_hsv(self, img):
        """Random Hue, Saturation, Value jitter."""
        if random.random() > self.hsv_p:
            return img
        h_gain, s_gain, v_gain = 0.015, 0.7, 0.4
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=np.int16)
        lut_h = ((x * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def random_flip(self, img, boxes):
        """Horizontal flip."""
        if random.random() < self.flip_p:
            img = cv2.flip(img, 1)
            if len(boxes):
                h, w = img.shape[:2]
                boxes[:, 1], boxes[:, 3] = w - boxes[:, 3], w - boxes[:, 1]
        return img, boxes

    def cutout(self, img):
        """Random erasing (simulates occlusion)."""
        if random.random() > self.cutout_p:
            return img
        h, w = img.shape[:2]
        n_cuts = random.randint(1, 5)
        for _ in range(n_cuts):
            bw = random.randint(w // 20, w // 5)
            bh = random.randint(h // 20, h // 5)
            x = random.randint(0, w - bw)
            y = random.randint(0, h - bh)
            img[y:y+bh, x:x+bw] = random.randint(64, 128)
        return img

    def simulate_rain(self, img):
        """Overlay synthetic rain streaks."""
        if random.random() > self.rain_p:
            return img
        h, w = img.shape[:2]
        rain = np.zeros_like(img)
        n_drops = random.randint(500, 2000)
        for _ in range(n_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(10, 30)
            angle = random.uniform(-20, 20)
            dx = int(length * np.sin(np.radians(angle)))
            dy = int(length * np.cos(np.radians(angle)))
            cv2.line(rain, (x, y), (min(x+dx, w-1), min(y+dy, h-1)),
                     (200, 200, 200), 1, cv2.LINE_AA)
        return cv2.addWeighted(img, 0.85, rain, 0.15, 0)

    def add_noise(self, img):
        """Gaussian noise for sensor noise robustness."""
        if random.random() > self.noise_p:
            return img
        sigma = random.uniform(5, 25)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def scale_translate(self, img, boxes):
        """Random scale + translation with border padding."""
        scale = random.uniform(0.5, 1.5)
        tx = random.uniform(-0.1, 0.1) * img.shape[1]
        ty = random.uniform(-0.1, 0.1) * img.shape[0]
        h, w = img.shape[:2]
        M = np.float32([[scale, 0, tx], [0, scale, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
        if len(boxes):
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + tx
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * scale + ty
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, w)
            boxes[:, [2, 4]] = np.clip(boxes[:, [2, 4]], 0, h)
            # Remove boxes that became too small
            valid = ((boxes[:, 3] - boxes[:, 1]) > 4) & ((boxes[:, 4] - boxes[:, 2]) > 4)
            boxes = boxes[valid]
        return img, boxes

    def __call__(self, img: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = self.random_hsv(img)
        img, boxes = self.random_flip(img, boxes)
        img, boxes = self.scale_translate(img, boxes)
        img = self.cutout(img)
        img = self.simulate_rain(img)
        img = self.add_noise(img)
        return img, boxes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "vehicle", "helmet_violation", "seatbelt_violation", "triple_riding",
    "wrong_side_driving", "stop_line_violation", "red_light_violation", "illegal_parking"
]

# Class weights to address long-tail distribution of violations
CLASS_WEIGHTS = torch.tensor([0.5, 2.0, 2.0, 3.0, 3.0, 2.5, 2.5, 2.0])


class TrafficViolationDataset(Dataset):
    """
    YOLO-format dataset for traffic violation detection.

    Expected structure:
        images/  *.jpg | *.png
        labels/  *.txt  (one per image, YOLO format)

    Data Sources (recommended for Kaggle):
        - https://universe.roboflow.com/traffic-violations
        - https://universe.roboflow.com/indian-traffic-dataset
        - https://www.kaggle.com/datasets/andrewmvd/traffic-sign-dataset-in-yolo-format
    """

    def __init__(self, img_dir: str, label_dir: str, img_size: int = 640,
                 augment: bool = True, cache: bool = False,
                 class_names: List[str] = None):
        self.img_dir   = Path(img_dir)
        self.lbl_dir   = Path(label_dir)
        self.img_size  = img_size
        self.augment   = augment
        self.augmentor = AugmentationPipeline() if augment else None
        self.class_names = class_names or CLASS_NAMES

        # Collect image paths
        self.img_paths = sorted(
            [p for p in self.img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

        # Optional: cache images in RAM (fast on Kaggle with enough RAM)
        self.cache = cache
        self._cached = {} if cache else None

    def __len__(self): return len(self.img_paths)

    def _load_image(self, idx: int) -> np.ndarray:
        if self.cache and idx in self._cached:
            return self._cached[idx].copy()
        img = cv2.imread(str(self.img_paths[idx]))
        assert img is not None, f"Cannot read {self.img_paths[idx]}"
        if self.cache:
            self._cached[idx] = img
        return img.copy()

    def __getitem__(self, idx: int):
        img  = self._load_image(idx)
        h, w = img.shape[:2]

        lbl_path = self.lbl_dir / (self.img_paths[idx].stem + ".txt")
        boxes = parse_yolo_label(str(lbl_path), w, h)  # Nx5 pixel coords

        if self.augment and self.augmentor:
            img, boxes = self.augmentor(img, boxes)
            h, w = img.shape[:2]

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # Re-normalise boxes to new size
        if len(boxes):
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * self.img_size / w
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * self.img_size / h
            boxes = torch.from_numpy(boxes)
        else:
            boxes = torch.zeros((0, 5))

        return img, boxes, str(self.img_paths[idx])


def collate_fn(batch):
    imgs, boxes, paths = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, list(boxes), list(paths)


def build_dataloader(img_dir: str, lbl_dir: str, batch_size: int = 16,
                     img_size: int = 640, augment: bool = True,
                     num_workers: int = 4, cache: bool = False) -> DataLoader:
    ds = TrafficViolationDataset(img_dir, lbl_dir, img_size=img_size,
                                  augment=augment, cache=cache)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True, drop_last=augment)


# ---------------------------------------------------------------------------
# Synthetic data generator (for Kaggle demo without real dataset)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(out_dir: str, n_images: int = 100,
                                 n_classes: int = 8, img_size: int = 640):
    """
    Creates a synthetic dataset of random traffic-like images for pipeline
    validation before real data is loaded.
    """
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        # Synthetic road-like background (grey + noise)
        img = np.full((img_size, img_size, 3), 100, dtype=np.uint8)
        img += np.random.randint(-30, 30, img.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)

        # Draw road marking
        cv2.rectangle(img, (0, img_size//2 - 20), (img_size, img_size//2 + 20), (200, 200, 200), -1)

        boxes_str = []
        n_boxes = random.randint(1, 5)
        for _ in range(n_boxes):
            cls = random.randint(0, n_classes - 1)
            bw  = random.randint(50, 200)
            bh  = random.randint(50, 150)
            cx  = random.randint(bw//2, img_size - bw//2)
            cy  = random.randint(bh//2, img_size - bh//2)
            col = [(200, 50, 50), (50, 200, 50), (50, 50, 200),
                   (200, 200, 50), (50, 200, 200), (200, 50, 200),
                   (150, 100, 50), (100, 150, 200)][cls]
            cv2.rectangle(img, (cx-bw//2, cy-bh//2), (cx+bw//2, cy+bh//2), col, 2)
            boxes_str.append(
                f"{cls} {cx/img_size:.6f} {cy/img_size:.6f} {bw/img_size:.6f} {bh/img_size:.6f}"
            )

        cv2.imwrite(str(out / "images" / f"img_{i:05d}.jpg"), img)
        with open(out / "labels" / f"img_{i:05d}.txt", "w") as f:
            f.write("\n".join(boxes_str))

    print(f"[Synthetic] Generated {n_images} images → {out}")
    return str(out)


if __name__ == "__main__":
    path = generate_synthetic_dataset("./data/synthetic", n_images=20)
    dl = build_dataloader(f"{path}/images", f"{path}/labels", batch_size=4, augment=True, num_workers=0)
    imgs, boxes, paths = next(iter(dl))
    print(f"Batch: imgs={imgs.shape}, boxes=[{[b.shape for b in boxes]}]")
