# AutoViolate-CV
## Flipkart Gridhackathon Round 2 | Theme 3 — Automated Traffic Violation Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-green.svg)](https://ultralytics.com/yolov8)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**AutoViolate-CV** is a production-grade, modular computer vision prototype for automated identification, classification, documentation, and reporting of traffic violations from image and video data. Built for Kaggle evaluation with GPU acceleration, it covers all 8 required violation categories end-to-end.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT (Image / Video)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  TASK 1: Image Preprocessing                                    │
│  ┌──────────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────┐  │
│  │ LowLight     │ │ Shadow    │ │ Rain         │ │ Deblur   │  │
│  │ Enhancer     │ │ Remover   │ │ Remover      │ │ (Wiener) │  │
│  │ (CLAHE+Gamma)│ │ (LAB)     │ │ (GuidedFilt) │ │          │  │
│  └──────────────┘ └───────────┘ └──────────────┘ └──────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  TASK 2 & 3: Object Detection                                   │
│  ┌───────────────────────────────────────────┐                  │
│  │  YOLOv8-L (Anchor-free, Decoupled Head)   │                  │
│  │  + Optional RT-DETR Ensemble (WBF)        │                  │
│  │  Detects: vehicles, riders, pedestrians   │                  │
│  └───────────────────────────────────────────┘                  │
└────────────────┬────────────────────────┬────────────────────────┘
                 ▼                        ▼
┌──────────────────────────┐  ┌───────────────────────────────────┐
│  TASK 4: Classification  │  │  TASK 5: License Plate Recognition│
│  EfficientNet-B3         │  │  YOLOv8-n (plate detect)          │
│  + MixUp / CutMix        │  │  → LightSRNet (4x super-res)      │
│  + Focal Loss            │  │  → EasyOCR + fuzzy validation     │
│  + Temperature Scaling   │  └───────────────────────────────────┘
└──────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  TASK 7: Analytics & Reporting                                  │
│  CSV records │ HTML Dashboard (Plotly) │ PDF Summary            │
│  Spatial Heatmap │ Time-series Trends │ Searchable Query        │
└─────────────────────────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  TASK 8: Evaluation                                             │
│  mAP@50/75 │ Accuracy │ Precision │ Recall │ F1 │ FPS          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Violation Categories

| ID | Class | Severity | Fine (INR) |
|----|-------|----------|-----------|
| 0 | vehicle (base) | info | — |
| 1 | helmet_violation | 🔴 high | ₹1,000 |
| 2 | seatbelt_violation | 🟠 medium | ₹1,000 |
| 3 | triple_riding | 🔴 high | ₹1,000 |
| 4 | wrong_side_driving | 🚨 critical | ₹5,000 |
| 5 | stop_line_violation | 🟠 medium | ₹500 |
| 6 | red_light_violation | 🚨 critical | ₹5,000 |
| 7 | illegal_parking | 🟡 low | ₹500 |

---

## Project Structure

```
gridthon/
├── config/
│   └── config.yaml              # Master configuration
├── src/
│   ├── preprocessing/
│   │   └── image_enhancer.py    # Task 1: All enhancement modules
│   ├── data/
│   │   └── dataset.py           # Data loading, augmentation, synthetic gen
│   ├── models/
│   │   ├── detector.py          # Task 2 & 3: YOLOv8 + WBF ensemble
│   │   └── classifier.py        # Task 4: EfficientNet-B3 + MixUp/CutMix
│   ├── lpr/
│   │   └── lpr_pipeline.py      # Task 5: Plate detect + SR + EasyOCR
│   ├── inference/
│   │   └── pipeline.py          # Unified inference orchestrator
│   ├── evaluation/
│   │   └── evaluator.py         # Task 8: mAP, F1, latency benchmarks
│   ├── reporting/
│   │   └── reporter.py          # Task 7: HTML, CSV, PDF reports
│   └── visualization/
│       └── visualizer.py        # Annotation drawing + metric plots
├── train.py                     # Training driver
├── showcase.py                  # End-to-end demo script
├── kaggle_notebook.py           # Kaggle notebook cells
├── requirements.txt
└── README.md
```

---

## Data Requirements & Sources

> **⚠️ No real traffic images are included.** The `generate_synthetic_dataset()` utility
> creates annotated placeholder images for smoke-testing the pipeline. For real performance,
> use one of the following:

### Recommended Kaggle/Roboflow Datasets

| Dataset | Violations Covered | Format | Link |
|---------|-------------------|--------|------|
| Traffic Violation Detection (Roboflow) | Helmet, seatbelt, triple riding | YOLO | [roboflow.com](https://universe.roboflow.com/traffic-violations) |
| Indian Traffic Dataset | Helmet, signal, lane | YOLO | [roboflow.com](https://universe.roboflow.com/indian-traffic-dataset) |
| Helmet Detection Dataset (Kaggle) | Helmet compliance | YOLO | kaggle.com/datasets/anshulmehtakaggl/helmet-detection |
| Traffic Sign YOLO (Kaggle) | Traffic signals | YOLO | kaggle.com/datasets/andrewmvd/traffic-sign-dataset-in-yolo-format |

### Expected Directory Layout (YOLO format)

```
data/
├── train/
│   ├── images/   *.jpg
│   └── labels/   *.txt  (class_id cx cy bw bh — normalised)
├── val/
│   ├── images/
│   └── labels/
└── test/
    └── images/
```

### Annotation Format (YOLO)
```
# label.txt for one image
1 0.512 0.433 0.124 0.198    # helmet_violation
0 0.230 0.500 0.300 0.400    # vehicle
```

---

## Setup & Execution

### 1. Local Environment

```bash
git clone https://github.com/your-team/autoviolate-cv.git
cd autoviolate-cv
pip install -r requirements.txt
```

### 2. Run Showcase (no real data needed)

```bash
python showcase.py --config config/config.yaml --out_dir showcase_output
```

### 3. Train Models

```bash
# Train detection model
python train.py --config config/config.yaml --task detect

# Train classifier
python train.py --config config/config.yaml --task classify

# Train both
python train.py --config config/config.yaml --task all
```

### 4. Run Inference on an Image

```python
import cv2, yaml
from src.inference.pipeline import TrafficViolationInference

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

pipeline = TrafficViolationInference.from_config(cfg)
img = cv2.imread("test_image.jpg")
result = pipeline.infer(img, image_id="test")

for v in result.violations:
    print(f"Violation: {v.class_name} | Conf: {v.confidence:.2f} | Plate: {v.plate_text}")
```

### 5. Run on Kaggle

Upload the project as a Kaggle dataset (`autoviolate-cv`), then run `kaggle_notebook.py` cells in a GPU notebook.

---

## Technical Choices & Innovations

### Detection: YOLOv8-L
- **Why**: Best mAP/latency tradeoff; anchor-free decoupled head reduces false positives; natively supports FP16, ONNX/TensorRT export; ~53 mAP@50 on COCO with ~15ms T4 latency.
- **Alternative considered**: RT-DETR (transformer-based, higher accuracy, ~2x slower) — included as optional ensemble via WBF.

### Classification: EfficientNet-B3
- **Why**: Compound scaling (depth × width × resolution) achieves best accuracy per FLOP. 12M params vs ~60M for ResNet-50 at comparable accuracy.
- **Innovation**: Per-class alpha-weighted Focal Loss with γ=2 handles long-tail violation distribution (rare classes like triple riding get 3× weight).

### Training Innovations
| Technique | Purpose |
|-----------|---------|
| MixUp (α=0.2) | Smooth class boundaries, reduce overfitting |
| CutMix (α=1.0) | Occlusion robustness (vehicles obscured by others) |
| EMA (decay=0.9999) | Stabilise training, improve val mAP by ~1-2% |
| Mosaic Augmentation | Handle dense traffic scenes (4-image mosaic) |
| Rain/Noise Simulation | Adverse weather robustness without real data |
| Label Smoothing (ε=0.1) | Prevent overconfident predictions |
| Temperature Scaling | Post-hoc calibration of confidence scores |

### LPR Pipeline Innovation: LightSRNet
- Custom 4× super-resolution network (ESRGAN-inspired, RDB blocks) applied to small/low-res plate crops before OCR.
- Reduces OCR error rate on low-resolution cameras by ~30-40% (estimated).

### Scalability
| GPU | Model | Expected FPS | Cameras Supported |
|-----|-------|-------------|-------------------|
| T4 (Kaggle) | YOLOv8-L FP16 | ~60 FPS | ~60 live feeds |
| T4 (Kaggle) | YOLOv8-n FP16 | ~200 FPS | ~200 live feeds |
| CPU (8-core) | YOLOv8-n INT8 | ~15 FPS | ~5 live feeds |

For real-time deployment: export to TensorRT with `detector.export("engine")`.

---

## Expected Performance (with real traffic violation dataset)

| Metric | Expected Value |
|--------|---------------|
| mAP@50 (detection) | 0.55 – 0.70 |
| mAP@75 (detection) | 0.30 – 0.45 |
| Classification Accuracy | 78 – 88% |
| Macro F1 | 0.72 – 0.85 |
| Plate Recognition Rate | 65 – 85% |
| Inference Latency (T4, FP16) | 15 – 25ms |
| Throughput | 40 – 65 FPS |

> Values depend heavily on dataset quality, annotation accuracy, and class balance.

---

## Output Files

After running `showcase.py`:

| File | Description |
|------|-------------|
| `showcase_output/detect_*.jpg` | Annotated detection images |
| `showcase_output/reports/dashboard.html` | Interactive Plotly dashboard |
| `showcase_output/reports/violation_records.csv` | Searchable violation log |
| `showcase_output/reports/report.pdf` | PDF summary |
| `showcase_output/metric_dashboard.png` | Static metric chart (4-panel) |
| `showcase_output/confusion_matrix.png` | Classification confusion matrix |
| `showcase_output/evaluation_report.json` | Full JSON metrics |
| `showcase_output/demo_*_original.jpg` | Original scene images |
| `showcase_output/demo_*_processed.jpg` | Preprocessed scene images |

---

## Team

**AutoViolate** | Flipkart Gridhackathon Round 2 — Theme 3

---

*"Every traffic violation is a preventable tragedy. Technology should make enforcement faster, fairer, and more consistent."*
