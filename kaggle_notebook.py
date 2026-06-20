# AutoViolate-CV Kaggle Notebook
# Flipkart Gridhackathon Round 2 | Theme 3
# Run all cells top-to-bottom in a Kaggle GPU (T4/P100) notebook

# ============================================================
# CELL 1: Environment Setup
# ============================================================
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pkgs = [
    "ultralytics>=8.2.0", "timm>=0.9.12", "easyocr>=1.7.1",
    "ensemble-boxes>=1.0.9", "fpdf2>=2.7.6",
]
for p in pkgs:
    install(p)

import os, sys
sys.path.insert(0, "/kaggle/working/autoviolate")

# ============================================================
# CELL 2: Clone / Copy project files
# ============================================================
# If submitting as a Kaggle dataset, files are already at:
# /kaggle/input/autoviolate-cv/
# Otherwise clone from GitHub:
# !git clone https://github.com/your-team/autoviolate-cv.git /kaggle/working/autoviolate

# ============================================================
# CELL 3: Data Setup
# ============================================================
# Option A: Use Roboflow API to download traffic violation dataset
try:
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
    project = rf.workspace("traffic-violations").project("traffic-violation-detection")
    dataset = project.version(1).download("yolov8")
    DATA_ROOT = dataset.location
except Exception as e:
    print(f"Roboflow not available ({e}), generating synthetic data...")
    from src.data.dataset import generate_synthetic_dataset
    DATA_ROOT = generate_synthetic_dataset("/kaggle/working/synthetic_data", n_images=500)

print(f"Data root: {DATA_ROOT}")

# ============================================================
# CELL 4: Generate data.yaml
# ============================================================
import yaml, os
from src.models.detector import generate_data_yaml
from src.data.dataset import CLASS_NAMES

data_yaml_path = "/kaggle/working/data.yaml"
generate_data_yaml(DATA_ROOT, CLASS_NAMES, data_yaml_path)
print("data.yaml created:")
print(open(data_yaml_path).read())

# ============================================================
# CELL 5: Load Configuration
# ============================================================
with open("/kaggle/working/autoviolate/config/config.yaml") as f:
    cfg = yaml.safe_load(f)

# Override paths for Kaggle
cfg["paths"]["data_root"]    = DATA_ROOT
cfg["paths"]["train_images"] = os.path.join(DATA_ROOT, "train/images")
cfg["paths"]["train_labels"] = os.path.join(DATA_ROOT, "train/labels")
cfg["paths"]["val_images"]   = os.path.join(DATA_ROOT, "valid/images")
cfg["paths"]["val_labels"]   = os.path.join(DATA_ROOT, "valid/labels")
cfg["paths"]["output_dir"]   = "/kaggle/working/outputs"
cfg["paths"]["model_dir"]    = "/kaggle/working/models"
cfg["paths"]["report_dir"]   = "/kaggle/working/reports"
cfg["training"]["epochs"]    = 30   # reduce for Kaggle time limit
cfg["training"]["batch_size"] = 8

# ============================================================
# CELL 6: Train Detection Model
# ============================================================
from src.models.detector import TrafficDetector
detector = TrafficDetector.from_config(cfg)
if detector._backend == "ultralytics":
    detector.train(
        data_yaml=data_yaml_path,
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch_size"],
        img_size=640,
        project=cfg["paths"]["model_dir"],
        name="detect_run",
    )

# ============================================================
# CELL 7: Run Showcase (full pipeline demo)
# ============================================================
from showcase import Showcase
showcase = Showcase(cfg, out_dir="/kaggle/working/showcase_output")
showcase.run_all()

# ============================================================
# CELL 8: View Dashboard
# ============================================================
from IPython.display import IFrame, display
display(IFrame("/kaggle/working/showcase_output/reports/dashboard.html", width=1100, height=750))

# ============================================================
# CELL 9: Display Annotated Images
# ============================================================
import cv2, matplotlib.pyplot as plt
import glob

annotated_imgs = sorted(glob.glob("/kaggle/working/showcase_output/detect_*.jpg"))
fig, axes = plt.subplots(1, min(5, len(annotated_imgs)), figsize=(20, 5))
for ax, path in zip(axes, annotated_imgs[:5]):
    img = cv2.imread(path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title(os.path.basename(path))
plt.suptitle("AutoViolate-CV Detection Results", fontsize=14)
plt.tight_layout()
plt.savefig("/kaggle/working/gallery.png", dpi=150)
plt.show()

# ============================================================
# CELL 10: Show Metric Dashboard
# ============================================================
from IPython.display import Image
Image("/kaggle/working/showcase_output/metric_dashboard.png")
