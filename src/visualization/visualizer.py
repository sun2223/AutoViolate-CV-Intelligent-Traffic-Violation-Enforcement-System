"""
Visualization Module
======================
Functions:
  - draw_violations_on_image
  - plot_confusion_matrix
  - plot_pr_curves
  - plot_metric_dashboard (static PNG)
  - save_annotated_gallery
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Colour palette (per severity)
# ---------------------------------------------------------------------------

SEVERITY_COLOURS = {
    "critical": (0, 0, 255),
    "high":     (0, 80, 255),
    "medium":   (0, 165, 255),
    "low":      (0, 220, 50),
    "info":     (180, 180, 180),
    "unknown":  (255, 255, 0),
}

CLASS_COLOURS = [
    (180, 180, 180), (0, 80, 255), (0, 165, 255),
    (0, 0, 255),     (0, 0, 200), (0, 220, 50),
    (0, 0, 255),     (255, 140, 0),
]


# ---------------------------------------------------------------------------
# Core annotation function
# ---------------------------------------------------------------------------

def draw_violations(image: np.ndarray, violations: List[Dict],
                     show_confidence: bool = True,
                     show_plate: bool = True) -> np.ndarray:
    """
    Draw violation bounding boxes with labels on a BGR image.
    violations: list of dicts with keys: class_name, confidence, severity, bbox, plate_text
    """
    vis = image.copy()
    for v in violations:
        bbox  = v.get("bbox", [0,0,100,100])
        sev   = v.get("severity", "unknown")
        col   = SEVERITY_COLOURS.get(sev, (255, 255, 0))
        x1, y1, x2, y2 = [int(c) for c in bbox]

        # Main box
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        # Corner accents
        L = max(12, (x2-x1)//8)
        for dx1, dy1, dx2, dy2 in [
            (x1, y1, x1+L, y1), (x1, y1, x1, y1+L),
            (x2-L, y1, x2, y1), (x2, y1, x2, y1+L),
            (x1, y2-L, x1, y2), (x1, y2, x1+L, y2),
            (x2-L, y2, x2, y2), (x2, y2-L, x2, y2),
        ]:
            cv2.line(vis, (dx1, dy1), (dx2, dy2), col, 3)

        # Label text
        name   = v.get("class_name", "violation")
        conf   = v.get("confidence", 0.0)
        plate  = v.get("plate_text", "")
        label  = name.replace("_", " ").title()
        if show_confidence:
            label += f" {conf:.2f}"
        if show_plate and plate:
            label += f"  🚘 {plate}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Background pill
        cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 8, y1), col, -1)
        cv2.putText(vis, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Overlay: violation count
    n = len(violations)
    cv2.rectangle(vis, (0, 0), (220, 36), (20, 20, 40), -1)
    cv2.putText(vis, f"Violations: {n}", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (233, 69, 96), 2)
    return vis


# ---------------------------------------------------------------------------
# Matplotlib-based static charts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: List[List[int]], class_names: List[str],
                           out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cm_arr = np.array(cm, dtype=np.float32)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = cm_arr / np.where(row_sums == 0, 1, row_sums)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        n = len(class_names)
        ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Normalised Confusion Matrix", fontsize=13, fontweight="bold")

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm_arr[i,j]:.0f}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i,j] > 0.6 else "black")

        fig.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"[Viz] Confusion matrix → {out_path}")
        plt.close(fig)
        return fig
    except ImportError:
        print("[Viz] matplotlib not available; skipping confusion matrix plot.")


def plot_pr_curves(per_class_ap: Dict[str, float], out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = list(per_class_ap.keys())
        aps     = [per_class_ap[c] for c in classes]
        colours = plt.cm.Set2(np.linspace(0, 1, len(classes))) if hasattr(plt, 'cm') else None

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(classes, aps, color=colours if colours is not None else "steelblue")
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Average Precision (AP)")
        ax.set_title("Per-Class AP @ IoU=0.50", fontsize=13, fontweight="bold")
        ax.axvline(np.mean(aps), color="red", linestyle="--", label=f"mAP={np.mean(aps):.3f}")
        ax.legend()
        fig.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"[Viz] PR curve → {out_path}")
        plt.close(fig)
    except ImportError:
        print("[Viz] matplotlib not available.")


def plot_metric_dashboard(metrics: Dict, out_path: str = None):
    """
    Four-panel dashboard: accuracy bars, confusion matrix preview, AP bars, latency gauge.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("AutoViolate-CV Performance Dashboard", fontsize=15, fontweight="bold", y=0.98)

        # 1. Classification metrics bar
        ax = axes[0, 0]
        cls_m = metrics.get("classification", {})
        m_keys = ["overall_accuracy", "macro_precision", "macro_recall", "macro_f1"]
        m_vals = [cls_m.get(k, 0) for k in m_keys]
        ax.bar([k.replace("macro_", "").replace("overall_", "").title() for k in m_keys],
               m_vals, color=["#4CAF50", "#2196F3", "#FF9800", "#E91E63"])
        ax.set_ylim(0, 1.15); ax.set_title("Classification Metrics", fontsize=11)
        for bar, val in zip(ax.patches, m_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                    ha="center", fontsize=10)

        # 2. mAP bars
        ax = axes[0, 1]
        iou_d = metrics.get("detection", {}).get("iou_thresholds", {})
        iou_keys = list(iou_d.keys())
        iou_vals = [iou_d[k]["map"] for k in iou_keys]
        ax.bar(iou_keys, iou_vals, color=["#9C27B0", "#3F51B5"])
        ax.set_ylim(0, 1.15); ax.set_title("Detection mAP", fontsize=11)
        for bar, val in zip(ax.patches, iou_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontsize=12)

        # 3. Per-class AP (first available IoU threshold)
        ax = axes[1, 0]
        if iou_keys:
            pc_ap = iou_d[iou_keys[0]].get("per_class_ap", {})
            names  = list(pc_ap.keys())
            vals   = list(pc_ap.values())
            cols   = plt.cm.tab10(np.linspace(0, 1, len(names)))
            ax.barh(names, vals, color=cols)
            ax.set_xlim(0, 1.1); ax.set_title("Per-Class AP @ IoU=0.50", fontsize=11)
            for bar, val in zip(ax.patches, vals):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=9)

        # 4. Efficiency
        ax = axes[1, 1]
        eff = metrics.get("efficiency", {}).get("single_image", {})
        eff_keys = ["latency_mean_ms", "latency_p95_ms", "latency_p99_ms"]
        eff_vals = [eff.get(k, 0) for k in eff_keys]
        eff_labels = ["Mean (ms)", "P95 (ms)", "P99 (ms)"]
        ax.bar(eff_labels, eff_vals, color=["#00BCD4", "#009688", "#4DB6AC"])
        ax.set_title(f"Inference Latency — {eff.get('throughput_fps', 0)} FPS", fontsize=11)
        for bar, val in zip(ax.patches, eff_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}", ha="center", fontsize=10)

        fig.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"[Viz] Dashboard → {out_path}")
        plt.close(fig)
    except ImportError:
        print("[Viz] matplotlib not available.")


# ---------------------------------------------------------------------------
# Annotated gallery saver
# ---------------------------------------------------------------------------

def save_annotated_gallery(images: List[np.ndarray], results: List,
                             out_dir: str, prefix: str = "annotated"):
    """Save annotated images to disk as a gallery."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, (img, result) in enumerate(zip(images, results)):
        violations = [v.to_dict() for v in result.violations] if hasattr(result, "violations") else []
        annotated = draw_violations(img, violations)
        path = out / f"{prefix}_{i:05d}.jpg"
        cv2.imwrite(str(path), annotated)
    print(f"[Gallery] Saved {len(images)} annotated images → {out_dir}")
