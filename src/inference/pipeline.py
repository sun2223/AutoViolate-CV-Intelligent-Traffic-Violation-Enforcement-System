"""
Inference Engine - Unified Pipeline Orchestrator
=================================================
Integrates: Preprocessor → Detector → Classifier → LPR
Supports: single image, batch, video stream.
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict, field


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class ViolationRecord:
    image_id:      str
    timestamp:     float
    class_id:      int
    class_name:    str
    confidence:    float
    severity:      str
    fine_inr:      int
    bbox:          List[int]
    plate_text:    Optional[str] = None
    plate_valid:   bool = False
    frame_index:   int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FrameResult:
    image_id:        str
    timestamp:       float
    detections:      List[Dict]           = field(default_factory=list)
    violations:      List[ViolationRecord] = field(default_factory=list)
    plate_records:   List[Dict]           = field(default_factory=list)
    inference_ms:    float = 0.0
    preprocess_ms:   float = 0.0


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class TrafficViolationInference:
    """
    Full inference pipeline:
      1. Preprocess image (enhancement + normalisation)
      2. Run detector (YOLOv8) on original BGR image
      3. For each detection ROI: classify violation
      4. Run LPR on entire image
      5. Return structured FrameResult

    The detector operates on the original (enhanced but not normalised) image
    to preserve pixel value range expected by YOLOv8.

    Usage
    -----
    pipeline = TrafficViolationInference.from_config(cfg)
    result   = pipeline.infer(image_bgr, image_id="img_001")
    """

    def __init__(self, preprocessor, detector, classifier, lpr,
                 skip_classes: List[int] = None):
        self.preprocessor = preprocessor
        self.detector     = detector
        self.classifier   = classifier
        self.lpr          = lpr
        # Classes that are detections only, not violations (e.g. class 0 = vehicle)
        self.skip_classes = skip_classes or [0]

    @classmethod
    def from_config(cls, cfg: dict) -> "TrafficViolationInference":
        from src.preprocessing.image_enhancer import TrafficImagePreprocessor
        from src.models.detector import TrafficDetector
        from src.models.classifier import ViolationClassifier
        from src.lpr.lpr_pipeline import LicensePlateRecognizer

        pp  = TrafficImagePreprocessor.from_config(cfg)
        det = TrafficDetector.from_config(cfg)
        clf = ViolationClassifier.from_config(cfg)
        lpr = LicensePlateRecognizer.from_config(cfg)
        return cls(pp, det, clf, lpr)

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Run only enhancement (no normalisation) for detector input."""
        from src.preprocessing.image_enhancer import LowLightEnhancer, ShadowRemover
        ll = LowLightEnhancer()
        sh = ShadowRemover()
        image = ll.enhance(image)
        image = sh.remove(image)
        return image

    def infer(self, image: np.ndarray, image_id: str = "unknown",
              frame_index: int = 0) -> FrameResult:
        ts = time.time()

        # --- Preprocessing ---
        t0 = time.perf_counter()
        enhanced = self._enhance_image(image.copy())
        preprocess_ms = (time.perf_counter() - t0) * 1000

        # --- Detection ---
        t0 = time.perf_counter()
        detections = self.detector.predict(enhanced)
        detect_ms  = (time.perf_counter() - t0) * 1000

        # --- Classification + LPR ---
        violations = []
        for det in detections:
            cls_id = det.get("class_id", -1)
            if cls_id in self.skip_classes:
                continue
            bbox = det.get("bbox", [0, 0, 100, 100])
            # Classify the ROI
            clf_result = self.classifier.classify_roi(enhanced, bbox)
            v = ViolationRecord(
                image_id    = image_id,
                timestamp   = ts,
                class_id    = clf_result["class_id"],
                class_name  = clf_result["class_name"],
                confidence  = clf_result["confidence"],
                severity    = clf_result.get("severity", "unknown"),
                fine_inr    = clf_result.get("fine_inr", 0),
                bbox        = bbox,
                frame_index = frame_index,
            )
            violations.append(v)

        # --- LPR on full image ---
        plate_records = self.lpr.process(enhanced)
        for v in violations:
            # Match plate by proximity
            best_plate = self._match_plate(v.bbox, plate_records)
            if best_plate:
                v.plate_text  = best_plate["plate_text"]
                v.plate_valid = best_plate["valid"]

        inference_ms = detect_ms + (time.perf_counter() - t0) * 1000

        return FrameResult(
            image_id      = image_id,
            timestamp     = ts,
            detections    = detections,
            violations    = violations,
            plate_records = plate_records,
            inference_ms  = round(inference_ms, 2),
            preprocess_ms = round(preprocess_ms, 2),
        )

    def _match_plate(self, vehicle_bbox: List[int],
                      plates: List[Dict]) -> Optional[Dict]:
        """Match nearest plate below the vehicle bounding box."""
        vx1, vy1, vx2, vy2 = vehicle_bbox
        best, best_dist = None, float("inf")
        for p in plates:
            px1, py1, px2, py2 = p["bbox"]
            # Plate should be below vehicle centre
            plate_cy = (py1 + py2) / 2
            if plate_cy < vy1 or plate_cy > vy2 + 100:
                continue
            cx_dist = abs((px1+px2)/2 - (vx1+vx2)/2)
            if cx_dist < best_dist:
                best_dist = cx_dist
                best = p
        return best

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def infer_batch(self, images: List[np.ndarray],
                    image_ids: List[str] = None) -> List[FrameResult]:
        ids = image_ids or [f"img_{i}" for i in range(len(images))]
        return [self.infer(img, iid) for img, iid in zip(images, ids)]

    # ------------------------------------------------------------------
    # Video inference
    # ------------------------------------------------------------------

    def infer_video(self, video_path: str, out_path: Optional[str] = None,
                    max_frames: int = None, skip_frames: int = 1) -> List[FrameResult]:
        """
        Process a video file frame by frame.
        Optionally write annotated output video.
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        fps  = cap.get(cv2.CAP_PROP_FPS) or 30
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        results, frame_idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if max_frames and frame_idx >= max_frames: break
            if frame_idx % (skip_frames + 1) == 0:
                fr = self.infer(frame, f"frame_{frame_idx:06d}", frame_idx)
                results.append(fr)
                if writer:
                    annotated = self._draw_result(frame.copy(), fr)
                    writer.write(annotated)
            frame_idx += 1

        cap.release()
        if writer: writer.release()
        print(f"[Video] Processed {frame_idx} frames → {len(results)} results")
        return results

    def _draw_result(self, image: np.ndarray, result: FrameResult) -> np.ndarray:
        """Annotate an image with violation bounding boxes and labels."""
        COLOURS = {
            "critical": (0, 0, 255), "high": (0, 100, 255),
            "medium": (0, 165, 255), "low": (0, 255, 0), "info": (200, 200, 200)
        }
        for v in result.violations:
            x1, y1, x2, y2 = v.bbox
            col = COLOURS.get(v.severity, (255, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), col, 2)
            label = f"{v.class_name} {v.confidence:.2f}"
            if v.plate_text:
                label += f" | {v.plate_text}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - lh - 6), (x1 + lw, y1), col, -1)
            cv2.putText(image, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # FPS overlay
        cv2.putText(image, f"Inf: {result.inference_ms:.1f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml, sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    pipeline = TrafficViolationInference.from_config(cfg)
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = pipeline.infer(img, "test_image")
    print(f"Detections: {len(result.detections)}, Violations: {len(result.violations)}")
    print(f"Latency: {result.inference_ms}ms")
