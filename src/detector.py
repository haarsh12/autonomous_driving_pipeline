"""
detector.py
Step 1: Real-time object detection using a YOLO model.

Detects road-relevant objects (cars, trucks, buses, motorcycles, bicycles,
pedestrians, traffic lights, stop signs) and returns clean, structured
detections for the rest of the pipeline to consume.

Extra:
  - Traffic light color classification (RED / YELLOW / GREEN) by HSV analysis
    of the cropped bounding box region.
  - Rickshaw reclassification heuristic.
  - Rider suppression heuristic (person on bicycle/motorcycle merged).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import cv2

# COCO class names relevant for autonomous driving perception
RELEVANT_CLASSES = {
    "person":        "pedestrian",
    "bicycle":       "bicycle",
    "car":           "car",
    "motorcycle":    "motorcycle",
    "bus":           "bus",
    "truck":         "truck",
    "traffic light": "traffic_light",
    "stop sign":     "stop_sign",
}

# Average real-world object heights in metres (for monocular pinhole distance)
REAL_WORLD_HEIGHT_M = {
    "pedestrian":   1.72,
    "bicycle":      1.15,
    "car":          1.55,
    "motorcycle":   1.30,
    "rickshaw":     1.80,
    "bus":          3.20,
    "truck":        3.00,
    "traffic_light": 0.90,
    "stop_sign":    0.75,
    "building":     8.00,   # approximate average floor height × 2
}


@dataclass
class Detection:
    class_name:    str           # normalised name, e.g. "pedestrian"
    confidence:    float         # 0-1
    box:           tuple         # (x1, y1, x2, y2) in pixel coords
    box_height_px: float
    box_width_px:  float
    center:        tuple         # (cx, cy) in pixel coords
    extra:         dict = field(default_factory=dict)
    # extra may contain: {"tl_color": "RED"|"YELLOW"|"GREEN"|"UNKNOWN"}


# ─────────────────────────────────────────────────────────────────────────────
# Traffic light colour classifier
# ─────────────────────────────────────────────────────────────────────────────

# HSV ranges for each light colour
_TL_HSV = {
    "RED":    [(np.array([0,   120,  90], np.uint8), np.array([12,  255, 255], np.uint8)),
               (np.array([168, 120,  90], np.uint8), np.array([180, 255, 255], np.uint8))],
    "YELLOW": [(np.array([18,  100, 120], np.uint8), np.array([38,  255, 255], np.uint8))],
    "GREEN":  [(np.array([42,  60,  60],  np.uint8), np.array([90,  255, 255], np.uint8))],
}


def classify_traffic_light_color(frame_bgr: np.ndarray,
                                  box: tuple,
                                  min_pixels: int = 15) -> str:
    """
    Crop the top-third of a traffic light bounding box and find the
    dominant lit colour using HSV pixel counts.
    Returns 'RED', 'YELLOW', 'GREEN', or 'UNKNOWN'.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = frame_bgr.shape[:2]
    x1, x2 = max(0, x1), min(W-1, x2)
    y1, y2 = max(0, y1), min(H-1, y2)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return "UNKNOWN"

    # Look at top 40% of the box where the active light sits
    crop_y2 = y1 + max(4, int((y2 - y1) * 0.40))
    crop = frame_bgr[y1:crop_y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    counts = {}
    for color, ranges in _TL_HSV.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        counts[color] = int(mask.sum() // 255)

    best_color = max(counts, key=counts.get)
    if counts[best_color] < min_pixels:
        return "UNKNOWN"
    return best_color


# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Wrapper around Ultralytics YOLO model filtered down to driving-relevant
    classes with:
      - Auto-rickshaw reclassification heuristic
      - Rider suppression (person on bicycle/motorcycle)
      - Traffic light colour classification (RED/YELLOW/GREEN)
    """

    def __init__(self, model_path: str = "yolo11s.pt",
                 conf_threshold: float = 0.45,
                 device: str = "cpu",
                 imgsz: int = 640):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print("[INFO] CUDA requested but not available — falling back to CPU.")
            device = "cpu"
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz  = imgsz
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    # ── Detection ────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.
        Returns a filtered, enriched list of Detection objects.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False
        )[0]

        raw = []
        names = results.names

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            raw_name = names[cls_id]
            if raw_name not in RELEVANT_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf      = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            norm_name = RELEVANT_CLASSES[raw_name]

            # ── Rickshaw reclassification ─────────────────────────────────
            if norm_name in ("truck", "car"):
                h_px = y2 - y1
                w_px = x2 - x1
                ar   = h_px / max(w_px, 1e-3)
                if 1.05 <= ar <= 2.2 and w_px < 350 and h_px < 400:
                    norm_name = "rickshaw"

            # ── Traffic light colour ───────────────────────────────────────
            extra = {}
            if norm_name == "traffic_light":
                extra["tl_color"] = classify_traffic_light_color(
                    frame, (x1, y1, x2, y2)
                )

            raw.append(Detection(
                class_name=norm_name,
                confidence=conf,
                box=(x1, y1, x2, y2),
                box_height_px=max(y2 - y1, 1e-3),
                box_width_px=max(x2 - x1, 1e-3),
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                extra=extra,
            ))

        # ── Rider suppression ─────────────────────────────────────────────
        # Only suppress if overlap > 55% (was 35% — too aggressive)
        vehicles  = [d for d in raw if d.class_name in ("motorcycle", "bicycle")]
        filtered  = []

        for d in raw:
            if d.class_name == "pedestrian":
                px1, py1, px2, py2 = d.box
                p_area = (px2 - px1) * (py2 - py1)
                is_rider = False
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v.box
                    ix1 = max(px1, vx1); iy1 = max(py1, vy1)
                    ix2 = min(px2, vx2); iy2 = min(py2, vy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / max(p_area, 1e-6) > 0.55:   # raised from 0.35
                            is_rider = True
                            break
                if not is_rider:
                    filtered.append(d)
            else:
                filtered.append(d)

        return filtered
