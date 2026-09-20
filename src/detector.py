"""
detector.py
Step 1: Real-time object detection using a YOLO model.

Detects road-relevant objects (cars, trucks, buses, motorcycles, bicycles,
pedestrians, traffic lights, stop signs) and returns clean, structured
detections for the rest of the pipeline to consume.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

# COCO class names relevant for autonomous driving perception
RELEVANT_CLASSES = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
}

# Average real-world object heights in meters, used for monocular pinhole distance estimation
REAL_WORLD_HEIGHT_M = {
    "pedestrian": 1.7,
    "bicycle": 1.1,
    "car": 1.55,
    "motorcycle": 1.3,
    "rickshaw": 1.8,
    "bus": 3.2,
    "truck": 3.0,
    "traffic_light": 0.9,   # housing only
    "stop_sign": 0.75,
}


@dataclass
class Detection:
    class_name: str          # normalized name, e.g. "pedestrian"
    confidence: float        # 0-1
    box: tuple                # (x1, y1, x2, y2) in pixel coords
    box_height_px: float
    box_width_px: float
    center: tuple              # (cx, cy) in pixel coords


class ObjectDetector:
    """
    Wrapper around Ultralytics YOLO model filtered down to driving-relevant classes
    with auto-rickshaw reclassification and rider suppression heuristics.
    """

    def __init__(self, model_path: str = "yolo11s.pt", conf_threshold: float = 0.50,
                 device: str = "cpu", imgsz: int = 960):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print("[INFO] CUDA requested but PyTorch CUDA is not available on this environment. Running on CPU.")
            device = "cpu"
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.
        Returns a list of Detection objects for relevant classes only.
        """
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device,
            imgsz=self.imgsz, verbose=False
        )[0]

        detections = []
        names = results.names
        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_name = names[cls_id]
            if raw_name not in RELEVANT_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            norm_name = RELEVANT_CLASSES[raw_name]

            # --- AUTO-RICKSHAW RECLASSIFICATION HEURISTIC ---
            if norm_name in ("truck", "car"):
                h_px = y2 - y1
                w_px = x2 - x1
                aspect_ratio = h_px / max(w_px, 1e-3)
                if 1.05 <= aspect_ratio <= 2.2 and w_px < 350 and h_px < 400:
                    norm_name = "rickshaw"

            detections.append(Detection(
                class_name=norm_name,
                confidence=conf,
                box=(x1, y1, x2, y2),
                box_height_px=max(y2 - y1, 1e-3),
                box_width_px=max(x2 - x1, 1e-3),
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            ))

        # --- RIDER SUPPRESSION / MERGING HEURISTIC ---
        filtered_detections = []
        vehicles = [d for d in detections if d.class_name in ("motorcycle", "bicycle")]
        
        for d in detections:
            if d.class_name == "pedestrian":
                is_rider = False
                px1, py1, px2, py2 = d.box
                p_area = (px2 - px1) * (py2 - py1)
                
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v.box
                    ix1, iy1 = max(px1, vx1), max(py1, vy1)
                    ix2, iy2 = min(px2, vx2), min(py2, vy2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        if inter_area / max(p_area, 1e-6) > 0.35:
                            is_rider = True
                            break
                if not is_rider:
                    filtered_detections.append(d)
            else:
                filtered_detections.append(d)

        return filtered_detections
