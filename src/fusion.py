"""
fusion.py
Fuses object detections with the depth map, computes estimated real-world
distance (in meters), assigns lane positions, relative speed, TTC, and risk levels.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from detector import Detection, REAL_WORLD_HEIGHT_M


@dataclass
class FusedObject:
    class_name: str
    confidence: float
    box: tuple                    # (x1, y1, x2, y2)
    center: tuple                 # (cx, cy)
    distance_m: float             # raw metric distance
    geometric_distance_m: float
    relative_depth_value: float   # MiDaS relative depth sample
    depth_rank: int               # 1 = closest object in frame
    track_id: Optional[int] = None

    # Enriched metrics
    smoothed_distance_m: float = 0.0
    relative_speed_m_s: float = 0.0   # positive = closing in / approaching
    relative_speed_kmh: float = 0.0
    ttc_s: float = float('inf')        # Time-to-Collision in seconds
    lane_position: str = "EGO LANE"    # "EGO LANE", "LEFT LANE", "RIGHT LANE", "OFF-ROAD"
    risk_level: str = "safe"           # "safe", "caution", "warning", "critical"
    extra: dict = field(default_factory=dict)   # passthrough from Detection (e.g. tl_color)


def estimate_focal_length_px(frame_width_px: int, horizontal_fov_deg: float = 70.0) -> float:
    """Estimate focal length in pixels from frame width and FOV."""
    fov_rad = np.deg2rad(horizontal_fov_deg)
    return (frame_width_px / 2.0) / np.tan(fov_rad / 2.0)


def classify_lane(cx: float, cy: float, frame_w: int, frame_h: int) -> str:
    """
    Classify object lateral lane position relative to ego vehicle's path
    using a trapezoidal projection model tuned for driving scenes.
    """
    norm_x = cx / float(frame_w)
    norm_y = cy / float(frame_h)

    y_horizon = 0.45
    y_hood = 0.95
    t = np.clip((norm_y - y_horizon) / max(y_hood - y_horizon, 1e-3), 0.0, 1.0)

    left_bound = 0.38 + (0.18 - 0.38) * t
    right_bound = 0.62 + (0.82 - 0.62) * t

    if norm_x < left_bound - 0.15:
        return "OFF-ROAD"
    elif norm_x < left_bound:
        return "LEFT LANE"
    elif norm_x <= right_bound:
        return "EGO LANE"
    elif norm_x <= right_bound + 0.15:
        return "RIGHT LANE"
    else:
        return "OFF-ROAD"


def sample_depth_in_box(depth_map: np.ndarray, box: tuple, shrink: float = 0.4) -> float:
    """Robust median depth sample inside shrunk box region."""
    x1, y1, x2, y2 = box
    h, w = depth_map.shape[:2]
    bw, bh = (x2 - x1), (y2 - y1)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx1 = int(np.clip(cx - bw * shrink / 2, 0, w - 1))
    sx2 = int(np.clip(cx + bw * shrink / 2, 0, w - 1))
    sy1 = int(np.clip(cy - bh * shrink / 2, 0, h - 1))
    sy2 = int(np.clip(cy + bh * shrink / 2, 0, h - 1))

    if sx2 <= sx1 or sy2 <= sy1:
        sx1, sy1, sx2, sy2 = int(x1), int(y1), int(x2), int(y2)

    region = depth_map[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return float(np.median(depth_map))
    return float(np.median(region))


def fuse_detections_with_depth(
    detections: List[Detection],
    depth_map: np.ndarray,
    frame_width_px: int,
    horizontal_fov_deg: float = 70.0,
    geometric_weight: float = 0.7,
    frame_height_px: int = 540,
) -> List[FusedObject]:
    """
    Combine YOLO detections with depth map and camera geometry to produce
    fused objects with metric distance, lane position, and depth ranking.
    """
    focal_length_px = estimate_focal_length_px(frame_width_px, horizontal_fov_deg)

    raw_results = []
    for det in detections:
        real_h = REAL_WORLD_HEIGHT_M.get(det.class_name, 1.5)
        geometric_distance = (real_h * focal_length_px) / max(det.box_height_px, 1.0)
        depth_val = sample_depth_in_box(depth_map, det.box)
        raw_results.append((det, geometric_distance, depth_val))

    sorted_by_depth = sorted(raw_results, key=lambda r: -r[2])
    depth_rank_lookup = {id(r[0]): i + 1 for i, r in enumerate(sorted_by_depth)}

    depth_vals = np.array([r[2] for r in raw_results])

    fused = []
    for det, geometric_distance, depth_val in raw_results:
        if len(depth_vals) >= 2 and depth_vals.std() > 1e-6:
            scene_median_depth = np.median(depth_vals)
            relative_factor = scene_median_depth / max(depth_val, 1e-6)
            depth_informed_distance = geometric_distance * np.clip(relative_factor, 0.6, 1.6)
        else:
            depth_informed_distance = geometric_distance

        final_distance = (
            geometric_weight * geometric_distance
            + (1.0 - geometric_weight) * depth_informed_distance
        )
        final_distance = round(float(np.clip(final_distance, 0.5, 100.0)), 1)

        lane = classify_lane(det.center[0], det.center[1], frame_width_px, frame_height_px)

        fused.append(FusedObject(
            class_name=det.class_name,
            confidence=det.confidence,
            box=det.box,
            center=det.center,
            distance_m=final_distance,
            geometric_distance_m=round(float(geometric_distance), 1),
            relative_depth_value=float(depth_val),
            depth_rank=depth_rank_lookup[id(det)],
            smoothed_distance_m=final_distance,
            lane_position=lane,
            extra=getattr(det, "extra", {}),
        ))

    return fused
