"""
tracker.py
Multi-Object Tracker with:
  - Majority-vote category stabilization
  - Distance smoothing & relative speed calculation (m/s and km/h)
  - Time-to-Collision (TTC) estimation
  - Multi-factor Risk Matrix (Distance + TTC + Speed + Lane awareness)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import deque
import numpy as np
import time

from fusion import FusedObject, classify_lane


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


@dataclass
class Track:
    track_id: int
    class_name: str
    box: tuple
    center: tuple

    # History buffers
    raw_class_history: deque = field(default_factory=lambda: deque(maxlen=20))
    distance_history: deque = field(default_factory=lambda: deque(maxlen=15))
    time_history: deque = field(default_factory=lambda: deque(maxlen=15))

    # Computed metrics
    smoothed_distance_m: float = 0.0
    relative_speed_m_s: float = 0.0    # positive = closing in / approaching
    relative_speed_kmh: float = 0.0
    ttc_s: float = float('inf')         # Time to collision in seconds
    lane_position: str = "EGO LANE"
    risk_level: str = "safe"           # "safe", "caution", "warning", "critical"

    age: int = 0
    missed_frames: int = 0
    last_velocity: tuple = (0.0, 0.0)   # (dx, dy) per frame

    def update_class_vote(self, detected_class: str, confidence: float):
        """Category voting / class stabilization across frames."""
        self.raw_class_history.append((detected_class, confidence))
        class_weights = {}
        for cls, conf in self.raw_class_history:
            class_weights[cls] = class_weights.get(cls, 0.0) + conf

        best_cls = max(class_weights.keys(), key=lambda c: class_weights[c])
        self.class_name = best_cls

    def update_motion_and_risk(self, new_dist: float, new_center: tuple, frame_dt: float = 0.1):
        """Calculate smoothed distance, relative speed, TTC, and risk level."""
        now = time.time()
        self.time_history.append(now)

        if self.smoothed_distance_m == 0.0:
            self.smoothed_distance_m = new_dist
        else:
            self.smoothed_distance_m = 0.35 * new_dist + 0.65 * self.smoothed_distance_m

        self.distance_history.append(new_dist)

        if len(self.distance_history) >= 3:
            dist_list = list(self.distance_history)
            d_start = dist_list[0]
            d_end = dist_list[-1]
            n_steps = len(dist_list) - 1
            dt = n_steps * frame_dt

            instant_speed = (d_start - d_end) / max(dt, 1e-3)
            self.relative_speed_m_s = 0.30 * instant_speed + 0.70 * self.relative_speed_m_s
        else:
            self.relative_speed_m_s = 0.0

        self.relative_speed_kmh = self.relative_speed_m_s * 3.6

        if self.relative_speed_m_s > 0.3:
            raw_ttc = self.smoothed_distance_m / max(self.relative_speed_m_s, 1e-3)
            if self.ttc_s == float('inf'):
                self.ttc_s = raw_ttc
            else:
                self.ttc_s = 0.40 * raw_ttc + 0.60 * self.ttc_s
        else:
            self.ttc_s = float('inf')

        self.risk_level = self._compute_risk_level()

    def _compute_risk_level(self) -> str:
        dist = self.smoothed_distance_m
        spd  = self.relative_speed_m_s
        ttc  = self.ttc_s
        lane = self.lane_position

        if lane == "EGO LANE":
            if (ttc <= 2.5) or (dist <= 4.5 and spd > 0.4) or (dist <= 3.2):
                return "critical"
            elif (ttc <= 4.5) or (dist <= 7.5 and spd > 0.2) or (dist <= 5.5):
                return "warning"
            elif (ttc <= 7.0) or (dist <= 11.0 and spd > 0.0) or (dist <= 8.5):
                return "caution"
            else:
                return "safe"
        elif lane in ("LEFT LANE", "RIGHT LANE"):
            if (ttc <= 2.2 and spd > 1.2) or (dist <= 3.5 and spd > 1.0):
                return "warning"
            elif (ttc <= 4.5 and spd > 0.5) or (dist <= 6.5 and spd > 0.2):
                return "caution"
            else:
                return "safe"
        else: # OFF-ROAD
            if dist <= 3.0:
                return "caution"
            return "safe"

    @property
    def movement_status(self) -> str:
        if self.relative_speed_m_s > 0.5:
            return "approaching"
        elif self.relative_speed_m_s < -0.5:
            return "receding"
        return "stable"


class CentroidIOUTracker:
    """
    Robust multi-object tracker combining IoU and centroid distance matching.
    """

    def __init__(self, iou_threshold: float = 0.25, max_centroid_dist_px: float = 90.0,
                 max_missed_frames: int = 12):
        self.iou_threshold = iou_threshold
        self.max_centroid_dist_px = max_centroid_dist_px
        self.max_missed_frames = max_missed_frames
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _centroid_dist(self, c1: tuple, c2: tuple) -> float:
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def update(self, fused_objects: List[FusedObject], frame_w: int = 960, frame_h: int = 540) -> List[FusedObject]:
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = list(range(len(fused_objects)))
        matches = []

        # 1. Match by IoU first
        for tid in list(unmatched_tracks):
            track = self.tracks[tid]
            best_det, best_iou = None, self.iou_threshold
            for di in unmatched_dets:
                det = fused_objects[di]
                if det.class_name != track.class_name and iou(track.box, det.box) < 0.40:
                    continue
                score = iou(track.box, det.box)
                if score > best_iou:
                    best_iou, best_det = score, di
            if best_det is not None:
                matches.append((tid, best_det))
                unmatched_tracks.discard(tid)
                unmatched_dets.remove(best_det)

        # 2. Match by centroid distance
        for tid in list(unmatched_tracks):
            track = self.tracks[tid]
            best_det, best_dist = None, self.max_centroid_dist_px
            for di in unmatched_dets:
                det = fused_objects[di]
                d = self._centroid_dist(track.center, det.center)
                if d < best_dist:
                    best_dist, best_det = d, di
            if best_det is not None:
                matches.append((tid, best_det))
                unmatched_tracks.discard(tid)
                unmatched_dets.remove(best_det)

        # 3. Update matched tracks
        for tid, di in matches:
            det = fused_objects[di]
            track = self.tracks[tid]

            dx = det.center[0] - track.center[0]
            dy = det.center[1] - track.center[1]
            track.last_velocity = (dx, dy)

            track.box = det.box
            track.center = det.center
            track.lane_position = classify_lane(det.center[0], det.center[1], frame_w, frame_h)

            track.update_class_vote(det.class_name, det.confidence)
            det.class_name = track.class_name

            track.update_motion_and_risk(det.distance_m, det.center)

            track.age += 1
            track.missed_frames = 0
            det.track_id = tid

            det.smoothed_distance_m = round(track.smoothed_distance_m, 1)
            det.relative_speed_m_s = round(track.relative_speed_m_s, 2)
            det.relative_speed_kmh = round(track.relative_speed_kmh, 1)
            det.ttc_s = round(track.ttc_s, 1) if track.ttc_s != float('inf') else float('inf')
            det.lane_position = track.lane_position
            det.risk_level = track.risk_level

        # 4. Handle unmatched tracks
        for tid in unmatched_tracks:
            track = self.tracks[tid]
            track.missed_frames += 1
            dx, dy = track.last_velocity
            cx, cy = track.center
            track.center = (cx + dx, cy + dy)

            if track.missed_frames > self.max_missed_frames:
                del self.tracks[tid]

        # 5. Create new tracks for unmatched detections
        for di in unmatched_dets:
            det = fused_objects[di]
            tid = self._next_id
            self._next_id += 1

            lane = classify_lane(det.center[0], det.center[1], frame_w, frame_h)
            new_track = Track(
                track_id=tid,
                class_name=det.class_name,
                box=det.box,
                center=det.center,
                lane_position=lane,
            )
            new_track.update_class_vote(det.class_name, det.confidence)
            new_track.update_motion_and_risk(det.distance_m, det.center)

            self.tracks[tid] = new_track
            det.track_id = tid

            det.smoothed_distance_m = round(new_track.smoothed_distance_m, 1)
            det.relative_speed_m_s = round(new_track.relative_speed_m_s, 2)
            det.relative_speed_kmh = round(new_track.relative_speed_kmh, 1)
            det.ttc_s = round(new_track.ttc_s, 1) if new_track.ttc_s != float('inf') else float('inf')
            det.lane_position = new_track.lane_position
            det.risk_level = new_track.risk_level

        return fused_objects

    def get_movement_status(self, track_id: int) -> str:
        track = self.tracks.get(track_id)
        return track.movement_status if track else "stable"
