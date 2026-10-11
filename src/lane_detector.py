"""Image-space, ground-aligned lane-marking detector.

The previous implementation projected every frame through one fixed bird's-eye
trapezoid and then filled that trapezoid. A fixed calibration is not valid for
all camera positions, bends, or road widths, so a weak edge response could turn
into a large synthetic lane that was visibly detached from the road.

This detector only draws a corridor after it has found a geometrically valid
left/right pair of *image-space* lane-marking candidates. The boundary points
therefore lie on the source frame's road markings. A short temporal hold keeps
the display stable through an occlusion, but there is intentionally no made-up
fallback corridor when a lane has never been detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _LineCandidate:
    """A lane-line candidate in ``x = slope * y + intercept`` form."""

    slope: float
    intercept: float
    score: float
    support: float
    min_y: float
    max_y: float

    def x_at(self, y: float) -> float:
        return self.slope * y + self.intercept


class LaneDetector:
    """Detect an ego-lane corridor from visible road markings."""

    def __init__(self, smooth_alpha: float = 0.30, hold_frames: int = 10):
        self.alpha = float(np.clip(smooth_alpha, 0.05, 1.0))
        self.hold_frames = max(0, int(hold_frames))

        # Normalised coefficients let one detector survive a resolution change.
        self._left_line_norm: Optional[Tuple[float, float]] = None
        self._right_line_norm: Optional[Tuple[float, float]] = None
        self._missed_frames = 0

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _road_roi(height: int, width: int) -> np.ndarray:
        """Return a conservative trapezoid covering the road ahead."""
        return np.array(
            [
                (int(width * 0.27), int(height * 0.53)),
                (int(width * 0.73), int(height * 0.53)),
                (width - 1, height - 1),
                (0, height - 1),
            ],
            dtype=np.int32,
        )

    def _lane_feature_mask(
        self, frame_bgr: np.ndarray, vehicle_boxes: Optional[List[tuple]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return edge/color features and a stricter painted-marking mask."""
        height, width = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # White and yellow are the usual lane-marking colours. The white range
        # allows shadowed paint while excluding most saturated vehicle colours.
        white = cv2.inRange(
            hsv,
            np.array([0, 0, 135], dtype=np.uint8),
            np.array([180, 105, 255], dtype=np.uint8),
        )
        yellow = cv2.inRange(
            hsv,
            np.array([10, 38, 55], dtype=np.uint8),
            np.array([48, 255, 255], dtype=np.uint8),
        )
        painted = cv2.bitwise_or(white, yellow)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 35, 100)
        # Hough voting stays edge-only.  Adding every bright colour pixel here
        # overwhelms short dashed markings with textured paint, shadows, and
        # vehicle reflections; colour is used later as a support score instead.
        features = edges

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [self._road_roi(height, width)], 255)
        features = cv2.bitwise_and(features, roi_mask)
        painted = cv2.bitwise_and(painted, roi_mask)

        # A car grille, tyre edge, or bumper is a very strong false lane cue.
        # Removing an expanded box is safer than letting it bend the lane fit.
        if vehicle_boxes:
            for box in vehicle_boxes:
                x1, y1, x2, y2 = (int(v) for v in box)
                pad_x = max(6, int((x2 - x1) * 0.08))
                pad_y = max(5, int((y2 - y1) * 0.06))
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(width - 1, x2 + pad_x)
                y2 = min(height - 1, y2 + pad_y)
                cv2.rectangle(features, (x1, y1), (x2, y2), 0, -1)
                cv2.rectangle(painted, (x1, y1), (x2, y2), 0, -1)

        return features, painted

    # ------------------------------------------------------------------
    # Candidate fitting and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_support(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        """Measure how much of a Hough segment sits on painted road pixels."""
        steps = max(12, int(np.hypot(x2 - x1, y2 - y1) / 4))
        xs = np.clip(np.linspace(x1, x2, steps).round().astype(np.int32), 0, mask.shape[1] - 1)
        ys = np.clip(np.linspace(y1, y2, steps).round().astype(np.int32), 0, mask.shape[0] - 1)
        return float(np.count_nonzero(mask[ys, xs]) / steps)

    @staticmethod
    def _dedupe_candidates(candidates: List[_LineCandidate], width: int) -> List[_LineCandidate]:
        """Keep one representative for repeated Hough fragments of one line."""
        selected: List[_LineCandidate] = []
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            duplicate = any(
                abs(candidate.slope - kept.slope) < 0.16
                and abs(candidate.intercept - kept.intercept) < width * 0.07
                for kept in selected
            )
            if not duplicate:
                selected.append(candidate)
            if len(selected) == 24:
                break
        return selected

    def _find_candidates(
        self, features: np.ndarray, painted: np.ndarray
    ) -> Tuple[List[_LineCandidate], List[_LineCandidate]]:
        """Find plausible left and right marking segments in the lower road ROI."""
        height, width = features.shape[:2]
        lines = cv2.HoughLinesP(
            features,
            rho=1,
            theta=np.pi / 180,
            threshold=12,
            minLineLength=max(12, int(width * 0.010)),
            maxLineGap=max(20, int(width * 0.019)),
        )
        if lines is None:
            return [], []

        # Dilating by a few pixels lets either edge of a painted stripe receive
        # credit for the nearby white/yellow paint.
        paint_support = cv2.dilate(painted, np.ones((5, 5), dtype=np.uint8))
        left: List[_LineCandidate] = []
        right: List[_LineCandidate] = []
        y_base = height * 0.96
        y_reference = height * 0.58

        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            dy = y2 - y1
            if abs(dy) < 4:
                continue

            slope = (x2 - x1) / float(dy)  # dx / dy, not dy / dx
            if not 0.12 <= abs(slope) <= 2.5:
                continue

            min_y, max_y = min(y1, y2), max(y1, y2)
            # Keep evidence on the road close to the vehicle. This rejects
            # building edges and traffic-light supports above the asphalt.
            if min_y < height * 0.60 or max_y < height * 0.75:
                continue

            intercept = x1 - slope * y1
            x_base = slope * y_base + intercept
            x_reference = slope * y_reference + intercept
            if not (width * 0.33 <= x_reference <= width * 0.67):
                continue
            if not (-width * 0.15 <= x_base <= width * 1.15):
                continue

            length = float(np.hypot(x2 - x1, y2 - y1))
            support = self._sample_support(paint_support, x1, y1, x2, y2)
            score = length * (0.45 + 1.55 * support) * (0.65 + 0.35 * (max_y / height))
            candidate = _LineCandidate(slope, intercept, score, support, min_y, max_y)

            if slope < -0.12 and x_base < width * 0.62:
                left.append(candidate)
            elif slope > 0.12 and x_base > width * 0.38:
                right.append(candidate)

        return self._dedupe_candidates(left, width), self._dedupe_candidates(right, width)

    @staticmethod
    def _line_pair(
        left: _LineCandidate, right: _LineCandidate, height: int, width: int
    ) -> Optional[Dict[str, float]]:
        """Validate a left/right pair as one ego-lane corridor."""
        denom = right.slope - left.slope
        if denom <= 1e-6:
            return None

        y_vanish = (left.intercept - right.intercept) / denom
        x_vanish = left.x_at(y_vanish)
        y_base = height * 0.96
        left_base = left.x_at(y_base)
        right_base = right.x_at(y_base)
        lane_width = right_base - left_base
        lane_center = (left_base + right_base) / 2.0

        # The road's vanishing point must be inside the lower road area. A
        # higher intersection usually comes from building edges or a vehicle
        # silhouette rather than a marking painted on the ground.
        if not (height * 0.58 <= y_vanish <= height * 0.82):
            return None
        if not (width * 0.40 <= x_vanish <= width * 0.60):
            return None
        if not (width * 0.18 <= lane_width <= width * 0.60):
            return None
        if not (width * 0.34 <= lane_center <= width * 0.66):
            return None

        # A normal camera lane in this project is ~35–40% of the image at the
        # lower ROI. This is a soft score, not a hard calibration assumption.
        width_score = float(np.exp(-abs(lane_width / width - 0.37) / 0.18))
        center_score = float(np.exp(-abs(lane_center / width - 0.50) / 0.14))
        vanishing_score = float(np.exp(-abs(x_vanish / width - 0.50) / 0.14))
        confidence = min(1.0, (left.support + right.support) / 1.25)
        score = (left.score + right.score) * (0.35 + 0.65 * width_score * center_score * vanishing_score)

        return {
            "score": score,
            "confidence": confidence,
            "x_vanish": x_vanish,
            "y_vanish": y_vanish,
            "lane_width": lane_width,
            "left_base": left_base,
            "right_base": right_base,
        }

    def _best_pair(
        self, left: List[_LineCandidate], right: List[_LineCandidate], height: int, width: int
    ) -> Optional[Tuple[_LineCandidate, _LineCandidate, Dict[str, float]]]:
        best = None
        best_score = -1.0
        previous = self._previous_lines(height, width)
        for left_line in left:
            for right_line in right:
                pair = self._line_pair(left_line, right_line, height, width)
                if pair is None:
                    continue
                score = pair["score"]
                if previous is not None:
                    prev_left, prev_right = previous
                    base_delta = (
                        abs(left_line.x_at(height * 0.96) - prev_left.x_at(height * 0.96))
                        + abs(right_line.x_at(height * 0.96) - prev_right.x_at(height * 0.96))
                    ) / (2.0 * width)
                    score *= 0.55 + 0.45 * float(np.exp(-base_delta / 0.16))
                if score > best_score:
                    best_score = score
                    best = (left_line, right_line, pair)
        return best

    # ------------------------------------------------------------------
    # Temporal state
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_line(line: _LineCandidate, height: int, width: int) -> Tuple[float, float]:
        return line.slope * height / width, line.intercept / width

    @staticmethod
    def _denormalise_line(line_norm: Tuple[float, float], height: int, width: int) -> _LineCandidate:
        slope_norm, intercept_norm = line_norm
        return _LineCandidate(
            slope=slope_norm * width / height,
            intercept=intercept_norm * width,
            score=0.0,
            support=0.0,
            min_y=height * 0.5,
            max_y=height,
        )

    def _previous_lines(self, height: int, width: int) -> Optional[Tuple[_LineCandidate, _LineCandidate]]:
        if self._left_line_norm is None or self._right_line_norm is None:
            return None
        return (
            self._denormalise_line(self._left_line_norm, height, width),
            self._denormalise_line(self._right_line_norm, height, width),
        )

    def _store_pair(self, left: _LineCandidate, right: _LineCandidate, height: int, width: int) -> None:
        left_norm = np.array(self._normalise_line(left, height, width), dtype=np.float64)
        right_norm = np.array(self._normalise_line(right, height, width), dtype=np.float64)
        if self._left_line_norm is not None:
            left_norm = (1.0 - self.alpha) * np.array(self._left_line_norm) + self.alpha * left_norm
        if self._right_line_norm is not None:
            right_norm = (1.0 - self.alpha) * np.array(self._right_line_norm) + self.alpha * right_norm
        self._left_line_norm = tuple(left_norm)
        self._right_line_norm = tuple(right_norm)
        self._missed_frames = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self, frame_bgr: np.ndarray, vehicle_boxes: Optional[List[tuple]] = None
    ) -> Tuple[np.ndarray, dict]:
        """Return a lane-only overlay and its source-frame geometry.

        The overlay is black outside the lane polygon. ``draw_overlay`` blends
        only non-black pixels, so it cannot darken the complete video frame.
        """
        height, width = frame_bgr.shape[:2]
        overlay = np.zeros_like(frame_bgr)
        features, painted = self._lane_feature_mask(frame_bgr, vehicle_boxes)
        left_candidates, right_candidates = self._find_candidates(features, painted)
        best = self._best_pair(left_candidates, right_candidates, height, width)

        left_detected = best is not None
        right_detected = best is not None
        tracking = False
        confidence = 0.0

        if best is not None:
            left_line, right_line, pair = best
            confidence = float(pair["confidence"])
            self._store_pair(left_line, right_line, height, width)
            # Draw the smoothed pair to prevent visible one-frame jitter.
            smoothed = self._previous_lines(height, width)
            if smoothed is not None:
                left_line, right_line = smoothed
            pair = self._line_pair(left_line, right_line, height, width)
            if pair is None:  # Defensive: retain the unsmoothed validated pair.
                left_line, right_line, pair = best
        else:
            self._missed_frames += 1
            previous = self._previous_lines(height, width)
            if previous is None or self._missed_frames > self.hold_frames:
                # Do not invent a lane when the road markings are unavailable.
                if self._missed_frames > self.hold_frames:
                    self._left_line_norm = None
                    self._right_line_norm = None
                return overlay, {
                    "left_pts": None,
                    "right_pts": None,
                    "vanishing_pt": None,
                    "corridor_polygon": None,
                    "left_detected": False,
                    "right_detected": False,
                    "tracking": False,
                    "confidence": 0.0,
                    "top_y": None,
                }
            left_line, right_line = previous
            pair = self._line_pair(left_line, right_line, height, width)
            if pair is None:
                return overlay, {
                    "left_pts": None,
                    "right_pts": None,
                    "vanishing_pt": None,
                    "corridor_polygon": None,
                    "left_detected": False,
                    "right_detected": False,
                    "tracking": False,
                    "confidence": 0.0,
                    "top_y": None,
                }
            tracking = True
            confidence = max(0.0, 0.65 - 0.05 * self._missed_frames)

        y_bottom = int(height * 0.985)
        # Starting just below the pair's intersection creates a narrow pointed
        # far end, rather than the large horizontal "floating" cap from the old
        # fixed BEV trapezoid.
        y_top = int(np.clip(pair["y_vanish"] + 5, height * 0.50, height * 0.86))
        left_top = (int(round(left_line.x_at(y_top))), y_top)
        right_top = (int(round(right_line.x_at(y_top))), y_top)
        left_bottom = (int(round(left_line.x_at(y_bottom))), y_bottom)
        right_bottom = (int(round(right_line.x_at(y_bottom))), y_bottom)

        polygon = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(overlay, [polygon], (20, 150, 35), lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [np.array([left_bottom, left_top], dtype=np.int32)], False, (255, 215, 0), 4, cv2.LINE_AA)
        cv2.polylines(overlay, [np.array([right_top, right_bottom], dtype=np.int32)], False, (255, 215, 0), 4, cv2.LINE_AA)

        return overlay, {
            "left_pts": (left_bottom, left_top),
            "right_pts": (right_bottom, right_top),
            "vanishing_pt": (int(round(pair["x_vanish"])), int(round(pair["y_vanish"]))),
            "corridor_polygon": polygon.tolist(),
            "left_detected": left_detected,
            "right_detected": right_detected,
            "tracking": tracking,
            "confidence": round(confidence, 2),
            "top_y": y_top,
        }
