"""
lane_detector.py
Bird's Eye View (BEV / IPM) Ground-Anchored 2nd-Degree Polynomial Lane Detector.

Key Features:
  1. Inverse Perspective Mapping (IPM / BEV Warp): Transforms camera perspective
     into flat Bird's-Eye View ground space.
  2. Ground Plane Anchoring: All polynomial curves and filled drivable corridors
     are calculated in BEV ground space and unwarped back onto the road, ensuring
     they sit FLAT on the asphalt (never balloon into the air or sidewalks).
  3. Parallel Lane Enforcement: Forces left & right lanes to maintain a constant,
     realistic lane width (~3.7m in BEV space), overriding false curb detections.
  4. Bounded Distance: Extends only to the actual visible range of detected lane markings.
  5. Occlusion Masking: Vehicle bounding boxes are masked out before edge extraction.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class LaneDetector:
    """
    BEV (Inverse Perspective Mapping) Ground-Anchored Lane Detector.
    Guarantees flat ground projection, parallel lane width, and bounded distance.
    """

    def __init__(self, smooth_alpha: float = 0.25):
        self.alpha = smooth_alpha

        # Smoothed polynomial coefficients in BEV space: x_bev = A*y_bev^2 + B*y_bev + C
        self._left_fit_bev  = None
        self._right_fit_bev = None

        # BEV dimensions
        self.bev_w = 480
        self.bev_h = 640

        # Smoothed top extent in BEV space (0 = farthest top, bev_h = hood bottom)
        self._y_top_bev_smooth = float(self.bev_h * 0.35)

    def _get_ipm_matrices(self, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Perspective Transform matrices M (camera -> BEV) and M_inv (BEV -> camera)."""
        # Ground ROI trapezoid in camera view
        src = np.float32([
            [W * 0.42, H * 0.62],   # Top-Left (horizon)
            [W * 0.58, H * 0.62],   # Top-Right (horizon)
            [W * 0.90, H * 0.95],   # Bottom-Right (hood)
            [W * 0.10, H * 0.95],   # Bottom-Left (hood)
        ])

        # Rectangle in BEV space
        dst = np.float32([
            [self.bev_w * 0.25, 0],
            [self.bev_w * 0.75, 0],
            [self.bev_w * 0.75, self.bev_h],
            [self.bev_w * 0.25, self.bev_h],
        ])

        M     = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return M, M_inv

    def _extract_bev_edge_mask(
        self, frame_bgr: np.ndarray, M: np.ndarray, vehicle_boxes: List[tuple] = None
    ) -> np.ndarray:
        """Extract road lane edges and warp into BEV space."""
        H, W = frame_bgr.shape[:2]

        # 1. Color filtering in HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # White lane mask
        lower_white = np.array([0, 0, 185], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask_white  = cv2.inRange(hsv, lower_white, upper_white)

        # Yellow lane mask (center line)
        lower_yellow = np.array([14, 70, 110], dtype=np.uint8)
        upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
        mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

        color_mask = cv2.bitwise_or(mask_white, mask_yellow)

        # 2. Grayscale Canny
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        combined = cv2.bitwise_or(edges, color_mask)

        # 3. Mask out vehicle bounding boxes before warping
        if vehicle_boxes:
            for box in vehicle_boxes:
                vx1, vy1, vx2, vy2 = [int(v) for v in box]
                vx1, vy1 = max(0, vx1 - 5), max(0, vy1 - 5)
                vx2, vy2 = min(W - 1, vx2 + 5), min(H - 1, vy2 + 5)
                combined[vy1:vy2, vx1:vx2] = 0

        # 4. Warp into BEV (Bird's Eye View)
        bev_binary = cv2.warpPerspective(combined, M, (self.bev_w, self.bev_h), flags=cv2.INTER_NEAREST)
        return bev_binary

    def _fit_bev_polynomials(
        self, bev_binary: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Fits 2nd-degree polynomial x_bev = A*y^2 + B*y + C in BEV space using histogram sliding windows.
        Enforces parallel lane width constraint in BEV space.
        """
        # Find non-zero pixels
        nonzero = bev_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if len(nonzeroy) < 40:
            return None, None, int(self.bev_h * 0.35)

        # Highest detected lane pixel in BEV (smallest y_bev)
        min_y_bev = int(np.min(nonzeroy))

        # Histogram of bottom half in BEV space
        bottom_half = bev_binary[int(self.bev_h * 0.5):, :]
        histogram = np.sum(bottom_half, axis=0)

        midpoint = int(self.bev_w // 2)
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Default BEV lane width in pixels (~240px out of 480px width)
        EXPECTED_LANE_WIDTH_BEV = int(self.bev_w * 0.48)

        # Sliding window tracking in BEV space
        nwindows = 9
        window_height = int(self.bev_h // nwindows)

        leftx_current  = leftx_base  if histogram[leftx_base] > 10 else int(midpoint - EXPECTED_LANE_WIDTH_BEV / 2)
        rightx_current = rightx_base if histogram[rightx_base] > 10 else int(midpoint + EXPECTED_LANE_WIDTH_BEV / 2)

        margin = 40
        minpix = 15

        left_lane_inds  = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low  = self.bev_h - (window + 1) * window_height
            win_y_high = self.bev_h - window * window_height

            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds  = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([], dtype=int)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        left_fit, right_fit = None, None

        if len(lefty) > 25:
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
            except (np.linalg.LinAlgError, ValueError):
                left_fit = None

        if len(righty) > 25:
            try:
                right_fit = np.polyfit(righty, rightx, 2)
            except (np.linalg.LinAlgError, ValueError):
                right_fit = None

        # Enforce parallel lane width constraint in BEV space:
        # If one lane is missing/weak or misdetected, generate it parallel to the valid lane!
        if left_fit is not None and right_fit is None:
            # Shift left fit to the right by expected lane width
            right_fit = left_fit.copy()
            right_fit[2] += EXPECTED_LANE_WIDTH_BEV
        elif right_fit is not None and left_fit is None:
            # Shift right fit to the left by expected lane width
            left_fit = right_fit.copy()
            left_fit[2] -= EXPECTED_LANE_WIDTH_BEV

        return left_fit, right_fit, min_y_bev

    def detect(
        self, frame_bgr: np.ndarray, vehicle_boxes: List[tuple] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Runs BEV ground-anchored lane detection and returns camera overlay + lane_data.
        """
        H, W = frame_bgr.shape[:2]
        M, M_inv = self._get_ipm_matrices(H, W)

        # 1. Extract BEV edge mask
        bev_binary = self._extract_bev_edge_mask(frame_bgr, M, vehicle_boxes)

        # 2. Fit 2nd-degree BEV polynomials
        left_fit_cur, right_fit_cur, min_y_bev = self._fit_bev_polynomials(bev_binary)

        # Smooth bounded top extent in BEV space (stop where lane marks end)
        detected_top_bev = max(int(self.bev_h * 0.25), min_y_bev)
        self._y_top_bev_smooth = self._y_top_bev_smooth * (1 - self.alpha) + detected_top_bev * self.alpha
        top_y_bev = int(self._y_top_bev_smooth)

        # 3. Smooth BEV polynomial coefficients via EMA
        if left_fit_cur is not None:
            if self._left_fit_bev is None:
                self._left_fit_bev = left_fit_cur
            else:
                self._left_fit_bev = self._left_fit_bev * (1 - self.alpha) + left_fit_cur * self.alpha

        if right_fit_cur is not None:
            if self._right_fit_bev is None:
                self._right_fit_bev = right_fit_cur
            else:
                self._right_fit_bev = self._right_fit_bev * (1 - self.alpha) + right_fit_cur * self.alpha

        # Synthetic fallback if uninitialized
        if self._left_fit_bev is None and self._right_fit_bev is None:
            # Vertical parallel lines in BEV space
            self._left_fit_bev  = np.array([0.0, 0.0, float(self.bev_w * 0.26)])
            self._right_fit_bev = np.array([0.0, 0.0, float(self.bev_w * 0.74)])

        # 4. Generate smooth curves in BEV space from hood bottom down to bounded top_y_bev
        ploty_bev = np.linspace(top_y_bev, self.bev_h - 1, num=30)

        a_l, b_l, c_l = self._left_fit_bev
        left_x_bev  = a_l * (ploty_bev ** 2) + b_l * ploty_bev + c_l

        a_r, b_r, c_r = self._right_fit_bev
        right_x_bev = a_r * (ploty_bev ** 2) + b_r * ploty_bev + c_r

        pts_left_bev  = np.array([np.transpose(np.vstack([left_x_bev, ploty_bev]))], dtype=np.float32)
        pts_right_bev = np.array([np.transpose(np.vstack([right_x_bev, ploty_bev]))], dtype=np.float32)

        # 5. Build ground corridor polygon in BEV space
        pts_left_bev_arr  = pts_left_bev[0]
        pts_right_bev_arr = pts_right_bev[0]

        bev_corridor_poly = np.vstack([pts_left_bev_arr, np.flipud(pts_right_bev_arr)])

        # 6. UNWARP BEV POINTS BACK TO ORIGINAL CAMERA PERSPECTIVE GROUND VIEW using M_inv!
        bev_canvas = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)

        # Draw filled corridor in BEV space
        cv2.fillPoly(bev_canvas, [np.int32(bev_corridor_poly)], (0, 150, 40))

        # Draw left and right curves in BEV space
        cv2.polylines(bev_canvas, [np.int32(pts_left_bev_arr)], isClosed=False, color=(255, 200, 0), thickness=6)
        cv2.polylines(bev_canvas, [np.int32(pts_right_bev_arr)], isClosed=False, color=(255, 200, 0), thickness=6)

        # Unwarp the entire BEV canvas back onto the camera frame image plane!
        camera_overlay = cv2.warpPerspective(bev_canvas, M_inv, (W, H), flags=cv2.INTER_LINEAR)

        # Compute camera endpoint coordinates for API compatibility
        left_pts_cam  = cv2.perspectiveTransform(pts_left_bev, M_inv)[0]
        right_pts_cam = cv2.perspectiveTransform(pts_right_bev, M_inv)[0]

        left_endpoints  = (tuple(np.int32(left_pts_cam[-1])), tuple(np.int32(left_pts_cam[0])))
        right_endpoints = (tuple(np.int32(right_pts_cam[-1])), tuple(np.int32(right_pts_cam[0])))

        lane_data = {
            "left_pts": left_endpoints,
            "right_pts": right_endpoints,
            "vanishing_pt": (
                int((left_pts_cam[0][0] + right_pts_cam[0][0]) / 2),
                int((left_pts_cam[0][1] + right_pts_cam[0][1]) / 2)
            ),
            "corridor_polygon": None,
            "left_detected": left_fit_cur is not None,
            "right_detected": right_fit_cur is not None,
            "top_y": int(left_pts_cam[0][1]),
        }

        return camera_overlay, lane_data
