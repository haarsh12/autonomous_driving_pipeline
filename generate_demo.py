"""
generate_demo.py
Creates a synthetic driving scene video (car approaching, pedestrian crossing,
motorcycle passing) and runs it through the fusion -> tracker -> visualizer pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np

from detector import Detection
from fusion import fuse_detections_with_depth
from tracker import CentroidIOUTracker
from visualizer import draw_overlay, generate_warnings, SIDEBAR_WIDTH

W, H = 960, 540
N_FRAMES = 90
OUT_PATH = "outputs/demo_pipeline_output.mp4"


def draw_road_background(frame):
    frame[:] = (60, 60, 60)
    cv2.rectangle(frame, (0, int(H * 0.55)), (W, H), (40, 40, 40), -1)
    for x in range(0, W, 60):
        cv2.line(frame, (x, int(H * 0.78)), (x + 30, int(H * 0.78)), (200, 200, 200), 4)
    cv2.rectangle(frame, (0, 0), (W, int(H * 0.55)), (90, 110, 130), -1)
    return frame


def make_synthetic_detections(frame_idx):
    dets = []
    t = frame_idx / N_FRAMES

    # Car approaching head-on
    car_h = 40 + t * 220
    car_w = car_h * 1.5
    cx = W * 0.62
    cy = H * 0.72
    x1, y1 = cx - car_w / 2, cy - car_h / 2
    x2, y2 = cx + car_w / 2, cy + car_h / 2
    dets.append(Detection("car", 0.95, (x1, y1, x2, y2), car_h, car_w, (cx, cy)))

    # Pedestrian crossing left to right
    if 15 < frame_idx < 75:
        px = 80 + (frame_idx - 15) * (W - 200) / 60
        ph, pw = 130, 45
        py = H * 0.68
        x1, y1 = px - pw / 2, py - ph / 2
        x2, y2 = px + pw / 2, py + ph / 2
        dets.append(Detection("pedestrian", 0.91, (x1, y1, x2, y2), ph, pw, (px, py)))

    # Motorcycle receding
    if frame_idx < 60:
        mt = frame_idx / 60
        mh = 90 - mt * 60
        mw = mh * 1.2
        mx = W * 0.85
        my = H * 0.70
        x1, y1 = mx - mw / 2, my - mh / 2
        x2, y2 = mx + mw / 2, my + mh / 2
        dets.append(Detection("motorcycle", 0.88, (x1, y1, x2, y2), mh, mw, (mx, my)))

    return dets


def main():
    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, 15.0, (W + SIDEBAR_WIDTH, H))

    tracker = CentroidIOUTracker()
    depth_map = np.full((H, W), 0.5, dtype=np.float32)

    log_lines = []
    for frame_idx in range(N_FRAMES):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_road_background(frame)

        dets = make_synthetic_detections(frame_idx)
        fused = fuse_detections_with_depth(dets, depth_map, W)
        fused = tracker.update(fused)
        warnings = generate_warnings(fused, tracker)

        annotated = draw_overlay(frame, fused, tracker)
        cv2.putText(annotated, f"Frame {frame_idx+1}/{N_FRAMES}", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(annotated)

        if warnings:
            log_lines.append(f"Frame {frame_idx+1}: " + " | ".join(warnings))

    writer.release()
    print(f"Saved demo video: {OUT_PATH}")
    print(f"Active tracks created: {tracker._next_id - 1}")


if __name__ == "__main__":
    main()
