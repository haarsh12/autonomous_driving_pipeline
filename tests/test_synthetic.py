"""
test_synthetic.py
Validates fusion (distance estimation), tracking, and alerting logic using
synthetic detections + synthetic depth maps without requiring neural net weights.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector import Detection, REAL_WORLD_HEIGHT_M  # noqa: E402
from fusion import fuse_detections_with_depth, estimate_focal_length_px  # noqa: E402
from tracker import CentroidIOUTracker  # noqa: E402
from visualizer import generate_warnings, risk_level  # noqa: E402


def make_detection(class_name, x1, y1, x2, y2, conf=0.9):
    return Detection(
        class_name=class_name,
        confidence=conf,
        box=(x1, y1, x2, y2),
        box_height_px=(y2 - y1),
        box_width_px=(x2 - x1),
        center=((x1 + x2) / 2, (y1 + y2) / 2),
    )


def test_focal_length_sane():
    f = estimate_focal_length_px(1280, horizontal_fov_deg=70.0)
    assert 800 < f < 1200, f"Focal length out of expected range: {f}"
    print(f"[OK] focal length estimate for 1280px/70deg FOV = {f:.1f}px")


def test_geometric_distance_decreases_as_box_grows():
    frame_w = 1280
    depth_map = np.full((720, 1280), 0.5, dtype=np.float32)

    near_car = make_detection("car", 400, 300, 700, 500)   # tall box = close
    far_car = make_detection("car", 600, 350, 680, 400)    # short box = far

    fused_near = fuse_detections_with_depth([near_car], depth_map, frame_w)[0]
    fused_far = fuse_detections_with_depth([far_car], depth_map, frame_w)[0]

    print(f"[OK] near car distance={fused_near.distance_m}m, far car distance={fused_far.distance_m}m")
    assert fused_near.distance_m < fused_far.distance_m, "Near car should be estimated closer than far car"


def test_depth_map_influences_ranking():
    frame_w = 1280
    depth_map = np.zeros((720, 1280), dtype=np.float32)
    depth_map[:, :640] = 0.9
    depth_map[:, 640:] = 0.2

    left_car = make_detection("car", 100, 300, 300, 420)
    right_car = make_detection("car", 900, 300, 1100, 420)

    fused = fuse_detections_with_depth([left_car, right_car], depth_map, frame_w)
    left_result = [f for f in fused if f.center[0] < 640][0]
    right_result = [f for f in fused if f.center[0] >= 640][0]

    print(f"[OK] left_car rank={left_result.depth_rank} dist={left_result.distance_m}m | "
          f"right_car rank={right_result.depth_rank} dist={right_result.distance_m}m")

    assert left_result.depth_rank == 1, "Left car (higher depth value) should rank as closest"
    assert left_result.distance_m < right_result.distance_m, \
        "Depth-map cue should pull left car's distance estimate closer than right car's"


def test_tracker_assigns_and_persists_ids():
    tracker = CentroidIOUTracker()
    frame_w = 1280
    depth_map = np.full((720, 1280), 0.5, dtype=np.float32)

    det1 = make_detection("pedestrian", 500, 300, 560, 480)
    fused1 = fuse_detections_with_depth([det1], depth_map, frame_w)
    fused1 = tracker.update(fused1)
    id_frame1 = fused1[0].track_id
    assert id_frame1 is not None
    print(f"[OK] pedestrian assigned track_id={id_frame1} on first frame")

    det2 = make_detection("pedestrian", 505, 280, 575, 480)
    fused2 = fuse_detections_with_depth([det2], depth_map, frame_w)
    fused2 = tracker.update(fused2)
    id_frame2 = fused2[0].track_id

    assert id_frame2 == id_frame1, "Same object across frames should keep the same track ID"
    print(f"[OK] track ID persisted across frames: {id_frame2}")


def test_approaching_status_and_warning_triggers():
    tracker = CentroidIOUTracker()
    frame_w = 1280
    depth_map = np.full((720, 1280), 0.5, dtype=np.float32)

    box_heights = [150, 200, 260, 340, 430]
    last_fused = None
    for h in box_heights:
        det = make_detection("pedestrian", 500, 300, 560, 300 + h)
        fused = fuse_detections_with_depth([det], depth_map, frame_w)
        fused = tracker.update(fused)
        last_fused = fused

    status = tracker.get_movement_status(last_fused[0].track_id)
    print(f"[OK] final movement status after approach sequence: '{status}', "
          f"final distance={last_fused[0].distance_m}m")
    assert status == "approaching", f"Expected 'approaching', got '{status}'"

    warnings = generate_warnings(last_fused, tracker)
    print(f"[OK] warnings generated: {warnings}")
    assert any("WARNING" in w or "Caution" in w or "EMERGENCY" in w for w in warnings), \
        "Expected at least one warning for a close, approaching pedestrian"


def test_risk_levels():
    frame_w = 1280
    depth_map = np.full((720, 1280), 0.5, dtype=np.float32)

    close_ped = make_detection("pedestrian", 500, 100, 580, 600)
    far_truck = make_detection("truck", 600, 350, 640, 370)

    tracker = CentroidIOUTracker()
    fused = fuse_detections_with_depth([close_ped, far_truck], depth_map, frame_w)
    fused = tracker.update(fused, frame_w=frame_w, frame_h=720)
    ped_result = [f for f in fused if f.class_name == "pedestrian"][0]
    truck_result = [f for f in fused if f.class_name == "truck"][0]

    print(f"[OK] pedestrian dist={ped_result.distance_m}m risk={risk_level(ped_result)} | "
          f"truck dist={truck_result.distance_m}m risk={risk_level(truck_result)}")

    assert risk_level(ped_result) in ("critical", "warning", "danger")
    assert risk_level(truck_result) == "safe"


if __name__ == "__main__":
    tests = [
        test_focal_length_sane,
        test_geometric_distance_decreases_as_box_grows,
        test_depth_map_influences_ranking,
        test_tracker_assigns_and_persists_ids,
        test_approaching_status_and_warning_triggers,
        test_risk_levels,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    if failed:
        sys.exit(1)
