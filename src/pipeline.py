"""
pipeline.py
End-to-end Perception Pipeline Orchestrator connecting video/webcam input,
YOLO detection, MiDaS depth, fusion, tracking, visualization, and reporting.

Architecture (threaded):
  - Reader  thread:    reads raw frames from disk into raw_q at full disk speed
  - Inference thread:  pulls raw frames → GPU inference → pushes annotated frames to out_q
  - Main    thread:    pulls annotated frames from out_q, displays at exact source FPS
                       using wall-clock sleep so it never goes faster or slower.
"""

import time
import datetime
import threading
import queue
import cv2

from detector import ObjectDetector
from depth_estimator import DepthEstimator
from fusion import fuse_detections_with_depth
from tracker import CentroidIOUTracker
from lane_detector import LaneDetector
from visualizer import draw_overlay, generate_warnings, SIDEBAR_WIDTH
from report_generator import PerceptionReportGenerator


class PerceptionPipeline:
    def __init__(self,
                 model_path: str = "yolo11s.pt",
                 depth_model: str = "MiDaS_small",
                 conf_threshold: float = 0.50,
                 device: str = "cpu",
                 horizontal_fov_deg: float = 70.0,
                 run_depth_every_n_frames: int = 3,
                 detect_every_n_frames: int = 4,
                 imgsz: int = 960,
                 output_dir: str = "outputs"):
        self.detector    = ObjectDetector(model_path, conf_threshold, device, imgsz=imgsz)
        self.depth_est   = DepthEstimator(depth_model, device)
        self.tracker     = CentroidIOUTracker()
        self.lane_det    = LaneDetector(smooth_alpha=0.30)
        self.fov_deg     = horizontal_fov_deg

        self.depth_every  = run_depth_every_n_frames
        self.detect_every = max(1, detect_every_n_frames)

        self._depth_map        = None
        self._last_detections  = []
        self._last_lane_overlay = None
        self._last_lane_data    = {}
        self._frame_count      = 0
        self.report_gen        = PerceptionReportGenerator(output_dir=output_dir)

    # ------------------------------------------------------------------
    def process_frame(self, frame, frame_idx: int = 0, total_frames: int = 0):
        t0 = time.time()
        self._frame_count += 1

        # 1. Detection (run every detect_every frames)
        if self._frame_count % self.detect_every == 1 or self._frame_count == 1:
            self._last_detections = self.detector.detect(frame)
        detections = self._last_detections

        # 2. Depth estimation (run every depth_every frames)
        if self._depth_map is None or self._frame_count % self.depth_every == 0:
            self._depth_map = self.depth_est.estimate(frame)

        # 3. Lane detection with vehicle occlusion masking
        vehicle_boxes = [d.box for d in detections if d.class_name in ("car", "truck", "bus", "rickshaw", "motorcycle")]
        lane_overlay, lane_data = self.lane_det.detect(frame, vehicle_boxes=vehicle_boxes)
        self._last_lane_overlay = lane_overlay
        self._last_lane_data    = lane_data

        # 4. Fusion & Tracking
        fused = fuse_detections_with_depth(
            detections, self._depth_map,
            frame.shape[1], self.fov_deg,
            frame_height_px=frame.shape[0]
        )
        fused    = self.tracker.update(fused, frame_w=frame.shape[1], frame_h=frame.shape[0])
        warnings = generate_warnings(fused, self.tracker)

        ts_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.report_gen.log_frame(frame_idx, ts_str, fused)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        annotated = draw_overlay(
            frame.copy(), fused, self.tracker,
            fps=fps, frame_idx=frame_idx, total_frames=total_frames,
            event_log=self.report_gen.events,
            lane_overlay=lane_overlay,
            lane_data=lane_data,
        )
        return annotated, fused, warnings

    # ------------------------------------------------------------------
    # run() — threaded pipeline
    # ------------------------------------------------------------------
    def run(self, source=0, output_path=None, show=False,
            max_frames=None, frame_skip: int = 1):

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps_in,
                (width + SIDEBAR_WIDTH, height)
            )

        frame_skip = max(1, int(frame_skip))
        target_spf = 1.0 / fps_in      # seconds between display frames at source speed

        # Show at FULL resolution — no scale degradation
        disp_w = width + SIDEBAR_WIDTH
        disp_h = height

        # Queues: bounded so reader never runs too far ahead of inference
        raw_q  = queue.Queue(maxsize=16)   # raw frames waiting for GPU
        out_q  = queue.Queue(maxsize=16)   # annotated frames ready to display
        stop   = threading.Event()
        t_start = time.time()

        # ── Thread 1 : Read frames from disk ─────────────────────────
        def reader_fn():
            idx = 0
            while not stop.is_set():
                ret, frame = cap.read()
                if not ret or (max_frames and idx >= max_frames):
                    raw_q.put(None)        # sentinel → stop inference
                    return
                try:
                    raw_q.put((idx, frame), timeout=2.0)
                except queue.Full:
                    pass                   # drop if downstream is stuck
                idx += 1

        # ── Thread 2 : GPU Inference ──────────────────────────────────
        def inference_fn():
            last_annotated = None
            while not stop.is_set():
                try:
                    item = raw_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is None:
                    out_q.put(None)        # propagate sentinel → stop display
                    return
                frame_idx, frame = item

                # Only run full GPU inference every frame_skip frames
                if frame_idx % frame_skip == 0 or last_annotated is None:
                    annotated, _, warnings = self.process_frame(
                        frame, frame_idx=frame_idx, total_frames=total_frames
                    )
                    last_annotated = annotated
                    if warnings:
                        print(f"Frame {frame_idx}: " + " | ".join(warnings), flush=True)
                else:
                    annotated = last_annotated   # reuse previous overlay

                # Write to disk (if requested) in inference thread
                if writer:
                    writer.write(annotated)

                try:
                    out_q.put((frame_idx, annotated), timeout=2.0)
                except queue.Full:
                    pass                   # drop if display is stuck

        t_reader    = threading.Thread(target=reader_fn,    daemon=True, name="reader")
        t_inference = threading.Thread(target=inference_fn, daemon=True, name="inference")
        t_reader.start()
        t_inference.start()

        # ── Main thread : Display at exact source FPS ─────────────────
        frame_idx     = 0
        display_t0    = None
        first_idx     = None

        try:
            while True:
                try:
                    item = out_q.get(timeout=5.0)
                except queue.Empty:
                    break
                if item is None:
                    break

                frame_idx, annotated = item

                if show:
                    # Initialise wall-clock anchor on first frame
                    if display_t0 is None:
                        display_t0 = time.perf_counter()
                        first_idx  = frame_idx

                    # Sleep until this frame's scheduled display time
                    expected  = (frame_idx - first_idx) * target_spf
                    remaining = expected - (time.perf_counter() - display_t0)
                    if remaining > 0.002:          # only sleep if > 2 ms
                        time.sleep(remaining)

                    # Display at full native resolution — no scale degradation
                    cv2.imshow("ADAS Perception", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop.set()
                        break

        finally:
            stop.set()
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        t_reader.join(timeout=3)
        t_inference.join(timeout=8)

        t_elapsed = time.time() - t_start
        avg_fps   = frame_idx / max(t_elapsed, 1e-6)

        report_files = self.report_gen.generate_report(str(source), t_elapsed, avg_fps)
        print("\n" + "=" * 60)
        print("PERCEPTION PIPELINE EXECUTION COMPLETE!")
        print("=" * 60)
        print(f"Processed {frame_idx} frames in {t_elapsed:.2f}s ({avg_fps:.2f} FPS).")
        print(f"Summary Report (Markdown): {report_files['markdown']}")
        print(f"Summary Report (JSON):     {report_files['json']}")
        print(f"Frame Log CSV:             {report_files.get('frame_csv', 'N/A')}")
        print(f"Incidents CSV:             {report_files.get('incidents_csv', 'N/A')}")
        if output_path:
            print(f"Annotated Video Output:    {output_path}")

        return frame_idx
