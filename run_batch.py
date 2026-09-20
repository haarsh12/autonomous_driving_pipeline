"""
run_batch.py
Batch processor for running the Perception System on multiple input video files.
"""

import os
import sys
import glob
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import PerceptionPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Batch test perception system on video feeds")
    p.add_argument("--pattern", default="*.mp4", help="Glob pattern for input videos")
    p.add_argument("--output-dir", default="outputs", help="Directory to save annotated videos & reports")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    p.add_argument("--depth-model", default="MiDaS_small", help="Depth estimation model")
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--fov", type=float, default=70.0, help="Camera FOV in degrees")
    p.add_argument("--depth-interval", type=int, default=2, help="Run depth estimation every N frames")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    return p.parse_args()


def run_batch():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    videos = sorted(glob.glob(args.pattern))
    if not videos:
        print(f"No videos matching pattern '{args.pattern}' found in {os.getcwd()}")
        sys.exit(1)

    print(f"Found {len(videos)} video file(s) matching '{args.pattern}':")
    for v in videos:
        print(f"  - {v}")

    print(f"\nInitializing Perception Pipeline...")
    pipeline = PerceptionPipeline(
        model_path=args.model,
        depth_model=args.depth_model,
        conf_threshold=args.conf,
        device=args.device,
        horizontal_fov_deg=args.fov,
        run_depth_every_n_frames=args.depth_interval,
    )

    batch_summary = []
    total_start = time.time()

    for idx, vid in enumerate(videos, 1):
        vid_name = Path(vid).stem
        out_path = os.path.join(args.output_dir, f"annotated_{vid_name}.mp4")
        print(f"\n[{idx}/{len(videos)}] Processing: {vid}")

        t0 = time.time()
        pipeline.tracker._next_id = 1
        pipeline.tracker.tracks = {}

        n_frames = pipeline.run(
            source=vid,
            output_path=out_path,
            show=False,
            max_frames=args.max_frames,
        )
        elapsed = time.time() - t0
        fps = n_frames / max(elapsed, 1e-6)

        video_stat = {
            "video": vid,
            "output": out_path,
            "frames_processed": n_frames,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(fps, 2),
            "total_tracks": pipeline.tracker._next_id - 1
        }
        batch_summary.append(video_stat)
        print(f"    Finished {n_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS). Tracks: {pipeline.tracker._next_id - 1}")

    total_elapsed = time.time() - total_start
    report_path = os.path.join(args.output_dir, "batch_test_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": vars(args),
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": batch_summary
        }, f, indent=2)

    print("\n" + "="*60)
    print("BATCH TEST COMPLETE!")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    run_batch()
