"""
main.py
CLI entry point for ADAS Perception Pipeline.

Examples:
  python main.py --source 0                      # webcam, live window
  python main.py --source driving.mp4 --output out.mp4
  python main.py --source driving.mp4 --output out.mp4 --show
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch  # noqa: E402


def print_device_banner(device: str):
    """Print a clear banner showing which compute device will be used."""
    print("=" * 60)
    print("  ADAS PERCEPTION SYSTEM — Device Info")
    print("=" * 60)
    if device == "cuda":
        if torch.cuda.is_available():
            gpu_idx  = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            vram_total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
            cuda_ver   = torch.version.cuda or "unknown"
            print(f"  ✅ Device      : GPU (CUDA)")
            print(f"  ✅ GPU Name    : {gpu_name}")
            print(f"  ✅ GPU Index   : cuda:{gpu_idx}")
            print(f"  ✅ VRAM        : {vram_total:.1f} GB total")
            print(f"  ✅ CUDA Version: {cuda_ver}")
        else:
            print("  ⚠️  CUDA requested but NOT available — falling back to CPU")
            print("  ❌ Device      : CPU (no GPU found)")
    else:
        print("  ℹ️  Device      : CPU (--device cpu was selected)")
        print("  💡 Tip: Pass --device cuda to use your GPU for faster inference")
    print("=" * 60)
    print()


from pipeline import PerceptionPipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Autonomous Driving Perception System")
    p.add_argument("--source", default="0",
                    help="Webcam index (e.g. 0) or path to a video file")
    p.add_argument("--output", default=None, help="Path to save annotated output video")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model weights path")
    p.add_argument("--imgsz", type=int, default=416,
                   help="YOLO input image size (default 416 for speed)")
    p.add_argument("--depth-model", default="MiDaS_small",
                    choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])
    p.add_argument("--conf", type=float, default=0.45,
                   help="Detection confidence threshold (default 0.45)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--fov", type=float, default=70.0, help="Assumed horizontal camera FOV (deg)")
    p.add_argument("--show", action="store_true", help="Show a live preview window")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing)")
    p.add_argument("--frame-skip", type=int, default=2,
                   help="Process 1 in every N frames for speed (default 2)")
    p.add_argument("--depth-interval", type=int, default=4,
                   help="Run depth model every N frames (default 4)")
    p.add_argument("--detect-interval", type=int, default=3,
                   help="Run YOLO every N frames; tracker runs every frame (default 3)")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # Show GPU / CPU status before loading any models
    print_device_banner(args.device)

    pipeline = PerceptionPipeline(
        model_path=args.model,
        depth_model=args.depth_model,
        conf_threshold=args.conf,
        device=args.device,
        horizontal_fov_deg=args.fov,
        run_depth_every_n_frames=args.depth_interval,
        detect_every_n_frames=args.detect_interval,
        imgsz=args.imgsz,
    )

    n = pipeline.run(source=source, output_path=args.output, show=args.show,
                      max_frames=args.max_frames, frame_skip=args.frame_skip)
    print(f"Processed {n} frames.")
    if args.output:
        print(f"Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()
