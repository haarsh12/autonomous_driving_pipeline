"""
run_test.py
Single easy-to-use presenter runner for testing on sample video files or webcam.
"""

import os
import sys

# 1. SET YOUR INPUT VIDEO PATH HERE (or 0 for webcam):
INPUT_VIDEO_PATH = r"input video/driving_city.mp4"

# 2. OPTIONAL CONFIGURATION:
OUTPUT_VIDEO_PATH = "outputs/annotated_driving_city.mp4"
SHOW_LIVE_PREVIEW = True     # Set to True for interactive live window preview
PROCESS_EVERY_N_FRAMES = 1   # Set to 1 for full quality

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Automatic VENV Detection
venv_site_packages = os.path.join(PROJECT_DIR, "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import PerceptionPipeline


def run():
    print("=" * 60)
    print("  AUTONOMOUS DRIVING PERCEPTION SYSTEM - PRESENTATION RUNNER  ")
    print("=" * 60)
    
    if not os.path.exists(INPUT_VIDEO_PATH) and not str(INPUT_VIDEO_PATH).isdigit():
        print(f"INPUT VIDEO '{INPUT_VIDEO_PATH}' not found.")
        print("Falling back to generating synthetic demo video...")
        import generate_demo
        generate_demo.main()
        vid_src = "outputs/demo_pipeline_output.mp4"
    else:
        vid_src = INPUT_VIDEO_PATH

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    
    print(f"Loading AI Models (YOLO11 + MiDaS Depth)...")
    pipeline = PerceptionPipeline(
        model_path="yolo11n.pt",
        depth_model="MiDaS_small",
        conf_threshold=0.45,
        device="cuda",              # Uses GPU (cuda) automatically with fallback if needed
        horizontal_fov_deg=70.0,
        run_depth_every_n_frames=2,
        detect_every_n_frames=2,
        imgsz=640,
    )

    print(f"Processing Video Source: {vid_src}")
    total_frames = pipeline.run(
        source=vid_src,
        output_path=OUTPUT_VIDEO_PATH,
        show=SHOW_LIVE_PREVIEW,
        frame_skip=PROCESS_EVERY_N_FRAMES,
    )

    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"Total Frames Processed : {total_frames}")
    print(f"Annotated Video Saved  : {os.path.abspath(OUTPUT_VIDEO_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    run()
