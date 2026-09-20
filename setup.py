"""
setup.py
Run this ONCE to configure the project:
  1. Installs Python dependencies from requirements.txt
  2. Pre-downloads YOLO11n weights (~6 MB) into models/
  3. Pre-downloads MiDaS small weights (~90 MB) into models/
  4. Runs synthetic unit tests to verify system logic
"""

import subprocess
import sys
import os
import shutil


def run(cmd, desc=""):
    print(f"\n>>> {desc or cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  Autonomous Driving Perception System — Setup")
    print("=" * 60)

    # 1. Install deps
    run(f'"{sys.executable}" -m pip install -r requirements.txt',
        "Installing Python dependencies...")

    # 2. Download YOLO weights
    print("\n>>> Downloading YOLO11n weights (~6 MB)...")
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    os.makedirs("models", exist_ok=True)
    if os.path.exists("yolo11n.pt"):
        shutil.move("yolo11n.pt", "models/yolo11n.pt")
    print("    Saved to models/yolo11n.pt")

    # 3. Download MiDaS weights
    print("\n>>> Downloading MiDaS_small weights (~90 MB)...")
    import torch
    import torch.hub
    torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    print("    MiDaS_small ready.")

    # 4. Run synthetic unit tests
    run(f'"{sys.executable}" tests/test_synthetic.py',
        "Running synthetic unit tests...")

    # 5. Done
    print("\n" + "=" * 60)
    print("  Setup complete! Try one of these:")
    print("=" * 60)
    print("  # Run presenter test script:")
    print("  python run_test.py")
    print()
    print("  # Webcam (live window):")
    print("  python main.py --source 0 --show")
    print()
    print("  # Video file run:")
    print("  python main.py --source driving.mp4 --output outputs/result.mp4 --show")
    print()


if __name__ == "__main__":
    main()
