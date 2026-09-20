.\venv\Scripts\python.exe main.py --source "input video/driving_city.mp4" --output "outputs/annotated_driving_city.mp4" --device cuda --show
 

# Autonomous Driving ADAS Perception System

A complete AI-based autonomous driving perception prototype that detects road objects, estimates their real-world metric distance, tracks them across video frames, and calculates Time-To-Collision (TTC) to trigger collision alerts — running locally using open-source models.

---

## Features

| Feature | Detail |
|---|---|
| **Object Detection** | YOLO11 (Standard/Nano) detects cars, trucks, buses, motorcycles, bicycles, pedestrians, rickshaws, traffic lights, stop signs |
| **Depth Estimation** | MiDaS small monocular per-pixel relative depth map estimation |
| **Distance Estimation** | Pinhole camera model (primary) fused with MiDaS relative depth ranking |
| **Object Tracking** | Centroid + IoU tracking with class majority voting & velocity extrapolation |
| **Domain Heuristics** | Rider suppression (pedestrian on motorcycle) & auto-rickshaw aspect-ratio reclassification |
| **Collision Alerts** | Multi-factor Risk Matrix (Distance + TTC + Speed + Lane awareness) with WARNING / EMERGENCY banners |
| **HUD & BEV Dashboard** | Real-time overlay featuring telemetry panel, Top-5 Threat Table, and Bird's-Eye-View (BEV) radar map |
| **Post-Drive Report** | Auto-generates Markdown + JSON report + frame-level CSV + critical incidents CSV |

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║            AUTONOMOUS DRIVING PERCEPTION PIPELINE                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [Camera / Video File]                                               ║
║         │                                                            ║
║         ▼                                                            ║
║  ┌──────────────────┐    ┌──────────────────────┐                   ║
║  │  detector.py     │    │  depth_estimator.py   │                   ║
║  │  YOLO11s / n     │    │  MiDaS_small          │                   ║
║  │  → bboxes,       │    │  → H×W depth map      │                   ║
║  │    class, conf   │    │    (relative)          │                   ║
║  └────────┬─────────┘    └──────────┬─────────────┘                 ║
║           │                         │                                ║
║           └──────────┬──────────────┘                                ║
║                      ▼                                               ║
║           ┌──────────────────────┐                                   ║
║           │  fusion.py           │                                   ║
║           │  Geometric dist +    │                                   ║
║           │  MiDaS depth rank    │                                   ║
║           │  → distance_m/object │                                   ║
║           └──────────┬───────────┘                                   ║
║                      ▼                                               ║
║           ┌──────────────────────┐                                   ║
║           │  tracker.py          │                                   ║
║           │  IOU + centroid      │                                   ║
║           │  → track IDs, TTC,   │                                   ║
║           │    approach/recede   │                                   ║
║           └──────────┬───────────┘                                   ║
║                      ▼                                               ║
║           ┌──────────────────────┐    ┌──────────────────────────┐  ║
║           │  visualizer.py       │    │  report_generator.py     │  ║
║           │  HUD overlay v5:     │    │  • Frame CSV log         │  ║
║           │  • Minimalist boxes  │    │  • Incidents CSV         │  ║
║           │  • TTC color coding  │    │  • Traffic composition   │  ║
║           │  • BEV threat map    │    │  • Markdown + JSON report│  ║
║           │  • Threat table      │    └──────────────────────────┘  ║
║           └──────────┬───────────┘                                   ║
║                      ▼                                               ║
║         [Annotated Video + Dashboard + Reports]                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Quick Start

### 1. Setup Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Run setup (installs requirements, downloads model weights, runs synthetic unit tests)
python setup.py
```

### 2. Run the Perception System

#### Presenter Script
```bash
python run_test.py
```

#### CLI Interface
```bash
# Webcam live feed
python main.py --source 0 --show

# Video file processing with annotated output
python main.py --source driving.mp4 --output outputs/result.mp4 --show

# High-performance mode (interleaved detection & depth intervals)
python main.py --source driving.mp4 --output outputs/result.mp4 --detect-interval 4 --depth-interval 3
```

---

## Directory Structure

```
d:\adas_ad\
├── .gitignore                # Excludes reference_project/, venv/, outputs/, models/, etc.
├── README.md                 # Complete documentation
├── requirements.txt          # PyTorch, Ultralytics, OpenCV, Timm, etc.
├── setup.py                  # Environment setup & model weights downloader
├── main.py                   # Primary CLI entrypoint
├── run_test.py               # Presenter runner
├── generate_demo.py          # Synthetic drive simulator
├── run_batch.py              # Batch processing runner
├── src/                      # ADAS Core Perception Package
│   ├── __init__.py
│   ├── detector.py           # Object Detection (YOLO11) + Domain Heuristics
│   ├── depth_estimator.py    # Monocular Depth Estimation (MiDaS)
│   ├── fusion.py             # Geometric Pinhole & MiDaS Relative Depth Fusion
│   ├── tracker.py            # Multi-Object Tracker & Time-To-Collision (TTC)
│   ├── visualizer.py         # Dashboard HUD & BEV Radar Threat Map
│   ├── report_generator.py   # Telemetry & Incident Reports (Markdown, JSON, CSV)
│   └── pipeline.py           # Perception Pipeline Orchestrator
└── tests/                    # Synthetic Unit Test Suite
    ├── __init__.py
    └── test_synthetic.py
```
