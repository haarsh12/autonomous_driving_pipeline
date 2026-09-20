"""
report_generator.py
Generates post-run perception summary reports (JSON, Markdown, CSV telemetry logs).
"""

import json
import os
import csv
import time
from collections import Counter
from typing import List, Dict, Any


class PerceptionReportGenerator:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.object_stats: Dict[int, Dict[str, Any]] = {}
        self.total_frames = 0
        self.risk_counts = {"safe": 0, "caution": 0, "warning": 0, "critical": 0}
        self._frame_rows: List[Dict[str, Any]] = []
        self._incident_rows: List[Dict[str, Any]] = []
        self._class_counter: Counter = Counter()

    def log_frame(self, frame_idx: int, timestamp_str: str, fused_objects: List[Any]):
        """Collect frame statistics and record critical/warning events."""
        self.total_frames += 1

        frame_max_risk = "safe"
        for obj in fused_objects:
            tid = obj.track_id
            if tid is None:
                continue

            r = getattr(obj, "risk_level", "safe")
            if r == "critical" and frame_max_risk != "critical":
                frame_max_risk = "critical"
            elif r == "warning" and frame_max_risk not in ("critical", "warning"):
                frame_max_risk = "warning"
            elif r == "caution" and frame_max_risk == "safe":
                frame_max_risk = "caution"

            if tid not in self.object_stats:
                self.object_stats[tid] = {
                    "track_id": tid,
                    "class_name": obj.class_name,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "min_distance_m": obj.distance_m,
                    "max_speed_kmh": obj.relative_speed_kmh,
                    "min_ttc_s": obj.ttc_s if obj.ttc_s != float('inf') else 999.0,
                    "primary_lane": obj.lane_position,
                    "critical_count": 0,
                    "warning_count": 0,
                    "lane_history": [obj.lane_position],
                }
            else:
                st = self.object_stats[tid]
                st["last_frame"] = frame_idx
                st["class_name"] = obj.class_name
                st["min_distance_m"] = min(st["min_distance_m"], obj.distance_m)
                st["max_speed_kmh"] = max(st["max_speed_kmh"], obj.relative_speed_kmh)
                if obj.ttc_s != float('inf'):
                    st["min_ttc_s"] = min(st["min_ttc_s"], obj.ttc_s)
                st["lane_history"].append(obj.lane_position)

            if r == "critical":
                self.object_stats[tid]["critical_count"] += 1
            elif r == "warning":
                self.object_stats[tid]["warning_count"] += 1

            self._class_counter[obj.class_name] += 1

            self._frame_rows.append({
                "frame_idx":    frame_idx,
                "timestamp":    timestamp_str,
                "track_id":     tid,
                "class_name":   obj.class_name,
                "distance_m":   round(obj.distance_m, 2),
                "rel_speed_kmh": round(obj.relative_speed_kmh, 1),
                "ttc_s":        round(obj.ttc_s, 2) if obj.ttc_s != float('inf') else -1,
                "lane":         obj.lane_position,
                "risk_level":   r,
            })

            if obj.ttc_s != float('inf') and obj.ttc_s < 3.0:
                recent_incidents = [i for i in self._incident_rows
                                    if i["track_id"] == tid and (frame_idx - i["frame_idx"]) < 10]
                if not recent_incidents:
                    self._incident_rows.append({
                        "frame_idx":    frame_idx,
                        "timestamp":    timestamp_str,
                        "track_id":     tid,
                        "class_name":   obj.class_name,
                        "distance_m":   round(obj.distance_m, 2),
                        "ttc_s":        round(obj.ttc_s, 2),
                        "lane":         obj.lane_position,
                        "risk_level":   r.upper(),
                    })

            if r in ("critical", "warning"):
                recent_logs = [e for e in self.events if e["track_id"] == tid and (frame_idx - e["frame_idx"]) < 15]
                if not recent_logs:
                    event_data = {
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_str,
                        "track_id": tid,
                        "class_name": obj.class_name,
                        "risk_level": r.upper(),
                        "est_distance_m": obj.distance_m,
                        "rel_speed_kmh": round(obj.relative_speed_kmh, 1),
                        "ttc_s": round(obj.ttc_s, 1) if obj.ttc_s != float('inf') else "N/A",
                        "lane_position": obj.lane_position,
                    }
                    self.events.append(event_data)

        self.risk_counts[frame_max_risk] += 1

    def generate_report(self, video_name: str, total_elapsed_s: float, avg_fps: float) -> Dict[str, str]:
        """Write JSON and Markdown report files and return output file paths."""
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        json_path = os.path.join(self.output_dir, f"perception_summary_{base_name}.json")
        md_path = os.path.join(self.output_dir, f"perception_summary_{base_name}.md")

        unique_objects = len(self.object_stats)
        critical_events = [e for e in self.events if e["risk_level"] == "CRITICAL"]
        warning_events = [e for e in self.events if e["risk_level"] == "WARNING"]

        obj_list = []
        for tid, st in sorted(self.object_stats.items()):
            lanes = st["lane_history"]
            primary_lane = max(set(lanes), key=lanes.count)
            min_ttc = f"{st['min_ttc_s']:.1f}s" if st['min_ttc_s'] < 900.0 else "N/A"
            obj_list.append({
                "track_id": tid,
                "class_name": st["class_name"].upper(),
                "min_est_distance_m": round(st["min_distance_m"], 1),
                "max_speed_kmh": round(st["max_speed_kmh"], 1),
                "min_ttc": min_ttc,
                "primary_lane": primary_lane,
                "critical_warnings": st["critical_count"],
                "total_warnings": st["warning_count"],
            })

        csv_path = os.path.join(self.output_dir, f"frame_log_{base_name}.csv")
        if self._frame_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._frame_rows[0].keys())
                writer.writeheader()
                writer.writerows(self._frame_rows)

        incidents_path = os.path.join(self.output_dir, f"incidents_{base_name}.csv")
        if self._incident_rows:
            with open(incidents_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._incident_rows[0].keys())
                writer.writeheader()
                writer.writerows(self._incident_rows)

        total_detections = sum(self._class_counter.values())
        traffic_composition = {
            cls: {"count": cnt, "pct": round(cnt / max(total_detections, 1) * 100, 1)}
            for cls, cnt in sorted(self._class_counter.items(), key=lambda x: -x[1])
        }

        report_dict = {
            "video_file": video_name,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance": {
                "total_frames_processed": self.total_frames,
                "elapsed_seconds": round(total_elapsed_s, 2),
                "average_fps": round(avg_fps, 2),
            },
            "summary_metrics": {
                "total_unique_tracks": unique_objects,
                "total_critical_events": len(critical_events),
                "total_warning_events": len(warning_events),
                "total_incidents_ttc_under_3s": len(self._incident_rows),
                "risk_distribution_pct": {
                    k: round((v / max(self.total_frames, 1)) * 100, 1) for k, v in self.risk_counts.items()
                },
                "traffic_composition": traffic_composition,
            },
            "tracked_objects": obj_list,
            "event_timeline": self.events,
        }

        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        comp_rows = ""
        for cls, data in traffic_composition.items():
            comp_rows += f"| `{cls.upper()}` | {data['count']} | {data['pct']}% |\n"

        md_content = f"""# Autonomous Perception System — Execution Summary Report

**Video Source:** `{video_name}`  
**Report Generated:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  

---

## Executive Summary & Telemetry

| Metric | Value |
|---|---|
| **Total Frames Processed** | `{self.total_frames}` |
| **Total Processing Time** | `{total_elapsed_s:.2f}s` |
| **Average Pipeline FPS** | `{avg_fps:.2f} FPS` |
| **Total Tracked Objects** | `{unique_objects}` |
| **Critical Near-Miss Events** | `{len(critical_events)}` |
| **Collision Warnings Triggered** | `{len(warning_events)}` |
| **Critical Incidents (TTC < 3s)** | `{len(self._incident_rows)}` |

### System Risk Distribution
- **SAFE**: {report_dict['summary_metrics']['risk_distribution_pct']['safe']}% of frames
- **CAUTION**: {report_dict['summary_metrics']['risk_distribution_pct']['caution']}% of frames
- **WARNING**: {report_dict['summary_metrics']['risk_distribution_pct']['warning']}% of frames
- **CRITICAL**: {report_dict['summary_metrics']['risk_distribution_pct']['critical']}% of frames

---

## Traffic Composition

| Object Class | Detection Count | % Share |
|:---:|:---:|:---:|
{comp_rows}
---

## Tracked Objects Overview

| Track ID | Class | Min Est. Distance | Max Rel. Speed | Min TTC | Primary Lane | Risk Alerts |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
        for o in obj_list:
            md_content += f"| **ID{o['track_id']}** | `{o['class_name']}` | `{o['min_est_distance_m']}m` | `{o['max_speed_kmh']} km/h` | `{o['min_ttc']}` | `{o['primary_lane']}` | `{o['critical_warnings']} Critical / {o['total_warnings']} Warning` |\n"

        md_content += """
---

## Critical & Warning Events Timeline

| Frame | Time | Track ID | Object Class | Risk Level | Est. Distance | Rel. Speed | TTC | Lane Position |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
        if self.events:
            for ev in self.events[:30]:
                risk_str = f"**{ev['risk_level']}**" if ev['risk_level'] == "CRITICAL" else ev['risk_level']
                md_content += f"| F{ev['frame_idx']} | `{ev['timestamp']}` | **ID{ev['track_id']}** | `{ev['class_name'].upper()}` | {risk_str} | `{ev['est_distance_m']}m` | `{ev['rel_speed_kmh']} km/h` | `{ev['ttc_s']}` | `{ev['lane_position']}` |\n"
            if len(self.events) > 30:
                md_content += f"\n*+ {len(self.events) - 30} additional events logged in JSON report.*\n"
        else:
            md_content += "| — | — | — | — | SAFE | — | — | — | — |\n"

        md_content += f"""
---

## Output Files
- **Annotated Video**: `outputs/`
- **Frame-by-Frame CSV Log**: `frame_log_{base_name}.csv`
- **Critical Incidents CSV**: `incidents_{base_name}.csv` ({len(self._incident_rows)} incidents)

---
*Report automatically generated by Autonomous Driving Perception System.*
"""

        with open(md_path, "w") as f:
            f.write(md_content)

        return {"json": json_path, "markdown": md_path, "frame_csv": csv_path, "incidents_csv": incidents_path}
