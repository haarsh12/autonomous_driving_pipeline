"""
visualizer.py
Visual Overlay & Sidebar Dashboard with:
  - Minimalist bounding box labels: [ID] [CLASS] | [Distance]
  - TTC-based dynamic color coding
  - Alert banners & HUD telemetry panel
  - BEV Radar Threat Map with lane corridors and semi-circle distance arcs
"""

from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import time
import datetime

from fusion import FusedObject

SIDEBAR_WIDTH = 390

# Palette (BGR)
C_CRITICAL = (0, 30, 220)       # Vivid Red
C_WARNING  = (0, 140, 255)      # Amber / Orange
C_CAUTION  = (0, 215, 255)      # Yellow
C_SAFE     = (0, 200, 70)       # Vivid Green

C_DARK_BG  = (12, 12, 14)
C_PANEL    = (22, 22, 26)
C_PANEL2   = (30, 30, 36)
C_BORDER   = (50, 50, 58)
C_BORDER2  = (38, 38, 44)
C_TEXT_DIM = (100, 100, 115)
C_TEXT_MID = (175, 175, 190)
C_TEXT_HI  = (230, 230, 245)
C_ACCENT   = (0, 210, 255)      # Cyan
C_EGO      = (255, 110, 20)     # Orange Ego vehicle
C_WHITE    = (255, 255, 255)

_START_TIME = time.time()


def get_risk_color(risk_str: str) -> tuple:
    if risk_str == "critical":
        return C_CRITICAL
    elif risk_str == "warning":
        return C_WARNING
    elif risk_str == "caution":
        return C_CAUTION
    return C_SAFE


def get_ttc_color(ttc_s: float) -> tuple:
    if ttc_s <= 3.0:
        return C_CRITICAL
    elif ttc_s <= 6.0:
        return C_WARNING
    elif ttc_s <= 10.0:
        return C_CAUTION
    return C_SAFE


def risk_level(obj: FusedObject) -> str:
    return getattr(obj, "risk_level", "safe")


def generate_warnings(objects: List[FusedObject], tracker=None) -> List[str]:
    warnings = []
    for obj in objects:
        r = getattr(obj, "risk_level", "safe")
        label = obj.class_name.replace("_", " ").capitalize()
        dist_str = f"Est. Dist: {obj.distance_m:.1f}m"
        ttc_str = f"TTC: {obj.ttc_s:.1f}s" if obj.ttc_s != float('inf') else ""
        lane_str = getattr(obj, "lane_position", "EGO LANE")

        if r == "critical":
            msg = f"EMERGENCY: {label} in {lane_str} at {dist_str}"
            if ttc_str:
                msg += f" | {ttc_str}"
            warnings.append(msg)
        elif r == "warning":
            msg = f"WARNING: {label} approaching in {lane_str} ({dist_str})"
            warnings.append(msg)
        elif r == "caution" and obj.ttc_s < 7.0:
            msg = f"Caution: {label} ({dist_str})"
            warnings.append(msg)
    return warnings


def _pt(img, text, pos, font, scale, color, thick=1):
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


def _bracket(img, x1, y1, x2, y2, color, thick=2, arm=14):
    corners = [
        [(x1, y1+arm), (x1, y1), (x1+arm, y1)],
        [(x2-arm, y1), (x2, y1), (x2, y1+arm)],
        [(x2, y2-arm), (x2, y2), (x2-arm, y2)],
        [(x1+arm, y2), (x1, y2), (x1, y2-arm)],
    ]
    for c in corners:
        cv2.polylines(img, [np.array(c, np.int32)], False, color, thick, cv2.LINE_AA)


def _hr(canvas, x1, x2, y, color=C_BORDER):
    cv2.line(canvas, (x1, y), (x2, y), color, 1, cv2.LINE_AA)


def draw_overlay(frame, objects: List[FusedObject], tracker=None,
                 fps: Optional[float] = None,
                 frame_idx: int = 0, total_frames: int = 0,
                 event_log: Optional[List[Dict[str, Any]]] = None):
    """
    Render the autonomous perception dashboard.
    Returns annotated canvas (H x W+SIDEBAR_WIDTH x 3).
    """
    H, W = frame.shape[:2]
    t = time.time()
    elapsed = t - _START_TIME
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    max_risk = "safe"
    closest_m = 999.0
    min_ttc_s = float('inf')

    for obj in objects:
        r = getattr(obj, "risk_level", "safe")
        if r == "critical":
            max_risk = "critical"
        elif r == "warning" and max_risk != "critical":
            max_risk = "warning"
        elif r == "caution" and max_risk == "safe":
            max_risk = "caution"

        if obj.distance_m < closest_m:
            closest_m = obj.distance_m
        if obj.ttc_s < min_ttc_s:
            min_ttc_s = obj.ttc_s

    # 1. ANNOTATE VIDEO PANE
    video = frame.copy()
    drawn_pills = []

    # Sort objects by distance (farthest first) so closer objects get primary label positioning
    sorted_objs = sorted(objects, key=lambda o: -o.distance_m)

    for obj in sorted_objs:
        x1, y1, x2, y2 = [int(v) for v in obj.box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)

        risk = getattr(obj, "risk_level", "safe")
        color = get_ttc_color(obj.ttc_s) if obj.ttc_s != float('inf') else get_risk_color(risk)
        status = tracker.get_movement_status(obj.track_id) if (tracker and obj.track_id) else ""

        fill_col = {"critical": (0, 0, 60), "warning": (0, 40, 60), "caution": (0, 45, 45), "safe": (0, 40, 0)}[risk]
        cv2.rectangle(video, (x1, y1), (x2, y2), fill_col, 1)

        arm = max(10, min(20, (x2-x1) // 4))
        _bracket(video, x1, y1, x2, y2, color, thick=2, arm=arm)

        cx_box = (x1+x2)//2
        arrow = "^" if status == "approaching" else ("v" if status == "receding" else "*")
        _pt(video, arrow, (cx_box-5, y2+15), FONT, 0.5, color, 2)

        id_str = f"ID{obj.track_id} " if obj.track_id else ""
        cls_str = obj.class_name.upper().replace("_", " ")
        label = f"{id_str}{cls_str} | {obj.distance_m:.1f}m"

        (lw, lh), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        lpad = 4
        pill_w = lw + lpad * 2
        pill_h = lh + lpad * 2

        lx = max(0, min(W - pill_w, x1))
        ly = max(0, y1 - pill_h - 2)

        # Smart Overlap Resolution: shift label vertically if it collides with another pill
        pill_rect = [lx, ly, lx + pill_w, ly + pill_h]
        for _ in range(4):
            collision = False
            for rx1, ry1, rx2, ry2 in drawn_pills:
                if not (pill_rect[2] < rx1 or pill_rect[0] > rx2 or pill_rect[3] < ry1 or pill_rect[1] > ry2):
                    collision = True
                    break
            if collision:
                ly = max(0, ly - pill_h - 3)
                pill_rect = [lx, ly, lx + pill_w, ly + pill_h]
            else:
                break

        drawn_pills.append(pill_rect)

        cv2.rectangle(video, (pill_rect[0], pill_rect[1]), (pill_rect[2], pill_rect[3]), (0, 0, 0), -1)
        cv2.rectangle(video, (pill_rect[0], pill_rect[1]), (pill_rect[2], pill_rect[3]), color, 1)
        _pt(video, label, (pill_rect[0] + lpad, pill_rect[1] + lh + lpad - 1), FONT, 0.45, color, 1)

    # Top-left warning banners
    warnings = generate_warnings(objects, tracker)
    wy = 32
    for warn in warnings[:3]:
        is_crit = "EMERGENCY" in warn
        wc = C_CRITICAL if is_crit else C_WARNING
        flash = (int(t*5) % 2 == 0) if is_crit else True
        if flash:
            (ww, wh), _ = cv2.getTextSize(warn, FONT, 0.55, 2)
            cv2.rectangle(video, (6, wy - wh - 6), (18 + ww, wy + 8), (0, 0, 0), -1)
            cv2.rectangle(video, (6, wy - wh - 6), (18 + ww, wy + 8), wc, 1)
            _pt(video, warn, (10, wy), FONT, 0.55, wc, 2)
        wy += 34

    # Bottom status strip
    strip_h = 28
    cv2.rectangle(video, (0, H-strip_h), (W, H), (0, 0, 0), -1)
    fps_label = f"FPS: {fps:.1f}" if fps else "FPS: --"
    time_label = f"{datetime.datetime.now().strftime('%H:%M:%S')}"
    frame_label = f"Frame {frame_idx}/{total_frames}" if total_frames else f"Frame {frame_idx}"
    _pt(video, fps_label, (10, H-10), FONT, 0.40, C_ACCENT, 1)
    _pt(video, time_label, (90, H-10), FONT, 0.40, C_TEXT_MID, 1)
    _pt(video, frame_label, (W-130, H-10), FONT, 0.40, C_TEXT_DIM, 1)

    if total_frames > 0:
        prog = frame_idx / total_frames
        cv2.rectangle(video, (0, H-4), (W, H), (30, 30, 30), -1)
        cv2.rectangle(video, (0, H-4), (int(W*prog), H), C_ACCENT, -1)

    # 2. SIDEBAR DASHBOARD CANVAS
    canvas = np.full((H, W + SIDEBAR_WIDTH, 3), C_DARK_BG, dtype=np.uint8)
    canvas[:, :W] = video
    cv2.line(canvas, (W, 0), (W, H), C_BORDER, 1, cv2.LINE_AA)

    SX  = W + 12
    SR  = W + SIDEBAR_WIDTH - 12
    MID = W + SIDEBAR_WIDTH // 2
    cy  = 14

    # HEADER
    cv2.rectangle(canvas, (W, 0), (W+SIDEBAR_WIDTH, 46), C_PANEL2, -1)
    _pt(canvas, "AUTONOMY HUD v4", (SX, cy+14), FONT, 0.55, C_ACCENT, 2)

    rec_alpha = 0.5 + 0.5 * abs(np.sin(t * 2.5))
    rec_col = tuple(int(c * rec_alpha) for c in C_CRITICAL)
    cv2.circle(canvas, (SR-8, cy+8), 5, rec_col, -1, cv2.LINE_AA)
    _pt(canvas, "REC", (SR-32, cy+13), FONT, 0.32, C_TEXT_DIM, 1)

    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    elapsed_str = f"+{int(elapsed//60):02d}:{int(elapsed%60):02d}"
    _pt(canvas, now_str, (SX, cy+30), FONT, 0.36, C_TEXT_DIM, 1)
    _pt(canvas, elapsed_str, (SX+75, cy+30), FONT, 0.36, C_TEXT_DIM, 1)
    cy = 50
    _hr(canvas, SX, SR, cy)
    cy += 8

    # DYNAMIC ALERT BANNER
    banner_h = 32
    if max_risk == "critical":
        flash_on = int(t*6)%2 == 0
        b_col = (0, 0, 200) if flash_on else (0, 0, 90)
        b_text = "! EMERGENCY BRAKE !"
        b_tc = C_WHITE
    elif max_risk == "warning":
        b_col = (0, 100, 185)
        b_text = "▲ COLLISION RISK ▲"
        b_tc = C_WHITE
    elif max_risk == "caution":
        b_col = (0, 130, 180)
        b_text = "CAUTION PRE-ALERT"
        b_tc = C_WHITE
    else:
        b_col = (0, 88, 28)
        b_text = "✔ SYSTEM ALL CLEAR"
        b_tc = (190, 255, 190)

    cv2.rectangle(canvas, (SX, cy), (SR, cy+banner_h), b_col, -1)
    cv2.rectangle(canvas, (SX, cy), (SR, cy+banner_h), C_BORDER, 1)
    (btw, bth), _ = cv2.getTextSize(b_text, FONT, 0.50, 2)
    _pt(canvas, b_text, (MID-btw//2, cy+(banner_h+bth)//2), FONT, 0.50, b_tc, 2)
    cy += banner_h + 6

    # PROXIMITY & TTC GAUGE
    gauge_h = 14
    gauge_max = 30.0
    gauge_ratio = max(0.0, 1.0 - min(closest_m, gauge_max) / gauge_max)
    fill_px = int((SR - SX) * gauge_ratio)

    cv2.rectangle(canvas, (SX, cy), (SR, cy+gauge_h), C_PANEL2, -1)
    cv2.rectangle(canvas, (SX, cy), (SR, cy+gauge_h), C_BORDER2, 1)

    if fill_px > 0:
        g_col = C_CRITICAL if gauge_ratio > 0.7 else C_WARNING if gauge_ratio > 0.4 else C_SAFE
        cv2.rectangle(canvas, (SX, cy), (SX+fill_px, cy+gauge_h), g_col, -1)

    ttc_disp = f" | Min TTC: {min_ttc_s:.1f}s" if min_ttc_s < 90.0 else ""
    dist_lbl = f"EST. DIST: {closest_m:.1f}m{ttc_disp}" if closest_m < 999 else "EST. DIST: --"
    _pt(canvas, dist_lbl, (SX+4, cy+gauge_h-3), FONT, 0.30, C_TEXT_HI, 1)
    cy += gauge_h + 8
    _hr(canvas, SX, SR, cy)
    cy += 6

    # TELEMETRY PANEL
    _pt(canvas, "TELEMETRY", (SX, cy+10), FONT, 0.38, C_TEXT_MID, 1)
    cy += 13
    _hr(canvas, SX, SR, cy, C_BORDER2)
    cy += 5

    active_tracks = len(set(o.track_id for o in objects if o.track_id is not None))
    fps_val = f"{fps:.1f}" if fps else "—"
    base_spd = 52.0 + 5.0 * np.sin(t * 0.25)
    spd = (max(8.0, base_spd-30) if max_risk == "critical" else
           max(36.0, base_spd-13) if max_risk == "warning" else base_spd)

    tel = [
        ("TARGETS IN VIEW", str(active_tracks)),
        ("PIPELINE FPS",    fps_val),
        ("CLOSEST EST. DIST", f"{closest_m:.1f}m" if closest_m < 999 else "—"),
        ("MIN TTC",         f"{min_ttc_s:.1f}s" if min_ttc_s < 90.0 else "N/A"),
        ("EGO SPEED",       f"{spd:.1f} km/h"),
        ("FRAME",           f"{frame_idx}/{total_frames}" if total_frames else str(frame_idx)),
    ]
    for lbl, val in tel:
        _pt(canvas, lbl, (SX, cy+9), FONT, 0.33, C_TEXT_MID, 1)
        vw = cv2.getTextSize(val, FONT, 0.37, 1)[0][0]
        _pt(canvas, val, (SR-vw, cy+9), FONT, 0.37, C_TEXT_HI, 1)
        cy += 14

    cy += 4
    _hr(canvas, SX, SR, cy)
    cy += 6

    # TOP-5 THREAT TABLE
    _pt(canvas, "TOP-5 THREAT TABLE  (by TTC)", (SX, cy+10), FONT, 0.36, C_TEXT_MID, 1)
    cy += 13
    _hr(canvas, SX, SR, cy, C_BORDER2)
    cy += 3

    COL = [SX+2, SX+88, SX+140, SX+192, SX+248, SR-32]
    for hdr, cx_h in zip(["CLASS", "DIST", "SPD", "TTC", "LANE", "RISK"], COL):
        _pt(canvas, hdr, (cx_h, cy+9), FONT, 0.29, C_ACCENT, 1)
    cy += 12
    _hr(canvas, SX, SR, cy, C_BORDER2)
    cy += 3

    def _sort_key_ttc(o):
        ttc = o.ttc_s if o.ttc_s != float('inf') else 9999.0
        return (ttc, o.distance_m)

    sorted_objs = sorted(objects, key=_sort_key_ttc)
    top5 = sorted_objs[:5]
    ROW_H = 16

    for obj in top5:
        if cy + ROW_H > H - 215:
            break

        risk = getattr(obj, "risk_level", "safe")
        color = get_ttc_color(obj.ttc_s) if obj.ttc_s != float('inf') else get_risk_color(risk)
        id_tag = f"[{obj.track_id}]" if obj.track_id else "[--]"
        cls_s  = f"{id_tag}{obj.class_name[:5].upper()}"
        dist_s = f"{obj.distance_m:.1f}m"
        speed_s = f"{obj.relative_speed_kmh:+.0f}k/h"
        ttc_s  = f"{obj.ttc_s:.1f}s" if obj.ttc_s != float('inf') else "--"
        lane_s = getattr(obj, "lane_position", "EGO")[:4]

        row_bg = (28, 0, 0) if risk == "critical" else (20, 18, 0) if risk == "warning" else (15, 15, 0) if risk == "caution" else (0, 20, 0)
        cv2.rectangle(canvas, (SX, cy), (SR, cy+ROW_H), row_bg, -1)

        _pt(canvas, cls_s,   (COL[0], cy+11), FONT, 0.31, color, 1)
        _pt(canvas, dist_s,  (COL[1], cy+11), FONT, 0.31, C_TEXT_HI, 1)
        _pt(canvas, speed_s, (COL[2], cy+11), FONT, 0.31, C_TEXT_MID, 1)
        ttc_color = C_CRITICAL if obj.ttc_s < 3.0 else C_WARNING if obj.ttc_s < 6.0 else C_TEXT_MID
        _pt(canvas, ttc_s,   (COL[3], cy+11), FONT, 0.31, ttc_color, 1)
        _pt(canvas, lane_s,  (COL[4], cy+11), FONT, 0.31, C_ACCENT if lane_s == "EGO" else C_TEXT_DIM, 1)
        _pt(canvas, risk[:4].upper(), (COL[5], cy+11), FONT, 0.31, color, 1)

        cy += ROW_H

    remaining = len(sorted_objs) - 5
    if remaining > 0:
        _pt(canvas, f"  +{remaining} more targets tracked...", (SX, cy+10), FONT, 0.30, C_TEXT_DIM, 1)
        cy += 14

    # RECENT EVENT LOG
    cy = max(cy + 4, H - 210)
    _hr(canvas, SX, SR, cy)
    cy += 6
    _pt(canvas, "RECENT EVENT LOG", (SX, cy+9), FONT, 0.35, C_TEXT_DIM, 1)
    cy += 12
    _hr(canvas, SX, SR, cy, C_BORDER2)
    cy += 3

    if event_log:
        for ev in event_log[-3:]:
            r_col = C_CRITICAL if ev.get("risk_level") == "CRITICAL" else C_WARNING
            ev_str = f"F{ev['frame_idx']}: {ev['class_name'].upper()} [{ev['lane_position'][:3]}] {ev['est_distance_m']}m"
            if ev.get('ttc_s') != "N/A":
                ev_str += f" TTC:{ev['ttc_s']}s"
            _pt(canvas, ev_str, (SX, cy+9), FONT, 0.30, r_col, 1)
            cy += 13
    else:
        _pt(canvas, "No critical risk events logged", (SX, cy+9), FONT, 0.30, C_TEXT_DIM, 1)
        cy += 13

    # BEV THREAT MAP
    R_MARGIN = 6
    R_TITLE  = 16
    R_H      = 165
    ry0 = H - R_H - R_MARGIN - R_TITLE
    ry1 = H - R_MARGIN
    rx0 = SX
    rx1 = SR
    rcx = MID

    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), C_PANEL, -1)
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), C_BORDER, 1)
    _pt(canvas, "BEV THREAT MAP", (rx0+8, ry0+12), FONT, 0.35, C_ACCENT, 1)
    _hr(canvas, rx0, rx1, ry0+R_TITLE, C_BORDER)

    py0 = ry0 + R_TITLE + 2
    py1 = ry1 - 4
    ph  = py1 - py0
    MAX_D = 50.0

    for d in [10, 20, 30, 40, 50]:
        ry_line = int(py1 - (d / MAX_D) * ph)
        if ry_line > py0:
            cv2.line(canvas, (rx0, ry_line), (rx1, ry_line), C_BORDER2, 1, cv2.LINE_AA)
            _pt(canvas, f"{d}m", (rx0+3, ry_line-2), FONT, 0.24, (60, 60, 70), 1)

    lane_w   = 20
    ego_l    = rcx - lane_w
    ego_r    = rcx + lane_w
    adj_l    = rcx - lane_w * 3
    adj_r    = rcx + lane_w * 3

    for seg_y in range(py0, py1, 10):
        if seg_y + 6 < py1:
            cv2.line(canvas, (ego_l, seg_y), (ego_l, seg_y+6), (80, 80, 105), 1, cv2.LINE_AA)
            cv2.line(canvas, (ego_r, seg_y), (ego_r, seg_y+6), (80, 80, 105), 1, cv2.LINE_AA)

    for seg_y in range(py0, py1, 12):
        if seg_y + 5 < py1:
            for lx_l in [adj_l, adj_r]:
                if rx0 < lx_l < rx1:
                    cv2.line(canvas, (lx_l, seg_y), (lx_l, seg_y+5), (45, 45, 55), 1, cv2.LINE_AA)

    ego_y = py1 - 12
    ego_pts = np.array([[rcx-6, ego_y], [rcx+6, ego_y], [rcx, ego_y-13]], np.int32)
    cv2.fillPoly(canvas, [ego_pts], C_EGO)
    cv2.polylines(canvas, [ego_pts], True, (255, 190, 130), 1, cv2.LINE_AA)

    for d_m, arc_col, lbl_col in [
        (10, (0, 100, 100), (80, 160, 160)),
        (20, (0,  70,  70), (60, 120, 120)),
        (30, (0,  50,  50), (50,  90,  90)),
    ]:
        arc_r = int((d_m / MAX_D) * ph)
        cv2.ellipse(canvas, (rcx, ego_y), (arc_r, arc_r), 0, 180, 360, arc_col, 1, cv2.LINE_AA)
        label_x = min(rcx + arc_r + 3, rx1 - 18)
        label_y = ego_y - 2
        if py0 < label_y < py1:
            _pt(canvas, f"{d_m}m", (label_x, label_y), FONT, 0.26, lbl_col, 1)

    sweep_angle = (t * 60) % 360
    sweep_rad = np.deg2rad(-sweep_angle + 90)
    sweep_len = (rx1 - rx0) // 2 - 8
    for seg in range(4):
        frac = seg / 4.0
        seg_angle = sweep_rad + np.deg2rad(frac * 25)
        ex = int(rcx + sweep_len * np.cos(seg_angle))
        ey = int(ego_y - abs(np.sin(seg_angle)) * ph)
        ey = int(np.clip(ey, py0, ego_y))
        alpha_c = int(50 * (1.0 - frac))
        cv2.line(canvas, (rcx, ego_y), (ex, ey), (0, alpha_c, 0), 1, cv2.LINE_AA)

    for obj in objects:
        d = obj.distance_m
        if d > MAX_D:
            continue
        obj_y = int(ego_y - (d / MAX_D) * ph)
        obj_y = int(np.clip(obj_y, py0, ego_y))
        offset = (obj.center[0] - W / 2.0) / (W / 2.0)
        obj_x = int(rcx + offset * ((rx1 - rcx) - 16))
        obj_x = int(np.clip(obj_x, rx0+6, rx1-6))

        risk = getattr(obj, "risk_level", "safe")
        rc = get_ttc_color(obj.ttc_s) if obj.ttc_s != float('inf') else get_risk_color(risk)

        if risk == "critical":
            pulse = 6 + int(4 * abs(np.sin(t*5)))
            cv2.circle(canvas, (obj_x, obj_y), pulse, rc, 1, cv2.LINE_AA)

        cv2.circle(canvas, (obj_x, obj_y), 4, rc, -1, cv2.LINE_AA)
        cv2.circle(canvas, (obj_x, obj_y), 4, C_TEXT_HI, 1, cv2.LINE_AA)

        if obj.track_id is not None:
            blip_lbl = f"{obj.track_id}"
            if obj.ttc_s < 10.0:
                blip_lbl += f" ({obj.ttc_s:.1f}s)"
            _pt(canvas, blip_lbl, (obj_x+6, obj_y+3), FONT, 0.28, C_TEXT_MID, 1)

    return canvas
