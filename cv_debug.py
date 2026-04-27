#!/usr/bin/env python3
"""
CV debug tool — runs needle detection on any gauge image and saves an
annotated output image showing exactly what OpenCV found.

Usage:
    python3 cv_debug.py <image_path> [scale_min] [scale_max] [min_clock] [max_clock]

Example (0-20000 gauge, standard 7→5 o'clock layout):
    python3 cv_debug.py gauge.jpg 0 20000 7 5
"""

import sys
import math
import cv2
import numpy as np
from PIL import Image


# ── Core CV functions (mirrors app.py) ───────────────────────────────────────

def clock_to_atan2(h: float) -> int:
    return int((270 + h * 30) % 360)


def find_gauge_center(gray):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=min(h, w) // 2,
        param1=100, param2=30,
        minRadius=min(h, w) // 5,
        maxRadius=min(h, w) // 2,
    )
    if circles is not None:
        cx, cy, r = map(int, np.round(circles[0][0]))
        print(f"  Circle found at ({cx},{cy})  r={r}")
        return cx, cy, r
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.45)
    print(f"  No circle — using image centre ({cx},{cy})  r={r}")
    return cx, cy, r


def detect_needle(gray, cx, cy, r):
    r_inner = max(4, int(r * 0.12))
    r_outer = int(r * 0.80)
    n_pts   = 100
    h, w    = gray.shape

    scores = np.zeros(360, dtype=np.float64)
    for a in range(360):
        rad = np.radians(a)
        t   = np.linspace(r_inner, r_outer, n_pts)
        xs  = np.clip((cx + t * np.cos(rad)).astype(int), 0, w - 1)
        ys  = np.clip((cy + t * np.sin(rad)).astype(int), 0, h - 1)
        scores[a] = np.sum(255.0 - gray[ys, xs])

    # Circular smooth (7°) to reduce tick-mark spikes
    k       = 7
    kernel  = np.ones(k) / k
    padded  = np.concatenate([scores[-k:], scores, scores[:k]])
    smooth  = np.convolve(padded, kernel, mode="same")[k: k + 360]

    candidate = int(np.argmax(smooth))

    # Choose tip (farther, darker) over tail
    def rim_dark(a):
        rad = np.radians(a)
        rx = int(np.clip(cx + r * 0.75 * np.cos(rad), 0, w - 1))
        ry = int(np.clip(cy + r * 0.75 * np.sin(rad), 0, h - 1))
        return float(255 - gray[ry, rx])

    opp = (candidate + 180) % 360
    if rim_dark(opp) > rim_dark(candidate):
        candidate = opp

    return candidate, smooth


def angle_to_reading(needle_angle, s_min, s_max, min_clock, max_clock):
    a_min  = clock_to_atan2(min_clock)
    a_max  = clock_to_atan2(max_clock)
    sweep  = (a_max - a_min) % 360
    if sweep < 45:
        sweep = 270
    delta    = (needle_angle - a_min) % 360
    fraction = float(np.clip(delta / sweep, 0.0, 1.0))
    return round(s_min + fraction * (s_max - s_min), 1), sweep


# ── Visualisation ─────────────────────────────────────────────────────────────

def annotate(img_bgr, cx, cy, r, needle_angle, reading, s_min, s_max,
             min_clock, max_clock, sweep, scores_smooth):
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # Draw detected circle
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)

    # Draw needle line (tip → centre)
    rad    = np.radians(needle_angle)
    tip_x  = int(cx + r * 0.85 * np.cos(rad))
    tip_y  = int(cy + r * 0.85 * np.sin(rad))
    cv2.line(out, (cx, cy), (tip_x, tip_y), (0, 0, 255), 3)
    cv2.circle(out, (tip_x, tip_y), 6, (0, 0, 255), -1)

    # Draw scale arc (min → max)
    a_min = clock_to_atan2(min_clock)
    a_max = clock_to_atan2(max_clock)
    for a_deg in range(0, int(sweep) + 1, 2):
        aa   = (a_min + a_deg) % 360
        rad2 = np.radians(aa)
        px   = int(cx + r * 0.92 * np.cos(rad2))
        py   = int(cy + r * 0.92 * np.sin(rad2))
        cv2.circle(out, (px, py), 1, (255, 200, 0), -1)

    # Mark MIN and MAX on arc
    for clk, label in [(min_clock, f"MIN {s_min:.0f}"), (max_clock, f"MAX {s_max:.0f}")]:
        aa  = clock_to_atan2(clk)
        rad2 = np.radians(aa)
        px  = int(cx + r * 0.92 * np.cos(rad2))
        py  = int(cy + r * 0.92 * np.sin(rad2))
        cv2.circle(out, (px, py), 6, (255, 200, 0), -1)
        cv2.putText(out, label, (px + 4, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # Reading label
    label = f"Reading: {reading:.1f}   needle={needle_angle}deg"
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, f"Scale {s_min:.0f}-{s_max:.0f}  sweep={sweep:.0f}deg", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    cv2.putText(out, f"Min@{min_clock}oclock  Max@{max_clock}oclock", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cv_debug.py <image> [min] [max] [min_clock] [max_clock]")
        sys.exit(1)

    path      = sys.argv[1]
    s_min     = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    s_max     = float(sys.argv[3]) if len(sys.argv) > 3 else 100
    min_clock = float(sys.argv[4]) if len(sys.argv) > 4 else 7.5
    max_clock = float(sys.argv[5]) if len(sys.argv) > 5 else 4.5

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        # Try via PIL for exotic formats
        pil = Image.open(path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    print(f"\nImage : {path}  ({img_bgr.shape[1]}×{img_bgr.shape[0]})")
    print(f"Scale : {s_min}–{s_max}  |  MIN @ {min_clock} o'clock  MAX @ {max_clock} o'clock\n")

    cx, cy, r = find_gauge_center(gray)
    needle, smooth = detect_needle(gray, cx, cy, r)
    reading, sweep = angle_to_reading(needle, s_min, s_max, min_clock, max_clock)

    print(f"  Needle angle : {needle}°")
    print(f"  a_min (atan2): {clock_to_atan2(min_clock)}°")
    print(f"  a_max (atan2): {clock_to_atan2(max_clock)}°")
    print(f"  Sweep        : {sweep}°")
    delta    = (needle - clock_to_atan2(min_clock)) % 360
    fraction = delta / sweep
    print(f"  Delta        : {delta}°  →  fraction={fraction:.3f}")
    print(f"\n  *** READING  : {reading} ***\n")

    # Top-5 candidate angles
    top5 = np.argsort(smooth)[-5:][::-1]
    print("  Top-5 radial scores:", [(int(a), round(float(smooth[a]), 0)) for a in top5])

    out_path = path.rsplit(".", 1)[0] + "_cv_debug.jpg"
    annotated = annotate(img_bgr, cx, cy, r, needle, reading, s_min, s_max,
                         min_clock, max_clock, sweep, smooth)
    cv2.imwrite(out_path, annotated)
    print(f"\n  Annotated image saved → {out_path}")


if __name__ == "__main__":
    main()
