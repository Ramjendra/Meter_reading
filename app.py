#!/usr/bin/env python3
"""
Meter Reading Web App  —  Pure OpenCV + Tesseract OCR  (no ML model)
Run:  python3 app.py
Open: http://localhost:5000
"""

import logging
import math
import re
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Known unit strings — searched inside OCR text from the dial face
KNOWN_UNITS = [
    (r"°\s*C",  "°C"),   (r"°\s*F",  "°F"),
    (r"\bPSI\b","PSI"),  (r"\bpsi\b","PSI"),
    (r"\bkPa\b","kPa"),  (r"\bMPa\b","MPa"),  (r"\bPa\b","Pa"),
    (r"\bbar\b","bar"),  (r"\bBAR\b","bar"),
    (r"\bRPM\b","RPM"),  (r"\brpm\b","RPM"),
    (r"\bHz\b", "Hz"),   (r"\bkHz\b","kHz"),  (r"\bMHz\b","MHz"),
    (r"\bV\b",  "V"),    (r"\bA\b",  "A"),
    (r"\b%\b",  "%"),
]

app = Flask(__name__)
progress_state = {"current": 0, "total": 0, "file": "", "done": False}

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = "meter_app.log"


class _NoProgressFilter(logging.Filter):
    def filter(self, record):
        return "GET /progress" not in record.getMessage()


_file_h   = logging.FileHandler(LOG_FILE, encoding="utf-8")
_stream_h = logging.StreamHandler()
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
for _h in (_file_h, _stream_h):
    _h.setFormatter(_fmt)
    _h.addFilter(_NoProgressFilter())

logging.basicConfig(level=logging.DEBUG, handlers=[_file_h, _stream_h])
log = logging.getLogger("meter_app")

for _n in ("PIL", "urllib3", "pytesseract"):
    logging.getLogger(_n).setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.INFO)

# ─── Startup banner (no model loading) ───────────────────────────────────────

log.info("=" * 60)
log.info("Meter Reader  —  Pure OpenCV + Tesseract  (no ML model)")
log.info(f"Uploads : {UPLOAD_FOLDER.resolve()}")
log.info(f"Log     : {Path(LOG_FILE).resolve()}")
log.info("=" * 60)



# ─── Layer 1 : OCR scale-number detection ────────────────────────────────────

def detect_scale_numbers(gray: np.ndarray, cx: int, cy: int, r: int) -> list:
    """
    Find numbers printed on the dial face.
    Returns [(angle_deg, value), …] in image-atan2 convention (0=right, CW, y-down).
    Only numbers in the annular scale ring (25 %–115 % of r) are kept.
    """
    h, w = gray.shape

    # Up-scale aggressively — Tesseract needs ≥ 20 px per character height
    # Target: dial diameter of at least 600 px after scaling
    target_diam = 600
    scale = max(1.0, target_diam / (2.0 * r))
    up = cv2.resize(gray, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)

    # Sharpen + contrast enhance for cleaner digit edges
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    up = clahe.apply(up)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    up = cv2.filter2D(up, -1, kernel)

    # Minimum text-box width in up-scaled pixels to reject stray digit fragments
    # (e.g. "2" split from "20" will be narrower than a full 2-char label)
    min_box_w = int(scale * r * 0.04)   # roughly 1 character width at scale

    data = pytesseract.image_to_data(
        up,
        config="--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DICT,
    )

    seen = {}   # value → (angle, conf) — keep highest confidence per value
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i])
        if not text or conf < 30:
            continue
        try:
            value = float(text)
        except ValueError:
            continue

        box_w = int(data["width"][i])
        # Single-digit boxes that are too narrow are likely split fragments
        if len(text) == 1 and box_w < min_box_w:
            log.debug(f"  OCR skip: '{text}' box_w={box_w} < min {min_box_w}")
            continue

        # Pixel centre of this text box in original image coordinates
        bx = (data["left"][i] + data["width"][i]  / 2.0) / scale
        by = (data["top"][i]  + data["height"][i] / 2.0) / scale
        dist = math.hypot(bx - cx, by - cy)

        if dist < r * 0.25 or dist > r * 1.20:
            log.debug(f"  OCR skip: {value:.0f} dist/r={dist/r:.2f} out of ring")
            continue

        angle = math.degrees(math.atan2(by - cy, bx - cx)) % 360
        if value not in seen or conf > seen[value][1]:
            seen[value] = (angle, conf)
            log.debug(f"  OCR keep: {value:.0f} @ {angle:.1f}°  conf={conf}  box_w={box_w}")

    # Drop stray small values when scale is clearly in thousands (e.g. 0-20000 gauge)
    if seen:
        max_v = max(seen.keys())
        if max_v >= 1000:
            seen = {v: info for v, info in seen.items()
                    if v == 0.0 or v >= max_v * 0.01}

    pts = [(a, v) for v, (a, _) in seen.items()]
    log.info(f"  OCR: {len(pts)} scale numbers  →  {sorted(v for _, v in pts)}")
    return pts


# ─── Layer 1 calibration : angle → value linear fit ──────────────────────────

def build_calibration(pts: list):
    """
    Fit a linear CW-angle → value mapping from OCR calibration points.
    Returns dict {a_ref, v_ref, slope} or None if < 2 points found.

        reading = v_ref + slope × (needle_angle − a_ref)  mod 360  (CW)
    """
    if len(pts) < 2:
        return None

    pts_sorted = sorted(pts, key=lambda p: p[1])   # by value ascending
    a_ref, v_ref = pts_sorted[0]

    # CW angular offsets from the lowest-value mark
    deltas = np.array([(a - a_ref) % 360 for a, _ in pts_sorted], dtype=float)
    values = np.array([v for _, v in pts_sorted], dtype=float)

    # If larger values have larger CW offsets — standard (CW-increasing) gauge
    if np.corrcoef(deltas, values)[0, 1] < 0:
        # CCW-increasing gauge — flip reference to the highest-value mark
        a_ref, v_ref = pts_sorted[-1]
        deltas = np.array([(a - a_ref) % 360 for a, _ in reversed(pts_sorted)], dtype=float)
        values = np.array([v for _, v in reversed(pts_sorted)], dtype=float)
        log.debug("  Calib: CCW direction detected")

    slope = float(np.polyfit(deltas, values, 1)[0])   # units per degree
    log.info(f"  Calib: ref={v_ref:.0f}@{a_ref:.1f}°  slope={slope:.2f} u/deg  ({len(pts)} pts)")
    return {"a_ref": a_ref, "v_ref": v_ref, "slope": slope}


def calibrated_reading(needle_angle: int, calib: dict) -> float:
    delta = (needle_angle - calib["a_ref"]) % 360
    return round(calib["v_ref"] + calib["slope"] * delta, 1)


# ─── Unit / type detection via OCR ──────────────────────────────────────────

def detect_unit_ocr(gray: np.ndarray, cx: int, cy: int, r: int) -> tuple:
    """
    Read all text on the dial face (wider whitelist) and match against KNOWN_UNITS.
    Returns (meter_type_guess, unit_str).  Falls back to ("gauge", "") on failure.
    """
    target_diam = 600
    scale = max(1.0, target_diam / (2.0 * r))
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    up = clahe.apply(up)

    raw = pytesseract.image_to_string(up, config="--psm 11 --oem 3")
    log.debug(f"  Unit OCR raw: {raw.strip()!r}")

    unit = ""
    for pattern, label in KNOWN_UNITS:
        if re.search(pattern, raw):
            unit = label
            break

    type_guess = "thermometer" if unit in ("°C", "°F") else \
                 "pressure gauge" if unit in ("PSI", "bar", "kPa", "MPa", "Pa") else \
                 "tachometer" if unit == "RPM" else \
                 "voltmeter" if unit == "V" else \
                 "ammeter" if unit == "A" else \
                 "gauge"

    log.info(f"  Unit OCR: type={type_guess!r}  unit={unit!r}")
    return type_guess, unit


# ─── Layer 2 : CV needle detection ───────────────────────────────────────────

def _find_gauge_center(gray: np.ndarray):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (11, 11), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=min(h, w) // 2, param1=100, param2=30,
        minRadius=min(h, w) // 5, maxRadius=min(h, w) // 2,
    )
    if circles is not None:
        cx, cy, r = map(int, np.round(circles[0][0]))
        log.debug(f"  CV: circle ({cx},{cy}) r={r}")
        return cx, cy, r
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.45)
    log.debug(f"  CV: fallback centre ({cx},{cy}) r={r}")
    return cx, cy, r


def detect_needle_angle(img_pil: Image.Image, cx=None, cy=None, r=None) -> tuple:
    """
    Radial-sweep needle detector.
    Returns (angle_deg, cx, cy, r).  angle_deg: 0=right, CW, y-down.
    """
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if cx is None:
        cx, cy, r = _find_gauge_center(gray)

    r_inner, r_outer, n_pts = max(4, int(r * 0.12)), int(r * 0.80), 80
    scores = np.zeros(360, dtype=np.float64)
    for a in range(360):
        rad = np.radians(a)
        t   = np.linspace(r_inner, r_outer, n_pts)
        xs  = np.clip((cx + t * np.cos(rad)).astype(int), 0, w - 1)
        ys  = np.clip((cy + t * np.sin(rad)).astype(int), 0, h - 1)
        scores[a] = np.sum(255.0 - gray[ys, xs])

    k = 7
    pad      = np.concatenate([scores[-k:], scores, scores[:k]])
    smoothed = np.convolve(pad, np.ones(k) / k, mode="same")[k: k + 360]
    candidate = int(np.argmax(smoothed))

    def rim_dark(a):
        rad = np.radians(a)
        rx = int(np.clip(cx + r * 0.75 * np.cos(rad), 0, w - 1))
        ry = int(np.clip(cy + r * 0.75 * np.sin(rad), 0, h - 1))
        return float(255 - gray[ry, rx])

    opp = (candidate + 180) % 360
    if rim_dark(opp) > rim_dark(candidate):
        candidate = opp

    log.debug(f"  CV: needle={candidate}°  centre=({cx},{cy}) r={r}")
    return candidate, cx, cy, r



# ─── Main pipeline ────────────────────────────────────────────────────────────

def read_meter(img: Image.Image) -> dict:
    t0 = time.time()

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Shared gauge centre (computed once, reused in both layers)
    cx, cy, r = _find_gauge_center(gray)

    # ── Layer 1: OCR finds scale numbers + builds calibration ─────────────────
    log.info("  Layer 1 (OCR): detecting scale numbers …")
    ocr_pts = detect_scale_numbers(gray, cx, cy, r)
    calib   = build_calibration(ocr_pts)

    # ── Layer 2: CV finds needle angle ────────────────────────────────────────
    log.info("  Layer 2 (CV): detecting needle angle …")
    needle_angle, cx, cy, r = detect_needle_angle(img, cx, cy, r)
    log.info(f"  Needle: {needle_angle}°")

    # ── Unit + type detection via OCR ─────────────────────────────────────────
    log.info("  Unit OCR: detecting unit …")
    meter_type, unit = detect_unit_ocr(gray, cx, cy, r)

    # ── Compute reading ───────────────────────────────────────────────────────
    if calib is not None:
        reading = calibrated_reading(needle_angle, calib)
        s_min   = min(v for _, v in ocr_pts)
        s_max   = max(v for _, v in ocr_pts)
        method  = f"OCR({len(ocr_pts)} pts)+CV"
    else:
        # Geometric fallback: standard 270° sweep (7→5 o'clock), scale 0–100
        log.warning("  OCR calibration failed — geometric fallback (0-100, std sweep)")
        a_min  = int((270 + 7.5 * 30) % 360)   # 7:30 o'clock = 315°
        sweep  = 270
        delta  = (needle_angle - a_min) % 360
        frac   = float(np.clip(delta / sweep, 0.0, 1.0))
        reading = round(frac * 100.0, 1)
        s_min, s_max = 0, 100
        method = "CV only (OCR failed)"

    value   = str(reading)
    raw     = f"{value} {unit}".strip()
    elapsed = time.time() - t0
    log.info(f"  Done {elapsed:.1f}s  [{method}]  →  {value} {unit}")

    return {
        "meter_type":  meter_type,
        "scale_range": f"{int(s_min)}–{int(s_max)}",
        "raw_reading": raw,
        "value":       value,
        "unit":        unit,
        "needle_angle": needle_angle,
        "ocr_points":  len(ocr_pts),
        "method":      method,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/progress")
def progress():
    return jsonify(progress_state)


@app.route("/")
def index():
    log.info(f"GET /  from {request.remote_addr}")
    return HTML_PAGE


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/read", methods=["POST"])
def read():
    log.info("=" * 60)
    log.info(f"POST /read  from {request.remote_addr}")

    if "images" not in request.files:
        log.warning("Request missing 'images' field")
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    log.info(f"Received {len(files)} file(s)")
    results = []
    batch_t0 = time.time()
    progress_state.update({"current": 0, "total": len(files), "file": "", "done": False})

    for idx, f in enumerate(files, 1):
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            log.warning(f"[{idx}] Skipped '{f.filename}' — unsupported extension '{ext}'")
            continue

        uid       = uuid.uuid4().hex[:8]
        filename  = f"{uid}{ext}"
        save_path = UPLOAD_FOLDER / filename
        f.save(save_path)
        size_kb   = save_path.stat().st_size // 1024
        progress_state.update({"current": idx, "file": f.filename})
        log.info(f"[{idx}/{len(files)}] Processing '{f.filename}'  →  saved as {filename}  ({size_kb} KB)")

        try:
            img  = Image.open(save_path).convert("RGB")
            log.debug(f"  Image size: {img.size[0]}×{img.size[1]} px")
            data = read_meter(img)
            data["status"]    = "success"
            data["image_url"] = f"/uploads/{filename}"
            data["filename"]  = f.filename
            log.info(f"[{idx}] OK  →  {data['value']} {data['unit']}  ({data['meter_type']})")
        except Exception as e:
            log.error(f"[{idx}] FAILED '{f.filename}': {e}")
            log.debug(traceback.format_exc())
            data = {
                "status":    "error",
                "error":     str(e),
                "image_url": f"/uploads/{filename}",
                "filename":  f.filename,
            }

        results.append(data)

    total_t = time.time() - batch_t0
    ok  = sum(1 for r in results if r["status"] == "success")
    err = len(results) - ok
    progress_state.update({"done": True})
    log.info(f"Batch done in {total_t:.1f}s  —  {ok} ok, {err} error(s)")
    log.info("-" * 60)
    return jsonify(results)


# ─── Embedded HTML ────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Meter Reader</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;padding:32px 16px}

  h1{text-align:center;font-size:1.8rem;color:#f0f4ff;margin-bottom:6px}
  .subtitle{text-align:center;color:#64748b;font-size:.85rem;margin-bottom:36px}

  /* Drop zone */
  .dropzone{
    border:2px dashed #334155;border-radius:14px;padding:48px 24px;
    text-align:center;cursor:pointer;transition:border-color .2s,background .2s;
    max-width:720px;margin:0 auto 28px;
  }
  .dropzone.drag{border-color:#38bdf8;background:#0e1e2e}
  .dropzone svg{margin-bottom:12px;color:#475569}
  .dropzone p{color:#64748b;font-size:.9rem}
  .dropzone p span{color:#38bdf8;font-weight:600}
  #fileInput{display:none}

  /* Previews before submission */
  .preview-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:12px;
    max-width:720px;margin:0 auto 20px}
  .preview-thumb{position:relative;border-radius:8px;overflow:hidden;aspect-ratio:1;background:#1e2230}
  .preview-thumb img{width:100%;height:100%;object-fit:cover}
  .preview-thumb .remove{position:absolute;top:4px;right:4px;background:rgba(0,0,0,.7);
    border:none;border-radius:50%;width:20px;height:20px;cursor:pointer;
    color:#f87171;font-size:12px;line-height:20px;text-align:center}

  /* Buttons */
  .btn-row{text-align:center;margin-bottom:40px}
  button.read-btn{
    background:#0ea5e9;color:#fff;border:none;border-radius:10px;
    padding:12px 36px;font-size:1rem;font-weight:600;cursor:pointer;
    transition:background .2s;
  }
  button.read-btn:hover{background:#0284c7}
  button.read-btn:disabled{background:#334155;color:#64748b;cursor:not-allowed}
  button.clear-btn{
    background:transparent;color:#64748b;border:1px solid #334155;
    border-radius:10px;padding:12px 24px;font-size:1rem;cursor:pointer;
    margin-left:12px;transition:border-color .2s,color .2s;
  }
  button.clear-btn:hover{border-color:#94a3b8;color:#94a3b8}

  /* Spinner */
  .spinner{display:none;text-align:center;margin-bottom:28px}
  .spinner.active{display:block}
  @keyframes spin{to{transform:rotate(360deg)}}
  .spin-icon{display:inline-block;animation:spin 1s linear infinite;margin-right:8px;font-size:1.1rem}
  .progress-wrap{background:#1e2230;border-radius:10px;overflow:hidden;height:6px;max-width:420px;margin:10px auto 0}
  .progress-bar{height:6px;background:#38bdf8;width:0%;transition:width .4s ease}
  .progress-label{font-size:.8rem;color:#64748b;margin-top:8px}

  /* Results */
  .results-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
    gap:24px;max-width:1280px;margin:0 auto}
  .card{background:#1e2230;border-radius:14px;overflow:hidden;border:1px solid #2d3348;
    box-shadow:0 4px 24px rgba(0,0,0,.4);transition:transform .2s,box-shadow .2s}
  .card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,0,0,.6)}
  .card img{width:100%;height:200px;object-fit:cover;display:block}
  .card-body{padding:16px 18px 18px}
  .card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
  .card-header span{font-size:.72rem;color:#94a3b8;text-transform:uppercase;
    letter-spacing:.8px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%}
  .badge{font-size:.68rem;font-weight:700;padding:3px 9px;border-radius:20px;text-transform:uppercase}
  .badge-ok{background:#14532d;color:#4ade80}
  .badge-err{background:#7f1d1d;color:#f87171}
  .reading-val{font-size:2.2rem;font-weight:800;color:#38bdf8;line-height:1;margin-bottom:2px}
  .reading-unit{font-size:.95rem;color:#7dd3fc;font-weight:500;margin-bottom:14px}
  table{width:100%;border-collapse:collapse;font-size:.8rem}
  tr{border-bottom:1px solid #2d3348}tr:last-child{border-bottom:none}
  td{padding:6px 3px;vertical-align:top}
  td:first-child{color:#64748b;font-weight:600;width:40%;text-transform:uppercase;
    font-size:.7rem;letter-spacing:.4px;padding-top:8px}
  td:last-child{color:#cbd5e1;word-break:break-word}

  /* Export bar */
  .export-bar{text-align:center;margin-top:32px;display:none}
  .export-bar.visible{display:block}
  .export-btn{background:transparent;color:#38bdf8;border:1px solid #38bdf8;
    border-radius:8px;padding:8px 20px;font-size:.85rem;cursor:pointer;margin:0 6px;
    transition:background .2s,color .2s}
  .export-btn:hover{background:#38bdf8;color:#0f1117}
</style>
</head>
<body>

<h1>Analog Meter Reader</h1>
<p class="subtitle">Upload meter images — OpenCV + Tesseract OCR, no AI model, fully offline</p>

<!-- Drop zone -->
<div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
  <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round"
      d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
  </svg>
  <p><span>Click to upload</span> or drag &amp; drop</p>
  <p style="margin-top:6px;font-size:.75rem">JPG, PNG, BMP, WebP</p>
</div>
<input type="file" id="fileInput" multiple accept="image/*"/>

<!-- Thumbnail previews -->
<div class="preview-grid" id="previewGrid"></div>

<!-- Action buttons -->
<div class="btn-row">
  <button class="read-btn" id="readBtn" disabled onclick="readMeters()">Read Meters</button>
  <button class="clear-btn" id="clearBtn" onclick="clearAll()">Clear</button>
</div>

<!-- Processing indicator -->
<div class="spinner" id="spinner">
  <span class="spin-icon">&#9696;</span> <span id="spinnerText">Reading meters…</span>
  <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>
  <div class="progress-label" id="progressLabel"></div>
</div>

<!-- Results -->
<div class="results-grid" id="resultsGrid"></div>

<!-- Export -->
<div class="export-bar" id="exportBar">
  <button class="export-btn" onclick="downloadXML()">Download XML</button>
  <button class="export-btn" onclick="downloadHTML()">Download HTML Report</button>
</div>

<script>
let selectedFiles = [];
let lastResults   = [];

// ── Drag & drop ───────────────────────────────────────────────────────────────
const dz = document.getElementById('dropzone');
dz.addEventListener('dragover',  e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave', ()=>dz.classList.remove('drag'));
dz.addEventListener('drop', e=>{
  e.preventDefault(); dz.classList.remove('drag');
  addFiles([...e.dataTransfer.files]);
});
document.getElementById('fileInput').addEventListener('change', e=>{
  addFiles([...e.target.files]);
  e.target.value='';
});

function addFiles(newFiles){
  newFiles.forEach(f=>{ if(f.type.startsWith('image/')) selectedFiles.push(f); });
  renderPreviews();
}

function renderPreviews(){
  const grid = document.getElementById('previewGrid');
  grid.innerHTML='';
  selectedFiles.forEach((f,i)=>{
    const url = URL.createObjectURL(f);
    grid.innerHTML += `<div class="preview-thumb">
      <img src="${url}" alt="${f.name}"/>
      <button class="remove" onclick="removeFile(${i})">✕</button>
    </div>`;
  });
  document.getElementById('readBtn').disabled = selectedFiles.length === 0;
}

function removeFile(i){ selectedFiles.splice(i,1); renderPreviews(); }

function clearAll(){
  selectedFiles=[];
  document.getElementById('previewGrid').innerHTML='';
  document.getElementById('resultsGrid').innerHTML='';
  document.getElementById('readBtn').disabled=true;
  document.getElementById('exportBar').classList.remove('visible');
  lastResults=[];
}

// ── Read ──────────────────────────────────────────────────────────────────────
let pollTimer = null;

function startProgressPolling(total){
  const bar   = document.getElementById('progressBar');
  const label = document.getElementById('progressLabel');
  const text  = document.getElementById('spinnerText');
  pollTimer = setInterval(async ()=>{
    try{
      const p = await (await fetch('/progress')).json();
      const pct = p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
      bar.style.width = pct + '%';
      if(p.file) label.textContent = `[${p.current}/${p.total}]  ${p.file}`;
      if(p.done){ clearInterval(pollTimer); pollTimer=null; }
    }catch(e){}
  }, 800);
}

async function readMeters(){
  if(!selectedFiles.length) return;
  document.getElementById('readBtn').disabled=true;
  document.getElementById('spinner').classList.add('active');
  document.getElementById('progressBar').style.width='0%';
  document.getElementById('progressLabel').textContent='';
  document.getElementById('spinnerText').textContent='Reading meters…';
  document.getElementById('resultsGrid').innerHTML='';
  document.getElementById('exportBar').classList.remove('visible');

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('images', f));

  startProgressPolling(selectedFiles.length);

  try {
    const res  = await fetch('/read', {method:'POST', body:fd});
    const data = await res.json();
    lastResults = data;
    renderResults(data);
    document.getElementById('exportBar').classList.add('visible');
    document.getElementById('progressBar').style.width='100%';
    document.getElementById('progressLabel').textContent=`Done — ${data.length} meter(s) read`;
  } catch(err){
    document.getElementById('resultsGrid').innerHTML =
      `<p style="color:#f87171;text-align:center">Error: ${err.message}</p>`;
  }

  if(pollTimer){ clearInterval(pollTimer); pollTimer=null; }
  document.getElementById('spinner').classList.remove('active');
  document.getElementById('readBtn').disabled = selectedFiles.length===0;
}

// ── Render result cards ───────────────────────────────────────────────────────
function renderResults(data){
  const grid = document.getElementById('resultsGrid');
  grid.innerHTML = data.map(r => {
    if(r.status==='error') return `
      <div class="card">
        <img src="${r.image_url}" alt="${r.filename}"/>
        <div class="card-body">
          <div class="card-header">
            <span>${r.filename}</span>
            <span class="badge badge-err">error</span>
          </div>
          <p style="color:#f87171;font-size:.82rem">${r.error}</p>
        </div>
      </div>`;

    return `
      <div class="card">
        <img src="${r.image_url}" alt="${r.filename}"/>
        <div class="card-body">
          <div class="card-header">
            <span>${r.filename}</span>
            <span class="badge badge-ok">ok</span>
          </div>
          <div class="reading-val">${r.value}</div>
          <div class="reading-unit">${r.unit}</div>
          <table>
            <tr><td>Type</td>      <td>${r.meter_type}</td></tr>
            <tr><td>Scale</td>     <td>${r.scale_range}</td></tr>
            <tr><td>Raw</td>       <td>${r.raw_reading}</td></tr>
            <tr><td>Needle</td>    <td>${r.needle_angle}°</td></tr>
            <tr><td>Method</td>    <td>${r.method}</td></tr>
            <tr><td>Time</td>      <td>${r.timestamp}</td></tr>
          </table>
        </div>
      </div>`;
  }).join('');
}

// ── Export ────────────────────────────────────────────────────────────────────
function downloadXML(){
  const now = new Date().toISOString();
  let xml = `<?xml version="1.0" ?>\n<MeterReadings generated_at="${now}" model="OpenCV+Tesseract" total_meters="${lastResults.length}">\n`;
  lastResults.forEach((r,i)=>{
    xml += `  <Meter id="${i+1}">\n`;
    xml += `    <Image>${r.filename}</Image>\n`;
    xml += `    <Status>${r.status}</Status>\n`;
    if(r.status==='success'){
      xml += `    <MeterType>${r.meter_type}</MeterType>\n`;
      xml += `    <ScaleRange>${r.scale_range}</ScaleRange>\n`;
      xml += `    <Value>${r.value}</Value>\n`;
      xml += `    <Unit>${r.unit}</Unit>\n`;
      xml += `    <RawReading>${r.raw_reading}</RawReading>\n`;
      xml += `    <Timestamp>${r.timestamp}</Timestamp>\n`;
    } else {
      xml += `    <Error>${r.error}</Error>\n`;
    }
    xml += `  </Meter>\n`;
  });
  xml += `</MeterReadings>`;
  download('meter_readings.xml', xml, 'application/xml');
}

function downloadHTML(){
  const now = new Date().toLocaleString();
  const cards = lastResults.map((r,i) => {
    if(r.status==='error') return `<div class="card"><img src="${r.image_url}"/><div class="card-body"><p>Error: ${r.error}</p></div></div>`;
    return `<div class="card">
      <img src="${r.image_url}" style="width:100%;height:200px;object-fit:cover"/>
      <div class="card-body">
        <div class="reading-val">${r.value} <span style="font-size:1.2rem">${r.unit}</span></div>
        <table><tr><td>Type</td><td>${r.meter_type}</td></tr>
        <tr><td>Scale</td><td>${r.scale_range}</td></tr>
        <tr><td>Raw</td><td>${r.raw_reading}</td></tr>
        <tr><td>Time</td><td>${r.timestamp}</td></tr></table>
      </div></div>`;
  }).join('');
  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/>
    <title>Meter Report</title>
    <style>body{font-family:sans-serif;background:#0f1117;color:#e2e8f0;padding:32px}
    h1{text-align:center;color:#f0f4ff;margin-bottom:8px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:24px;max-width:1280px;margin:0 auto}
    .card{background:#1e2230;border-radius:14px;overflow:hidden;border:1px solid #2d3348}.card-body{padding:16px}
    .reading-val{font-size:2rem;font-weight:800;color:#38bdf8;margin-bottom:12px}
    table{width:100%;border-collapse:collapse;font-size:.82rem}tr{border-bottom:1px solid #2d3348}
    td{padding:6px 3px}td:first-child{color:#64748b;font-weight:600;text-transform:uppercase;font-size:.7rem;width:35%}
    </style></head><body>
    <h1>Meter Readings Report</h1><p style="text-align:center;color:#64748b;margin-bottom:32px">Generated: ${now}</p>
    <div class="grid">${cards}</div></body></html>`;
  download('meter_report.html', html, 'text/html');
}

function download(name, content, type){
  const a  = document.createElement('a');
  a.href   = URL.createObjectURL(new Blob([content],{type}));
  a.download = name;
  a.click();
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    log.info("Server starting on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
