#!/usr/bin/env python3
"""
Meter Reading Web App
Run:  python3 app.py
Open: http://localhost:5000
"""

import logging
import os
import re
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

UNIT_RE = re.compile(
    r"°[CF]|PSI|psi|bar|kPa|MPa|Pa|kg/cm[²2]|rpm|RPM"
    r"|[Kk]Hz|MHz|Hz|[Vv]olts?|[Aa]mps?|kW|MW|m³/h|L/min"
    r"|%|inHg|mmHg|atm",
    re.IGNORECASE,
)

app = Flask(__name__)
progress_state = {"current": 0, "total": 0, "file": "", "done": False}

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FILE = "meter_app.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),          # also print to terminal
    ],
)
log = logging.getLogger("meter_app")

# Silence noisy third-party loggers
for _noisy in ("transformers", "torch", "PIL", "httpcore", "httpx",
               "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger("werkzeug").setLevel(logging.INFO)

# ─── Load model once at startup ───────────────────────────────────────────────

log.info("=" * 60)
log.info("Meter Reader App starting")
log.info(f"Model  : {MODEL_ID}")
log.info(f"Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
log.info(f"Uploads: {UPLOAD_FOLDER.resolve()}")
log.info(f"Log    : {Path(LOG_FILE).resolve()}")
log.info("=" * 60)

t0 = time.time()
log.info("Loading processor …")
processor = AutoProcessor.from_pretrained(MODEL_ID)
log.info("Loading model weights …")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
log.info(f"Model ready on {device}  (load time: {time.time()-t0:.1f}s)")
log.info("-" * 60)


# ─── Inference helpers ────────────────────────────────────────────────────────

COMBINED_PROMPT = (
    "Look at this analog meter image and answer all three questions in one line, "
    "using exactly this format:\n"
    "TYPE: <instrument type> | SCALE: <min>-<max> | READING: <value> <unit>\n\n"
    "Example: TYPE: pressure gauge | SCALE: 0-300 | READING: 120 PSI\n"
    "Now answer for this image:"
)


def read_meter(img: Image.Image) -> dict:
    log.info("  Running single inference pass …")
    t0 = time.time()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": COMBINED_PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    log.debug(f"  Raw answer: {answer}  ({elapsed:.2f}s)")

    # Parse structured response
    meter_type  = _extract_field(answer, "TYPE")
    scale_range = _extract_field(answer, "SCALE")
    reading     = _extract_field(answer, "READING")

    numbers = re.findall(r"-?\d+\.?\d*", reading)
    m = UNIT_RE.search(reading)
    result = {
        "meter_type":  meter_type,
        "scale_range": scale_range,
        "raw_reading": reading,
        "value":       numbers[0] if numbers else "—",
        "unit":        m.group() if m else "",
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    log.info(f"  Done in {elapsed:.1f}s  →  {result['value']} {result['unit']}  ({result['meter_type']})")
    return result


def _extract_field(text: str, key: str) -> str:
    m = re.search(rf"{key}\s*:\s*([^|]+)", text, re.IGNORECASE)
    return m.group(1).strip().rstrip(".") if m else text.strip()


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
<p class="subtitle">Upload meter images — Qwen2.5-VL-3B reads them offline, no internet needed</p>

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
            <tr><td>Time</td>      <td>${r.timestamp}</td></tr>
          </table>
        </div>
      </div>`;
  }).join('');
}

// ── Export ────────────────────────────────────────────────────────────────────
function downloadXML(){
  const now = new Date().toISOString();
  let xml = `<?xml version="1.0" ?>\n<MeterReadings generated_at="${now}" model="SmolVLM2-SLM" total_meters="${lastResults.length}">\n`;
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
