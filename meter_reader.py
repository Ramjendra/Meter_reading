#!/usr/bin/env python3
"""
Analog Meter Reader — Qwen2.5-VL-3B (best accuracy, offline after first download)
The model reads each meter visually — no image preprocessing.

Laptop:  python3 meter_reader.py 1.jpeg 2.jpeg 3.jpeg 4.jpeg
Android: python3 meter_reader.py --model mobile *.jpeg   (uses Qwen2-VL-2B)
"""

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip3 install transformers torch pillow accelerate qwen-vl-utils torchvision")
    sys.exit(1)


# ─── Model options ────────────────────────────────────────────────────────────

MODELS = {
    # Best accuracy — laptop/desktop  (~6 GB RAM fp32, ~3 GB fp16)
    "best":   ("Qwen/Qwen2.5-VL-3B-Instruct",  "qwen25"),
    # Good accuracy — lighter laptop  (~4 GB RAM fp32)
    "medium": ("Qwen/Qwen2-VL-2B-Instruct",     "qwen2"),
    # Android / low-RAM devices       (~1 GB RAM)
    "mobile": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "smolvlm"),
}

COMBINED_PROMPT = (
    "You are an expert at reading analog meters. Follow these steps carefully:\n\n"
    "1. SCALE: Look only at the numbers actually printed on the dial face. "
    "Report the smallest and largest printed numbers — do NOT infer or add numbers not visible.\n"
    "2. NEEDLE: Trace the needle from its pivot (center) to its tip. "
    "Note which printed number or tick mark the tip is closest to.\n"
    "3. READING: Report only the value the needle tip points at, "
    "interpolating between ticks if needed.\n\n"
    "Reply in exactly this one-line format:\n"
    "TYPE: <type> | SCALE: <min printed>-<max printed> | READING: <needle value> <unit>\n\n"
    "Example: TYPE: thermometer | SCALE: -20-120 | READING: 95 °C\n\n"
    "Now read this meter image:"
)

UNIT_RE = re.compile(
    r"°[CF]|PSI|psi|bar|kPa|MPa|Pa|kg/cm[²2]|rpm|RPM"
    r"|[Kk]Hz|MHz|Hz|[Vv]olts?|[Aa]mps?|kW|MW|m³/h|L/min"
    r"|%|inHg|mmHg|atm",
    re.IGNORECASE,
)


# ─── Reader class ─────────────────────────────────────────────────────────────

class MeterReader:

    def __init__(self, model_size: str = "best"):
        model_id, family = MODELS[model_size]
        self.family = family
        print(f"Loading {model_id}")
        print("  First run downloads the model — cached, then fully offline.\n")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)

        if family == "qwen25":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto"
            )
        elif family == "qwen2":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto"
            )
        else:
            from transformers import SmolVLMForConditionalGeneration
            self.model = SmolVLMForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            self.model.to(self.device)

        self.model.eval()
        print(f"Model ready on {self.device}.\n")

    def _infer(self, img: Image.Image) -> str:
        if self.family in ("qwen25", "qwen2"):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": COMBINED_PROMPT},
            ]}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt], images=[img], return_tensors="pt"
            ).to(self.device)
        else:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": COMBINED_PROMPT},
            ]}]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt, images=[img], return_tensors="pt"
            ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def read_meter(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        t0  = time.time()

        raw = self._infer(img)
        elapsed = time.time() - t0

        meter_type  = self._field(raw, "TYPE")
        scale_range = self._field(raw, "SCALE")
        reading     = self._field(raw, "READING")
        value, unit = self._extract(reading)

        print(f"  Raw    : {raw}  ({elapsed:.1f}s)")
        print(f"  Reading: {value} {unit}")

        return {
            "image":       os.path.basename(image_path),
            "meter_type":  meter_type,
            "scale_range": scale_range,
            "raw_reading": reading,
            "value":       value,
            "unit":        unit,
            "timestamp":   datetime.now().isoformat(),
            "status":      "success",
        }

    def _field(self, text: str, key: str) -> str:
        m = re.search(rf"{key}\s*:\s*([^|]+)", text, re.IGNORECASE)
        return m.group(1).strip().rstrip(".") if m else text.strip()

    def _extract(self, text: str):
        numbers = re.findall(r"-?\d+\.?\d*", text)
        m = UNIT_RE.search(text)
        return (numbers[0] if numbers else "—"), (m.group() if m else "")

    def process_images(self, paths: list) -> list:
        results = []
        for i, p in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}]  {p}")
            try:
                r = self.read_meter(p)
                print()
            except Exception as exc:
                print(f"  ✗ {exc}\n")
                r = {
                    "image":     os.path.basename(p),
                    "error":     str(exc),
                    "timestamp": datetime.now().isoformat(),
                    "status":    "failed",
                }
            results.append(r)
        return results

    # ── XML ───────────────────────────────────────────────────────────────────

    def save_xml(self, results: list, out_path: str = "meter_readings.xml") -> str:
        root = ET.Element("MeterReadings")
        root.set("generated_at", datetime.now().isoformat())
        root.set("model",        "Qwen2.5-VL-3B")
        root.set("total_meters", str(len(results)))

        for idx, r in enumerate(results, 1):
            meter = ET.SubElement(root, "Meter")
            meter.set("id", str(idx))
            ET.SubElement(meter, "Image").text  = r.get("image", "")
            ET.SubElement(meter, "Status").text = r.get("status", "failed")
            if r["status"] == "success":
                ET.SubElement(meter, "MeterType").text  = r["meter_type"]
                ET.SubElement(meter, "ScaleRange").text = r["scale_range"]
                ET.SubElement(meter, "Value").text      = r["value"]
                ET.SubElement(meter, "Unit").text       = r["unit"]
                ET.SubElement(meter, "RawReading").text = r["raw_reading"]
                ET.SubElement(meter, "Timestamp").text  = r["timestamp"]
            else:
                ET.SubElement(meter, "ErrorDetail").text = r.get("error", "")

        raw    = ET.tostring(root, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="  ")
        xml_str = "\n".join(ln for ln in pretty.split("\n") if ln.strip())

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        return xml_str

    # ── HTML ──────────────────────────────────────────────────────────────────

    def save_html(self, results: list, out_path: str = "meter_report.html") -> None:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cards = ""
        for idx, r in enumerate(results, 1):
            image  = r.get("image", "")
            status = r.get("status", "failed")
            badge  = "badge-success" if status == "success" else "badge-fail"
            if status == "success":
                value = r["value"]; unit = r["unit"]
                rows  = (f'<tr><td>Type</td><td>{r["meter_type"]}</td></tr>'
                         f'<tr><td>Scale</td><td>{r["scale_range"]}</td></tr>'
                         f'<tr><td>Raw</td><td>{r["raw_reading"]}</td></tr>'
                         f'<tr><td>Time</td><td>{r["timestamp"][:19].replace("T"," ")}</td></tr>')
            else:
                value, unit, rows = "—", "", f'<tr><td>Error</td><td>{r.get("error","")}</td></tr>'

            cards += f"""
  <div class="card">
    <img class="card-image" src="{image}" alt="Meter {idx}"/>
    <div class="card-body">
      <div class="card-title">
        <span>Meter #{idx} &mdash; {image}</span>
        <span class="badge {badge}">{status}</span>
      </div>
      <div class="reading-value">{value}</div>
      <div class="reading-unit">{unit}</div>
      <table>{rows}</table>
    </div>
  </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Meter Readings Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f1117;color:#e2e8f0;padding:32px 16px}}
  header{{text-align:center;margin-bottom:40px}}
  header h1{{font-size:1.8rem;font-weight:700;color:#f0f4ff}}
  header p{{font-size:.85rem;color:#64748b;margin-top:6px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:28px;max-width:1280px;margin:0 auto}}
  .card{{background:#1e2230;border-radius:14px;overflow:hidden;border:1px solid #2d3348;box-shadow:0 4px 24px rgba(0,0,0,.4);transition:transform .2s}}
  .card:hover{{transform:translateY(-3px)}}
  .card-image{{width:100%;height:220px;object-fit:cover;display:block}}
  .card-body{{padding:18px 20px 20px}}
  .card-title{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}}
  .card-title span{{font-size:.75rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px}}
  .badge{{font-size:.7rem;font-weight:600;padding:3px 10px;border-radius:20px;text-transform:uppercase}}
  .badge-success{{background:#14532d;color:#4ade80}}
  .badge-fail{{background:#7f1d1d;color:#f87171}}
  .reading-value{{font-size:2.4rem;font-weight:800;color:#38bdf8;line-height:1;margin-bottom:4px}}
  .reading-unit{{font-size:1rem;color:#7dd3fc;font-weight:500;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:.82rem}}
  tr{{border-bottom:1px solid #2d3348}}tr:last-child{{border-bottom:none}}
  td{{padding:7px 4px;vertical-align:top}}
  td:first-child{{color:#64748b;font-weight:600;width:40%;text-transform:uppercase;font-size:.72rem;letter-spacing:.4px;padding-top:9px}}
  td:last-child{{color:#cbd5e1;word-break:break-word}}
  footer{{text-align:center;margin-top:48px;font-size:.75rem;color:#334155}}
</style></head>
<body>
<header>
  <h1>Analog Meter Readings</h1>
  <p>Generated: {generated_at} &nbsp;|&nbsp; Model: Qwen2.5-VL-3B &nbsp;|&nbsp; {len(results)} meters</p>
</header>
<div class="grid">{cards}</div>
<footer>Qwen2.5-VL-3B &mdash; Analog Meter Reader &mdash; Offline</footer>
</body></html>"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analog Meter Reader — Qwen2.5-VL-3B, fully offline"
    )
    parser.add_argument("images", nargs="*", help="Image files to process")
    parser.add_argument(
        "--model", choices=["best", "medium", "mobile"], default="best",
        help="best=Qwen2.5-VL-3B (laptop)  medium=Qwen2-VL-2B  mobile=SmolVLM2-500M (Android)",
    )
    parser.add_argument("--output", default="meter_readings.xml")
    parser.add_argument("--dir",    default=".", help="Scan folder when no filenames given")
    args = parser.parse_args()

    if args.images:
        paths = args.images
    else:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        paths = sorted(str(p) for p in Path(args.dir).iterdir() if p.suffix.lower() in exts)

    if not paths:
        print("No images found. Pass filenames or use --dir <folder>")
        sys.exit(1)

    print(f"Found {len(paths)} image(s)\n")
    reader  = MeterReader(model_size=args.model)
    results = reader.process_images(paths)

    xml_out   = reader.save_xml(results, args.output)
    html_path = args.output.replace(".xml", ".html")
    reader.save_html(results, html_path)

    print("─" * 55)
    print(f"XML  → {args.output}")
    print(f"HTML → {html_path}")
    print("─" * 55)
    print(xml_out)


if __name__ == "__main__":
    main()
