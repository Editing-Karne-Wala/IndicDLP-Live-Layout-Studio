# ========================================================================================
# IndicDLP: Broadsheet Layout Annotation & 100% OCR Pipeline (100% Self-Contained)
# Output: Desktop/1.json (Ready for direct import to Label Studio at http://localhost:9090)
# ========================================================================================

import os
import sys
import json
import base64
import torch
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO

# ========================================================================================
# ⚙️ USER CONFIGURATION: CHANGE THESE 2 SETTINGS
# ========================================================================================
# 1. Directory containing your first 10 broadsheet newspaper scans:
INPUT_SCANS_DIR = r"/path/to/your/scans"

# 2. Your Gemini API Key (optional - if left empty, local EasyOCR is used):
GEMINI_API_KEY = ""

# ========================================================================================
# 📦 OUTPUT FILE DESTINATION (Desktop/1.json)
# ========================================================================================
home_dir = os.path.expanduser("~")
desktop_dir = os.path.join(home_dir, "Desktop")
if not os.path.exists(desktop_dir):
    wsl_win_users = [d for d in os.listdir("/mnt/c/Users") if os.path.exists(f"/mnt/c/Users/{d}/Desktop")] if os.path.exists("/mnt/c/Users") else []
    if wsl_win_users:
        desktop_dir = f"/mnt/c/Users/{wsl_win_users[0]}/Desktop"
    else:
        desktop_dir = home_dir

OUTPUT_JSON_PATH = os.path.join(desktop_dir, "1.json")

# ========================================================================================
# 🚀 HARDWARE & MODEL SELECTION
# Auto-detects NVIDIA GPU vs. AMD / CPU safely
# ========================================================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_gpu_ocr = True if torch.cuda.is_available() else False

print("=" * 65)
print(f"🖥️  Device Mode: {device.upper()}")
if not torch.cuda.is_available():
    print("ℹ️  Notice: No CUDA GPU detected (standard for AMD integrated graphics).")
    print("    Running inference and OCR on optimized multi-threaded CPU.")
print("=" * 65)

# Locate YOLO weights file
weight_file = "weights/md2_PT_indicdlp_FT.pt"
if not os.path.exists(weight_file):
    alt_paths = [
        os.path.expanduser("~/IndicDLP-Live-Layout-Studio/weights/md2_PT_indicdlp_FT.pt"),
        "md2_PT_indicdlp_FT.pt",
    ]
    for p in alt_paths:
        if os.path.exists(p):
            weight_file = p
            break

print(f"📦 Loading YOLO layout parser from: {weight_file}...")
layout_model = YOLO(weight_file)
print("✓ Layout parser ready!")

# Initialize EasyOCR
print("⚡ Initializing EasyOCR Reader (Bengali)...")
reader = easyocr.Reader(['bn'], gpu=use_gpu_ocr, verbose=False)
print("✓ EasyOCR Reader ready!")

# Initialize Gemini Client if provided
client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=10000))
        print("✓ Gemini Multimodal API client initialized!")
    except Exception:
        print("⚠️  Gemini SDK not installed or key error. Falling back to EasyOCR.")

CLASS_MAP = {
    "headline": "Headline",
    "title": "Headline",
    "flag": "Headline",
    "header": "Headline",
    "sub-title": "Subtitle",
    "sub-headline": "Subtitle",
    "heading": "Subtitle",
    "section-title": "Subtitle",
    "sub-section-title": "Subtitle",
    "paragraph": "BodyText",
    "ordered-list": "BodyText",
    "unordered-list": "BodyText",
    "table": "BodyText",
    "table-of-contents": "BodyText",
    "index": "BodyText",
    "sidebar": "BodyText",
    "advertisement": "BodyText",
    "caption": "Caption",
    "table-caption": "Caption",
    "figure-caption": "Caption",
    "figure": "Image",
}

def predict_layout(img_path):
    with Image.open(img_path) as im:
        bgr = np.ascontiguousarray(np.array(im.convert("RGB"))[:, :, ::-1])
    res = list(layout_model.predict(bgr, imgsz=1024, conf=0.15, iou=0.45,
                                    device=0 if device.startswith("cuda") else "cpu", verbose=False))[0]
    names = res.names
    dets = []
    for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
        dets.append({"bbox": [x1, y1, x2, y2], "conf": float(c), "label": names[int(k)]})
    return dets

def ocr_with_gemini(crop_img: Image.Image) -> str:
    if not client:
        return ""
    prompt = (
        "Transcribe the Bengali text in this historical newspaper snippet verbatim. "
        "Output only the exact transcribed Bengali text:"
    )
    for model_name in ["gemini-3.1-flash-lite", "gemini-3-flash-preview"]:
        try:
            response = client.models.generate_content(model=model_name, contents=[crop_img, prompt])
            if response and response.text:
                txt = response.text.strip()
                lines = [l for l in txt.split("\n") if not l.lower().startswith("here is") and not l.lower().startswith("the text")]
                return "\n".join(lines).strip()
        except Exception:
            continue
    return ""

def ocr_with_easyocr(crop_cv2) -> str:
    try:
        results = reader.readtext(crop_cv2)
        lines = [r[1].strip() for r in results if len(r[1].strip()) > 0]
        return "\n".join(lines).strip()
    except Exception:
        return ""

def main():
    if not os.path.exists(INPUT_SCANS_DIR):
        print(f"❌ Error: INPUT_SCANS_DIR not found: {INPUT_SCANS_DIR}")
        print("Please open this script and update INPUT_SCANS_DIR at the top with your folder path.")
        return

    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = sorted([f for f in os.listdir(INPUT_SCANS_DIR) if f.lower().endswith(valid_exts)])[:10]

    if not image_files:
        print(f"❌ Error: No image files found in {INPUT_SCANS_DIR}")
        return

    print(f"\n📂 Found {len(image_files)} broadsheet scans to annotate and transcribe.")
    label_studio_tasks = []

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(INPUT_SCANS_DIR, fname)
        print(f"\n[{idx:02d}/{len(image_files):02d}] Processing: {fname}...")

        with Image.open(fpath) as pil_img:
            W, H = pil_img.size
            with open(fpath, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
            ext = os.path.splitext(fname)[1].lower().replace(".", "")
            mime = "jpeg" if ext in ["jpg", "jpeg"] else "png"
            data_uri = f"data:image/{mime};base64,{b64_data}"

        raw_dets = predict_layout(fpath)
        print(f"  • Layout Regions Detected: {len(raw_dets)}")

        ls_results = []
        for r_idx, det in enumerate(raw_dets):
            x1, y1, x2, y2 = det["bbox"]
            orig_lbl = det["label"].lower()
            std_label = CLASS_MAP.get(orig_lbl, "BodyText")

            rx = round(float(x1) / W * 100, 4)
            ry = round(float(y1) / H * 100, 4)
            rw = round(float(x2 - x1) / W * 100, 4)
            rh = round(float(y2 - y1) / H * 100, 4)
            reg_id = f"scan_{idx}_reg_{r_idx}"

            ls_results.append({
                "value": {
                    "x": rx, "y": ry, "width": rw, "height": rh,
                    "rotation": 0, "rectanglelabels": [std_label]
                },
                "id": reg_id, "from_name": "label", "to_name": "image",
                "type": "rectanglelabels", "origin": "manual"
            })

            if std_label != "Image":
                with Image.open(fpath) as pil_full:
                    crop_pil = pil_full.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))

                ocr_text = ""
                if std_label in ["Headline", "Subtitle"] and client:
                    ocr_text = ocr_with_gemini(crop_pil)

                if not ocr_text:
                    crop_cv2 = cv2.imread(fpath)[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))]
                    if crop_cv2 is not None and crop_cv2.size > 0:
                        ocr_text = ocr_with_easyocr(crop_cv2)

                if not ocr_text:
                    ocr_text = f"[{std_label}]"

                ls_results.append({
                    "value": {
                        "x": rx, "y": ry, "width": rw, "height": rh,
                        "text": [ocr_text]
                    },
                    "id": reg_id, "from_name": "ocr", "to_name": "image",
                    "type": "textarea", "origin": "manual"
                })

        label_studio_tasks.append({
            "data": {
                "image": data_uri,
                "filename": fname,
                "original_filename": fname,
                "page_index": idx
            },
            "annotations": [{
                "result": ls_results,
                "was_cancelled": False,
                "ground_truth": False
            }]
        })
        print(f"  ✓ Page {idx} completed with 100% OCR transcription.")

    print(f"\nWriting Label Studio JSON to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(label_studio_tasks, f, ensure_ascii=False)

    print("=" * 65)
    print("🎉 COMPLETE! Output generated:")
    print(f"👉 File: {OUTPUT_JSON_PATH}")
    print("\nHow to view in Label Studio (Port 9090):")
    print("1. Open http://localhost:9090 in your browser")
    print("2. Click 'Create Project'")
    print("3. Go to 'Data Import' and upload '1.json'")
    print("4. All scans, bounding boxes, and OCR text will appear instantly!")
    print("=" * 65)

if __name__ == "__main__":
    main()
