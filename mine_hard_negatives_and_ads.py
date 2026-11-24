"""
========================================================================================
Project: National Indic Document Layout Parsing (IndicDLP) | 22 Scheduled Indian Languages
Component: High-Speed Batched Ad Detection & Hard-Negative Mining Engine (700 Scans)
Institution: IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata / DST
Author: Research Intern Team & Antigravity
Timestamp: 2026-11-24 23:15:00 IST

Architecture:
-------------
1. Fast GPU Layout Region Proposal (YOLOv10x, FP16 Autocast, NMS-free)
2. Information-Theoretic Disambiguation (Typographic Entropy H, Perplexity 2^H, Faraday Cage Gamma)
3. Deep Pretrained ResNet-18 & LeNet Validation on Batched GPU Crops
4. Precise Hard-Negative Mining:
   - Boxed Announcements / Special Communiques (Border = True, High Entropy) -> hard_negative_boxed_news
   - High Aspect Ratio Skinny Columns (H/W > 4.5, W/H > 4.5) -> hard_negative_aspect_ratio
   - Tabular / Financial / Sports Grid Layouts -> hard_negative_tables
   - Masthead / Page Furniture false alarms -> hard_negative_headers
5. Pure-Text & Display Advertisement Harvesting:
   - ResNet/Entropy Confirmed Ads -> ad_crops
========================================================================================
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

REPO_DIR = Path(r"C:\Users\shiny\Documents\IndicDLP-Live-Layout-Studio")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from model_inference import load_model, predict_once
from ad_image_disambiguator import disambiguate_region, compute_typographic_entropy, compute_boundary_insulation
from models_benchmark import ModifiedLeNet5, build_resnet18

# Transform for 224x224 letterboxed inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_ad_classifiers(device: str):
    """Loads best LeNet and ResNet18 checkpoints"""
    lenet = ModifiedLeNet5(num_classes=2).to(device)
    lenet_path = REPO_DIR / "checkpoints" / "best_lenet.pt"
    if lenet_path.exists():
        lenet.load_state_dict(torch.load(lenet_path, map_location=device, weights_only=False))
    lenet.eval()

    resnet = build_resnet18(num_classes=2, pretrained=False).to(device)
    resnet_path = REPO_DIR / "checkpoints" / "best_resnet18.pt"
    if resnet_path.exists():
        resnet.load_state_dict(torch.load(resnet_path, map_location=device, weights_only=False))
    resnet.eval()

    return lenet, resnet

def process_batch(batch_name: str, scan_dir: Path, output_base: Path, device: str = "cuda:0"):
    print(f"\n=======================================================", flush=True)
    print(f"🚀 Launching Mining Pipeline for {batch_name}", flush=True)
    print(f"📂 Source: {scan_dir}", flush=True)
    print(f"📁 Destination: {output_base}", flush=True)
    print(f"=======================================================", flush=True)

    batch_output = output_base / batch_name
    ad_crops_dir = batch_output / "ad_crops"
    hard_neg_boxed_dir = batch_output / "hard_negative_boxed_news"
    hard_neg_aspect_dir = batch_output / "hard_negative_aspect_ratio"
    hard_neg_tables_dir = batch_output / "hard_negative_tables"
    all_non_ads_dir = batch_output / "non_ad_crops"
    metadata_path = batch_output / "batch_mining_metadata.json"

    for d in [ad_crops_dir, hard_neg_boxed_dir, hard_neg_aspect_dir, hard_neg_tables_dir, all_non_ads_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Models
    print("📦 Loading YOLOv10x, ResNet-18, and LeNet on GPU...", flush=True)
    load_model("yolov10x", device=device)
    lenet, resnet = load_ad_classifiers(device)
    print("✓ All models loaded into GPU VRAM!", flush=True)

    image_files = sorted([f for f in scan_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])
    print(f"📄 Found {len(image_files)} pages to process in {batch_name}.\n", flush=True)

    batch_manifest = []
    total_ads_found = 0
    total_hard_negatives = 0
    start_batch_time = time.time()

    for p_idx, img_p in enumerate(image_files, 1):
        t0 = time.time()
        orig_img = Image.open(img_p).convert("RGB")
        w_orig, h_orig = orig_img.size
        full_np = np.array(orig_img)
        full_gray = cv2.cvtColor(full_np, cv2.COLOR_RGB2GRAY)

        # 1. GPU Layout Inference
        raw_dets = predict_once("yolov10x", orig_img, conf=0.18, iou=0.45, imgsz=1024, device=device)

        page_records = []
        page_tensors = []
        page_candidate_crops = []
        page_candidate_meta = []

        # 2. Extract Candidate Regions & Structural Physics
        for r_idx, d in enumerate(raw_dets):
            lbl = d["label"].lower()
            x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 15 or bh < 15:
                continue

            aspect_ratio = bw / float(bh + 1e-4)
            rel_area = (bw * bh) / float(w_orig * h_orig)

            crop_rgb = orig_img.crop((x1, y1, x2, y2))
            crop_gray = full_gray[y1:y2, x1:x2]

            # Fast boundary and typographic entropy
            typo_info = compute_typographic_entropy(crop_gray)
            shannon_H = typo_info.get("run_length_entropy", 0.0)
            perplexity = round(float(2.0 ** shannon_H), 3) if shannon_H > 0 else 1.0
            char_count = typo_info.get("char_candidates", 0)
            gamma = compute_boundary_insulation(full_gray, (x1, y1, x2, y2))

            crop_tensor = val_transforms(crop_rgb)
            page_tensors.append(crop_tensor)
            page_candidate_crops.append(crop_rgb)
            page_candidate_meta.append({
                "box": [x1, y1, x2, y2],
                "label": lbl,
                "bw": bw, "bh": bh,
                "aspect_ratio": round(aspect_ratio, 3),
                "rel_area": round(rel_area, 4),
                "shannon_H": round(shannon_H, 4),
                "perplexity": perplexity,
                "gamma": round(gamma, 4),
                "char_count": char_count,
            })

        # 3. Batched GPU Inference for ResNet-18 & LeNet
        if page_tensors:
            batch_tensor = torch.stack(page_tensors).to(device)
            with torch.no_grad():
                resnet_logits = resnet(batch_tensor)
                resnet_probs = torch.softmax(resnet_logits, dim=1)[:, 1].cpu().numpy()

                lenet_logits = lenet(batch_tensor)
                lenet_probs = torch.softmax(lenet_logits, dim=1)[:, 1].cpu().numpy()
        else:
            resnet_probs = []
            lenet_probs = []

        # 4. Classification & Hard Negative Mining Logic
        page_ad_count = 0
        page_hn_count = 0

        for i, meta in enumerate(page_candidate_meta):
            r_prob = float(resnet_probs[i])
            l_prob = float(lenet_probs[i])
            lbl = meta["label"]
            x1, y1, x2, y2 = meta["box"]
            bw, bh = meta["bw"], meta["bh"]
            gamma = meta["gamma"]
            perp = meta["perplexity"]
            aspect = meta["aspect_ratio"]
            crop_img = page_candidate_crops[i]

            crop_id = f"{batch_name}_P{p_idx:03d}_R{i:03d}_x{x1}_y{y1}"

            # Detection Rule for Ads (Display or Pure Text):
            # A region is an Ad if:
            # - ResNet is very confident (p >= 0.70)
            # - OR high boundary insulation (closed box/Faraday cage) with high entropy
            # - OR labelled advertisement by YOLO with ResNet p >= 0.40
            is_ad = (
                (r_prob >= 0.70) or
                (gamma >= 0.50 and perp >= 1.70 and r_prob >= 0.35) or
                (lbl in ["advertisement", "figure"] and r_prob >= 0.45)
            )

            # Exclusion: Masthead headers (top 6% of page) are never ads
            if y1 < (h_orig * 0.055):
                is_ad = False

            if is_ad:
                save_fn = f"Ad_{crop_id}.png"
                crop_img.save(ad_crops_dir / save_fn)
                meta["class"] = "ad"
                meta["resnet_p"] = round(r_prob, 4)
                meta["lenet_p"] = round(l_prob, 4)
                meta["saved_file"] = str(ad_crops_dir / save_fn)
                page_records.append(meta)
                page_ad_count += 1
                total_ads_found += 1
            else:
                # ==============================================================
                # HARD NEGATIVE MINING LOGIC
                # ==============================================================
                saved_as_hn = False

                # 1. Hard Negative: Boxed Editorial Announcements (High gamma / closed border, but news text)
                if gamma >= 0.35 and r_prob < 0.70:
                    hn_fn = f"HN_BoxedNews_{crop_id}.png"
                    crop_img.save(hard_neg_boxed_dir / hn_fn)
                    meta["class"] = "hard_negative_boxed_news"
                    saved_as_hn = True

                # 2. Hard Negative: Extreme Aspect Ratio Single Column text (tall skinny or wide ribbon)
                elif (bh / float(bw + 1e-4) >= 4.5 or bw / float(bh + 1e-4) >= 4.5) and (bw * bh > 5000):
                    hn_fn = f"HN_ExtremeAspect_{crop_id}.png"
                    crop_img.save(hard_neg_aspect_dir / hn_fn)
                    meta["class"] = "hard_negative_aspect_ratio"
                    saved_as_hn = True

                # 3. Hard Negative: Tables, Commodities, Sports scores
                elif lbl in ["table", "table-of-contents", "index"] or (meta["char_count"] > 30 and perp > 2.0 and gamma < 0.3):
                    hn_fn = f"HN_TableScore_{crop_id}.png"
                    crop_img.save(hard_neg_tables_dir / hn_fn)
                    meta["class"] = "hard_negative_tables"
                    saved_as_hn = True
                
                # Standard Non-Ad Crop
                else:
                    std_fn = f"NonAd_{lbl}_{crop_id}.png"
                    crop_img.save(all_non_ads_dir / std_fn)
                    meta["class"] = "non_ad"

                if saved_as_hn:
                    page_hn_count += 1
                    total_hard_negatives += 1

                meta["resnet_p"] = round(r_prob, 4)
                meta["lenet_p"] = round(l_prob, 4)
                page_records.append(meta)

        elapsed = time.time() - t0
        print(f"[{p_idx:03d}/{len(image_files):03d}] {img_p.name} ({elapsed:.1f}s) -> "
              f"{len(raw_dets)} regions | {page_ad_count} Ads | {page_hn_count} Hard Negatives", flush=True)

        batch_manifest.append({
            "page_index": p_idx,
            "filename": img_p.name,
            "regions_total": len(raw_dets),
            "ads_found": page_ad_count,
            "hard_negatives": page_hn_count,
            "regions": page_records
        })

    # Save manifest
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(batch_manifest, f, indent=2)

    total_batch_time = time.time() - start_batch_time
    print(f"\n=======================================================", flush=True)
    print(f"✅ {batch_name} Completed in {total_batch_time/60.0:.2f} minutes!", flush=True)
    print(f"⭐ Total Ads Mined: {total_ads_found}", flush=True)
    print(f"🎯 Total Hard Negatives Mined: {total_hard_negatives}", flush=True)
    print(f"📄 Metadata written: {metadata_path}", flush=True)
    print(f"=======================================================\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, required=True, help="e.g. batch_2")
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to batch image folder")
    parser.add_argument("--output_base", type=str, default=r"C:\Users\shiny\Desktop\Detected_Advertisements_700")
    args = parser.parse_args()

    process_batch(args.batch, Path(args.scan_dir), Path(args.output_base))
