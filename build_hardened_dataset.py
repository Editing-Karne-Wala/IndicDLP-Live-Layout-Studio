# -*- coding: utf-8 -*-
import os
import sys
import shutil
import random
import json
from pathlib import Path

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SOURCE_700_DIR = Path(r"C:\Users\shiny\Desktop\Detected_Advertisements_700")
PREV_DATASET_DIR = Path(r"C:\Users\shiny\Desktop\Ad_Dataset")
OUTPUT_DIR = Path(r"C:\Users\shiny\Desktop\Hardened_Ad_Dataset")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def compile_hardened_dataset():
    print("==================================================================", flush=True)
    print("🚀 Initializing Stratified Compilation of Hardened Dataset", flush=True)
    print(f"📁 Destination: {OUTPUT_DIR}", flush=True)
    print("==================================================================\n", flush=True)

    # Setup directories
    for split in ["train", "val", "test"]:
        for cls in ["ad", "non_ad"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    ad_files = []
    non_ad_files = []

    # 1. Harvest from Detected_Advertisements_700 (batches 2-8)
    print("🔍 Scanning Detected_Advertisements_700 (batches 2 through 8)...", flush=True)
    for batch_dir in sorted(SOURCE_700_DIR.iterdir()):
        if not batch_dir.is_dir():
            continue

        # Ads
        ad_crops_dir = batch_dir / "ad_crops"
        if ad_crops_dir.exists():
            for f in ad_crops_dir.glob("*.png"):
                ad_files.append((f, f"700_{batch_dir.name}_{f.name}"))

        # Hard Negatives - Boxed News
        boxed_dir = batch_dir / "hard_negative_boxed_news"
        if boxed_dir.exists():
            for f in boxed_dir.glob("*.png"):
                non_ad_files.append((f, f"HN_Boxed_{batch_dir.name}_{f.name}"))

        # Hard Negatives - Extreme Aspect Ratio
        aspect_dir = batch_dir / "hard_negative_aspect_ratio"
        if aspect_dir.exists():
            for f in aspect_dir.glob("*.png"):
                non_ad_files.append((f, f"HN_Aspect_{batch_dir.name}_{f.name}"))

        # Standard non_ads
        non_ad_dir = batch_dir / "non_ad_crops"
        if non_ad_dir.exists():
            for f in non_ad_dir.glob("*.png"):
                non_ad_files.append((f, f"700_StdNonAd_{batch_dir.name}_{f.name}"))

    print(f"  • Found {len(ad_files)} Ads and {len(non_ad_files)} Non-Ads in 700-scan archive.", flush=True)

    # 2. Integrate Previous Benchmark Dataset (if available)
    if PREV_DATASET_DIR.exists():
        print("🔍 Integrating previous benchmark dataset from Ad_Dataset...", flush=True)
        prev_ads = 0
        prev_non_ads = 0
        for split in ["train", "val", "test"]:
            p_ad = PREV_DATASET_DIR / split / "ad"
            if p_ad.exists():
                for f in p_ad.glob("*.png"):
                    ad_files.append((f, f"prev_{split}_{f.name}"))
                    prev_ads += 1

            p_non = PREV_DATASET_DIR / split / "non_ad"
            if p_non.exists():
                for f in p_non.glob("*.png"):
                    non_ad_files.append((f, f"prev_{split}_{f.name}"))
                    prev_non_ads += 1

        print(f"  • Integrated {prev_ads} previous Ads and {prev_non_ads} previous Non-Ads.", flush=True)

    # Deduplicate and shuffle
    print("\n🔀 Shuffling and stratifying with seed 42...", flush=True)
    random.shuffle(ad_files)
    random.shuffle(non_ad_files)

    total_ads = len(ad_files)
    total_non_ads = len(non_ad_files)
    total_samples = total_ads + total_non_ads

    print(f"📊 Total Combined Samples: {total_samples}")
    print(f"  ⭐ Ads: {total_ads}")
    print(f"  🛡️ Non-Ads (including Hard Negatives): {total_non_ads}\n")

    manifest = {
        "dataset_name": "Hardened_Ad_Dataset",
        "description": "Historical Bengali Newspaper Ad & Hard-Negative Editorial Corpus",
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "counts": {
            "total_samples": total_samples,
            "total_ads": total_ads,
            "total_non_ads": total_non_ads,
            "train": {"ad": 0, "non_ad": 0, "total": 0},
            "val": {"ad": 0, "non_ad": 0, "total": 0},
            "test": {"ad": 0, "non_ad": 0, "total": 0}
        }
    }

    def split_and_copy(file_list, class_label):
        n = len(file_list)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        train_slice = file_list[:n_train]
        val_slice = file_list[n_train:n_train + n_val]
        test_slice = file_list[n_train + n_val:]

        slices = [("train", train_slice), ("val", val_slice), ("test", test_slice)]

        for split_name, items in slices:
            target_dir = OUTPUT_DIR / split_name / class_label
            print(f"  🚚 Copying {len(items):5d} {class_label} items -> {split_name}...", flush=True)
            for src_path, unique_name in items:
                dst_path = target_dir / unique_name
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
            manifest["counts"][split_name][class_label] = len(items)
            manifest["counts"][split_name]["total"] += len(items)

    print("📦 Processing class: 'ad'...", flush=True)
    split_and_copy(ad_files, "ad")

    print("\n📦 Processing class: 'non_ad' (with Boxed News & Aspect Ratio Negatives)...", flush=True)
    split_and_copy(non_ad_files, "non_ad")

    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n==================================================================", flush=True)
    print("✅ Hardened Dataset Compilation Complete!", flush=True)
    print(f"📄 Manifest saved to: {manifest_path}", flush=True)
    print(f"📊 Train: {manifest['counts']['train']['total']} crops ({manifest['counts']['train']['ad']} ads / {manifest['counts']['train']['non_ad']} non-ads)")
    print(f"📊 Val:   {manifest['counts']['val']['total']} crops ({manifest['counts']['val']['ad']} ads / {manifest['counts']['val']['non_ad']} non-ads)")
    print(f"📊 Test:  {manifest['counts']['test']['total']} crops ({manifest['counts']['test']['ad']} ads / {manifest['counts']['test']['non_ad']} non-ads)")
    print("==================================================================", flush=True)

if __name__ == "__main__":
    compile_hardened_dataset()
