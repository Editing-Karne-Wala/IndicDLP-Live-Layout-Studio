"""
========================================================================================
Project: National Indic Document Layout Parsing (IndicDLP) | 22 Scheduled Indian Languages
Component: Sequential Multi-Batch Mining Runner (Batches 3 through 8)
Institution: IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata / DST
Author: Research Intern Team & Antigravity
Timestamp: 2026-11-25 00:01:00 IST
========================================================================================
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

REPO_DIR = Path(r"C:\Users\shiny\Documents\IndicDLP-Live-Layout-Studio")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mine_hard_negatives_and_ads import process_batch

SCANS_ROOT = Path(r"C:\Users\shiny\Downloads\Newspaper_Scans\Extracted\Further_Extractions")
OUTPUT_BASE = Path(r"C:\Users\shiny\Desktop\Detected_Advertisements_700")

BATCHES_TO_RUN = ["batch_3", "batch_4", "batch_5", "batch_6", "batch_7", "batch_8"]

def run_all_batches():
    total_start = time.time()
    print("==================================================================", flush=True)
    print(f"🌟 Starting Autonomous Mining Pipeline for Batches 3 through 8", flush=True)
    print(f"📊 Total Batches: {len(BATCHES_TO_RUN)} (600 Broadsheet Scans Total)", flush=True)
    print("==================================================================\n", flush=True)

    summary = {}
    for idx, batch_name in enumerate(BATCHES_TO_RUN, 1):
        scan_dir = SCANS_ROOT / batch_name
        if not scan_dir.exists():
            print(f"⚠️ Warning: Directory {scan_dir} does not exist! Skipping.", flush=True)
            continue

        print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", flush=True)
        print(f"[{idx}/{len(BATCHES_TO_RUN)}] Executing Batch: {batch_name}", flush=True)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", flush=True)
        
        b_start = time.time()
        try:
            process_batch(batch_name=batch_name, scan_dir=scan_dir, output_base=OUTPUT_BASE, device="cuda:0")
            elapsed = time.time() - b_start
            summary[batch_name] = {"status": "SUCCESS", "time_min": round(elapsed / 60.0, 2)}
        except Exception as e:
            elapsed = time.time() - b_start
            print(f"❌ Error processing {batch_name}: {e}", flush=True)
            summary[batch_name] = {"status": f"FAILED: {str(e)}", "time_min": round(elapsed / 60.0, 2)}

    total_time = (time.time() - total_start) / 60.0
    print("\n==================================================================", flush=True)
    print(f"🎉 All Batches (3-8) Complete in {total_time:.2f} minutes!", flush=True)
    for b, res in summary.items():
        print(f"  • {b}: {res['status']} ({res['time_min']} mins)", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_all_batches()
