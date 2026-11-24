"""
========================================================================================
Project: National Indic Document Layout Parsing (IndicDLP) | Task 2.C
Component: High-Speed GPU Benchmark & Training Engine on Hardened Dataset (68,000 Crops)
Institution: IDEAS-TIH, Indian Statistical Institute (ISI) Kolkata / DST
Author: Research Intern Team & Antigravity
Timestamp: 2026-11-25 02:51:00 IST
========================================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

REPO_DIR = Path(r"C:\Users\shiny\Documents\IndicDLP-Live-Layout-Studio")
DATASET_DIR = Path(r"C:\Users\shiny\Desktop\Hardened_Ad_Dataset")
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from ad_dataset import HistoricalAdDataset
from models_benchmark import ModifiedLeNet5, build_resnet18

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=("cuda" in str(device))):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_probs": all_probs
    }

def train_one_model(model_name: str, epochs: int = 5, batch_size: int = 64, lr: float = 2e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"🚀 Training {model_name.upper()} on Hardened Dataset (68,000 crops)", flush=True)
    print(f"⚡ Device: {device} | Batch Size: {batch_size} | Epochs: {epochs}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Datasets & Loaders
    train_dataset = HistoricalAdDataset(root_dir=DATASET_DIR, split="train")
    val_dataset = HistoricalAdDataset(root_dir=DATASET_DIR, split="val")
    test_dataset = HistoricalAdDataset(root_dir=DATASET_DIR, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"📊 Samples -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}", flush=True)

    # Class balance weights
    num_ads = sum(1 for _, t in train_dataset.samples if t == 1)
    num_non_ads = len(train_dataset) - num_ads
    weights = torch.tensor([1.0, float(num_non_ads) / float(max(1, num_ads))], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 2. Build Model
    if model_name.lower() == "lenet":
        model = ModifiedLeNet5(num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        model = build_resnet18(num_classes=2, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in str(device)))

    best_val_f1 = 0.0
    checkpoint_path = CHECKPOINTS_DIR / f"hardened_best_{model_name.lower()}.pt"
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=("cuda" in str(device))):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_avg_loss = train_loss / len(train_dataset)

        # Evaluate on validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_avg_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_avg_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"Val F1: {val_metrics['f1_score']:.4f} | "
              f"Val Rec: {val_metrics['recall']*100:.2f}%", flush=True)

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ⭐ Saved Best Checkpoint -> {checkpoint_path.name} (F1: {best_val_f1:.4f})", flush=True)

    # Final Test Set Evaluation
    print(f"\n🧪 Running Final Test Set Evaluation on {len(test_dataset)} unseen samples...", flush=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"🏆 Final Test Metrics for {model_name.upper()}:")
    print(f"  • Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  • Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  • Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  • F1 Score:  {test_metrics['f1_score']:.4f}")
    print(f"  • ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  • Confusion Matrix [TN, FP / FN, TP]: {test_metrics['confusion_matrix']}", flush=True)

    # Return summary dict
    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1_score": test_metrics["f1_score"],
            "roc_auc": test_metrics["roc_auc"],
            "confusion_matrix": test_metrics["confusion_matrix"]
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    results = {}
    print("==================================================================", flush=True)
    print("🌟 IndicDLP: Full Benchmark Execution (LeNet-5 vs ResNet-18)", flush=True)
    print("==================================================================", flush=True)

    # 1. Train LeNet-5 Baseline
    results["lenet"] = train_one_model("lenet", epochs=args.epochs, batch_size=args.batch_size, lr=3e-4)

    # 2. Train ResNet-18 Deep Residual Model
    results["resnet18"] = train_one_model("resnet18", epochs=args.epochs, batch_size=args.batch_size, lr=1.5e-4)

    out_json = REPO_DIR / "hardened_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ All Benchmark Training Complete! Results saved -> {out_json}", flush=True)
