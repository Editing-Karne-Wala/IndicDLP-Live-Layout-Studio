# ========================================================================================
# IndicDLP: Training Engine with Weighted Cross-Entropy Loss & Evaluation Metrics
# Usage: python train_benchmark.py --model lenet --epochs 8
#        python train_benchmark.py --model resnet18 --epochs 8
# ========================================================================================

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

REPO_DIR = Path(r"C:\Users\shiny\Documents\IndicDLP-Live-Layout-Studio")
DATASET_DIR = Path(r"C:\Users\shiny\Desktop\Ad_Dataset")
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from ad_dataset import HistoricalAdDataset
from models_benchmark import ModifiedLeNet5, build_resnet18

def parse_args():
    parser = argparse.ArgumentParser(description="Train Ad vs Non-Ad Classifier")
    parser.add_argument("--model", type=str, default="resnet18", choices=["lenet", "resnet18"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    return parser.parse_args()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
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
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", pos_label=1, zero_division=0)
    macro_f1 = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)[2]
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_targets, all_preds)
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "macro_f1": macro_f1,
        "auc": auc,
        "confusion_matrix": cm
    }

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"🚀 Initializing Benchmark Training for: {args.model.upper()}")
    print(f"🖥️  Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    # 1. Datasets & Loaders
    print("📦 Loading Train, Val, and Test DataLoaders with AspectRatioPad (224x224)...")
    train_dataset = HistoricalAdDataset(DATASET_DIR, split="train")
    val_dataset = HistoricalAdDataset(DATASET_DIR, split="val")
    test_dataset = HistoricalAdDataset(DATASET_DIR, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"   • Train Samples: {len(train_dataset):,}")
    print(f"   • Val Samples:   {len(val_dataset):,}")
    print(f"   • Test Samples:  {len(test_dataset):,}")

    # 2. Weighted Loss Calculation
    # Class 0 (non_ad), Class 1 (ad)
    n_ad = sum(1 for _, l in train_dataset.samples if l == 1)
    n_non_ad = len(train_dataset) - n_ad
    w_non_ad = len(train_dataset) / (2.0 * n_non_ad)
    w_ad = len(train_dataset) / (2.0 * n_ad)
    class_weights = torch.tensor([w_non_ad, w_ad], dtype=torch.float32).to(device)
    print(f"⚖️  Inverse Frequency Weights: Non-Ad = {w_non_ad:.4f} | Ad = {w_ad:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model & Optimizer
    if args.model == "lenet":
        model = ModifiedLeNet5(num_classes=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 2, weight_decay=1e-4)
    else:
        model = build_resnet18(num_classes=2, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_f1 = 0.0
    best_model_path = CHECKPOINTS_DIR / f"best_{args.model}.pt"

    print(f"\n🏋️ Starting Training ({args.epochs} Epochs)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"Val Ad-P: {val_metrics['precision']:.3f} | "
              f"Val Ad-R: {val_metrics['recall']:.3f} | "
              f"Val F1: {val_metrics['f1']:.3f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ Best Model Saved to {best_model_path.name} (Val Ad-F1: {best_val_f1:.3f})")

    # 4. Final Test Set Evaluation
    print("\n" + "=" * 65)
    print(f"🧪 Evaluating Best {args.model.upper()} on Unseen TEST SET ({len(test_dataset)} samples)...")
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print("=" * 65)
    print(f"🎯 Final Test Results for {args.model.upper()}:")
    print(f"   • Test Accuracy:        {test_metrics['accuracy']*100:.2f}%")
    print(f"   • Test Ad-Precision:    {test_metrics['precision']*100:.2f}%")
    print(f"   • Test Ad-Recall:       {test_metrics['recall']*100:.2f}%")
    print(f"   • Test Ad-F1 Score:     {test_metrics['f1']*100:.2f}%")
    print(f"   • Test Macro-F1:        {test_metrics['macro_f1']*100:.2f}%")
    print(f"   • Test ROC-AUC:         {test_metrics['auc']:.4f}")
    print("   • Confusion Matrix [TN, FP / FN, TP]:")
    print(test_metrics["confusion_matrix"])
    print("=" * 65)

if __name__ == "__main__":
    main()
