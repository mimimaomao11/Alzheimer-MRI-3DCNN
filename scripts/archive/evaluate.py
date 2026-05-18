from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADNINpyDataset
from models.baseline_cnn import Baseline3DCNN


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
    return np.asarray(labels_all), np.asarray(probs_all)


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NC", "AD"], yticklabels=["NC", "AD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_roc_curve(labels: np.ndarray, probs: np.ndarray, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AD vs NC baseline 3D CNN")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", choices=["ad_nc"], default="ad_nc")
    parser.add_argument("--test_csv", type=Path, default=Path("data/splits/test_list.csv"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location=device)
    norm_stats = checkpoint.get("norm_stats")
    if not norm_stats:
        raise ValueError("Checkpoint does not contain norm_stats.")

    dataset = ADNINpyDataset(
        args.test_csv,
        task=args.task,
        augment=False,
        mean=float(norm_stats["mean"]),
        std=float(norm_stats["std"]),
    )
    if len(dataset) == 0:
        raise ValueError("Test split has no NC/AD samples.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = Baseline3DCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels, probs = predict(model, loader, device)
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(labels, preds))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")

    cm_path = args.results_dir / "cm_baseline.png"
    roc_path = args.results_dir / "roc_baseline.png"
    txt_path = args.results_dir / "baseline_results.txt"
    save_confusion_matrix(cm, cm_path)
    if not np.isnan(auc):
        save_roc_curve(labels, probs, auc, roc_path)

    report = [
        "Baseline 3D CNN: AD vs NC",
        "",
        f"Model: {args.model}",
        f"Test CSV: {args.test_csv}",
        f"Samples: {len(dataset)}",
        "",
        f"Accuracy: {accuracy:.4f}",
        f"Sensitivity (AD recall): {sensitivity:.4f}",
        f"Specificity (NC recall): {specificity:.4f}",
        f"AUC-ROC: {auc:.4f}",
        "",
        "Confusion matrix [[TN, FP], [FN, TP]]:",
        str(cm.tolist()),
        "",
        "Reference note:",
        "Basaia et al. 2019 reported AUC around 0.98-0.99 on a much larger dataset.",
        "For this smaller AD vs NC subset (~151 images), an AUC around 0.80-0.90 is a more realistic baseline expectation.",
    ]
    txt_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"Saved confusion matrix: {cm_path}")
    if not np.isnan(auc):
        print(f"Saved ROC curve: {roc_path}")
    print(f"Saved report: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
