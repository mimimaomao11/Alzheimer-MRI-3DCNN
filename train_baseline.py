from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADNINpyDataset, compute_norm_stats, filter_task_df
from models.baseline_cnn import Baseline3DCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc="Train" if training else "Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            use_amp = scaler is not None and device.type == "cuda"

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]
            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.asarray(all_labels), np.asarray(all_probs)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    stats: dict,
    batch_size: int,
    preload: bool,
) -> tuple[DataLoader, DataLoader]:
    train_ds = ADNINpyDataset(
        train_df,
        augment=True,
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        preload=preload,
    )
    val_ds = ADNINpyDataset(
        val_df,
        augment=False,
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        preload=preload,
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    return train_loader, val_loader


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    print(f"\n=== Fold {fold}/{args.n_splits} ===")
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print("Train class counts:")
    print(train_df["group"].value_counts().reindex(["NC", "AD"], fill_value=0).to_string())
    print("Val class counts:")
    print(val_df["group"].value_counts().reindex(["NC", "AD"], fill_value=0).to_string())

    stats_path = args.results_dir / f"norm_stats_fold{fold}.json"
    stats = compute_norm_stats(train_df, stats_path, target_shape=(96, 96, 96), task="ad_nc")
    train_loader, val_loader = make_loaders(train_df, val_df, stats, args.batch_size, args.preload)

    model = Baseline3DCNN(num_classes=2, dropout3d=args.dropout3d).to(device)
    counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0)
    n_nc = int(counts.loc[0])
    n_ad = int(counts.loc[1])
    class_weight = torch.tensor([1.0, n_nc / max(n_ad, 1)], dtype=torch.float32, device=device)
    print(f"Class weight: NC={class_weight[0].item():.6f}, AD={class_weight[1].item():.6f}")

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    fold_log_path = args.results_dir / f"training_log_fold{fold}.csv"
    checkpoint_path = args.checkpoint_dir / f"best_baseline_fold{fold}.pth"
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics: dict | None = None
    best_epoch = 0

    with fold_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_sensitivity",
                "val_specificity",
                "val_auc",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, device)
            metrics = compute_metrics(val_labels, val_probs)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            writer.writerow(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{metrics['accuracy']:.6f}",
                    "val_sensitivity": f"{metrics['sensitivity']:.6f}",
                    "val_specificity": f"{metrics['specificity']:.6f}",
                    "val_auc": f"{metrics['auc']:.6f}",
                    "lr": f"{lr:.8f}",
                }
            )
            f.flush()

            print(
                f"Fold {fold} Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} acc={metrics['accuracy']:.4f} "
                f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f} "
                f"auc={metrics['auc']:.4f} lr={lr:.2e}"
            )

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_metrics = metrics
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "fold": fold,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "val_metrics": metrics,
                        "norm_stats": stats,
                        "label_map": {"NC": 0, "AD": 1},
                        "target_shape": (96, 96, 96),
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(
                        f"Early stopping fold {fold} at epoch {epoch}. "
                        f"Best epoch={best_epoch}, best val_loss={best_val_loss:.6f}"
                    )
                    break

    if best_metrics is None:
        raise RuntimeError(f"Fold {fold} did not produce metrics.")

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        **best_metrics,
        "checkpoint": str(checkpoint_path),
    }
    print(
        f"Fold {fold}: Acc={result['accuracy']:.4f} "
        f"Sensitivity={result['sensitivity']:.4f} "
        f"Specificity={result['specificity']:.4f} "
        f"AUC={result['auc']:.4f}"
    )
    return result


def summarize_results(results: list[dict], output_csv: Path) -> None:
    df = pd.DataFrame(results)
    metric_cols = ["accuracy", "sensitivity", "specificity", "auc"]
    summary_rows = []
    for metric in metric_cols:
        values = df[metric].astype(float)
        summary_rows.append(
            {
                "fold": "mean",
                "metric": metric,
                "value": values.mean(),
                "std": values.std(ddof=1),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("\n=== 5-Fold CV Results ===")
    for metric in metric_cols:
        values = df[metric].astype(float)
        print(f"{metric.capitalize():<12}: {values.mean():.4f} ± {values.std(ddof=1):.4f}")
    print(f"\nSaved fold results: {output_csv}")
    print(f"Saved summary: {summary_csv}")


def load_cv_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject_id" not in df.columns:
        raise ValueError(f"'subject_id' column missing from {path}. Required for subject-level CV split.")
    df = filter_task_df(df, task="ad_nc")
    if df.empty:
        raise ValueError(f"No NC/AD rows found in {path}")
    missing = [p for p in df["file_path"].astype(str) if not Path(p).exists()]
    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(f"Some file_path entries do not exist. First missing paths:\n{preview}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold CV training for AD vs NC baseline 3D CNN")
    parser.add_argument("--data_csv", type=Path, default=Path("data/processed_list.csv"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_csv", type=Path, default=Path("results/cv_baseline_results.csv"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout3d", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    df = load_cv_dataframe(args.data_csv)
    labels = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()
    n_unique = len(set(groups))
    print(f"CV data CSV: {args.data_csv}")
    print(f"Total scans: {len(df)} | Unique subjects: {n_unique}")
    print(df["group"].value_counts().reindex(["NC", "AD"], fill_value=0).to_string())

    # StratifiedGroupKFold ensures no subject appears in both train and val
    # (each subject has 2 near-identical scans; leakage would inflate metrics)
    kf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, labels, groups=groups), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        assert len(set(train_df["subject_id"]) & set(val_df["subject_id"])) == 0, "Subject leakage detected!"
        fold_results.append(train_one_fold(fold, train_df, val_df, args, device))

    summarize_results(fold_results, args.output_csv)

    best_fold = max(fold_results, key=lambda r: r["auc"])
    best_checkpoint = Path(best_fold["checkpoint"])
    if best_checkpoint.exists():
        shutil.copy2(best_checkpoint, args.checkpoint_dir / "best_baseline.pth")
        print(f"Copied highest-AUC fold checkpoint to: {args.checkpoint_dir / 'best_baseline.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
