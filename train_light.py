"""5-fold CV training — LightCNN3D (lightweight) for AD vs NC.

Key differences from train_densenet.py:
  - Model: LightCNN3D (587K params) instead of DenseNet121 (11M params)
  - Loss:  Focal loss to prevent sensitivity collapse
  - LR:    OneCycleLR (warmup + cosine decay) instead of ReduceLROnPlateau
  - Label smoothing: 0.1 to prevent overconfident predictions
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADNINpyDataset, compute_norm_stats, filter_task_df
from models.light_cnn3d import build_light_cnn3d


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss extended to multi-class via per-class weighting."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")

    # Youden's J: find threshold maximising sensitivity + specificity - 1
    youden_thresh, youden_sens, youden_spec = 0.5, float("nan"), float("nan")
    if len(np.unique(labels)) == 2:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j_scores = tpr + (1 - fpr) - 1
        best_idx = int(np.argmax(j_scores))
        youden_thresh = float(thresholds[best_idx])
        preds_y = (probs >= youden_thresh).astype(int)
        cm_y = confusion_matrix(labels, preds_y, labels=[0, 1])
        tn_y, fp_y, fn_y, tp_y = cm_y.ravel()
        youden_sens = float(tp_y / (tp_y + fn_y)) if (tp_y + fn_y) else float("nan")
        youden_spec = float(tn_y / (tn_y + fp_y)) if (tn_y + fp_y) else float("nan")

    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "auc":         auc,
        "youden_threshold": youden_thresh,
        "youden_sensitivity": youden_sens,
        "youden_specificity": youden_spec,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler=None,
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
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.asarray(all_labels), np.asarray(all_probs)


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------

def make_loaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    stats: dict, batch_size: int, preload: bool,
) -> tuple[DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    train_ds = ADNINpyDataset(train_df, augment=True,
                              mean=float(stats["mean"]), std=float(stats["std"]),
                              preload=preload)
    val_ds   = ADNINpyDataset(val_df,   augment=False,
                              mean=float(stats["mean"]), std=float(stats["std"]),
                              preload=preload)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin),
    )


def train_one_fold(
    fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
    args: argparse.Namespace, device: torch.device,
) -> dict:
    print(f"\n=== Fold {fold}/{args.n_splits} ===")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    print(train_df["group"].value_counts().reindex(["NC", "AD"], fill_value=0).to_string())

    stats_path = args.results_dir / f"norm_stats_light_fold{fold}.json"
    stats = compute_norm_stats(train_df, stats_path, target_shape=(128, 128, 128), task="ad_nc")
    train_loader, val_loader = make_loaders(train_df, val_df, stats, args.batch_size, args.preload)

    model = build_light_cnn3d(num_classes=2, dropout=args.dropout).to(device)

    class_weight = torch.tensor([1.0, args.ad_weight], dtype=torch.float32, device=device)
    print(f"Class weight: NC={class_weight[0]:.2f}, AD={class_weight[1]:.2f}")

    if args.focal_gamma > 0:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weight,
                              label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight,
                                        label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.use_onecycle:
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.1, anneal_strategy="cos",
            div_factor=10.0, final_div_factor=1e4,
        )
        epoch_scheduler = None
    else:
        # CosineAnnealingLR: stepped once per epoch, more stable for small datasets
        batch_scheduler = None
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path  = args.results_dir / f"training_log_light_fold{fold}.csv"
    ckpt_path = args.checkpoint_dir / f"best_light_fold{fold}.pth"
    best_val_auc = 0.0
    patience_counter = 0
    best_metrics: dict | None = None
    best_epoch = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "fold", "epoch", "train_loss", "val_loss",
            "val_accuracy", "val_sensitivity", "val_specificity", "val_auc", "lr",
        ])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, batch_scheduler)
            val_loss, val_labels, val_probs = run_epoch(
                model, val_loader, criterion, device)
            metrics = compute_metrics(val_labels, val_probs)
            if epoch_scheduler is not None:
                epoch_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow({
                "fold": fold, "epoch": epoch,
                "train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}",
                "val_accuracy":    f"{metrics['accuracy']:.6f}",
                "val_sensitivity": f"{metrics['sensitivity']:.6f}",
                "val_specificity": f"{metrics['specificity']:.6f}",
                "val_auc":         f"{metrics['auc']:.6f}", "lr": f"{lr:.8f}",
            })
            f.flush()

            print(
                f"Fold {fold} Epoch {epoch:03d} | "
                f"train={train_loss:.4f} val={val_loss:.4f} "
                f"acc={metrics['accuracy']:.4f} sens={metrics['sensitivity']:.4f} "
                f"spec={metrics['specificity']:.4f} auc={metrics['auc']:.4f} lr={lr:.2e}"
            )

            if metrics["auc"] > best_val_auc + args.min_delta:
                best_val_auc = metrics["auc"]
                patience_counter = 0
                best_metrics = metrics
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "fold": fold, "epoch": epoch,
                    "best_val_auc": best_val_auc,
                    "val_metrics": metrics,
                    "norm_stats": stats,
                    "label_map": {"NC": 0, "AD": 1},
                    "target_shape": (128, 128, 128),
                    "model_name": "LightCNN3D",
                }, ckpt_path)
                print(f"  -> Saved (val_auc={best_val_auc:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}. Best={best_epoch}")
                    break

    if best_metrics is None:
        raise RuntimeError(f"Fold {fold} produced no valid metrics.")

    result = {"fold": fold, "best_epoch": best_epoch,
              "best_val_auc": best_val_auc, **best_metrics,
              "checkpoint": str(ckpt_path)}
    print(f"Fold {fold}: Acc={result['accuracy']:.4f} Sens={result['sensitivity']:.4f} "
          f"Spec={result['specificity']:.4f} AUC={result['auc']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_results(results: list[dict], output_csv: Path) -> None:
    df = pd.DataFrame(results)
    metric_cols = ["accuracy", "sensitivity", "specificity", "auc"]
    summary_rows = []
    for m in metric_cols:
        values = df[m].astype(float)
        summary_rows.append({"metric": m, "mean": values.mean(), "std": values.std(ddof=1)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("\n=== 5-Fold CV Results (LightCNN3D) ===")
    for row in summary_rows:
        print(f"{row['metric'].capitalize():<12}: {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"\nSaved: {output_csv}\nSaved: {summary_csv}")


# ---------------------------------------------------------------------------
# Arg parsing + main
# ---------------------------------------------------------------------------

def load_cv_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject_id" not in df.columns:
        raise ValueError(f"Missing 'subject_id' column in {path}")
    df = filter_task_df(df, task="ad_nc")
    if df.empty:
        raise ValueError(f"No NC/AD rows in {path}")
    missing = [p for p in df["file_path"].astype(str) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing[:3]}")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-fold CV — LightCNN3D AD vs NC")
    p.add_argument("--data_csv",       type=Path, default=Path("data/processed_list.csv"))
    p.add_argument("--results_dir",    type=Path, default=Path("results"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--output_csv",     type=Path, default=Path("results/cv_light_results.csv"))
    p.add_argument("--n_splits",       type=int,   default=5)
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-3)
    p.add_argument("--dropout",        type=float, default=0.5)
    p.add_argument("--ad_weight",      type=float, default=1.0)
    p.add_argument("--focal_gamma",    type=float, default=2.0)
    p.add_argument("--label_smoothing",type=float, default=0.1)
    p.add_argument("--patience",       type=int,   default=25)
    p.add_argument("--min_delta",      type=float, default=1e-4)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--preload",        action="store_true", default=True)
    p.add_argument("--use_onecycle",   action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: LightCNN3D (~587K params) — batch_size={args.batch_size}")

    df = load_cv_dataframe(args.data_csv)
    labels = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()
    print(f"Total scans: {len(df)} | Unique subjects: {len(set(groups))}")
    print(df["group"].value_counts().reindex(["NC", "AD"], fill_value=0).to_string())

    kf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, labels, groups=groups), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        assert len(set(train_df["subject_id"]) & set(val_df["subject_id"])) == 0
        fold_results.append(train_one_fold(fold, train_df, val_df, args, device))

    summarize_results(fold_results, args.output_csv)

    best_fold = max(fold_results, key=lambda r: r["auc"])
    best_ckpt = Path(best_fold["checkpoint"])
    if best_ckpt.exists():
        shutil.copy2(best_ckpt, args.checkpoint_dir / "best_light.pth")
        print(f"Best fold AUC={best_fold['auc']:.4f} -> checkpoints/best_light.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
