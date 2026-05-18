from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import (ADNINpyDataset, compute_norm_stats,
                     CLINICAL_COLS, merge_clinical, compute_clinical_stats)
from models.light_cnn3d import build_light_cnn3d


# ---------- label helpers ----------

MCI_LABEL_MAP = {"sMCI": 0, "pMCI": 1}


def filter_mci_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["conversion_group"].isin(MCI_LABEL_MAP)].copy()
    df["label"] = df["conversion_group"].map(MCI_LABEL_MAP).astype(int)
    df["group"] = df["conversion_group"]   # ADNINpyDataset expects 'group' col
    return df.reset_index(drop=True)


# ---------- reuse helpers from train_densenet ----------

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
    use_clinical: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="Train" if training else "Val", leave=False):
            if use_clinical:
                images, clinical, labels = batch
                clinical = clinical.to(device, non_blocking=True)
            else:
                images, labels = batch
                clinical = None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            use_amp = scaler is not None and device.type == "cuda"

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, clinical)
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

            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            total_loss += float(loss.item()) * labels.size(0)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.asarray(all_labels), np.asarray(all_probs)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")

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
        "accuracy": float(accuracy_score(labels, preds)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "auc": auc,
        "youden_threshold": youden_thresh,
        "youden_sensitivity": youden_sens,
        "youden_specificity": youden_spec,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    stats: dict,
    batch_size: int,
    preload: bool,
    clinical_cols: list[str] | None = None,
    clinical_stats: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = ADNINpyDataset(
        train_df, task="mci_conversion", augment=True,
        mean=float(stats["mean"]), std=float(stats["std"]), preload=preload,
        clinical_cols=clinical_cols, clinical_stats=clinical_stats,
    )
    val_ds = ADNINpyDataset(
        val_df, task="mci_conversion", augment=False,
        mean=float(stats["mean"]), std=float(stats["std"]), preload=preload,
        clinical_cols=clinical_cols, clinical_stats=clinical_stats,
    )
    pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin),
    )


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    print(f"\n=== Fold {fold}/{args.n_splits} ===")
    print(f"Train: {len(train_df)} scans ({train_df['subject_id'].nunique()} subj) | "
          f"Val: {len(val_df)} scans ({val_df['subject_id'].nunique()} subj)")
    print(f"  Train pMCI/sMCI: {(train_df.group=='pMCI').sum()}/{(train_df.group=='sMCI').sum()}")
    print(f"  Val   pMCI/sMCI: {(val_df.group=='pMCI').sum()}/{(val_df.group=='sMCI').sum()}")

    clinical_cols = CLINICAL_COLS if args.adnimerge_csv else []
    clinical_stats = compute_clinical_stats(train_df, clinical_cols) if clinical_cols else {}
    if clinical_cols:
        print(f"  [clinical] Using {len(clinical_cols)} features: {clinical_cols}")

    stats_path = args.results_dir / f"norm_stats_mci_fold{fold}.json"
    stats = compute_norm_stats(train_df, stats_path, target_shape=(128, 128, 128), task="mci_conversion")
    train_loader, val_loader = make_loaders(
        train_df, val_df, stats, args.batch_size, args.preload,
        clinical_cols=clinical_cols or None,
        clinical_stats=clinical_stats or None,
    )

    model = build_light_cnn3d(num_classes=2, dropout=args.dropout_prob,
                               num_clinical=len(clinical_cols)).to(device)

    class_weight = torch.tensor([1.0, args.pmci_weight], dtype=torch.float32, device=device)
    print(f"Class weight: sMCI={class_weight[0].item():.2f}, pMCI={class_weight[1].item():.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path = args.results_dir / f"training_log_mci_fold{fold}.csv"
    ckpt_path = args.checkpoint_dir / f"best_mci_fold{fold}.pth"
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

        use_clinical = len(clinical_cols) > 0
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(model, train_loader, criterion, device,
                                         optimizer, scaler, use_clinical=use_clinical)
            val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, device,
                                                        use_clinical=use_clinical)
            metrics = compute_metrics(val_labels, val_probs)
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow({
                "fold": fold, "epoch": epoch,
                "train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{metrics['accuracy']:.6f}",
                "val_sensitivity": f"{metrics['sensitivity']:.6f}",
                "val_specificity": f"{metrics['specificity']:.6f}",
                "val_auc": f"{metrics['auc']:.6f}", "lr": f"{lr:.8f}",
            })
            f.flush()

            scheduler.step()
            print(f"Fold {fold} Epoch {epoch:03d} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f} "
                  f"auc={metrics['auc']:.4f} lr={lr:.2e}")

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
                    "label_map": {"sMCI": 0, "pMCI": 1},
                    "target_shape": (128, 128, 128),
                    "model_name": "LightCNN3D_MCI",
                    "num_clinical": len(clinical_cols),
                    "clinical_cols": clinical_cols,
                    "clinical_stats": clinical_stats,
                }, ckpt_path)
                print(f"  -> Saved (val_auc={best_val_auc:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}")
                    break

    if best_metrics is None:
        raise RuntimeError(f"Fold {fold} produced no valid metrics.")

    result = {"fold": fold, "best_epoch": best_epoch,
              "best_val_auc": best_val_auc, **best_metrics,
              "checkpoint": str(ckpt_path)}
    print(f"Fold {fold}: Acc={result['accuracy']:.4f} "
          f"Sens={result['sensitivity']:.4f} Spec={result['specificity']:.4f} "
          f"AUC={result['auc']:.4f}")
    return result


def summarize_results(results: list[dict], output_csv: Path) -> None:
    df = pd.DataFrame(results)
    metric_cols = ["accuracy", "sensitivity", "specificity", "auc"]
    summary_rows = []
    for metric in metric_cols:
        values = df[metric].astype(float)
        summary_rows.append({"metric": metric, "mean": values.mean(), "std": values.std(ddof=1)})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("\n=== 5-Fold CV Results (MCI Conversion: pMCI vs sMCI) ===")
    for row in summary_rows:
        print(f"{row['metric'].capitalize():<12}: {row['mean']:.4f} ± {row['std']:.4f}")
    print(f"\nSaved: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold CV — MCI conversion prediction (pMCI vs sMCI)")
    parser.add_argument("--data_csv", type=Path, default=Path("data/mci_conversion_list.csv"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output_csv", type=Path, default=Path("results/cv_mci_v2_results.csv"))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--pmci_weight", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload", action="store_true", default=True)
    parser.add_argument("--adnimerge_csv", type=Path, default=None,
                        help="Path to ADNIMERGE.csv; enables clinical feature fusion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.data_csv)
    df = filter_mci_df(df)
    if df.empty:
        raise ValueError(f"No pMCI/sMCI rows in {args.data_csv}")
    if "subject_id" not in df.columns:
        raise ValueError("'subject_id' column missing.")

    if args.adnimerge_csv:
        print(f"Merging clinical features from {args.adnimerge_csv}")
        df = merge_clinical(df, args.adnimerge_csv)

    labels = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()
    print(f"Total MCI scans: {len(df)} | Unique subjects: {df['subject_id'].nunique()}")
    print(f"pMCI: {(df.group=='pMCI').sum()} scans / {df[df.group=='pMCI']['subject_id'].nunique()} subj")
    print(f"sMCI: {(df.group=='sMCI').sum()} scans / {df[df.group=='sMCI']['subject_id'].nunique()} subj")

    kf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, labels, groups=groups), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        assert len(set(train_df["subject_id"]) & set(val_df["subject_id"])) == 0, "Subject leakage!"
        fold_results.append(train_one_fold(fold, train_df, val_df, args, device))

    summarize_results(fold_results, args.output_csv)

    best_fold = max(fold_results, key=lambda r: r["auc"])
    best_ckpt = Path(best_fold["checkpoint"])
    if best_ckpt.exists():
        shutil.copy2(best_ckpt, args.checkpoint_dir / "best_mci.pth")
        print(f"Best fold (AUC={best_fold['auc']:.4f}) → checkpoints/best_mci.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
