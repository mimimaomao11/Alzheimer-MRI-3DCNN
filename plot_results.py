"""Generate report figures from 5-fold CV training logs and results CSVs.

Usage (after training completes):
    python plot_results.py --task ad_nc
    python plot_results.py --task mci_conversion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(log_dir: Path, prefix: str, out_path: Path, n_folds: int = 5) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10.colors

    for fold in range(1, n_folds + 1):
        log_file = log_dir / f"training_log_{prefix}_fold{fold}.csv"
        if not log_file.exists():
            continue
        df = pd.read_csv(log_file)
        c = colors[(fold - 1) % len(colors)]
        axes[0].plot(df["epoch"], df["train_loss"], color=c, alpha=0.7, linewidth=1.2,
                     label=f"Fold {fold}")
        axes[0].plot(df["epoch"], df["val_loss"], color=c, alpha=0.4, linewidth=1.2,
                     linestyle="--")
        axes[1].plot(df["epoch"], df["val_auc"], color=c, linewidth=1.2,
                     label=f"Fold {fold}")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train loss (solid) / Val loss (dashed)")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation AUC per fold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-fold ROC
# ---------------------------------------------------------------------------

def _load_best_val_probs(log_dir: Path, prefix: str, cv_csv: Path, n_folds: int = 5):
    """
    Reconstruct (labels, probs) at best epoch per fold from per-epoch CSV + cv results.
    Falls back to just returning the AUC values from the cv results.
    """
    cv_df = pd.read_csv(cv_csv)
    return cv_df


def plot_metrics_summary(cv_csv: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(cv_csv)
    metric_cols = [c for c in ["accuracy", "sensitivity", "specificity", "auc"] if c in df.columns]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metric_cols))
    means = [df[c].astype(float).mean() for c in metric_cols]
    stds  = [df[c].astype(float).std(ddof=1) for c in metric_cols]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.8,
                  error_kw=dict(elinewidth=1.5, ecolor="black"))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.01,
                f"{mean:.3f}±{std:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_cols], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Confusion matrix aggregated across folds (from best-fold metrics in cv csv)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cv_csv: Path, out_path: Path, class_names: list[str], title: str) -> None:
    df = pd.read_csv(cv_csv)
    if not all(c in df.columns for c in ["tn", "fp", "fn", "tp"]):
        print(f"Skipping confusion matrix (no tn/fp/fn/tp columns in {cv_csv})")
        return

    tn = int(df["tn"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())
    tp = int(df["tp"].sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-fold AUC bar chart
# ---------------------------------------------------------------------------

def plot_per_fold_auc(cv_csv: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(cv_csv)
    if "auc" not in df.columns or "fold" not in df.columns:
        return

    folds = df["fold"].astype(int).tolist()
    aucs  = df["auc"].astype(float).tolist()
    mean_auc = np.mean(aucs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Fold {f}" for f in folds], aucs, color="#4C72B0", alpha=0.8)
    ax.axhline(mean_auc, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "ad_nc": {
        "log_prefix": "densenet",
        "cv_csv": "results/cv_densenet_results.csv",
        "class_names": ["NC", "AD"],
        "label": "DenseNet121 AD vs NC",
        "fig_suffix": "adnc",
    },
    "baseline": {
        "log_prefix": "baseline",
        "cv_csv": "results/cv_baseline_v2_results.csv",
        "class_names": ["NC", "AD"],
        "label": "Baseline 3D CNN AD vs NC",
        "fig_suffix": "baseline",
    },
    "light": {
        "log_prefix": "light",
        "cv_csv": "results/cv_light_v3_results.csv",
        "class_names": ["NC", "AD"],
        "label": "LightCNN3D v3 AD vs NC",
        "fig_suffix": "light",
    },
    "light_v4": {
        "log_prefix": "light",
        "cv_csv": "results/cv_light_v4_results.csv",
        "class_names": ["NC", "AD"],
        "label": "LightCNN3D v4 AD vs NC",
        "fig_suffix": "light_v4",
    },
    "mci_conversion": {
        "log_prefix": "mci",
        "cv_csv": "results/cv_mci_results.csv",
        "class_names": ["sMCI", "pMCI"],
        "label": "DenseNet121 MCI Conversion",
        "fig_suffix": "mci",
    },
    "mci_v2": {
        "log_prefix": "mci",
        "cv_csv": "results/cv_mci_v2_results.csv",
        "class_names": ["sMCI", "pMCI"],
        "label": "LightCNN3D MCI Conversion v2",
        "fig_suffix": "mci_v2",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate report figures from CV training logs")
    p.add_argument("--task", choices=list(TASK_CONFIG), default="ad_nc")
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--n_folds", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TASK_CONFIG[args.task]
    out_dir = args.results_dir / f"figures_{cfg['fig_suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_csv = Path(cfg["cv_csv"])
    if not cv_csv.exists():
        print(f"CV results not found: {cv_csv} — run training first.")
        return

    plot_learning_curves(
        args.results_dir, cfg["log_prefix"],
        out_dir / "learning_curves.png", args.n_folds
    )
    plot_metrics_summary(
        cv_csv,
        out_dir / "metrics_summary.png",
        f"{cfg['label']}: 5-Fold CV Metrics"
    )
    plot_confusion_matrix(
        cv_csv,
        out_dir / "confusion_matrix.png",
        cfg["class_names"],
        f"{cfg['label']}: Aggregate Confusion Matrix"
    )
    plot_per_fold_auc(
        cv_csv,
        out_dir / "per_fold_auc.png",
        f"{cfg['label']}: AUC per Fold"
    )
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
