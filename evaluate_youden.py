"""Evaluate saved checkpoints with Youden's J optimal threshold.

Reconstructs the exact same 5-fold splits used during training,
loads each fold's best checkpoint, runs inference, then reports
both @0.5 metrics and Youden's J optimal-threshold metrics.

Usage:
    python evaluate_youden.py --task ad_nc
    python evaluate_youden.py --task mci_conversion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ADNINpyDataset, filter_task_df, merge_clinical
from models.light_cnn3d import build_light_cnn3d


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_adnc_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return filter_task_df(df, task="ad_nc")


def load_mci_df(csv_path: Path) -> pd.DataFrame:
    MCI_LABEL_MAP = {"sMCI": 0, "pMCI": 1}
    df = pd.read_csv(csv_path)
    df = df[df["conversion_group"].isin(MCI_LABEL_MAP)].copy()
    df["label"] = df["conversion_group"].map(MCI_LABEL_MAP).astype(int)
    df["group"] = df["conversion_group"]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _tta_flip(images: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply one of 8 flip combinations to a batch of 3D volumes.

    aug_idx is a 3-bit mask: bit0=flip D, bit1=flip H, bit2=flip W.
    aug_idx=0 is the identity (no flip).
    """
    dims = [d for d, bit in zip([2, 3, 4], [1, 2, 4]) if aug_idx & bit]
    return torch.flip(images, dims=dims) if dims else images


def run_inference(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    norm_stats: dict,
    batch_size: int,
    device: torch.device,
    task: str,
    clinical_cols: list[str] | None = None,
    clinical_stats: dict | None = None,
    n_tta: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    val_ds = ADNINpyDataset(
        val_df, task=task, augment=False,
        mean=float(norm_stats["mean"]), std=float(norm_stats["std"]),
        preload=False,
        clinical_cols=clinical_cols,
        clinical_stats=clinical_stats,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_labels, all_probs = [], []
    use_clinical = bool(clinical_cols)
    n_tta = max(1, min(n_tta, 8))
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            if use_clinical:
                images, clinical, labels = batch
                clinical = clinical.to(device)
            else:
                images, labels = batch
                clinical = None
            images = images.to(device)
            prob_sum = None
            for aug_idx in range(n_tta):
                aug_images = _tta_flip(images, aug_idx)
                logits = model(aug_images, clinical)
                p = torch.softmax(logits, dim=1)[:, 1]
                prob_sum = p if prob_sum is None else prob_sum + p
            probs = prob_sum / n_tta
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    return np.asarray(all_labels), np.asarray(all_probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_full_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    # --- @0.5 ---
    preds_05 = (probs >= 0.5).astype(int)
    cm_05 = confusion_matrix(labels, preds_05, labels=[0, 1])
    tn, fp, fn, tp = cm_05.ravel()
    sens_05 = tp / (tp + fn) if (tp + fn) else float("nan")
    spec_05 = tn / (tn + fp) if (tn + fp) else float("nan")
    acc_05  = (tp + tn) / len(labels)

    # --- AUC ---
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")

    # --- Youden's J ---
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr + (1 - fpr) - 1
    best_idx = int(np.argmax(j_scores))
    youden_thresh = float(thresholds[best_idx])

    preds_y = (probs >= youden_thresh).astype(int)
    cm_y = confusion_matrix(labels, preds_y, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_y.ravel()
    youden_sens = tp_y / (tp_y + fn_y) if (tp_y + fn_y) else float("nan")
    youden_spec = tn_y / (tn_y + fp_y) if (tn_y + fp_y) else float("nan")
    youden_acc  = (tp_y + tn_y) / len(labels)

    return {
        "auc": auc,
        # @0.5
        "thresh_05": 0.5,
        "sens_05": sens_05, "spec_05": spec_05, "acc_05": acc_05,
        "tp_05": int(tp), "fn_05": int(fn), "tn_05": int(tn), "fp_05": int(fp),
        # Youden
        "youden_thresh": youden_thresh,
        "youden_sens": youden_sens, "youden_spec": youden_spec, "youden_acc": youden_acc,
        "tp_y": int(tp_y), "fn_y": int(fn_y), "tn_y": int(tn_y), "fp_y": int(fp_y),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_fold_table(results: list[dict], task: str) -> None:
    label_pos = "AD" if task == "ad_nc" else "pMCI"
    label_neg = "NC" if task == "ad_nc" else "sMCI"

    header = (
        f"\n{'='*72}\n"
        f"  Youden's J Evaluation — {'AD vs NC' if task == 'ad_nc' else 'MCI Conversion'}\n"
        f"{'='*72}\n"
        f"  Pos={label_pos}, Neg={label_neg}\n"
        f"{'='*72}"
    )
    print(header)

    # Per-fold table
    print(f"\n{'Fold':>4}  {'AUC':>6}  "
          f"{'Sens@0.5':>8}  {'Spec@0.5':>8}  "
          f"{'Youden_T':>8}  {'Youden_Sens':>11}  {'Youden_Spec':>11}  {'Youden_Acc':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['fold']:>4}  {r['auc']:>6.4f}  "
              f"{r['sens_05']:>8.4f}  {r['spec_05']:>8.4f}  "
              f"{r['youden_thresh']:>8.4f}  {r['youden_sens']:>11.4f}  "
              f"{r['youden_spec']:>11.4f}  {r['youden_acc']:>10.4f}")

    # Averages
    cols = ["auc", "sens_05", "spec_05", "youden_thresh",
            "youden_sens", "youden_spec", "youden_acc"]
    means = {c: np.mean([r[c] for r in results]) for c in cols}
    stds  = {c: np.std([r[c] for r in results], ddof=1) for c in cols}
    print("-" * 80)
    print(f"{'Mean':>4}  {means['auc']:>6.4f}  "
          f"{means['sens_05']:>8.4f}  {means['spec_05']:>8.4f}  "
          f"{means['youden_thresh']:>8.4f}  {means['youden_sens']:>11.4f}  "
          f"{means['youden_spec']:>11.4f}  {means['youden_acc']:>10.4f}")
    print(f"{'Std':>4}  {stds['auc']:>6.4f}  "
          f"{stds['sens_05']:>8.4f}  {stds['spec_05']:>8.4f}  "
          f"{stds['youden_thresh']:>8.4f}  {stds['youden_sens']:>11.4f}  "
          f"{stds['youden_spec']:>11.4f}  {stds['youden_acc']:>10.4f}")

    # Aggregated confusion matrices
    print(f"\n--- Aggregate Confusion Matrix @0.5 ---")
    tp_sum = sum(r['tp_05'] for r in results)
    fn_sum = sum(r['fn_05'] for r in results)
    tn_sum = sum(r['tn_05'] for r in results)
    fp_sum = sum(r['fp_05'] for r in results)
    total  = tp_sum + fn_sum + tn_sum + fp_sum
    print(f"  Pred {label_pos}  Pred {label_neg}")
    print(f"  True {label_pos}: {tp_sum:>4}  {fn_sum:>4}   (sens = {tp_sum/(tp_sum+fn_sum):.3f})")
    print(f"  True {label_neg}: {fp_sum:>4}  {tn_sum:>4}   (spec = {tn_sum/(tn_sum+fp_sum):.3f})")
    print(f"  Total samples: {total}")

    print(f"\n--- Aggregate Confusion Matrix @Youden threshold ---")
    tp_y_sum = sum(r['tp_y'] for r in results)
    fn_y_sum = sum(r['fn_y'] for r in results)
    tn_y_sum = sum(r['tn_y'] for r in results)
    fp_y_sum = sum(r['fp_y'] for r in results)
    print(f"  Pred {label_pos}  Pred {label_neg}")
    print(f"  True {label_pos}: {tp_y_sum:>4}  {fn_y_sum:>4}   (sens = {tp_y_sum/(tp_y_sum+fn_y_sum):.3f})")
    print(f"  True {label_neg}: {fp_y_sum:>4}  {tn_y_sum:>4}   (spec = {tn_y_sum/(tn_y_sum+fp_y_sum):.3f})")

    # Summary
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  AUC (門檻無關):          {means['auc']:.4f} ± {stds['auc']:.4f}")
    print(f"  Sensitivity @0.5:        {means['sens_05']:.4f} ± {stds['sens_05']:.4f}")
    print(f"  Sensitivity @Youden:     {means['youden_sens']:.4f} ± {stds['youden_sens']:.4f}  "
          f"  ▲ +{means['youden_sens']-means['sens_05']:+.4f}")
    print(f"  Specificity @0.5:        {means['spec_05']:.4f} ± {stds['spec_05']:.4f}")
    print(f"  Specificity @Youden:     {means['youden_spec']:.4f} ± {stds['youden_spec']:.4f}  "
          f"  {'▲' if means['youden_spec'] >= means['spec_05'] else '▼'} {means['youden_spec']-means['spec_05']:+.4f}")
    print(f"  Youden threshold (mean): {means['youden_thresh']:.4f} ± {stds['youden_thresh']:.4f}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc Youden's J evaluation on saved checkpoints")
    p.add_argument("--task", choices=["ad_nc", "mci_conversion"], default="ad_nc")
    p.add_argument("--data_csv",       type=Path, default=None)
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--n_splits",  type=int,   default=5)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--batch_size",type=int,   default=4)
    p.add_argument("--output_csv",     type=Path, default=None)
    p.add_argument("--adnimerge_csv",  type=Path, default=None,
                   help="Required when evaluating checkpoints trained with clinical features")
    p.add_argument("--tta", type=int, default=1, metavar="N",
                   help="Test-Time Augmentation: number of flip variants to average (1=off, 8=all flips)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Default CSV paths
    if args.data_csv is None:
        args.data_csv = (
            Path("data/processed_list.csv") if args.task == "ad_nc"
            else Path("data/mci_conversion_list.csv")
        )
    if args.output_csv is None:
        tag = "adnc" if args.task == "ad_nc" else "mci"
        args.output_csv = Path(f"results/youden_eval_{tag}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task:   {args.task}")
    print(f"CSV:    {args.data_csv}")
    print(f"TTA:    {args.tta} augmentation(s)")

    # Load and filter DataFrame
    if args.task == "ad_nc":
        df = load_adnc_df(args.data_csv)
        ckpt_prefix = "best_light_fold"
    else:
        df = load_mci_df(args.data_csv)
        ckpt_prefix = "best_mci_fold"

    print(f"Loaded {len(df)} scans / {df['subject_id'].nunique()} subjects")

    labels  = df["label"].to_numpy()
    groups  = df["subject_id"].to_numpy()

    # Pre-merge clinical features once if adnimerge_csv is provided
    df_clinical: pd.DataFrame | None = None
    if args.adnimerge_csv and args.adnimerge_csv.exists():
        print(f"Pre-merging clinical features from {args.adnimerge_csv}")
        df_clinical = merge_clinical(df, args.adnimerge_csv)

    kf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, labels, groups=groups), start=1):
        ckpt_path = args.checkpoint_dir / f"{ckpt_prefix}{fold}.pth"

        if not ckpt_path.exists():
            print(f"[Fold {fold}] Checkpoint not found: {ckpt_path} — skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        norm_stats   = ckpt["norm_stats"]
        best_epoch   = ckpt.get("epoch", "?")
        num_clinical = ckpt.get("num_clinical", 0)
        clinical_cols  = ckpt.get("clinical_cols", []) if num_clinical > 0 else []
        clinical_stats = ckpt.get("clinical_stats", {}) if num_clinical > 0 else {}

        print(f"\n[Fold {fold}] checkpoint epoch={best_epoch}, "
              f"stored_auc={ckpt.get('best_val_auc', float('nan')):.4f}"
              + (f", clinical={len(clinical_cols)} features" if clinical_cols else ""))

        # Use clinical-merged df if the checkpoint was trained with clinical features
        if clinical_cols and df_clinical is not None:
            val_df = df_clinical.iloc[val_idx].reset_index(drop=True)
        else:
            val_df = df.iloc[val_idx].reset_index(drop=True)
            clinical_cols = []
            clinical_stats = {}

        model = build_light_cnn3d(num_classes=2, dropout=0.0,
                                   num_clinical=num_clinical).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_labels, val_probs = run_inference(
            model, val_df, norm_stats, args.batch_size, device, args.task,
            clinical_cols=clinical_cols or None,
            clinical_stats=clinical_stats or None,
            n_tta=args.tta,
        )

        m = compute_full_metrics(val_labels, val_probs)
        m["fold"] = fold
        fold_results.append(m)

        print(f"  AUC={m['auc']:.4f}  "
              f"Sens@0.5={m['sens_05']:.4f}  Spec@0.5={m['spec_05']:.4f}  "
              f"Youden_T={m['youden_thresh']:.4f}  "
              f"Youden_Sens={m['youden_sens']:.4f}  Youden_Spec={m['youden_spec']:.4f}")

    if not fold_results:
        print("No folds evaluated.")
        return

    print_fold_table(fold_results, args.task)

    # Save CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
