"""5-fold Ensemble evaluation with leave-one-fold-out (LOO) strategy.

For each fold k's validation set, averages predictions from the OTHER 4 models
(LOO ensemble — fully honest, no data leakage). Also reports full 5-model ensemble
for comparison.

Usage:
    python evaluate_ensemble.py --task ad_nc
    python evaluate_ensemble.py --task mci_conversion
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

from dataset import ADNINpyDataset, filter_task_df
from models.light_cnn3d import build_light_cnn3d


# ---------------------------------------------------------------------------
# Data helpers  (identical to evaluate_youden.py)
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
# Inference — returns (labels, probs) for a given val_df + model
# ---------------------------------------------------------------------------

def run_inference(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    norm_stats: dict,
    batch_size: int,
    device: torch.device,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    val_ds = ADNINpyDataset(
        val_df, task=task, augment=False,
        mean=float(norm_stats["mean"]), std=float(norm_stats["std"]),
        preload=False,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    return np.asarray(all_labels), np.asarray(all_probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def youden_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) == 2 else float("nan")
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr + (1 - fpr) - 1
    best_idx = int(np.argmax(j_scores))
    thresh = float(thresholds[best_idx])

    for t, name in [(0.5, "05"), (thresh, "y")]:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if name == "05":
            sens_05 = tp / (tp + fn) if (tp + fn) else float("nan")
            spec_05 = tn / (tn + fp) if (tn + fp) else float("nan")
            tp_05, fn_05, tn_05, fp_05 = int(tp), int(fn), int(tn), int(fp)
        else:
            sens_y = tp / (tp + fn) if (tp + fn) else float("nan")
            spec_y = tn / (tn + fp) if (tn + fp) else float("nan")
            tp_y, fn_y, tn_y, fp_y = int(tp), int(fn), int(tn), int(fp)

    return dict(
        auc=auc,
        sens_05=sens_05, spec_05=spec_05,
        tp_05=tp_05, fn_05=fn_05, tn_05=tn_05, fp_05=fp_05,
        youden_thresh=thresh,
        sens_y=sens_y, spec_y=spec_y,
        tp_y=tp_y, fn_y=fn_y, tn_y=tn_y, fp_y=fp_y,
    )


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_comparison(
    single_results: list[dict],
    loo_results: list[dict],
    full5_results: list[dict],
    task: str,
) -> None:
    label_pos = "AD"   if task == "ad_nc" else "pMCI"
    label_neg = "NC"   if task == "ad_nc" else "sMCI"
    task_str  = "AD vs NC" if task == "ad_nc" else "MCI Conversion"

    def avg(results, key):
        return np.mean([r[key] for r in results])
    def std(results, key):
        return np.std([r[key] for r in results], ddof=1)

    print(f"\n{'='*76}")
    print(f"  Ensemble Evaluation — {task_str}   (Pos={label_pos}, Neg={label_neg})")
    print(f"{'='*76}")

    # Per-fold table (LOO)
    print(f"\n  LOO Ensemble per-fold (4 models, none trained on this fold's val data):")
    print(f"  {'Fold':>4}  {'AUC':>6}  {'Sens@0.5':>8}  {'Spec@0.5':>8}  "
          f"{'Youden_T':>8}  {'Youden_Sens':>11}  {'Youden_Spec':>11}")
    print("  " + "-"*72)
    for r in loo_results:
        print(f"  {r['fold']:>4}  {r['auc']:>6.4f}  {r['sens_05']:>8.4f}  {r['spec_05']:>8.4f}  "
              f"{r['youden_thresh']:>8.4f}  {r['sens_y']:>11.4f}  {r['spec_y']:>11.4f}")
    print("  " + "-"*72)
    print(f"  {'Mean':>4}  {avg(loo_results,'auc'):>6.4f}  "
          f"{avg(loo_results,'sens_05'):>8.4f}  {avg(loo_results,'spec_05'):>8.4f}  "
          f"{avg(loo_results,'youden_thresh'):>8.4f}  "
          f"{avg(loo_results,'sens_y'):>11.4f}  {avg(loo_results,'spec_y'):>11.4f}")
    print(f"  {'Std':>4}  {std(loo_results,'auc'):>6.4f}  "
          f"{std(loo_results,'sens_05'):>8.4f}  {std(loo_results,'spec_05'):>8.4f}  "
          f"{std(loo_results,'youden_thresh'):>8.4f}  "
          f"{std(loo_results,'sens_y'):>11.4f}  {std(loo_results,'spec_y'):>11.4f}")

    # Aggregate confusion matrix
    def agg_cm(results, prefix):
        tp = sum(r[f'tp_{prefix}'] for r in results)
        fn = sum(r[f'fn_{prefix}'] for r in results)
        tn = sum(r[f'tn_{prefix}'] for r in results)
        fp = sum(r[f'fp_{prefix}'] for r in results)
        s  = tp/(tp+fn) if (tp+fn) else float("nan")
        sp = tn/(tn+fp) if (tn+fp) else float("nan")
        return tp, fn, tn, fp, s, sp

    print(f"\n  Aggregate Confusion (LOO @0.5):")
    tp,fn,tn,fp,s,sp = agg_cm(loo_results, '05')
    print(f"    TP={tp:>4}  FN={fn:>4}  (sens={s:.3f})   TN={tn:>4}  FP={fp:>4}  (spec={sp:.3f})")
    print(f"  Aggregate Confusion (LOO @Youden):")
    tp,fn,tn,fp,s,sp = agg_cm(loo_results, 'y')
    print(f"    TP={tp:>4}  FN={fn:>4}  (sens={s:.3f})   TN={tn:>4}  FP={fp:>4}  (spec={sp:.3f})")

    # AUC comparison table
    print(f"\n{'='*76}")
    print(f"  AUC COMPARISON")
    print(f"{'='*76}")
    print(f"  {'Method':<32}  {'Mean AUC':>9}  {'Std':>7}  {'Sens@0.5':>9}  {'Spec@0.5':>9}")
    print(f"  {'-'*70}")

    def row(name, results):
        print(f"  {name:<32}  {avg(results,'auc'):>9.4f}  {std(results,'auc'):>7.4f}  "
              f"{avg(results,'sens_05'):>9.4f}  {avg(results,'spec_05'):>9.4f}")

    row("Single model (fold k only)", single_results)
    row("LOO ensemble (4 models)", loo_results)
    row("Full 5-model ensemble", full5_results)

    # AUC improvements
    delta_loo  = avg(loo_results, 'auc')  - avg(single_results, 'auc')
    delta_full = avg(full5_results, 'auc') - avg(single_results, 'auc')
    print(f"\n  LOO  ensemble vs single:   AUC {delta_loo:+.4f}")
    print(f"  Full ensemble vs single:   AUC {delta_full:+.4f}")
    print(f"{'='*76}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["ad_nc", "mci_conversion"], default="ad_nc")
    p.add_argument("--data_csv",       type=Path, default=None)
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--n_splits",  type=int,  default=5)
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--batch_size",type=int,  default=4)
    p.add_argument("--output_csv",type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_csv is None:
        args.data_csv = (
            Path("data/processed_list.csv") if args.task == "ad_nc"
            else Path("data/mci_conversion_list.csv")
        )
    if args.output_csv is None:
        tag = "adnc" if args.task == "ad_nc" else "mci"
        args.output_csv = Path(f"results/ensemble_eval_{tag}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Task: {args.task}")

    # Load DataFrame
    df = load_adnc_df(args.data_csv) if args.task == "ad_nc" else load_mci_df(args.data_csv)
    ckpt_prefix = "best_light_fold" if args.task == "ad_nc" else "best_mci_fold"
    print(f"Loaded {len(df)} scans / {df['subject_id'].nunique()} subjects")

    labels_all = df["label"].to_numpy()
    groups_all = df["subject_id"].to_numpy()
    kf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # ── Step 1: load all 5 checkpoints ──────────────────────────────────────
    print("\nLoading all 5 checkpoints...")
    ckpts: list[dict] = []
    models: list[torch.nn.Module] = []
    for fold in range(1, args.n_splits + 1):
        path = args.checkpoint_dir / f"{ckpt_prefix}{fold}.pth"
        ck = torch.load(path, map_location=device, weights_only=False)
        ckpts.append(ck)
        m = build_light_cnn3d(num_classes=2, dropout=0.0).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  Fold {fold}: epoch={ck.get('epoch','?')}  "
              f"stored_auc={ck.get('best_val_auc', float('nan')):.4f}")

    # ── Step 2: per-fold inference with all models ───────────────────────────
    # fold_val_probs[fold_idx][model_idx] = np.ndarray of probs for that val set
    fold_splits = list(kf.split(df, labels_all, groups=groups_all))

    print("\nRunning inference (all models × all validation sets)...")
    # Collect: for each fold k, run all 5 models on val_k
    all_fold_labels:      list[np.ndarray] = []
    all_fold_probs_mat:   list[np.ndarray] = []  # shape (5, n_val) per fold

    for fold_idx, (_, val_idx) in enumerate(fold_splits):
        fold = fold_idx + 1
        val_df = df.iloc[val_idx].reset_index(drop=True)
        n_val  = len(val_df)
        probs_matrix = np.zeros((args.n_splits, n_val), dtype=np.float32)

        for m_idx, (model, ck) in enumerate(zip(models, ckpts)):
            desc = f"Fold {fold} model {m_idx+1}"
            val_ds = ADNINpyDataset(
                val_df, task=args.task, augment=False,
                mean=float(ck["norm_stats"]["mean"]),
                std=float(ck["norm_stats"]["std"]),
                preload=False,
            )
            loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
            probs_list = []
            with torch.no_grad():
                for images, _ in tqdm(loader, desc=desc, leave=False):
                    logits = model(images.to(device))
                    probs_list.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            probs_matrix[m_idx] = probs_list

        # labels: use fold k's own model (consistent)
        val_ds_labels = ADNINpyDataset(
            val_df, task=args.task, augment=False,
            mean=float(ckpts[fold_idx]["norm_stats"]["mean"]),
            std=float(ckpts[fold_idx]["norm_stats"]["std"]),
            preload=False,
        )
        labels_this_fold = np.asarray(
            [val_ds_labels[i][1].item() for i in range(n_val)]
        )
        all_fold_labels.append(labels_this_fold)
        all_fold_probs_mat.append(probs_matrix)

    # ── Step 3: compute metrics for each strategy ────────────────────────────
    single_results, loo_results, full5_results = [], [], []

    for fold_idx in range(args.n_splits):
        fold = fold_idx + 1
        labels    = all_fold_labels[fold_idx]
        prob_mat  = all_fold_probs_mat[fold_idx]   # shape (5, n_val)

        # Single model (fold k's own model)
        probs_single = prob_mat[fold_idx]

        # LOO: average of the OTHER 4 models
        other_idx = [i for i in range(args.n_splits) if i != fold_idx]
        probs_loo  = prob_mat[other_idx].mean(axis=0)

        # Full 5-model average
        probs_full5 = prob_mat.mean(axis=0)

        m_single = youden_metrics(labels, probs_single)
        m_loo    = youden_metrics(labels, probs_loo)
        m_full5  = youden_metrics(labels, probs_full5)

        m_single["fold"] = fold
        m_loo["fold"]    = fold
        m_full5["fold"]  = fold

        single_results.append(m_single)
        loo_results.append(m_loo)
        full5_results.append(m_full5)

        print(f"Fold {fold}  "
              f"Single AUC={m_single['auc']:.4f}  "
              f"LOO AUC={m_loo['auc']:.4f}  "
              f"Full5 AUC={m_full5['auc']:.4f}")

    # ── Step 4: print and save ───────────────────────────────────────────────
    print_comparison(single_results, loo_results, full5_results, args.task)

    # Save CSVs
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    for name, results in [("single", single_results),
                          ("loo",    loo_results),
                          ("full5",  full5_results)]:
        out = args.output_csv.with_name(
            args.output_csv.stem + f"_{name}.csv"
        )
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
