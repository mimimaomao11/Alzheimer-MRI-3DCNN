"""Grad-CAM visualisation for DenseNet121 (single-channel, 3-D MRI).

Generates axial / coronal / sagittal slice overlays that show *which voxels*
the model attends to, so we can verify it is looking at clinically-relevant
regions (hippocampus, temporal lobe) rather than skull edges or artefacts.

Usage
-----
# AD vs NC model
python visualize_gradcam.py --task ad_nc --checkpoint checkpoints/best_densenet.pth \
    --data_csv data/processed_list.csv --out_dir results/gradcam_adnc

# MCI conversion model
python visualize_gradcam.py --task mci_conversion --checkpoint checkpoints/best_mci.pth \
    --data_csv data/mci_conversion_list.csv --out_dir results/gradcam_mci
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

from dataset import ADNINpyDataset, filter_task_df, resize_volume
from models.densenet_monai import build_densenet121


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Compute Grad-CAM using MONAI's built-in implementation.

    Returns (cam_3d normalised to [0,1], predicted_class, confidence).
    Falls back to input-gradient saliency if MONAI GradCAM fails.
    """
    from monai.visualize import GradCAM as MonaiGradCAM

    model.eval()

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_class = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, pred_class].item())

    if target_class is None:
        target_class = pred_class

    # Pick target layer based on model type
    model_type = type(model).__name__
    if model_type == "LightCNN3D":
        target_layer = "stage2.0"
    elif model_type == "Baseline3DCNN":
        target_layer = "features.2"   # last ConvBlock3D before global pool
    else:
        target_layer = "features.denseblock4"

    # MONAI GradCAM
    try:
        gcam = MonaiGradCAM(nn_module=model, target_layers=target_layer)
        cam_tensor = gcam(x=input_tensor, class_idx=target_class)  # (1, 1, D, H, W)
        cam = cam_tensor.squeeze().cpu().numpy()

        # Check that the map is non-trivial
        if cam.max() <= cam.min():
            raise ValueError("Trivial CAM — falling back to saliency")

    except Exception as exc:
        # Fallback: smoothed input-gradient saliency
        print(f"  [GradCAM fallback to saliency: {exc}]")
        inp = input_tensor.detach().clone().requires_grad_(True)
        out = model(inp)
        model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1.0
        out.backward(gradient=one_hot)
        cam = inp.grad.abs().squeeze().cpu().numpy()
        # Light Gaussian smoothing for readability
        cam = ndimage.gaussian_filter(cam, sigma=2.0)

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam, pred_class, confidence


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _blend_overlay(mri_norm: np.ndarray, cam_slice: np.ndarray, alpha: float = 0.45):
    """Return an RGB image: grey MRI (already [0,1]) + heatmap overlay."""
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)   # (H,W,3) grey
    heatmap = cm.jet(cam_slice)[:, :, :3]          # (H,W,3) colour
    blended = (1 - alpha) * mri_rgb + alpha * heatmap
    return np.clip(blended, 0, 1)


def _extract_brain_centered(
    volume: np.ndarray, cam: np.ndarray, half: int = 44
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vol_out, cam_out) cubes of side 2*half with brain CoM at centre.

    Uses tissue CoM (vol > -1.5, not zero) to handle both old files (where
    skull-stripped background is z-scored to ≈-1.9) and new files (background=0).
    The output is zero-padded so the brain is always centred even when the CoM
    sits near the edge of the source volume.
    """
    # Exclude both background (z-scored to ≈-1.9) and exact zeros (FOV padding)
    tissue = (volume > -1.5) & (volume != 0.0) & np.isfinite(volume)
    if not tissue.any():
        tissue = (volume != 0.0) & np.isfinite(volume)
    if tissue.any():
        coords = np.argwhere(tissue)
        cx, cy, cz = map(int, coords.mean(axis=0))
    else:
        cx, cy, cz = [s // 2 for s in volume.shape]

    D, H, W = volume.shape
    size = 2 * half
    vol_out = np.zeros((size, size, size), dtype=volume.dtype)
    cam_out = np.zeros((size, size, size), dtype=cam.dtype)

    src_lo = [max(c - half, 0) for c in (cx, cy, cz)]
    src_hi = [min(c + half, s) for c, s in zip((cx, cy, cz), (D, H, W))]

    dst_lo = [half - (c - lo) for c, lo in zip((cx, cy, cz), src_lo)]
    dst_hi = [dl + (sh - sl) for dl, sh, sl in zip(dst_lo, src_hi, src_lo)]

    vol_out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        volume[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    cam_out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        cam[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]

    return vol_out, cam_out


def _norm_mri(mri_sl: np.ndarray) -> np.ndarray:
    """Normalise MRI slice for display.

    Computes percentile range from non-zero (brain) voxels only so that
    CSF (very negative z-scores) stays dark-grey rather than being clipped
    to black alongside the background.  Exact-zero background is blacked out.
    """
    brain = mri_sl[mri_sl != 0.0]
    if brain.size < 10:
        brain = mri_sl
    lo, hi = np.percentile(brain, 2), np.percentile(brain, 98)
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((mri_sl - lo) / (hi - lo), 0.0, 1.0)
    # Black out exact-zero background only (brain tissue kept at all intensities)
    out[mri_sl == 0.0] = 0.0
    return out


def _crop2d(mri_sl: np.ndarray, cam_sl: np.ndarray, pad: int = 8):
    """Crop a 2D slice to the 2D tissue bounding box + padding."""
    tissue = (mri_sl > -1.5) & (mri_sl != 0.0) & np.isfinite(mri_sl)
    if not tissue.any():
        tissue = mri_sl != 0.0
    if tissue.any():
        rows = np.any(tissue, axis=1)
        cols = np.any(tissue, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        H, W = mri_sl.shape
        r0, r1 = max(r0 - pad, 0), min(r1 + pad, H - 1)
        c0, c1 = max(c0 - pad, 0), min(c1 + pad, W - 1)
        return mri_sl[r0:r1+1, c0:c1+1], cam_sl[r0:r1+1, c0:c1+1]
    return mri_sl, cam_sl


def plot_subject(
    volume: np.ndarray,
    cam: np.ndarray,
    subject_id: str,
    true_label: str,
    pred_label: str,
    confidence: float,
    out_path: Path,
) -> None:
    """Save a 3×3 grid: raw MRI | Grad-CAM | overlay, for axial/coronal/sagittal.

    Slices through the 3D tissue CoM, then tightly crops each 2D slice to its
    own tissue bounding box so the brain is always centred in the panel.
    """
    tissue3d = (volume > -1.5) & (volume != 0.0) & np.isfinite(volume)
    if not tissue3d.any():
        tissue3d = volume != 0.0
    if tissue3d.any():
        coords = np.argwhere(tissue3d)
        cx, cy, cz = map(int, coords.mean(axis=0))
    else:
        cx, cy, cz = [s // 2 for s in volume.shape]

    raw_slices = {
        "Axial":    (volume[cx, :, :],  cam[cx, :, :]),
        "Coronal":  (volume[:, cy, :],  cam[:, cy, :]),
        "Sagittal": (volume[:, :, cz],  cam[:, :, cz]),
    }
    slices = {k: _crop2d(m, c) for k, (m, c) in raw_slices.items()}

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(
        f"Subject: {subject_id} | True: {true_label} | Pred: {pred_label} "
        f"({confidence:.1%})",
        fontsize=13,
    )
    col_titles = ["MRI", "Grad-CAM", "Overlay"]

    for row, (plane, (mri_sl, cam_sl)) in enumerate(slices.items()):
        for col, title in enumerate(col_titles):
            ax = axes[row, col]
            ax.axis("off")

            if col == 0:
                norm_sl = _norm_mri(mri_sl)
                ax.imshow(norm_sl.T, cmap="gray", origin="lower",
                          interpolation="bilinear")
            elif col == 1:
                # Mask background out of CAM so heatmap only shows brain region
                brain_mask = (mri_sl > -1.0).astype(float)
                cam_masked = cam_sl * brain_mask
                ax.imshow(cam_masked.T, cmap="jet", origin="lower", vmin=0, vmax=1)
            else:
                norm_sl = _norm_mri(mri_sl)
                ax.imshow(_blend_overlay(norm_sl.T, cam_sl.T), origin="lower")

            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(plane, fontsize=10, rotation=90, va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Anatomical-level visualisation (hippocampus, entorhinal, etc.)
# ---------------------------------------------------------------------------

# Fraction of brain's inferior→superior extent for each anatomical level.
# Derived from MNI atlas typical coordinates mapped to ~107-voxel brain height.
# Accuracy: ±8–12 voxels (no MNI registration, so approximate).
ANATOMICAL_LEVELS = [
    (0.25, "Entorhinal Cortex",       "Earliest AD atrophy site"),
    (0.38, "Hippocampus / Amygdala",  "Key MCI→AD predictor"),
    (0.52, "Mid-Temporal",            "Temporal lobe body"),
    (0.68, "Temporoparietal",         "Late-stage AD region"),
]


def _brain_z_range(volume: np.ndarray) -> tuple[int, int]:
    """Return (z_min, z_max) of non-zero brain voxels along axis 0 (axial)."""
    mask = (volume != 0.0) & np.isfinite(volume)
    if not mask.any():
        return 0, volume.shape[0] - 1
    z_coords = np.any(mask, axis=(1, 2))
    indices = np.where(z_coords)[0]
    return int(indices[0]), int(indices[-1])


def plot_anatomical_levels(
    volume: np.ndarray,
    cam: np.ndarray,
    subject_id: str,
    true_label: str,
    pred_label: str,
    confidence: float,
    out_path: Path,
) -> None:
    """Save a figure with one row per anatomical level, 3 cols: MRI | CAM | Overlay.

    Slice positions are estimated from the per-subject brain bounding box so
    the result is robust to the wedge-shaped FOV artefact (brain in corner).
    Labels are approximate (no MNI registration) but land in the right region.
    """
    z_min, z_max = _brain_z_range(volume)
    z_extent = max(z_max - z_min, 1)

    n_levels = len(ANATOMICAL_LEVELS)
    fig, axes = plt.subplots(n_levels, 3, figsize=(12, 3.5 * n_levels))
    fig.suptitle(
        f"Anatomical Grad-CAM  |  Subject: {subject_id}\n"
        f"True: {true_label}  |  Pred: {pred_label} ({confidence:.1%})",
        fontsize=12,
    )
    col_titles = ["MRI", "Grad-CAM", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row, (frac, region_name, region_note) in enumerate(ANATOMICAL_LEVELS):
        z_idx = int(z_min + frac * z_extent)
        z_idx = np.clip(z_idx, 0, volume.shape[0] - 1)

        mri_sl = volume[z_idx, :, :]
        cam_sl = cam[z_idx, :, :]
        mri_sl, cam_sl = _crop2d(mri_sl, cam_sl)
        norm_sl = _norm_mri(mri_sl)

        axes[row, 0].imshow(norm_sl.T, cmap="gray", origin="lower",
                            interpolation="bilinear")
        axes[row, 0].axis("off")

        brain_mask = (mri_sl > -1.0).astype(float)
        axes[row, 1].imshow((cam_sl * brain_mask).T, cmap="jet",
                            origin="lower", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(_blend_overlay(norm_sl.T, cam_sl.T), origin="lower")
        axes[row, 2].axis("off")

        axes[row, 0].set_ylabel(
            f"{region_name}\n({region_note})\nz≈{z_idx}",
            fontsize=9, rotation=0, ha="right", va="center", labelpad=4,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved (anatomical): {out_path}")


# ---------------------------------------------------------------------------
# pMCI vs sMCI group comparison figure
# ---------------------------------------------------------------------------

def plot_group_comparison(
    gradcam_dir: Path,
    out_path: Path,
) -> None:
    """Side-by-side anatomical CAM comparison: pMCI columns vs sMCI columns.

    Reads all *_anatomical.png files from gradcam_dir, splits by label in
    filename, and builds one composite figure:
      rows  = anatomical levels (Entorhinal, Hippocampus, Mid-Temporal, Temporoparietal)
      cols  = pMCI subjects | sMCI subjects (separated by a thin divider)

    Each cell shows the CAM overlay slice at that anatomical level.
    If the pre-rendered PNGs exist the function simply re-assembles them; no
    model inference needed.
    """
    import matplotlib.image as mpimg

    pmci_files = sorted(gradcam_dir.glob("*_pMCI_anatomical.png"))
    smci_files = sorted(gradcam_dir.glob("*_sMCI_anatomical.png"))

    if not pmci_files and not smci_files:
        print(f"  [SKIP] No *_anatomical.png files found in {gradcam_dir}")
        return

    n_levels = len(ANATOMICAL_LEVELS)
    n_pmci   = len(pmci_files)
    n_smci   = len(smci_files)
    n_cols   = n_pmci + n_smci + 1   # +1 for divider column

    fig, axes = plt.subplots(
        n_levels, n_cols,
        figsize=(3.5 * n_cols, 3.8 * n_levels),
        gridspec_kw={"width_ratios": [1] * n_pmci + [0.05] + [1] * n_smci},
    )
    if n_levels == 1:
        axes = axes[np.newaxis, :]

    level_labels = [lvl[1] for lvl in ANATOMICAL_LEVELS]

    for row in range(n_levels):
        col = 0
        for fpath in pmci_files:
            img = mpimg.imread(str(fpath))
            # Each anatomical PNG has n_levels rows × 3 cols — extract the overlay col (col 2)
            h, w = img.shape[:2]
            row_h  = h // n_levels
            crop   = img[row * row_h:(row + 1) * row_h, (w * 2) // 3:, :]
            axes[row, col].imshow(crop)
            axes[row, col].axis("off")
            if row == 0:
                sid = fpath.stem.replace("_pMCI_anatomical", "")
                axes[row, col].set_title(f"pMCI\n{sid}", fontsize=7, color="#c0392b")
            col += 1

        # Divider
        axes[row, col].set_facecolor("#cccccc")
        axes[row, col].axis("off")
        col += 1

        for fpath in smci_files:
            img = mpimg.imread(str(fpath))
            h, w = img.shape[:2]
            row_h  = h // n_levels
            crop   = img[row * row_h:(row + 1) * row_h, (w * 2) // 3:, :]
            axes[row, col].imshow(crop)
            axes[row, col].axis("off")
            if row == 0:
                sid = fpath.stem.replace("_sMCI_anatomical", "")
                axes[row, col].set_title(f"sMCI\n{sid}", fontsize=7, color="#2980b9")
            col += 1

        axes[row, 0].set_ylabel(level_labels[row], fontsize=9,
                                rotation=0, ha="right", va="center", labelpad=4)

    fig.suptitle("Grad-CAM Anatomical Comparison: pMCI vs sMCI\n"
                 "(Overlay col only — red=hot activation, blue=low)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved (comparison): {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LABEL_NAMES = {
    "ad_nc":        {0: "NC", 1: "AD"},
    "mci_conversion": {0: "sMCI", 1: "pMCI"},
}


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    in_ch = ckpt.get("in_channels", 1)

    # Auto-detect model from state dict keys when model_name not stored
    sd_keys = set(ckpt["model_state_dict"].keys())
    if ckpt.get("model_name"):
        model_name = ckpt["model_name"]
    elif any("stage3" in k for k in sd_keys):
        model_name = "LightCNN3D"
    elif any("features.2.block" in k for k in sd_keys):
        model_name = "BaselineCNN"
    else:
        model_name = "DenseNet121"

    if model_name.startswith("LightCNN3D"):
        from models.light_cnn3d import build_light_cnn3d
        model = build_light_cnn3d(num_classes=2).to(device)
    elif model_name == "BaselineCNN":
        from models.baseline_cnn import Baseline3DCNN
        model = Baseline3DCNN(num_classes=2).to(device)
    elif in_ch == 3:
        from models.densenet_monai import build_densenet121_3ch
        model = build_densenet121_3ch(num_classes=2).to(device)
    else:
        model = build_densenet121(num_classes=2).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    norm_stats = ckpt.get("norm_stats", {})
    print(f"  Loaded model: {model_name} (fold={ckpt.get('fold')}, "
          f"epoch={ckpt.get('epoch')}, "
          f"AUC={ckpt.get('val_metrics', {}).get('auc', '?'):.3f})")
    return model, norm_stats, model_name


def select_subjects(df: pd.DataFrame, task: str, n_per_class: int = 3) -> pd.DataFrame:
    """Pick up to n_per_class subjects per class, deduplicated by subject_id."""
    label_map = {"ad_nc": {"NC": 0, "AD": 1}, "mci_conversion": {"sMCI": 0, "pMCI": 1}}[task]
    group_col = "conversion_group" if task == "mci_conversion" else "group"
    selected = []
    for grp in label_map:
        subset = df[df[group_col] == grp].drop_duplicates("subject_id")
        selected.append(subset.head(n_per_class))
    return pd.concat(selected, ignore_index=True)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, norm_stats, model_name = load_checkpoint(args.checkpoint, device)
    model.eval()

    # Load data
    df_all = pd.read_csv(args.data_csv)
    group_col = "conversion_group" if args.task == "mci_conversion" else "group"

    subjects = select_subjects(df_all, args.task, n_per_class=args.n_per_class)
    names = LABEL_NAMES[args.task]

    mean = float(norm_stats.get("mean", 0.0))
    std = float(norm_stats.get("std", 1.0))
    std = max(std, 1e-6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Grad-CAM for {len(subjects)} subjects → {args.out_dir}")

    for _, row in subjects.iterrows():
        sid = row["subject_id"]
        true_grp = row[group_col]
        true_label_name = true_grp

        # Load volume
        vol = np.load(row["file_path"]).astype(np.float32)
        vol = resize_volume(vol, (128, 128, 128))
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise
        vol_norm = (vol - mean) / std
        tensor = torch.from_numpy(vol_norm[None, None, ...]).to(device)  # (1,1,D,H,W)

        # Compute Grad-CAM
        cam, pred_class, conf = compute_gradcam(model, tensor)
        pred_label_name = names.get(pred_class, str(pred_class))

        # Save standard CoM figure
        fname = f"{sid}_{true_label_name}.png"
        plot_subject(vol, cam, sid, true_label_name, pred_label_name, conf,
                     args.out_dir / fname)

        # Save anatomical-level figure
        if args.anatomical:
            anat_path = args.out_dir / f"{sid}_{true_label_name}_anatomical.png"
            plot_anatomical_levels(vol, cam, sid, true_label_name,
                                   pred_label_name, conf, anat_path)

    # Group comparison figure (pMCI vs sMCI)
    if args.compare and args.task == "mci_conversion" and args.anatomical:
        plot_group_comparison(
            args.out_dir,
            args.out_dir / "comparison_pmci_vs_smci.png",
        )

    print(f"\nDone. Open '{args.out_dir}' to inspect the Grad-CAM overlays.")
    print("What to look for:")
    print("  [OK]  heatmap concentrated on medial temporal lobe / hippocampus")
    print("  [CHK] heatmap on ventricles (possible, but indirect signal)")
    print("  [BAD] heatmap on skull edges, outside brain, or uniform noise")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM for 3-D DenseNet121")
    p.add_argument("--task", choices=["ad_nc", "mci_conversion"], default="ad_nc")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_light.pth"))
    p.add_argument("--data_csv", type=Path, default=Path("data/processed_list.csv"))
    p.add_argument("--out_dir", type=Path, default=Path("results/gradcam_light"))
    p.add_argument("--n_per_class", type=int, default=3,
                   help="Number of subjects per class to visualise")
    p.add_argument("--anatomical", action="store_true", default=False,
                   help="Also save per-subject anatomical-level figures "
                        "(entorhinal / hippocampus / temporoparietal slices)")
    p.add_argument("--compare", action="store_true", default=False,
                   help="After generating anatomical figures, build a pMCI vs sMCI "
                        "side-by-side comparison figure (mci_conversion task only)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
