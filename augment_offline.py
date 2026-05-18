"""Offline augmentation: generate N augmented copies of each training scan.

Writes augmented .npy files to data/processed/<GROUP>/aug/ and appends
rows to a new CSV with the same subject_id so StratifiedGroupKFold keeps
each subject's augmented copies in the same fold as the original.

Usage:
    python augment_offline.py --n_aug 5 --groups AD NC
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


def augment_volume(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random spatial + intensity augmentation to a z-scored MRI volume."""
    vol = vol.copy()

    # Random flip on each axis
    for axis in range(3):
        if rng.random() < 0.5:
            vol = np.flip(vol, axis=axis)

    # Random rotation (±20° — slightly more aggressive than online ±15°)
    angle = float(rng.uniform(-20.0, 20.0))
    axes_pair = [(0, 1), (0, 2), (1, 2)][rng.integers(0, 3)]
    vol = ndimage.rotate(vol, angle, axes=axes_pair, reshape=False,
                         order=1, mode="nearest")

    # Random intensity scale ±15%
    scale = float(rng.uniform(0.85, 1.15))
    # Only scale non-zero (brain) voxels to preserve background
    mask = vol != 0.0
    vol[mask] = vol[mask] * scale

    # Gaussian noise σ=0.03
    noise = rng.normal(0.0, 0.03, vol.shape).astype(np.float32)
    vol[mask] = vol[mask] + noise[mask]

    # Random gamma on brain voxels (simulate MRI contrast variation)
    # Map brain to [0,1], apply gamma, map back
    brain = vol[mask]
    if brain.size == 0:
        return np.ascontiguousarray(vol, dtype=np.float32)
    b_min, b_max = brain.min(), brain.max()
    if b_max > b_min:
        brain_norm = (brain - b_min) / (b_max - b_min)
        gamma = float(rng.uniform(0.8, 1.2))
        brain_norm = np.power(np.clip(brain_norm, 1e-6, 1.0), gamma)
        vol[mask] = brain_norm * (b_max - b_min) + b_min

    return np.ascontiguousarray(vol, dtype=np.float32)


def run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv)
    groups = args.groups
    df_target = df[df["group"].isin(groups)].reset_index(drop=True)
    print(f"Source scans: {len(df_target)} ({df_target['group'].value_counts().to_dict()})")
    print(f"Generating {args.n_aug} augmented copies per scan → "
          f"{len(df_target) * args.n_aug} new files")

    rng = np.random.default_rng(args.seed)
    new_rows = []

    for _, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Augmenting"):
        src_path = Path(row["file_path"])
        if not src_path.exists():
            print(f"  [SKIP] Not found: {src_path}")
            continue

        vol = np.load(src_path).astype(np.float32)
        aug_dir = src_path.parent / "aug"
        aug_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.n_aug):
            aug_vol = augment_volume(vol, rng)
            aug_name = src_path.stem + f"_aug{i:02d}.npy"
            aug_path = aug_dir / aug_name
            np.save(aug_path, aug_vol)

            new_row = row.to_dict()
            new_row["file_path"] = str(aug_path)
            new_row["source_path"] = str(src_path)
            new_rows.append(new_row)

    aug_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, aug_df], ignore_index=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path, index=False)

    print(f"\nDone. Original: {len(df)}, Augmented added: {len(aug_df)}")
    print(f"Combined CSV: {out_path} ({len(combined_df)} rows)")
    print(combined_df["group"].value_counts().reindex(["NC", "AD", "MCI"],
                                                       fill_value=0).to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline 3D MRI augmentation")
    p.add_argument("--input_csv",  type=Path, default=Path("data/processed_list.csv"))
    p.add_argument("--output_csv", type=Path, default=Path("data/augmented_list.csv"))
    p.add_argument("--n_aug",      type=int,  default=5,
                   help="Number of augmented copies per scan")
    p.add_argument("--groups",     nargs="+", default=["AD", "NC"],
                   help="Which groups to augment")
    p.add_argument("--seed",       type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
