from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm import tqdm


LABEL_MAP = {"NC": 0, "AD": 1}
MCI_LABEL_MAP = {"sMCI": 0, "pMCI": 1}

# Clinical features pulled from ADNIMERGE baseline rows.
CLINICAL_COLS: list[str] = ["AGE", "PTGENDER", "APOE4", "MMSE_bl", "CDRSB_bl"]


def merge_clinical(df: pd.DataFrame, adnimerge_path: Path) -> pd.DataFrame:
    """Left-join ADNIMERGE baseline clinical features into a scan-level DataFrame.

    Matches on subject_id (scan df) == PTID (ADNIMERGE). PTGENDER is encoded
    as Male=1 / Female=0. Missing values are filled with per-column medians
    computed from the merged result.
    """
    adni = pd.read_csv(adnimerge_path, low_memory=False)

    bl = adni[adni["VISCODE"] == "bl"].drop_duplicates("PTID", keep="first")
    if bl.empty:
        bl = adni.drop_duplicates("PTID", keep="first")

    available = [c for c in CLINICAL_COLS if c in bl.columns]
    bl = bl[["PTID"] + available].copy()

    if "PTGENDER" in bl.columns:
        bl["PTGENDER"] = (bl["PTGENDER"].str.strip().str.lower() == "male").astype(float)

    merged = df.merge(bl, left_on="subject_id", right_on="PTID", how="left")
    if "PTID" in merged.columns:
        merged = merged.drop(columns=["PTID"])

    n_missing = merged[available].isna().any(axis=1).sum()
    if n_missing:
        print(f"  [clinical] {n_missing}/{len(merged)} scans missing → filling with median")
        for col in available:
            merged[col] = merged[col].fillna(merged[col].median())

    print(f"  [clinical] Merged {len(available)} features: {available}")
    return merged


def compute_clinical_stats(train_df: pd.DataFrame,
                            cols: list[str]) -> dict[str, tuple[float, float]]:
    """Compute per-feature (mean, std) from the training fold only."""
    stats: dict[str, tuple[float, float]] = {}
    for col in cols:
        m = float(train_df[col].mean())
        s = float(train_df[col].std(ddof=1))
        stats[col] = (m, max(s, 1e-6))
    return stats


def filter_task_df(df: pd.DataFrame, task: str = "ad_nc") -> pd.DataFrame:
    if task == "ad_nc":
        label_map = LABEL_MAP
    elif task == "mci_conversion":
        label_map = MCI_LABEL_MAP
    else:
        raise ValueError(f"Unsupported task: {task}")
    df = df[df["group"].isin(label_map)].copy()
    df["label"] = df["group"].map(label_map).astype(int)
    return df.reset_index(drop=True)


def load_split_csv(csv_path: str | Path, task: str = "ad_nc") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return filter_task_df(df, task=task)


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape == target_shape:
        return volume
    zoom = [t / s for t, s in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, zoom=zoom, order=1).astype(np.float32)


def compute_norm_stats(
    csv_path: str | Path | pd.DataFrame,
    output_path: str | Path = "data/norm_stats.json",
    target_shape: Tuple[int, int, int] = (96, 96, 96),
    task: str = "ad_nc",
) -> dict:
    if isinstance(csv_path, pd.DataFrame):
        df = filter_task_df(csv_path, task=task)
    else:
        df = load_split_csv(csv_path, task=task)
    if df.empty:
        raise ValueError(f"No NC/AD rows found in {csv_path}")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Computing train mean/std"):
        arr = np.load(row.file_path).astype(np.float32)
        arr = resize_volume(arr, target_shape)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        total_sum += float(arr.sum(dtype=np.float64))
        total_sq_sum += float(np.square(arr, dtype=np.float64).sum())
        total_count += int(arr.size)

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 1e-8)
    stats = {"mean": mean, "std": float(np.sqrt(variance)), "num_samples": int(len(df))}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_norm_stats(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing norm stats file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class ADNINpyDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame,
        task: str = "ad_nc",
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        augment: bool = False,
        norm_stats_path: str | Path = "data/norm_stats.json",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        preload: bool = False,
        clinical_cols: Optional[list[str]] = None,
        clinical_stats: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        if isinstance(csv_path, pd.DataFrame):
            self.df = filter_task_df(csv_path, task=task)
        else:
            self.df = load_split_csv(csv_path, task=task)
        self.target_shape = target_shape
        self.augment = augment
        self.preload = preload
        self.clinical_cols: list[str] = clinical_cols or []
        self.clinical_stats: dict[str, tuple[float, float]] = clinical_stats or {}

        if mean is None or std is None:
            stats = load_norm_stats(norm_stats_path)
            mean = float(stats["mean"])
            std = float(stats["std"])
        self.mean = float(mean)
        self.std = max(float(std), 1e-6)
        self._cache: list[np.ndarray] | None = None
        if self.preload:
            self._cache = []
            for row in tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Preloading volumes"):
                volume = np.load(row.file_path).astype(np.float32)
                volume = resize_volume(volume, self.target_shape)
                volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
                self._cache.append(volume)

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, volume: np.ndarray) -> np.ndarray:
        for axis in range(3):
            if np.random.rand() < 0.5:
                volume = np.flip(volume, axis=axis)

        angle = float(np.random.uniform(-15.0, 15.0))
        axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
        volume = ndimage.rotate(volume, angle=angle, axes=axes, reshape=False, order=1, mode="nearest")

        scale = float(np.random.uniform(0.9, 1.1))
        volume = volume * scale
        noise = np.random.normal(loc=0.0, scale=0.02, size=volume.shape).astype(np.float32)
        volume = volume + noise
        return volume.astype(np.float32)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        if self._cache is not None:
            volume = self._cache[index].copy()
        else:
            volume = np.load(row["file_path"]).astype(np.float32)
            volume = resize_volume(volume, self.target_shape)
            volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            volume = self._augment(volume)

        volume = (volume - self.mean) / self.std
        volume = np.ascontiguousarray(volume[None, ...], dtype=np.float32)
        label = int(row["label"])
        img_tensor = torch.from_numpy(volume)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.clinical_cols:
            clin = np.zeros(len(self.clinical_cols), dtype=np.float32)
            for i, col in enumerate(self.clinical_cols):
                val = float(row[col]) if pd.notna(row[col]) else 0.0
                if col in self.clinical_stats:
                    m, s = self.clinical_stats[col]
                    val = (val - m) / s
                clin[i] = val
            return img_tensor, torch.from_numpy(clin), label_tensor

        return img_tensor, label_tensor


def _tissue_maps(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive approximate GM/WM probability maps from a preprocessed T1.

    The .npy files are skull-stripped T1 volumes z-scored with brain-only voxel
    statistics during preprocessing.  Empirical intensity ranges:
      out-of-FOV (NaN→0): 0.0
      skull-stripped background: ≈ −2.1
      CSF: ≈ −1.0 to −0.3
      GM:  ≈ −0.3 to  0.5
      WM:  ≈  0.4 to  1.8

    Brain mask excludes exact zeros (out-of-FOV) and the −2.1 background cluster.
    """
    brain_mask = (raw != 0.0) & (raw > -1.7)

    # WM: high-intensity region (sigmoid centred at 0.45)
    wm = 1.0 / (1.0 + np.exp(-5.0 * (raw - 0.45)))
    wm = (wm * brain_mask).astype(np.float32)

    # GM: medium-intensity region (Gaussian bell centred at 0.05, σ = 0.5)
    gm = np.exp(-0.5 * ((raw - 0.05) / 0.5) ** 2)
    gm = (gm * brain_mask).astype(np.float32)

    return gm, wm


class ADNIMultiChannelDataset(Dataset):
    """3-channel dataset: z-scored T1 + approximate GM map + approximate WM map.

    GM and WM probability maps are derived from T1 intensity thresholding
    (no external segmentation tool required).  They are computed on the raw
    (pre-normalisation) .npy values so that the tissue-map thresholds remain
    stable across folds.
    """

    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame,
        task: str = "mci_conversion",
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        augment: bool = False,
        norm_stats_path: str | Path = "data/norm_stats.json",
        mean: Optional[float] = None,
        std: Optional[float] = None,
        preload: bool = False,
    ) -> None:
        if isinstance(csv_path, pd.DataFrame):
            self.df = filter_task_df(csv_path, task=task)
        else:
            self.df = load_split_csv(csv_path, task=task)
        self.target_shape = target_shape
        self.augment = augment
        self.preload = preload

        if mean is None or std is None:
            stats = load_norm_stats(norm_stats_path)
            mean = float(stats["mean"])
            std = float(stats["std"])
        self.mean = float(mean)
        self.std = max(float(std), 1e-6)

        self._cache: list[np.ndarray] | None = None
        if self.preload:
            self._cache = []
            for row in tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Preloading volumes"):
                volume = np.load(row.file_path).astype(np.float32)
                volume = resize_volume(volume, self.target_shape)
                volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
                self._cache.append(volume)

    def __len__(self) -> int:
        return len(self.df)

    def _augment_spatial(self, volume: np.ndarray) -> np.ndarray:
        """Spatial-only augmentation (flip + rotate) applied before tissue map derivation."""
        for axis in range(3):
            if np.random.rand() < 0.5:
                volume = np.flip(volume, axis=axis)
        angle = float(np.random.uniform(-15.0, 15.0))
        axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
        volume = ndimage.rotate(volume, angle=angle, axes=axes, reshape=False, order=1, mode="nearest")
        return volume.astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        if self._cache is not None:
            volume = self._cache[index].copy()
        else:
            volume = np.load(row["file_path"]).astype(np.float32)
            volume = resize_volume(volume, self.target_shape)
            volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            # Spatial transform first (preserves intensity relationships for tissue maps)
            volume = self._augment_spatial(volume)
            # Intensity augmentation for T1 only (scale + noise)
            scale = float(np.random.uniform(0.9, 1.1))
            volume = volume * scale
            volume = volume + np.random.normal(0.0, 0.02, volume.shape).astype(np.float32)

        # Derive tissue channels from raw (pre-normalisation) T1 values
        gm_map, wm_map = _tissue_maps(volume)

        # Normalise T1 channel
        t1_norm = ((volume - self.mean) / self.std).astype(np.float32)

        # Stack into (3, D, H, W)
        volume_3ch = np.ascontiguousarray(np.stack([t1_norm, gm_map, wm_map], axis=0))
        label = int(row["label"])
        return torch.from_numpy(volume_3ch), torch.tensor(label, dtype=torch.long)
