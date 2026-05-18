#!/usr/bin/env python
"""ADNI MRI preprocessing pipeline.

Supports NIfTI files directly and ADNI-style DICOM series via SimpleITK.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage
    from tqdm import tqdm
except ModuleNotFoundError:
    nib = None  # type: ignore
    np = None  # type: ignore
    sitk = None  # type: ignore
    ndimage = None  # type: ignore
    tqdm = None  # type: ignore


VALID_GROUPS = {"NC", "MCI", "AD"}
SUBJECT_RE = re.compile(r"\d{3}_S_\d{4,5}", re.IGNORECASE)


def require_dependencies() -> None:
    missing = []
    for module_name, module_value in {
        "nibabel": nib,
        "numpy": np,
        "SimpleITK": sitk,
        "scipy": ndimage,
        "tqdm": tqdm,
    }.items():
        if module_value is None:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {joined}\n"
            "Create and activate the project virtual environment, then run:\n"
            "  python -m pip install -r requirements.txt"
        )


@dataclass
class SubjectImage:
    subject_id: str
    group: str
    file_path: str
    input_type: str


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=output_dir / "preprocessing_log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def normalize_subject_id(value: str) -> str:
    value = str(value).strip()
    match = SUBJECT_RE.search(value)
    return match.group(0) if match else value


def extract_subject_id(path: Path) -> str:
    for part in path.parts:
        match = SUBJECT_RE.fullmatch(part)
        if match:
            return match.group(0)
    match = SUBJECT_RE.search(str(path))
    return match.group(0) if match else path.stem


def normalize_group(value: str) -> str:
    text = str(value).strip().upper()
    aliases = {
        "CN": "NC",
        "NORMAL": "NC",
        "NORMAL CONTROL": "NC",
        "CONTROL": "NC",
        "NL": "NC",
        "NCI": "NC",
        "MCI": "MCI",
        "EMCI": "MCI",
        "LMCI": "MCI",
        "AD": "AD",
        "DEMENTIA": "AD",
        "ALZHEIMER'S DISEASE": "AD",
        "ALZHEIMERS DISEASE": "AD",
    }
    return aliases.get(text, text if text in VALID_GROUPS else "Unknown")


def read_metadata(metadata_csv: Optional[Path]) -> Dict[str, str]:
    if not metadata_csv:
        return {}
    mapping: Dict[str, str] = {}
    subject_cols = [
        "subject_id",
        "subject",
        "ptid",
        "participant_id",
        "image data id",
        "imageuid",
    ]
    group_cols = ["group", "diagnosis", "dx", "dx_bl", "research group", "label"]
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping
        lower_to_name = {name.lower().strip(): name for name in reader.fieldnames}
        subject_col = next((lower_to_name[c] for c in subject_cols if c in lower_to_name), None)
        group_col = next((lower_to_name[c] for c in group_cols if c in lower_to_name), None)
        if subject_col is None or group_col is None:
            logging.warning("Metadata CSV lacks recognizable subject/group columns: %s", metadata_csv)
            return mapping
        for row in reader:
            sid = normalize_subject_id(row.get(subject_col, ""))
            group = normalize_group(row.get(group_col, ""))
            if sid and group in VALID_GROUPS:
                mapping[sid] = group
    return mapping


def infer_group_from_path(path: Path) -> str:
    for part in path.parts:
        group = normalize_group(part)
        if group in VALID_GROUPS:
            return group
    return "Unknown"


def find_nifti_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.nii", "*.nii.gz"):
        files.extend(input_dir.rglob(pattern))
    return sorted(files)


def find_dicom_series(input_dir: Path) -> List[Path]:
    dicom_dirs = set()
    for ext in ("*.dcm", "*.DCM"):
        for p in input_dir.rglob(ext):
            dicom_dirs.add(p.parent)
    return sorted(dicom_dirs)


def parse_include_terms(text: str) -> List[str]:
    terms = [v.strip().lower() for v in text.split(",") if v.strip()]
    return terms


def scan_dataset(input_dir: Path, metadata: Dict[str, str], series_include: Sequence[str]) -> List[SubjectImage]:
    niftis = find_nifti_files(input_dir)
    records: List[SubjectImage] = []
    if niftis:
        for p in tqdm(niftis, desc="Scanning NIfTI"):
            sid = extract_subject_id(p)
            group = metadata.get(sid, infer_group_from_path(p))
            records.append(SubjectImage(sid, group, str(p), "nifti"))
        return records

    dicom_dirs = find_dicom_series(input_dir)
    for p in tqdm(dicom_dirs, desc="Scanning DICOM series"):
        try:
            path_text = str(p).lower()
            if series_include and not any(term in path_text for term in series_include):
                continue
            sid = extract_subject_id(p)
            group = metadata.get(sid, infer_group_from_path(p))
            records.append(SubjectImage(sid, group, f"{p}::AUTO", "dicom"))
        except Exception as exc:
            logging.exception("Failed to scan DICOM directory %s: %s", p, exc)
    return records


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dicom_to_nifti(dicom_ref: str) -> nib.Nifti1Image:
    dir_text, series_id = dicom_ref.split("::", 1)
    if series_id == "AUTO":
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir_text)
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dir_text}")
        series_id = series_ids[0]
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dir_text, series_id)
    if not names:
        raise ValueError(f"No DICOM files found for series {dicom_ref}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.transpose(arr, (2, 1, 0))
    spacing = img.GetSpacing()
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return nib.Nifti1Image(arr, affine)


def load_image(record: SubjectImage) -> nib.Nifti1Image:
    if record.input_type == "dicom":
        return dicom_to_nifti(record.file_path)
    return nib.load(record.file_path)


def qc_image(img: nib.Nifti1Image) -> Tuple[dict, bool, str]:
    shape = tuple(int(v) for v in img.shape[:3])
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    orientation = "".join(nib.aff2axcodes(img.affine))
    issues: List[str] = []
    if len(shape) != 3 or any(v < 32 for v in shape):
        issues.append("shape_abnormal")
    if any((not np.isfinite(v)) or v <= 0 or v > 5 for v in zooms):
        issues.append("spacing_abnormal")
    return (
        {"shape": "x".join(map(str, shape)), "spacing": "x".join(f"{v:.4g}" for v in zooms), "orientation": orientation},
        not issues,
        ";".join(issues),
    )


def save_temp_nifti(img: nib.Nifti1Image, path: Path) -> None:
    nib.save(img, str(path))


def skull_strip_fsl(img: nib.Nifti1Image, bet_bin: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.nii.gz"
        out = Path(tmp) / "brain.nii.gz"
        save_temp_nifti(img, src)
        subprocess.run([bet_bin, str(src), str(out), "-m", "-R"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        brain = nib.load(str(out))
        mask = nib.load(str(Path(tmp) / "brain_mask.nii.gz")).get_fdata() > 0
        return brain, mask


def skull_strip_antspynet(img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, np.ndarray]:
    import antspynet  # type: ignore
    import ants  # type: ignore

    data = img.get_fdata().astype(np.float32)
    ants_img = ants.from_numpy(data)
    prob = antspynet.brain_extraction(ants_img, modality="t1")
    mask = prob.numpy() > 0.5
    stripped = data * mask
    return nib.Nifti1Image(stripped.astype(np.float32), img.affine, img.header), mask


def skull_strip_nilearn(img: nib.Nifti1Image) -> Tuple[nib.Nifti1Image, np.ndarray]:
    from nilearn.masking import compute_brain_mask

    mask_img = compute_brain_mask(img)
    mask = mask_img.get_fdata() > 0
    data = img.get_fdata().astype(np.float32)
    if not np.any(mask):
        mask = data > np.percentile(data[np.isfinite(data)], 10)
    return nib.Nifti1Image((data * mask).astype(np.float32), img.affine, img.header), mask


def n4_bias_correct(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Apply N4 ITK bias field correction using SimpleITK.

    Applied to the raw image before skull stripping.  Uses an Otsu-threshold
    mask so the algorithm focuses on tissue rather than the air background.
    Axes: nibabel stores (nx, ny, nz); SimpleITK GetImageFromArray expects the
    array in (nz, ny, nx) order — hence the transposes below.
    """
    data = img.get_fdata().astype(np.float32)

    # nibabel (nx,ny,nz) → SimpleITK (nz,ny,nx)
    sitk_img = sitk.GetImageFromArray(data.T)
    sitk_img.SetSpacing([float(s) for s in img.header.get_zooms()[:3]])
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

    # Otsu mask: background=0, tissue=1
    mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # 4 multi-resolution levels × 50 iterations — standard ANTs defaults
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrected = corrector.Execute(sitk_img, mask)

    # SimpleITK (nz,ny,nx) → nibabel (nx,ny,nz)
    corrected_data = sitk.GetArrayFromImage(corrected).T.astype(np.float32)
    return nib.Nifti1Image(corrected_data, img.affine, img.header)


def skull_strip(img: nib.Nifti1Image, method: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    if method == "fsl" or (method == "auto" and shutil.which("bet")):
        return skull_strip_fsl(img, shutil.which("bet") or "bet")
    if method == "antspynet":
        return skull_strip_antspynet(img)
    if method == "auto":
        try:
            return skull_strip_antspynet(img)
        except Exception as exc:
            logging.warning("ANTsPyNet unavailable or failed, using nilearn mask: %s", exc)
    return skull_strip_nilearn(img)


def intensity_normalize(data: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(data)
    mask = mask & finite
    if not np.any(mask):
        mask = finite
    values = data[mask]
    if mode == "minmax":
        lo, hi = np.percentile(values, [1, 99])
        if hi <= lo:
            return np.zeros_like(data, dtype=np.float32)
        out = (data - lo) / (hi - lo)
        return np.clip(out, 0, 1).astype(np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    out = (data - mean) / std
    out[~finite] = 0
    return out.astype(np.float32)


def spatial_normalize(img: nib.Nifti1Image, mni_resolution: int) -> nib.Nifti1Image:
    from nilearn import datasets, image

    template = datasets.load_mni152_template(resolution=mni_resolution)
    return image.resample_to_img(img, template, interpolation="continuous", force_resample=True, copy_header=True)


def resize_array(data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, data.shape[:3])]
    resized = ndimage.zoom(data, zoom=factors, order=1)
    return resized.astype(np.float32)


def crop_to_brain_bbox(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_vox: int = 4,
) -> np.ndarray:
    """Crop to the brain bounding box, then resize to target_shape.

    Requires non-brain voxels to be exactly 0 before this function is called
    (achieved by setting norm_data[~mask] = 0 after z-scoring in _process_one).
    After MNI registration, out-of-FOV voxels are NaN.  The mask
    (finite & != 0) therefore selects only real brain voxels.
    """
    brain_mask = np.isfinite(data) & (data != 0.0)
    if not brain_mask.any():
        return np.zeros(target_shape, dtype=np.float32)

    coords = np.where(brain_mask)
    lo = [max(int(c.min()) - pad_vox, 0) for c in coords]
    hi = [min(int(c.max()) + pad_vox + 1, s) for c, s in zip(coords, data.shape)]

    cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return resize_array(cropped, target_shape)


def _center_by_shift(arr: np.ndarray, tissue_threshold: float = 0.1) -> np.ndarray:
    """Shift the volume so the brain centre of mass lands at the volume centre.

    Works regardless of where the brain ended up after MNI registration or
    bounding-box cropping.  Brain voxels are identified by |z-score| > threshold
    (excludes the zero-valued skull-stripped background and interpolation noise).
    """
    brain = (np.abs(arr) > tissue_threshold) & np.isfinite(arr)
    if not brain.any():
        return arr
    coords = np.argwhere(brain)
    cx, cy, cz = coords.mean(axis=0)
    target = tuple(s // 2 for s in arr.shape)   # (48, 48, 48) for 96³
    delta = (target[0] - cx, target[1] - cy, target[2] - cz)
    shifted = ndimage.shift(arr, delta, order=1, mode="constant", cval=0.0)
    return shifted.astype(np.float32)


def safe_name(record: SubjectImage, index: int) -> str:
    suffix = f"{index:05d}"
    return f"{record.subject_id}_{suffix}.npy"


def _process_one(
    record: SubjectImage,
    output_dir: Path,
    target_shape: Tuple[int, int, int],
    norm_mode: str,
    skull_strip_method: str,
    mni_resolution: int,
    skip_spatial: bool,
    n4: bool,
    center_crop: bool,
    index: int,
) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    """Process a single subject. Returns (processed_row, qc_row, error_msg)."""
    try:
        if record.group not in VALID_GROUPS:
            return None, None, f"Skipping {record.file_path}: group={record.group}"

        img = load_image(record)
        if n4:
            img = n4_bias_correct(img)

        qc, qc_ok, issues = qc_image(img)
        qc_row = {
            "subject_id": record.subject_id,
            "group": record.group,
            "file_path": record.file_path,
            "shape": qc["shape"],
            "spacing": qc["spacing"],
            "orientation": qc["orientation"],
            "qc_pass": qc_ok,
            "issues": issues,
        }

        brain_img, mask = skull_strip(img, skull_strip_method)
        norm_data = intensity_normalize(brain_img.get_fdata().astype(np.float32), mask, norm_mode)
        norm_data[~mask] = 0.0
        norm_img = nib.Nifti1Image(norm_data, brain_img.affine, brain_img.header)
        if not skip_spatial:
            norm_img = spatial_normalize(norm_img, mni_resolution)
        raw_arr = norm_img.get_fdata().astype(np.float32)
        if center_crop:
            arr = crop_to_brain_bbox(raw_arr, target_shape)
            arr = _center_by_shift(arr)
        else:
            arr = resize_array(raw_arr, target_shape)

        out_path = output_dir / "processed" / record.group / safe_name(record, index)
        np.save(out_path, arr)
        processed_row = {
            "subject_id": record.subject_id,
            "group": record.group,
            "file_path": str(out_path),
            "source_path": record.file_path,
        }
        return processed_row, qc_row, None

    except Exception:
        return None, None, f"Failed {record.file_path}:\n{traceback.format_exc()}"


def _scan_existing(output_dir: Path) -> dict:
    """Return {subject_id: [file_path, ...]} for already-processed .npy files."""
    done: dict = {}
    for group in VALID_GROUPS:
        group_dir = output_dir / "processed" / group
        if not group_dir.exists():
            continue
        for f in sorted(group_dir.glob("*.npy")):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                sid = "_".join(parts[:3])
                done.setdefault(sid, []).append((group, str(f)))
    return done


def process_images(
    records: Sequence[SubjectImage],
    output_dir: Path,
    target_shape: Tuple[int, int, int],
    norm_mode: str,
    skull_strip_method: str,
    mni_resolution: int,
    skip_spatial: bool,
    n4: bool = False,
    center_crop: bool = True,
    num_workers: int = 1,
) -> List[dict]:
    processed_rows: List[dict] = []
    qc_rows: List[dict] = []
    for group in VALID_GROUPS:
        (output_dir / "processed" / group).mkdir(parents=True, exist_ok=True)

    already_done = _scan_existing(output_dir)
    if already_done:
        print(f"  [skip] {len(already_done)} subjects already processed — skipping re-processing.")

    common_kwargs = dict(
        output_dir=output_dir,
        target_shape=target_shape,
        norm_mode=norm_mode,
        skull_strip_method=skull_strip_method,
        mni_resolution=mni_resolution,
        skip_spatial=skip_spatial,
        n4=n4,
        center_crop=center_crop,
    )

    def _collect(processed_row, qc_row, error):
        if error:
            logging.warning(error)
        if qc_row:
            qc_rows.append(qc_row)
        if processed_row:
            processed_rows.append(processed_row)

    def _add_existing(record):
        for group, fpath in already_done[record.subject_id]:
            processed_rows.append({
                "subject_id": record.subject_id,
                "group": group,
                "file_path": fpath,
                "source_path": record.file_path,
            })

    todo = [r for r in records if r.subject_id not in already_done]
    for r in records:
        if r.subject_id in already_done:
            _add_existing(r)

    if num_workers <= 1:
        for i, record in enumerate(tqdm(todo, desc="Preprocessing")):
            _collect(*_process_one(record, index=i, **common_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_process_one, record, index=i, **common_kwargs): i
                for i, record in enumerate(todo)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Preprocessing ({num_workers} workers)"):
                _collect(*fut.result())

    write_csv(
        output_dir / "quality_report.csv",
        qc_rows,
        ["subject_id", "group", "file_path", "shape", "spacing", "orientation", "qc_pass", "issues"],
    )
    return processed_rows


def split_subjects(rows: Sequence[dict], seed: int) -> Dict[str, List[dict]]:
    rng = np.random.default_rng(seed)
    by_group: Dict[str, Dict[str, List[dict]]] = {g: {} for g in VALID_GROUPS}
    for row in rows:
        group = row["group"]
        if group in VALID_GROUPS:
            by_group[group].setdefault(row["subject_id"], []).append(row)

    splits = {"train": [], "val": [], "test": []}
    for group, subjects in by_group.items():
        ids = list(subjects)
        rng.shuffle(ids)
        n = len(ids)
        if n == 0:
            continue
        n_test = max(1, int(round(n * 0.15))) if n >= 3 else 0
        n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = 1 if n >= 2 else 0
            n_val = 0
        test_ids = set(ids[:n_test])
        val_ids = set(ids[n_test : n_test + n_val])
        train_ids = set(ids[n_test + n_val :])
        for sid in train_ids:
            splits["train"].extend(subjects[sid])
        for sid in val_ids:
            splits["val"].extend(subjects[sid])
        for sid in test_ids:
            splits["test"].extend(subjects[sid])
    return splits


def parse_shape(text: str) -> Tuple[int, int, int]:
    parts = tuple(int(v.strip()) for v in text.split(","))
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise argparse.ArgumentTypeError("shape must be like 128,128,128")
    return parts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADNI MRI preprocessing pipeline")
    parser.add_argument("--input_dir", required=True, type=Path, help="ADNI root directory")
    parser.add_argument("--output_dir", default=Path("data"), type=Path, help="Output directory")
    parser.add_argument("--metadata_csv", type=Path, default=None, help="Optional ADNI metadata CSV")
    parser.add_argument("--target_shape", type=parse_shape, default=(128, 128, 128), help="Output shape, e.g. 128,128,128")
    parser.add_argument("--normalization", choices=["zscore", "minmax"], default="zscore")
    parser.add_argument("--skull_strip", choices=["auto", "fsl", "antspynet", "nilearn"], default="auto")
    parser.add_argument("--mni_resolution", choices=[1, 2], type=int, default=2)
    parser.add_argument("--skip_spatial", action="store_true", help="Skip MNI resampling")
    parser.add_argument("--scan_only", action="store_true", help="Only scan inputs and write subject_list.csv")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N scanned records for testing")
    parser.add_argument(
        "--series_include",
        default="mprage,mp-rage,spgr,t1",
        help="Comma-separated DICOM path keywords to include. Use an empty string to include all DICOM series.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n4", action="store_true",
        help="Apply N4 bias field correction before skull stripping (recommended for T1 MRI)",
    )
    parser.add_argument(
        "--no_center_crop", action="store_true",
        help="Disable brain bounding-box crop (not recommended — leaves brain in corner)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Parallel worker processes (default: 4). Use 1 for sequential/debug mode.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    require_dependencies()
    configure_logging(args.output_dir)
    metadata = read_metadata(args.metadata_csv)
    records = scan_dataset(args.input_dir, metadata, parse_include_terms(args.series_include))
    if args.limit is not None:
        records = records[: args.limit]

    subject_rows = [
        {
            "subject_id": r.subject_id,
            "group": r.group,
            "file_path": r.file_path,
            "input_type": r.input_type,
        }
        for r in records
    ]
    write_csv(args.output_dir / "subject_list.csv", subject_rows, ["subject_id", "group", "file_path", "input_type"])
    if args.scan_only:
        logging.info("Scan-only mode wrote %d records", len(records))
        print(f"Scanned {len(records)} records")
        print(f"subject_list.csv written to: {args.output_dir / 'subject_list.csv'}")
        return 0

    processed_rows = process_images(
        records,
        args.output_dir,
        args.target_shape,
        args.normalization,
        args.skull_strip,
        args.mni_resolution,
        args.skip_spatial,
        args.n4,
        center_crop=not args.no_center_crop,
        num_workers=args.num_workers,
    )
    write_csv(
        args.output_dir / "processed_list.csv",
        processed_rows,
        ["subject_id", "group", "file_path", "source_path"],
    )

    splits = split_subjects(processed_rows, args.seed)
    split_dir = args.output_dir / "splits"
    for split_name, rows in splits.items():
        write_csv(split_dir / f"{split_name}_list.csv", rows, ["subject_id", "group", "file_path", "source_path"])

    logging.info("Scanned %d records, processed %d records", len(records), len(processed_rows))
    print(f"Scanned {len(records)} records")
    print(f"Processed {len(processed_rows)} records")
    print(f"Outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
