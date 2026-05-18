#!/usr/bin/env python
"""Create ADNI diagnosis metadata from ADNIMERGE.csv or DXSUM.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DX_MAP = {
    "CN": "NC",
    "MCI": "MCI",
    "EMCI": "MCI",
    "LMCI": "MCI",
    "AD": "AD",
    "SMC": "NC",
}

DXSUM_DIAGNOSIS_MAP = {
    1: "NC",
    2: "MCI",
    3: "AD",
}


def find_default_adnimerge() -> Path:
    candidates = [
        Path("ADNIMERGE.csv"),
        Path("dataset") / "ADNIMERGE.csv",
        Path("data") / "ADNIMERGE.csv",
        Path("dataset") / "DXSUM_10May2026.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = list(Path(".").rglob("ADNIMERGE.csv"))
    if not matches:
        matches = list(Path(".").rglob("DXSUM*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Cannot find ADNIMERGE.csv or DXSUM*.csv. Put it in the project root or pass --input <path>."
    )


def normalize_dx(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return DX_MAP.get(text)


def normalize_dxsum_diagnosis(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(float(value))
    except ValueError:
        return None
    return DXSUM_DIAGNOSIS_MAP.get(code)


def build_metadata(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if {"PTID", "VISCODE", "DX_bl"}.issubset(df.columns):
        label_col = "DX_bl"
        mapper = normalize_dx
    elif {"PTID", "VISCODE", "DIAGNOSIS"}.issubset(df.columns):
        label_col = "DIAGNOSIS"
        mapper = normalize_dxsum_diagnosis
    else:
        required = {"PTID", "VISCODE", "DX_bl"}
        dxsum_required = {"PTID", "VISCODE", "DIAGNOSIS"}
        missing_adnimerge = required - set(df.columns)
        missing_dxsum = dxsum_required - set(df.columns)
        raise ValueError(
            "Input CSV must look like ADNIMERGE or DXSUM. "
            f"Missing ADNIMERGE columns: {sorted(missing_adnimerge)}; "
            f"missing DXSUM columns: {sorted(missing_dxsum)}"
        )

    baseline = df[df["VISCODE"].astype(str).str.strip().str.lower() == "bl"].copy()
    baseline["group"] = baseline[label_col].map(mapper)
    out = baseline[["PTID", "group"]].rename(columns={"PTID": "subject_id"})
    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    out = out.dropna(subset=["group"])
    out = out.drop_duplicates(subset=["subject_id"], keep="first")
    out = out.sort_values("subject_id")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ADNI metadata from ADNIMERGE.csv")
    parser.add_argument("--input", type=Path, default=None, help="Path to ADNIMERGE.csv")
    parser.add_argument("--output", type=Path, default=Path("adni_metadata.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = args.input or find_default_adnimerge()
    out = build_metadata(input_csv, args.output)

    print(f"Input: {input_csv}")
    print(f"Output: {args.output}")
    print(f"Rows: {len(out)}")
    print("Group counts:")
    print(out["group"].value_counts().reindex(["NC", "MCI", "AD"], fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
