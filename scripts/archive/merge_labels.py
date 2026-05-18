#!/usr/bin/env python
"""Merge ADNI diagnosis labels into data/subject_list.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


VALID_GROUPS = {"NC", "MCI", "AD"}
SUBJECT_RE = re.compile(r"(\d{3})_?S_?(\d{4})", re.IGNORECASE)
RID_RE = re.compile(r"(\d{4})(?!.*\d)")


def normalize_subject_id(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    match = SUBJECT_RE.search(text)
    if match:
        return f"{match.group(1)}_S_{match.group(2)}"
    return text


def extract_rid(value: object) -> str | None:
    sid = normalize_subject_id(value)
    match = SUBJECT_RE.search(sid)
    if match:
        return match.group(2)
    fallback = RID_RE.search(sid)
    return fallback.group(1) if fallback else None


def direct_merge(subjects: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    left = subjects.copy()
    right = metadata.copy()
    left["_subject_key"] = left["subject_id"].map(normalize_subject_id)
    right["_subject_key"] = right["subject_id"].map(normalize_subject_id)
    right = right.drop_duplicates(subset=["_subject_key"], keep="first")

    merged = left.merge(
        right[["_subject_key", "group"]].rename(columns={"group": "_meta_group"}),
        on="_subject_key",
        how="left",
    )
    success_rate = merged["_meta_group"].notna().mean() if len(merged) else 0.0
    return merged, float(success_rate)


def rid_merge(subjects: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    left = subjects.copy()
    right = metadata.copy()
    left["_rid"] = left["subject_id"].map(extract_rid)
    right["_rid"] = right["subject_id"].map(extract_rid)

    rid_counts = right["_rid"].value_counts()
    unique_rids = rid_counts[rid_counts == 1].index
    right = right[right["_rid"].isin(unique_rids)].drop_duplicates(subset=["_rid"], keep="first")

    merged = left.merge(
        right[["_rid", "group"]].rename(columns={"group": "_meta_group"}),
        on="_rid",
        how="left",
    )
    success_rate = merged["_meta_group"].notna().mean() if len(merged) else 0.0
    return merged, float(success_rate)


def merge_labels(subject_list_csv: Path, metadata_csv: Path, output_csv: Path) -> tuple[pd.DataFrame, str, float]:
    subjects = pd.read_csv(subject_list_csv)
    metadata = pd.read_csv(metadata_csv)

    required_subject = {"subject_id", "group"}
    required_meta = {"subject_id", "group"}
    if not required_subject.issubset(subjects.columns):
        raise ValueError(f"{subject_list_csv} must contain columns: {sorted(required_subject)}")
    if not required_meta.issubset(metadata.columns):
        raise ValueError(f"{metadata_csv} must contain columns: {sorted(required_meta)}")

    metadata = metadata.copy()
    metadata["group"] = metadata["group"].astype(str).str.strip().str.upper()
    metadata = metadata[metadata["group"].isin(VALID_GROUPS)]

    merged, success_rate = direct_merge(subjects, metadata)
    method = "direct subject_id"
    if success_rate < 0.50:
        rid_merged, rid_success_rate = rid_merge(subjects, metadata)
        if rid_success_rate > success_rate:
            merged, success_rate = rid_merged, rid_success_rate
            method = "last-four RID"

    merged["group"] = merged["_meta_group"].fillna(merged["group"]).fillna("Unknown")
    merged.loc[~merged["group"].isin(VALID_GROUPS), "group"] = "Unknown"

    cleanup_cols = [c for c in merged.columns if c.startswith("_")]
    merged = merged.drop(columns=cleanup_cols)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged, method, success_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ADNI metadata labels into subject_list.csv")
    parser.add_argument("--subject_list", type=Path, default=Path("data") / "subject_list.csv")
    parser.add_argument("--metadata", type=Path, default=Path("adni_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("data") / "subject_list.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged, method, success_rate = merge_labels(args.subject_list, args.metadata, args.output)

    counts = merged["group"].value_counts().reindex(["NC", "MCI", "AD", "Unknown"], fill_value=0)
    print(f"Merge method: {method}")
    print(f"Merge success rate: {success_rate:.2%}")
    print("Group counts:")
    print(counts.to_string())
    print("\nPreview:")
    print(merged.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
