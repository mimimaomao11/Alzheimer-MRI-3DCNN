from __future__ import annotations

import pandas as pd
from pathlib import Path


def prepare_mci_conversion_labels(
    processed_csv: str = "data/processed_list.csv",
    dxsum_csv: str = "dataset/DXSUM_10May2026.csv",
    output_csv: str = "data/mci_conversion_list.csv",
) -> pd.DataFrame:
    df = pd.read_csv(processed_csv)
    mci_df = df[df["group"] == "MCI"].copy()
    mci_subjects = mci_df["subject_id"].unique()

    dx = pd.read_csv(dxsum_csv)
    dx_mci = dx[dx["PTID"].isin(mci_subjects)].copy()

    # pMCI: baseline MCI, later diagnosed as AD (DIAGNOSIS=3) at any follow-up
    # sMCI: baseline MCI, never diagnosed as AD throughout all follow-ups
    converters = set(dx_mci[dx_mci["DIAGNOSIS"] == 3.0]["PTID"].unique())

    mci_df["conversion_label"] = mci_df["subject_id"].apply(
        lambda sid: 1 if sid in converters else 0
    )
    mci_df["conversion_group"] = mci_df["conversion_label"].map({1: "pMCI", 0: "sMCI"})

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mci_df.to_csv(output_path, index=False)

    n_pmci = (mci_df["conversion_label"] == 1).sum()
    n_smci = (mci_df["conversion_label"] == 0).sum()
    n_pmci_subj = mci_df[mci_df["conversion_label"] == 1]["subject_id"].nunique()
    n_smci_subj = mci_df[mci_df["conversion_label"] == 0]["subject_id"].nunique()

    print(f"MCI conversion labels saved to: {output_csv}")
    print(f"pMCI: {n_pmci_subj} subjects ({n_pmci} scans)")
    print(f"sMCI: {n_smci_subj} subjects ({n_smci} scans)")
    print(f"Total: {n_pmci_subj + n_smci_subj} subjects ({len(mci_df)} scans)")
    return mci_df


if __name__ == "__main__":
    prepare_mci_conversion_labels()
