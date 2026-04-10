from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REVIEW_KEY_COLUMNS = (
    "preprocess_case_id",
    "preprocess_series_instance_uid",
)
REVIEW_FLAG_COLUMNS = (
    "qc_bucket",
    "review_status",
    "use_for_training",
    "review_note",
)
DEFAULT_FLAG_CONTEXT_COLUMNS = (
    "preprocess_num_slices",
    "preprocess_slice_thickness",
    "preprocess_spacing_z",
    "CT_train_val_split",
    "样本类型",
    "CT dicom路径",
    "CT_numpy路径",
    "CT_numpy_cloud路径",
)


def _require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required columns: {joined}")


def _dedupe_columns(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        output.append(column)
    return output


def _optional_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def build_review_flags_dataframe(
    manifest_df: pd.DataFrame,
    thick_slice_mm: float = 5.0,
    thick_spacing_z_mm: float = 5.0,
) -> pd.DataFrame:
    _require_columns(manifest_df, REVIEW_KEY_COLUMNS, "Manifest CSV")

    slice_thickness = _optional_numeric_column(manifest_df, "preprocess_slice_thickness")
    spacing_z = _optional_numeric_column(manifest_df, "preprocess_spacing_z")
    thick_mask = slice_thickness.ge(thick_slice_mm).fillna(False) | spacing_z.ge(thick_spacing_z_mm).fillna(False)

    keep_columns = list(REVIEW_KEY_COLUMNS)
    keep_columns.extend(column for column in DEFAULT_FLAG_CONTEXT_COLUMNS if column in manifest_df.columns)
    flags_df = manifest_df.loc[:, _dedupe_columns(keep_columns)].copy()
    flags_df["qc_bucket"] = np.where(thick_mask, "thick5_borderline", "standard")
    flags_df["review_status"] = np.where(thick_mask, "pending", "auto_pass")
    flags_df["use_for_training"] = np.where(thick_mask, 0, 1).astype(int)
    flags_df["review_note"] = ""
    return flags_df


def apply_review_flags_dataframe(
    manifest_df: pd.DataFrame,
    flags_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(manifest_df, REVIEW_KEY_COLUMNS, "Manifest CSV")
    _require_columns(flags_df, (*REVIEW_KEY_COLUMNS, *REVIEW_FLAG_COLUMNS), "Flags CSV")

    duplicate_mask = flags_df.duplicated(list(REVIEW_KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicate_rows = flags_df.loc[duplicate_mask, list(REVIEW_KEY_COLUMNS)].head(5)
        raise ValueError(
            "Flags CSV contains duplicate review keys, for example: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    flag_columns = [
        column
        for column in flags_df.columns
        if column not in manifest_df.columns or column in REVIEW_FLAG_COLUMNS
    ]
    merged_df = manifest_df.merge(
        flags_df.loc[:, _dedupe_columns([*REVIEW_KEY_COLUMNS, *flag_columns])],
        on=list(REVIEW_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )

    missing_mask = merged_df["use_for_training"].isna()
    if missing_mask.any():
        missing_rows = merged_df.loc[missing_mask, list(REVIEW_KEY_COLUMNS)].head(5)
        raise ValueError(
            "Some manifest rows do not have review flags, for example: "
            f"{missing_rows.to_dict(orient='records')}"
        )

    merged_df["use_for_training"] = (
        pd.to_numeric(merged_df["use_for_training"], errors="raise").astype(int)
    )
    filtered_df = merged_df.loc[merged_df["use_for_training"] == 1].copy()
    return filtered_df, merged_df


def generate_review_flags_csv(
    manifest_csv: Path,
    output_csv: Path,
    thick_slice_mm: float = 5.0,
    thick_spacing_z_mm: float = 5.0,
) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_csv)
    flags_df = build_review_flags_dataframe(
        manifest_df,
        thick_slice_mm=thick_slice_mm,
        thick_spacing_z_mm=thick_spacing_z_mm,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    flags_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return flags_df


def apply_review_flags_csv(
    manifest_csv: Path,
    flags_csv: Path,
    output_csv: Path,
    audit_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_df = pd.read_csv(manifest_csv)
    flags_df = pd.read_csv(flags_csv)
    filtered_df, merged_df = apply_review_flags_dataframe(manifest_df, flags_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if audit_csv is not None:
        audit_csv.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    return filtered_df, merged_df


def _build_flags_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a review-flags template from an intranet CT manifest."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--thick-slice-mm", type=float, default=5.0)
    parser.add_argument("--thick-spacing-z-mm", type=float, default=5.0)
    return parser


def _build_apply_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply review flags to an intranet CT manifest and keep only training-approved rows."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--flags-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, default=None)
    return parser


def main_generate_review_flags() -> None:
    args = _build_flags_parser().parse_args()
    flags_df = generate_review_flags_csv(
        manifest_csv=args.manifest_csv,
        output_csv=args.output_csv,
        thick_slice_mm=args.thick_slice_mm,
        thick_spacing_z_mm=args.thick_spacing_z_mm,
    )
    flagged = int((flags_df["use_for_training"] == 0).sum())
    print("=" * 72)
    print("Review-flags template created")
    print(f"Rows: {len(flags_df)}")
    print(f"Pending manual review: {flagged}")
    print(f"Flags CSV: {args.output_csv}")
    print("=" * 72)


def main_apply_review_flags() -> None:
    args = _build_apply_parser().parse_args()
    filtered_df, merged_df = apply_review_flags_csv(
        manifest_csv=args.manifest_csv,
        flags_csv=args.flags_csv,
        output_csv=args.output_csv,
        audit_csv=args.audit_csv,
    )
    dropped = int(len(merged_df) - len(filtered_df))
    print("=" * 72)
    print("Review flags applied")
    print(f"Manifest rows kept: {len(filtered_df)}")
    print(f"Manifest rows dropped: {dropped}")
    print(f"Output CSV: {args.output_csv}")
    if args.audit_csv is not None:
        print(f"Audit CSV: {args.audit_csv}")
    print("=" * 72)
