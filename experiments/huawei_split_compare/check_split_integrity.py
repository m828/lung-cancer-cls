#!/usr/bin/env python3
"""Integrity checks for fixed split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from huawei_split_tools import LABEL_NAMES, modality_counts, sha256_ids, split_label_distribution


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Check split manifest integrity.")
    parser.add_argument("--split_root", type=Path, default=root / "splits")
    parser.add_argument("--report_path", type=Path, default=root / "reports" / "integrity_report.json")
    parser.add_argument("--fail_on_patient_leakage", action="store_true")
    return parser.parse_args()


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def _split_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    return {
        split: set(df.loc[df["split"] == split, "sample_id"].astype(str).str.strip())
        for split in ["train", "val", "test"]
    }


def _check_manifest(df: pd.DataFrame, name: str, fail_on_patient_leakage: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {"sample_id", "label", "split", "ct_path", "has_text", "has_gene", "row_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append(f"missing required columns: {missing}")
        return {"name": name, "errors": errors, "warnings": warnings}

    valid_splits = {"train", "val", "test"}
    bad_splits = sorted(set(df["split"].astype(str)) - valid_splits - {""})
    if bad_splits:
        errors.append(f"unknown split values: {bad_splits}")

    duplicate_ids = df.loc[df["sample_id"].astype(str).str.strip().duplicated(), "sample_id"].astype(str).tolist()
    if duplicate_ids:
        errors.append(f"duplicate sample_id rows: {duplicate_ids[:20]}")

    split_sets = _split_sets(df)
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            errors.append(f"sample_id overlap {left}/{right}: {sorted(overlap)[:20]}")

    patient_leaks: dict[str, list[str]] = {}
    if "patient_id" in df.columns:
        for patient_id, group in df.groupby(df["patient_id"].astype(str).str.strip()):
            if not patient_id:
                continue
            splits = sorted(set(group["split"].astype(str)) & valid_splits)
            if len(splits) > 1:
                patient_leaks[patient_id] = splits
        if patient_leaks:
            message = f"patient_id crosses splits: {dict(list(patient_leaks.items())[:20])}"
            if fail_on_patient_leakage:
                errors.append(message)
            else:
                warnings.append(message)

    # Because every row in the manifest is a CT master row, any has_text/has_gene
    # row is automatically constrained to the row's CT split. These checks catch
    # malformed manifests where modality flags are present on rows without split.
    for modality_col in ["has_text", "has_gene"]:
        modality_rows = df.loc[df[modality_col].astype(bool)]
        outside = modality_rows.loc[~modality_rows["split"].isin(valid_splits)]
        if not outside.empty:
            errors.append(f"{modality_col} rows outside CT split: {outside['sample_id'].astype(str).head(20).tolist()}")

    final_mm = df.loc[df["has_text"].astype(bool) & df["has_gene"].astype(bool)]
    final_mm_outside = final_mm.loc[~final_mm["split"].isin(valid_splits)]
    if not final_mm_outside.empty:
        errors.append(f"final multimodal rows outside CT split: {final_mm_outside['sample_id'].astype(str).head(20).tolist()}")

    return {
        "name": name,
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "label_distribution": split_label_distribution(df),
        "modality_counts": modality_counts(df),
        "test_ids_sha256": sha256_ids(df.loc[df["split"] == "test", "sample_id"].tolist()),
    }


def _overlap_report(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_test = set(left.loc[left["split"] == "test", "sample_id"].astype(str))
    right_test = set(right.loc[right["split"] == "test", "sample_id"].astype(str))
    overlap = left_test & right_test
    denom = max(1, len(left_test | right_test))
    return {
        "huawei_8_2_test_count": int(len(left_test)),
        "split_7_1_2_test_count": int(len(right_test)),
        "overlap_count": int(len(overlap)),
        "overlap_ratio_union": float(len(overlap) / denom),
        "only_huawei_8_2_test": sorted(left_test - right_test),
        "only_split_7_1_2_test": sorted(right_test - left_test),
    }


def main() -> None:
    args = parse_args()
    reports: dict[str, Any] = {"manifests": {}, "folds": {}, "test_overlap": None}
    all_errors: list[str] = []

    huawei_path = args.split_root / "huawei_8_2" / "split_manifest.csv"
    split712_path = args.split_root / "split_7_1_2" / "split_manifest.csv"

    huawei_df = None
    split712_df = None
    for name, path in [("huawei_8_2", huawei_path), ("split_7_1_2", split712_path)]:
        try:
            df = _read_manifest(path)
            report = _check_manifest(df, name, args.fail_on_patient_leakage)
            reports["manifests"][name] = report
            all_errors.extend(f"{name}: {err}" for err in report.get("errors", []))
            if name == "huawei_8_2":
                huawei_df = df
            else:
                split712_df = df
        except Exception as exc:  # noqa: BLE001
            reports["manifests"][name] = {"errors": [str(exc)], "warnings": []}
            all_errors.append(f"{name}: {exc}")

    fold_dir = args.split_root / "split_5_fold"
    for fold_path in sorted(fold_dir.glob("fold_*.csv")):
        name = fold_path.stem
        try:
            df = _read_manifest(fold_path)
            report = _check_manifest(df, name, args.fail_on_patient_leakage)
            reports["folds"][name] = report
            all_errors.extend(f"{name}: {err}" for err in report.get("errors", []))
        except Exception as exc:  # noqa: BLE001
            reports["folds"][name] = {"errors": [str(exc)], "warnings": []}
            all_errors.append(f"{name}: {exc}")

    if huawei_df is not None and split712_df is not None:
        reports["test_overlap"] = _overlap_report(huawei_df, split712_df)

    reports["status"] = "pass" if not all_errors else "fail"
    reports["errors"] = all_errors
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
