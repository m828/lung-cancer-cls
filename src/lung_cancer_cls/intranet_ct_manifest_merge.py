from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_COL = "样本类型"
SPLIT_COL = "CT_train_val_split"
SAMPLE_ID_COL = "SampleID"
NPU_ABS_COL = "CT_numpy路径"
NPU_REL_COL = "CT_numpy_cloud路径"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "pandasnan"} else text


def _is_windows_abs(path_text: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_text))


def _normalize_rel_path(path_text: str) -> str:
    return str(path_text).replace("\\", "/").lstrip("/")


def _relative_to_root(path_text: str, ct_root: Path) -> str:
    text = _clean_text(path_text)
    if not text:
        return ""

    normalized = text.replace("\\", "/")
    if not normalized.startswith("/") and not _is_windows_abs(normalized):
        return _normalize_rel_path(normalized)

    absolute_path = Path(text)
    try:
        return absolute_path.relative_to(ct_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Path is not under ct_root: path={absolute_path}, ct_root={ct_root}"
        ) from exc


def _prepare_manifest(df: pd.DataFrame, ct_root: Path | None = None) -> pd.DataFrame:
    required = [LABEL_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    prepared = df.copy()
    if NPU_REL_COL not in prepared.columns:
        prepared[NPU_REL_COL] = ""

    def pick_rel_path(row: pd.Series) -> str:
        abs_path = _clean_text(row.get(NPU_ABS_COL))
        rel_path = _clean_text(row.get(NPU_REL_COL))
        if ct_root is not None and abs_path:
            return _relative_to_root(abs_path, ct_root)
        if rel_path:
            return _normalize_rel_path(rel_path)
        if abs_path:
            return _normalize_rel_path(abs_path)
        return ""

    prepared[NPU_REL_COL] = prepared.apply(pick_rel_path, axis=1)
    prepared = prepared.loc[prepared[NPU_REL_COL].astype(str).str.strip() != ""].copy()
    return prepared


def merge_intranet_manifests(
    base_df: pd.DataFrame,
    append_df: pd.DataFrame,
    ct_root: Path | None = None,
    dedupe_col: str = SAMPLE_ID_COL,
) -> pd.DataFrame:
    base_prepared = _prepare_manifest(base_df, ct_root=ct_root)
    append_prepared = _prepare_manifest(append_df, ct_root=ct_root)

    merged = pd.concat([base_prepared, append_prepared], ignore_index=True, sort=False)

    if dedupe_col and dedupe_col in merged.columns:
        dedupe_key = merged[dedupe_col].astype(str).str.strip()
        merged = merged.loc[~dedupe_key.duplicated(keep="last")].copy()

    return merged.reset_index(drop=True)


def merge_intranet_manifest_csvs(
    base_csv: Path,
    append_csv: Path,
    output_csv: Path,
    ct_root: Path | None = None,
    dedupe_col: str = SAMPLE_ID_COL,
) -> pd.DataFrame:
    base_df = pd.read_csv(base_csv).fillna("")
    append_df = pd.read_csv(append_csv).fillna("")
    merged = merge_intranet_manifests(
        base_df=base_df,
        append_df=append_df,
        ct_root=ct_root,
        dedupe_col=dedupe_col,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge an old intranet CT manifest with a newly rebuilt manifest."
    )
    parser.add_argument("--base-csv", type=Path, required=True, help="Old / existing manifest CSV.")
    parser.add_argument("--append-csv", type=Path, required=True, help="New manifest CSV to append.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Merged output CSV.")
    parser.add_argument(
        "--ct-root",
        type=Path,
        default=None,
        help="Unified CT root. If provided, CT_numpy路径 will be rewritten into CT_numpy_cloud路径 relative to this root.",
    )
    parser.add_argument(
        "--dedupe-col",
        type=str,
        default=SAMPLE_ID_COL,
        help="Column used for duplicate removal. Keep the later row on conflicts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merged = merge_intranet_manifest_csvs(
        base_csv=args.base_csv,
        append_csv=args.append_csv,
        output_csv=args.output_csv,
        ct_root=args.ct_root,
        dedupe_col=args.dedupe_col,
    )
    print("=" * 72)
    print("Merged intranet CT manifests")
    print(f"Rows: {len(merged)}")
    print(f"Output CSV: {args.output_csv}")
    print("=" * 72)
