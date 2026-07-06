#!/usr/bin/env python3
"""Utilities for Huawei split comparison experiments.

This module is intentionally self-contained so the comparison experiment can
live under experiments/huawei_split_compare without changing the main project.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CT_CSV = Path("/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_文本划分0205_修复.csv")
DEFAULT_CT_ROOT = Path("/home/apulis-dev/userdata/Data/CT1500")
DEFAULT_GENE_TSV = Path("/home/apulis-dev/userdata/Data/Gene/FDEM_CNV_merge_pcc.tsv")

LABEL_MAP: Mapping[str, int] = {
    "健康对照": 0,
    "良性结节": 1,
    "肺癌": 2,
}
LABEL_NAMES: Mapping[int, str] = {v: k for k, v in LABEL_MAP.items()}

MISSING_VALUES = {"", "nan", "none", "null", "pandasnan"}


def read_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep).fillna("")


def choose_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of the candidate columns exist: {list(candidates)}")
    return None


def clean_id(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in MISSING_VALUES else text


def normalize_split(value: object) -> str:
    split = str(value).strip().lower()
    if split in {"valid", "validation", "dev"}:
        return "val"
    if split in {"train", "val", "test"}:
        return split
    return ""


def label_to_int(value: object) -> int | None:
    text = str(value).strip()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        parsed = int(float(text))
    except (TypeError, ValueError):
        return None
    return parsed if parsed in LABEL_NAMES else None


def label_to_name(value: object) -> str:
    parsed = label_to_int(value)
    if parsed is not None:
        return LABEL_NAMES[parsed]
    return str(value).strip()


def resolve_ct_path(ct_root: Path, rel_or_abs: object) -> tuple[str, str]:
    raw = clean_id(rel_or_abs).replace("\\", "/")
    if not raw:
        return "", ""
    path = Path(raw)
    abs_path = path if path.is_absolute() else Path(ct_root) / raw.lstrip("/")
    return raw, str(abs_path.resolve())


def sha256_ids(ids: Iterable[object]) -> str:
    normalized = "\n".join(sorted(clean_id(x) for x in ids if clean_id(x)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_label_distribution(df: pd.DataFrame, split_col: str = "split") -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    if df.empty:
        return result
    for split, group in df.groupby(split_col, dropna=False):
        key = str(split)
        counts = group["label"].value_counts().sort_index()
        result[key] = {LABEL_NAMES.get(int(label), str(label)): int(count) for label, count in counts.items()}
    return result


def modality_counts(df: pd.DataFrame, split_col: str = "split") -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    if df.empty:
        return result
    for split, group in df.groupby(split_col, dropna=False):
        has_text = group.get("has_text", pd.Series(False, index=group.index)).astype(bool)
        has_gene = group.get("has_gene", pd.Series(False, index=group.index)).astype(bool)
        result[str(split)] = {
            "ct": int(len(group)),
            "text": int(has_text.sum()),
            "gene": int(has_gene.sum()),
            "multimodal_intersection": int((has_text & has_gene).sum()),
        }
    return result


def write_ids(path: Path, ids: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(clean_id(x) for x in ids if clean_id(x)) + "\n", encoding="utf-8")


def write_manifest_bundle(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.csv"
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    for split in ["train", "val", "test"]:
        part = df.loc[df["split"] == split].copy()
        if part.empty:
            continue
        part.to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8-sig")
        write_ids(out_dir / f"{split}_ids.txt", part["sample_id"].tolist())

    summary = {
        "name": name,
        "rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "label_distribution": split_label_distribution(df),
        "modality_counts": modality_counts(df),
        "test_ids_sha256": sha256_ids(df.loc[df["split"] == "test", "sample_id"].tolist()),
    }
    (out_dir / "split_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
