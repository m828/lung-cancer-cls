#!/usr/bin/env python3
"""Generate fixed split manifests for Huawei split comparison.

The manifests use CT as the master cohort. Text and gene availability are
computed inside each CT split only; no modality is allowed to create its own
random split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from huawei_split_tools import (
    DEFAULT_CT_CSV,
    DEFAULT_CT_ROOT,
    DEFAULT_GENE_TSV,
    LABEL_MAP,
    choose_column,
    clean_id,
    label_to_int,
    label_to_name,
    modality_counts,
    normalize_split,
    read_table,
    resolve_ct_path,
    sha256_ids,
    split_label_distribution,
    write_ids,
    write_manifest_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Huawei 8:2, 7:1:2, and 5-fold split manifests.")
    parser.add_argument("--ct_csv", type=Path, default=DEFAULT_CT_CSV)
    parser.add_argument("--ct_root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--gene_tsv", type=Path, default=DEFAULT_GENE_TSV)
    parser.add_argument("--text_feature_tsv", type=Path, default=None)
    parser.add_argument("--text_cache_dir", type=Path, default=None, help="Optional Huawei text cache with processed_df.csv.")
    parser.add_argument("--output_root", type=Path, default=Path(__file__).resolve().parent / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--sample_id_col", type=str, default=None)
    parser.add_argument("--patient_id_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--split_col", type=str, default="CT_train_val_split")
    parser.add_argument("--ct_path_col", type=str, default="CT_numpy_cloud路径")
    parser.add_argument("--text_id_col", type=str, default=None)
    parser.add_argument("--text_record_id_col", type=str, default=None)
    parser.add_argument("--gene_id_col", type=str, default=None)
    parser.add_argument("--allow_missing_ct_files", action="store_true")
    parser.add_argument("--allow_missing_gene_tsv", action="store_true")
    return parser.parse_args()


def _best_matching_gene_column(ct_df: pd.DataFrame, gene_df: pd.DataFrame, candidates: Sequence[str]) -> tuple[str | None, str | None, set[str]]:
    if gene_df.empty:
        return None, None, set()

    gene_col_candidates = list(gene_df.columns[:3])
    best: tuple[int, str | None, str | None, set[str]] = (-1, None, None, set())
    for ct_col in candidates:
        if ct_col not in ct_df.columns:
            continue
        ct_ids = {clean_id(v) for v in ct_df[ct_col].tolist()}
        ct_ids.discard("")
        for gene_col in gene_col_candidates:
            gene_ids = {clean_id(v) for v in gene_df[gene_col].tolist()}
            gene_ids.discard("")
            overlap = ct_ids & gene_ids
            if len(overlap) > best[0]:
                best = (len(overlap), ct_col, gene_col, gene_ids)
    return best[1], best[2], best[3]


def _load_text_ids(args: argparse.Namespace) -> tuple[set[str], str]:
    if args.text_feature_tsv and args.text_feature_tsv.exists():
        df = read_table(args.text_feature_tsv, sep="\t")
        id_col = args.text_record_id_col or choose_column(df, ["样本编号", "record_id", "SampleID"], required=True)
        return {clean_id(v) for v in df[id_col].tolist() if clean_id(v)}, f"text_feature_tsv:{args.text_feature_tsv}:{id_col}"

    if args.text_cache_dir:
        df_path = args.text_cache_dir / "processed_df.csv"
        if df_path.exists():
            df = pd.read_csv(df_path).fillna("")
            id_col = args.text_record_id_col or choose_column(df, ["样本编号", "record_id", "SampleID"], required=True)
            return {clean_id(v) for v in df[id_col].tolist() if clean_id(v)}, f"text_cache:{df_path}:{id_col}"

    return set(), "metadata_nonempty_text_id"


def build_ct_master_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if not args.ct_csv.exists():
        raise FileNotFoundError(f"CT CSV not found: {args.ct_csv}")
    if not args.ct_root.exists() and not args.allow_missing_ct_files:
        raise FileNotFoundError(f"CT root not found: {args.ct_root}")

    raw = read_table(args.ct_csv)
    sample_id_col = args.sample_id_col or choose_column(raw, ["SampleID", "样本编号", "测序样本ID"], required=True)
    label_col = args.label_col or choose_column(raw, ["样本类型", "样本类型（处理）", "label"], required=True)
    text_id_col = args.text_id_col or choose_column(raw, ["样本编号", "record_id", "SampleID"], required=False)
    patient_id_col = args.patient_id_col or choose_column(raw, ["patient_id", "PatientID", "病人ID", "患者ID", "样本编号"], required=False)

    if args.ct_path_col not in raw.columns:
        raise KeyError(f"CT path column not found: {args.ct_path_col}")
    if args.split_col not in raw.columns:
        raise KeyError(f"Split column not found: {args.split_col}")

    text_ids, text_source = _load_text_ids(args)

    gene_df = pd.DataFrame()
    gene_ids: set[str] = set()
    gene_ct_col = args.gene_id_col
    gene_table_col = None
    if args.gene_tsv.exists():
        gene_df = read_table(args.gene_tsv, sep="\t")
        if args.gene_id_col:
            if args.gene_id_col not in raw.columns:
                raise KeyError(f"Gene metadata ID column not found in CT CSV: {args.gene_id_col}")
            gene_ct_col = args.gene_id_col
            gene_table_col = choose_column(gene_df, [args.gene_id_col, "SampleID", "测序样本ID", gene_df.columns[0]], required=True)
            gene_ids = {clean_id(v) for v in gene_df[gene_table_col].tolist() if clean_id(v)}
        else:
            gene_ct_col, gene_table_col, gene_ids = _best_matching_gene_column(
                raw,
                gene_df,
                ["SampleID", "测序样本ID", "样本编号"],
            )
    elif not args.allow_missing_gene_tsv:
        raise FileNotFoundError(f"Gene TSV not found: {args.gene_tsv}")

    records = []
    missing_ct = 0
    invalid_label = 0
    for row_index, row in raw.iterrows():
        label = label_to_int(row[label_col])
        if label is None:
            invalid_label += 1
            continue

        ct_rel_path, ct_path = resolve_ct_path(args.ct_root, row[args.ct_path_col])
        if not ct_rel_path:
            missing_ct += 1
            continue
        if not args.allow_missing_ct_files and not Path(ct_path).exists():
            missing_ct += 1
            continue

        sample_id = clean_id(row[sample_id_col])
        if not sample_id:
            continue
        text_id = clean_id(row[text_id_col]) if text_id_col else ""
        gene_id = clean_id(row[gene_ct_col]) if gene_ct_col and gene_ct_col in raw.columns else ""

        records.append(
            {
                "sample_id": sample_id,
                "patient_id": clean_id(row[patient_id_col]) if patient_id_col else sample_id,
                "label": int(label),
                "label_name": label_to_name(label),
                "source_split": normalize_split(row[args.split_col]),
                "ct_path": ct_path,
                "ct_rel_path": ct_rel_path,
                "text_id": text_id,
                "gene_id": gene_id,
                "has_text": bool(text_id) if not text_ids else text_id in text_ids,
                "has_gene": bool(gene_id) if not gene_ids else gene_id in gene_ids,
                "row_index": int(row_index),
            }
        )

    master = pd.DataFrame(records)
    if master.empty:
        raise RuntimeError("No valid CT samples remain after metadata/label/path filtering.")
    master = master.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)

    metadata = {
        "ct_csv": str(args.ct_csv),
        "ct_root": str(args.ct_root),
        "gene_tsv": str(args.gene_tsv),
        "sample_id_col": sample_id_col,
        "patient_id_col": patient_id_col,
        "label_col": label_col,
        "split_col": args.split_col,
        "ct_path_col": args.ct_path_col,
        "text_id_col": text_id_col,
        "text_source": text_source,
        "gene_metadata_id_col": gene_ct_col,
        "gene_table_id_col": gene_table_col,
        "raw_rows": int(len(raw)),
        "valid_ct_rows": int(len(master)),
        "invalid_label_rows": int(invalid_label),
        "missing_ct_rows": int(missing_ct),
    }
    return master, metadata


def make_huawei_8_2(master: pd.DataFrame) -> pd.DataFrame:
    df = master.loc[master["source_split"].isin(["train", "val", "test"])].copy()
    df["split"] = df["source_split"]
    df["fold_idx"] = ""
    return df.reset_index(drop=True)


def make_7_1_2(master: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    assignments: dict[int, str] = {}
    for _, group in master.groupby("label", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)
        for i in idx[:n_train]:
            assignments[int(i)] = "train"
        for i in idx[n_train:n_train + n_val]:
            assignments[int(i)] = "val"
        for i in idx[n_train + n_val:]:
            assignments[int(i)] = "test"

    df = master.copy()
    df["split"] = [assignments[int(i)] for i in df.index]
    df["fold_idx"] = ""
    return df.reset_index(drop=True)


def make_5_fold(master: pd.DataFrame, seed: int, n_folds: int, out_dir: Path) -> list[pd.DataFrame]:
    labels = master["label"].to_numpy()
    indices = np.arange(len(master))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    fold_frames: list[pd.DataFrame] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(indices, labels)):
        df = master.copy()
        df["split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df["fold_idx"] = int(fold_idx)
        df = df.reset_index(drop=True)
        fold_frames.append(df)
        df.to_csv(out_dir / f"fold_{fold_idx}.csv", index=False, encoding="utf-8-sig")
        write_ids(out_dir / f"fold_{fold_idx}_train_ids.txt", df.loc[df["split"] == "train", "sample_id"].tolist())
        write_ids(out_dir / f"fold_{fold_idx}_val_ids.txt", df.loc[df["split"] == "val", "sample_id"].tolist())
    return fold_frames


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    master, metadata = build_ct_master_frame(args)
    (args.output_root / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    huawei = make_huawei_8_2(master)
    write_manifest_bundle(huawei, args.output_root / "huawei_8_2", "huawei_8_2")

    split_712 = make_7_1_2(master, seed=args.seed)
    write_manifest_bundle(split_712, args.output_root / "split_7_1_2", "split_7_1_2")

    fold_frames = make_5_fold(master, seed=args.seed, n_folds=args.n_folds, out_dir=args.output_root / "split_5_fold")
    fold_summary = {
        "name": "split_5_fold",
        "n_folds": int(args.n_folds),
        "folds": {
            f"fold_{i}": {
                "rows": int(len(df)),
                "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
                "label_distribution": split_label_distribution(df),
                "modality_counts": modality_counts(df),
                "val_ids_sha256": sha256_ids(df.loc[df["split"] == "val", "sample_id"].tolist()),
            }
            for i, df in enumerate(fold_frames)
        },
    }
    (args.output_root / "split_5_fold" / "split_stats.json").write_text(
        json.dumps(fold_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[generate_split_manifests] done")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
