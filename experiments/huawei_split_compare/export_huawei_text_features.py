#!/usr/bin/env python3
"""Export Huawei text cache to lung-cancer-cls text feature TSV.

Input:
  processed_df.csv + features.npz from Huawei train_text.py cache.

Output:
  A TSV accepted by src/lung_cancer_cls/text_clinical.py:
  - ID column, default 样本编号
  - num__0000... numeric features
  - bert_0000... text embedding features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert Huawei text cache to lung-cancer-cls text feature TSV.")
    parser.add_argument("--text_cache_dir", type=Path, required=True)
    parser.add_argument("--output_tsv", type=Path, default=root / "text_features" / "huawei_text_features.tsv")
    parser.add_argument("--id_col", type=str, default="样本编号")
    parser.add_argument("--fallback_numeric_dim", type=int, default=39)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_path = args.text_cache_dir / "processed_df.csv"
    feat_path = args.text_cache_dir / "features.npz"
    if not df_path.exists():
        raise FileNotFoundError(df_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    df = pd.read_csv(df_path).fillna("")
    if args.id_col not in df.columns:
        raise KeyError(f"id_col not found in processed_df.csv: {args.id_col}")
    feats = np.load(feat_path)
    if "X_text" not in feats:
        raise KeyError(f"X_text not found in {feat_path}")

    x_text = feats["X_text"].astype(np.float32)
    if "X_numeric" in feats:
        x_numeric = feats["X_numeric"].astype(np.float32)
    else:
        x_numeric = np.zeros((len(df), args.fallback_numeric_dim), dtype=np.float32)

    if len(df) != len(x_text) or len(df) != len(x_numeric):
        raise ValueError(f"row mismatch: df={len(df)}, X_text={len(x_text)}, X_numeric={len(x_numeric)}")

    out = pd.DataFrame({args.id_col: df[args.id_col].astype(str).str.strip()})
    for i in range(x_numeric.shape[1]):
        out[f"num__{i:04d}"] = x_numeric[:, i]
    for i in range(x_text.shape[1]):
        out[f"bert_{i:04d}"] = x_text[:, i]
    out = out.loc[out[args.id_col].ne("")].drop_duplicates(subset=[args.id_col], keep="first")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_tsv, sep="\t", index=False)
    meta = {
        "source_processed_df": str(df_path),
        "source_features_npz": str(feat_path),
        "record_id_col": args.id_col,
        "num_output_cols": [col for col in out.columns if col.startswith("num__")],
        "bert_dim": int(x_text.shape[1]),
        "num_rows": int(len(out)),
    }
    args.output_tsv.with_suffix(args.output_tsv.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[export_huawei_text_features] wrote {args.output_tsv} rows={len(out)}")


if __name__ == "__main__":
    main()
