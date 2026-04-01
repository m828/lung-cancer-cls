from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None


DEFAULT_TEXT_PATTERNS = [
    "入院记录",
    "主诉",
    "现病史",
    "既往史",
    "病史",
    "查体",
    "诊断",
    "病理",
    "影像",
    "出院记录",
]

DEFAULT_NUMERIC_HINTS = [
    "<SEP2>",
    "肝功",
    "肾功",
    "肿瘤标志",
    "凝血",
    "血常规",
    "血生",
    "年龄",
    "性别",
]

GENDER_MAP = {
    "男": 1.0,
    "女": 0.0,
    "male": 1.0,
    "female": 0.0,
    "m": 1.0,
    "f": 0.0,
}


@dataclass
class TextClinicalFeatureConfig:
    output_tsv: Path
    text_feature_tsv: Path | None = None
    text_health_csv: Path | None = None
    text_disease_csv: Path | None = None
    bert_model_path: Path | None = None
    embedding_backend: str = "bert"
    hash_dim: int = 128
    batch_size: int = 8
    max_length: int = 128
    record_id_col: str = "record_id"
    label_col: str = "样本类型"
    text_cols: str | None = None
    numeric_cols: str | None = None
    text_cache_tsv: Path | None = None


def _parse_optional_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or None


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", str(name)).strip("_")


def _coerce_gender(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(GENDER_MAP)
    return mapped.where(~mapped.isna(), series)


def infer_text_columns(
    df: pd.DataFrame,
    explicit_cols: Sequence[str] | None = None,
) -> List[str]:
    if explicit_cols:
        return [col for col in explicit_cols if col in df.columns]
    matches = [
        col for col in df.columns
        if any(pattern in str(col) for pattern in DEFAULT_TEXT_PATTERNS)
    ]
    return sorted(set(matches))


def infer_numeric_columns(
    df: pd.DataFrame,
    record_id_col: str,
    label_col: str,
    text_cols: Sequence[str],
    explicit_cols: Sequence[str] | None = None,
) -> List[str]:
    if explicit_cols:
        return [col for col in explicit_cols if col in df.columns]

    excluded = {record_id_col, label_col, *text_cols}
    numeric_cols: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        series_for_ratio = _coerce_gender(series.copy())
        numeric_series = pd.to_numeric(series_for_ratio, errors="coerce")
        non_null_ratio = float(numeric_series.notna().mean())
        if non_null_ratio >= 0.3 or any(hint in str(col) for hint in DEFAULT_NUMERIC_HINTS):
            numeric_cols.append(col)
    return numeric_cols


def _merge_text_csvs(config: TextClinicalFeatureConfig) -> pd.DataFrame:
    if config.text_feature_tsv is not None:
        return pd.read_csv(config.text_feature_tsv, sep="\t")

    if config.text_cache_tsv is not None and config.text_cache_tsv.exists():
        return pd.read_csv(config.text_cache_tsv, sep="\t")

    if config.text_health_csv is None or config.text_disease_csv is None:
        raise ValueError(
            "Provide either --text-feature-tsv or both --text-health-csv and --text-disease-csv."
        )

    df_health = pd.read_csv(config.text_health_csv)
    df_disease = pd.read_csv(config.text_disease_csv)
    all_cols = df_health.columns.union(df_disease.columns)
    df_health = df_health.reindex(columns=all_cols, fill_value=np.nan)
    df_disease = df_disease.reindex(columns=all_cols, fill_value=np.nan)
    df = pd.concat([df_health, df_disease], axis=0, ignore_index=True)
    if config.record_id_col not in df.columns:
        raise ValueError(f"record_id_col not found in text CSVs: {config.record_id_col}")
    df = df.dropna(subset=[config.record_id_col]).copy()
    df[config.record_id_col] = df[config.record_id_col].astype(str).str.strip()
    df = df.loc[df[config.record_id_col] != ""].drop_duplicates(subset=[config.record_id_col], keep="first")
    return df.reset_index(drop=True)


def _combine_text(df: pd.DataFrame, text_cols: Sequence[str]) -> List[str]:
    if not text_cols:
        return ["无文本"] * len(df)
    text_df = df[list(text_cols)].fillna("无")
    return text_df.apply(lambda row: " ".join(str(v) for v in row.tolist()), axis=1).tolist()


def _compute_hash_embeddings(texts: Sequence[str], dim: int) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for text in texts:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.normal(loc=0.0, scale=1.0, size=dim).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        vectors.append(vector)
    return np.stack(vectors, axis=0) if vectors else np.zeros((0, dim), dtype=np.float32)


def _compute_bert_embeddings(
    texts: Sequence[str],
    model_path: Path,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError(
            "transformers is required for BERT text features. Install it with `pip install transformers`."
        )
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), max(1, batch_size)):
            batch_texts = list(texts[start:start + max(1, batch_size)])
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            chunks.append(cls_vectors)
    if not chunks:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _build_numeric_feature_frame(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    if not numeric_cols:
        return pd.DataFrame(index=df.index), []

    num_df = df[list(numeric_cols)].copy()
    for col in num_df.columns:
        if "性别" in str(col):
            num_df[col] = _coerce_gender(num_df[col])
        num_df[col] = pd.to_numeric(num_df[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(num_df)

    clipped = values.astype(np.float32, copy=True)
    for col_idx in range(clipped.shape[1]):
        column = clipped[:, col_idx]
        mean = float(np.mean(column))
        std = float(np.std(column))
        if std > 0:
            clipped[:, col_idx] = np.clip(column, mean - 3.0 * std, mean + 3.0 * std)

    selector = VarianceThreshold(threshold=1e-6)
    selected = selector.fit_transform(clipped)
    selected_cols = [numeric_cols[idx] for idx, keep in enumerate(selector.get_support()) if keep]

    if selected.size == 0:
        return pd.DataFrame(index=df.index), []

    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected).astype(np.float32)
    out_cols = [f"num__{_sanitize_name(col)}" for col in selected_cols]
    return pd.DataFrame(scaled, columns=out_cols, index=df.index), out_cols


def prepare_text_feature_table(
    config: TextClinicalFeatureConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if config.output_tsv is None:
        raise ValueError("output_tsv is required.")

    if config.text_feature_tsv is not None and config.text_feature_tsv.exists():
        out_df = pd.read_csv(config.text_feature_tsv, sep="\t")
        if config.record_id_col not in out_df.columns:
            raise ValueError(f"record_id_col not found in text feature TSV: {config.record_id_col}")
        out_df = out_df.dropna(subset=[config.record_id_col]).copy()
        out_df[config.record_id_col] = out_df[config.record_id_col].astype(str).str.strip()
        out_df = out_df.loc[out_df[config.record_id_col] != ""].drop_duplicates(subset=[config.record_id_col], keep="first")
        out_df.to_csv(config.output_tsv, sep="\t", index=False)

        src_meta_path = config.text_feature_tsv.with_suffix(config.text_feature_tsv.suffix + ".meta.json")
        meta: Dict[str, Any] = {
            "copied_from": str(config.text_feature_tsv),
            "record_id_col": config.record_id_col,
            "label_col": config.label_col,
            "num_output_cols": [col for col in out_df.columns if str(col).startswith("num__")],
            "text_cols": [],
            "bert_dim": int(sum(1 for col in out_df.columns if str(col).startswith("bert_"))),
            "embedding_backend": "precomputed",
            "num_rows": int(len(out_df)),
        }
        if src_meta_path.exists():
            meta = json.loads(src_meta_path.read_text(encoding="utf-8"))
            meta["copied_from"] = str(config.text_feature_tsv)
        meta_path = config.output_tsv.with_suffix(config.output_tsv.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_df.reset_index(drop=True), meta

    df = _merge_text_csvs(config)
    if config.record_id_col not in df.columns:
        raise ValueError(f"record_id_col not found: {config.record_id_col}")

    explicit_text_cols = _parse_optional_list(config.text_cols)
    explicit_numeric_cols = _parse_optional_list(config.numeric_cols)
    text_cols = infer_text_columns(df, explicit_cols=explicit_text_cols)
    numeric_cols = infer_numeric_columns(
        df,
        record_id_col=config.record_id_col,
        label_col=config.label_col,
        text_cols=text_cols,
        explicit_cols=explicit_numeric_cols,
    )
    combined_texts = _combine_text(df, text_cols)

    numeric_frame, numeric_output_cols = _build_numeric_feature_frame(df, numeric_cols)
    backend = config.embedding_backend.lower().strip()
    if backend == "hash":
        embedding_matrix = _compute_hash_embeddings(combined_texts, config.hash_dim)
    elif backend == "bert":
        if config.bert_model_path is None:
            raise ValueError("--bert-model-path is required when embedding_backend=bert")
        embedding_matrix = _compute_bert_embeddings(
            combined_texts,
            model_path=config.bert_model_path,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
    else:
        raise ValueError(f"Unknown embedding_backend: {config.embedding_backend}")

    bert_cols = [f"bert_{idx:04d}" for idx in range(embedding_matrix.shape[1])]
    bert_frame = pd.DataFrame(embedding_matrix, columns=bert_cols, index=df.index)

    out_df = pd.concat(
        [
            df[[config.record_id_col]].reset_index(drop=True),
            numeric_frame.reset_index(drop=True),
            bert_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    out_df.to_csv(config.output_tsv, sep="\t", index=False)
    meta = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "record_id_col": config.record_id_col,
        "label_col": config.label_col,
        "num_input_cols": list(numeric_cols),
        "num_output_cols": list(numeric_output_cols),
        "text_cols": list(text_cols),
        "bert_dim": int(embedding_matrix.shape[1]),
        "embedding_backend": backend,
        "num_rows": int(len(out_df)),
    }
    meta_path = config.output_tsv.with_suffix(config.output_tsv.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_df, meta


def load_text_feature_table(
    path: Path,
    record_id_col: str = "record_id",
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
    df = pd.read_csv(path, sep="\t")
    if record_id_col not in df.columns:
        raise ValueError(f"record_id_col not found in text feature TSV: {record_id_col}")
    df = df.dropna(subset=[record_id_col]).copy()
    df[record_id_col] = df[record_id_col].astype(str).str.strip()
    df = df.loc[df[record_id_col] != ""].drop_duplicates(subset=[record_id_col], keep="first")

    num_cols = [col for col in df.columns if str(col).startswith("num__")]
    bert_cols = [col for col in df.columns if str(col).startswith("bert_")]
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df.reset_index(drop=True), num_cols, bert_cols, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare text-clinical features from raw CSVs or a cached TSV.")
    parser.add_argument("--output-tsv", type=Path, required=True, help="Output TSV path for prepared text features.")
    parser.add_argument("--text-feature-tsv", type=Path, default=None, help="Existing prepared text feature TSV. If provided, it will be copied/rewritten to output.")
    parser.add_argument("--text-health-csv", type=Path, default=None, help="Healthy text CSV path.")
    parser.add_argument("--text-disease-csv", type=Path, default=None, help="Disease text CSV path.")
    parser.add_argument("--bert-model-path", type=Path, default=None, help="Local BERT model path.")
    parser.add_argument("--embedding-backend", type=str, choices=["bert", "hash"], default="bert")
    parser.add_argument("--hash-dim", type=int, default=128, help="Hash embedding dimension for smoke/testing.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--record-id-col", type=str, default="record_id")
    parser.add_argument("--label-col", type=str, default="样本类型")
    parser.add_argument("--text-cols", type=str, default=None, help="Optional comma-separated explicit text columns.")
    parser.add_argument("--numeric-cols", type=str, default=None, help="Optional comma-separated explicit numeric columns.")
    parser.add_argument("--text-cache-tsv", type=Path, default=None, help="Optional prepared TSV cache to reuse before rebuilding.")
    return parser


def parse_args() -> TextClinicalFeatureConfig:
    args = build_parser().parse_args()
    return TextClinicalFeatureConfig(
        output_tsv=args.output_tsv,
        text_feature_tsv=args.text_feature_tsv,
        text_health_csv=args.text_health_csv,
        text_disease_csv=args.text_disease_csv,
        bert_model_path=args.bert_model_path,
        embedding_backend=args.embedding_backend,
        hash_dim=args.hash_dim,
        batch_size=args.batch_size,
        max_length=args.max_length,
        record_id_col=args.record_id_col,
        label_col=args.label_col,
        text_cols=args.text_cols,
        numeric_cols=args.numeric_cols,
        text_cache_tsv=args.text_cache_tsv,
    )


def main() -> None:
    config = parse_args()
    prepare_text_feature_table(config)


if __name__ == "__main__":
    main()
