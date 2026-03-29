from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from lung_cancer_cls.train import (
    build_epoch_log,
    compute_classification_metrics,
    resolve_selection_metric,
    resolve_selection_score,
)

LABEL_NAME_TO_ID: Dict[str, int] = {
    "健康对照": 0,
    "良性结节": 1,
    "肺癌": 2,
}

MULTICLASS_NAMES = {
    0: "normal",
    1: "benign",
    2: "malignant",
}

UNSPECIFIED_SPLITS = {"", "none", "nan", "null", "unspecified"}


@dataclass
class CNVXGBoostConfig:
    metadata_csv: Path
    gene_tsv: Path
    output_dir: Path
    label_col: str = "样本类型"
    metadata_gene_id_col: str = "SampleID"
    split_col: str = "CT_train_val_split"
    gene_id_col: str | None = None
    gene_label_col: str | None = None
    class_mode: str = "binary"
    binary_task: str = "malignant_vs_normal"
    selection_metric: str = "auto"
    split_mode: str = "train_val_test"
    use_predefined_split: bool = False
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_weight: float = 2.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    gamma: float = 0.0
    early_stopping_rounds: int = 30
    n_jobs: int = 4
    class_weight_strategy: str = "balanced"
    save_predictions: bool = True


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if np.isnan(value_f) or np.isinf(value_f):
        return None
    return value_f


def _jsonable_config(config: CNVXGBoostConfig) -> Dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _normalize_split(value: Any) -> str:
    split = str(value).strip().lower()
    if split in {"valid", "validation", "dev"}:
        return "val"
    if split not in {"train", "val", "test"}:
        return ""
    return split


def _safe_split(
    indices: Sequence[int],
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not indices:
        return [], []
    if len(indices) == 1:
        return list(indices), []

    unique_labels, counts = np.unique(labels, return_counts=True)
    can_stratify = len(unique_labels) > 1 and counts.min() >= 2
    kwargs = {
        "test_size": test_size,
        "random_state": seed,
        "shuffle": True,
    }
    if can_stratify:
        kwargs["stratify"] = labels

    try:
        train_idx, holdout_idx = train_test_split(list(indices), **kwargs)
    except ValueError:
        train_idx, holdout_idx = train_test_split(
            list(indices),
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return list(train_idx), list(holdout_idx)


def resolve_label_config(
    class_mode: str,
    binary_task: str,
) -> Tuple[Dict[int, str], Any]:
    mode = class_mode.lower().strip()
    if mode == "multiclass":
        return dict(MULTICLASS_NAMES), lambda label: label
    if mode != "binary":
        raise ValueError(f"Unknown class_mode: {class_mode}")

    task = binary_task.lower().strip()
    if task == "abnormal_vs_normal":
        return {0: "normal", 1: "abnormal"}, lambda label: 0 if label == 0 else 1
    if task == "malignant_vs_rest":
        return {0: "non_malignant", 1: "malignant"}, lambda label: 1 if label == 2 else 0
    if task == "malignant_vs_normal":
        return {0: "normal", 1: "malignant"}, lambda label: None if label == 1 else (1 if label == 2 else 0)
    raise ValueError(f"Unknown binary_task: {binary_task}")


def load_gene_feature_table(
    gene_tsv: Path,
    gene_id_col: str | None = None,
    gene_label_col: str | None = None,
) -> Tuple[pd.DataFrame, str, List[str], str | None]:
    df_gene = pd.read_csv(gene_tsv, sep="\t")
    if df_gene.empty:
        raise RuntimeError(f"Gene TSV is empty: {gene_tsv}")

    gene_id_col = gene_id_col or str(df_gene.columns[0])
    if gene_id_col not in df_gene.columns:
        raise ValueError(f"gene_id_col not found: {gene_id_col}")

    inferred_gene_label_col = gene_label_col
    if inferred_gene_label_col is None and len(df_gene.columns) > 1:
        inferred_gene_label_col = str(df_gene.columns[1])

    drop_cols = [gene_id_col]
    if inferred_gene_label_col and inferred_gene_label_col in df_gene.columns:
        drop_cols.append(inferred_gene_label_col)

    df_gene = df_gene.dropna(subset=[gene_id_col]).copy()
    df_gene[gene_id_col] = df_gene[gene_id_col].astype(str).str.strip()
    df_gene = df_gene.loc[df_gene[gene_id_col] != ""].drop_duplicates(subset=[gene_id_col], keep="first")

    feature_df = df_gene.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce")
    feature_names = feature_df.columns.tolist()
    if not feature_names:
        raise RuntimeError("No gene feature columns remain after dropping ID and label columns.")

    feature_matrix = feature_df.to_numpy(dtype=np.float32, copy=True)
    row_means = np.nanmean(feature_matrix, axis=1)
    row_means = np.where(np.isnan(row_means), 0.0, row_means)
    nan_rows, nan_cols = np.where(np.isnan(feature_matrix))
    if len(nan_rows) > 0:
        feature_matrix[nan_rows, nan_cols] = row_means[nan_rows]

    out_df = pd.DataFrame(feature_matrix, columns=feature_names)
    out_df.insert(0, gene_id_col, df_gene[gene_id_col].to_numpy())
    return out_df, gene_id_col, feature_names, inferred_gene_label_col


def build_cohort_table(
    metadata_csv: Path,
    gene_df: pd.DataFrame,
    config: CNVXGBoostConfig,
) -> Tuple[pd.DataFrame, Dict[int, str], Dict[str, int]]:
    df_meta = pd.read_csv(metadata_csv).fillna("")
    if df_meta.empty:
        raise RuntimeError(f"Metadata CSV is empty: {metadata_csv}")

    if config.metadata_gene_id_col not in df_meta.columns:
        raise ValueError(f"metadata_gene_id_col not found: {config.metadata_gene_id_col}")
    if config.label_col not in df_meta.columns:
        raise ValueError(f"label_col not found: {config.label_col}")

    class_names, label_mapper = resolve_label_config(config.class_mode, config.binary_task)
    split_series = pd.Series([""] * len(df_meta), index=df_meta.index, dtype=object)
    if config.split_col in df_meta.columns:
        split_series = df_meta[config.split_col].map(_normalize_split)

    cohort = pd.DataFrame(
        {
            "gene_id": df_meta[config.metadata_gene_id_col].astype(str).str.strip(),
            "label_name": df_meta[config.label_col].astype(str).str.strip(),
            "split": split_series,
        }
    )
    cohort = cohort.loc[cohort["gene_id"] != ""].copy()
    cohort["label_3class"] = cohort["label_name"].map(LABEL_NAME_TO_ID)
    cohort = cohort.dropna(subset=["label_3class"]).copy()
    cohort["label_3class"] = cohort["label_3class"].astype(int)
    cohort["label"] = cohort["label_3class"].map(label_mapper)
    cohort = cohort.dropna(subset=["label"]).copy()
    cohort["label"] = cohort["label"].astype(int)

    gene_id_col = config.gene_id_col or str(gene_df.columns[0])
    cohort = cohort.merge(gene_df, left_on="gene_id", right_on=gene_id_col, how="inner")

    stats = {
        "metadata_rows": int(len(df_meta)),
        "rows_after_label_filter": int(len(cohort)),
    }
    return cohort, class_names, stats


def deduplicate_gene_rows(cohort: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    kept_rows: List[pd.Series] = []
    duplicate_rows = 0
    label_conflicts = 0
    split_conflicts = 0

    for _, group in cohort.groupby("gene_id", sort=False):
        duplicate_rows += max(0, len(group) - 1)

        label_values = set(group["label"].tolist())
        if len(label_values) > 1:
            label_conflicts += 1
            continue

        split_values = {
            split for split in group["split"].tolist()
            if split not in UNSPECIFIED_SPLITS
        }
        if len(split_values) > 1:
            split_conflicts += 1
            continue

        row = group.iloc[0].copy()
        row["split"] = sorted(split_values)[0] if split_values else ""
        kept_rows.append(row)

    deduped = pd.DataFrame(kept_rows).reset_index(drop=True)
    stats = {
        "rows_before_dedup": int(len(cohort)),
        "rows_after_dedup": int(len(deduped)),
        "duplicate_rows_collapsed": int(duplicate_rows),
        "duplicate_gene_label_conflicts_dropped": int(label_conflicts),
        "duplicate_gene_split_conflicts_dropped": int(split_conflicts),
    }
    return deduped, stats


def split_cohort(
    cohort: pd.DataFrame,
    config: CNVXGBoostConfig,
) -> Tuple[List[int], List[int], List[int], str]:
    labels = cohort["label"].to_numpy(dtype=np.int64)
    all_indices = list(range(len(cohort)))

    if config.use_predefined_split:
        splits = cohort["split"].tolist()
        train_pool = [idx for idx, split in enumerate(splits) if split in {"train", ""}]
        val_idx = [idx for idx, split in enumerate(splits) if split == "val"]
        test_idx = [idx for idx, split in enumerate(splits) if split == "test"]

        if not train_pool and not val_idx and not test_idx:
            fallback_cfg = CNVXGBoostConfig(**{**asdict(config), "use_predefined_split": False})
            return split_cohort(cohort, fallback_cfg)

        if config.split_mode == "train_val":
            if not val_idx:
                train_idx, val_idx = _safe_split(
                    train_pool,
                    labels[train_pool],
                    test_size=config.val_ratio,
                    seed=config.seed,
                )
            else:
                train_idx = train_pool
            return train_idx, val_idx, [], "predefined"

        if not val_idx and train_pool:
            train_idx, val_idx = _safe_split(
                train_pool,
                labels[train_pool],
                test_size=config.val_ratio,
                seed=config.seed,
            )
        else:
            train_idx = train_pool

        if not test_idx:
            combined_pool = train_idx + val_idx
            train_plus_val_idx, test_idx = _safe_split(
                combined_pool,
                labels[combined_pool],
                test_size=config.test_ratio,
                seed=config.seed,
            )
            if not val_idx:
                train_idx, val_idx = _safe_split(
                    train_plus_val_idx,
                    labels[train_plus_val_idx],
                    test_size=config.val_ratio,
                    seed=config.seed,
                )
            else:
                train_idx = train_plus_val_idx
        return train_idx, val_idx, test_idx, "predefined"

    if config.split_mode == "train_val":
        train_idx, val_idx = _safe_split(
            all_indices,
            labels,
            test_size=config.val_ratio,
            seed=config.seed,
        )
        return train_idx, val_idx, [], "stratified_random"

    train_val_idx, test_idx = _safe_split(
        all_indices,
        labels,
        test_size=config.test_ratio,
        seed=config.seed,
    )
    train_idx, val_idx = _safe_split(
        train_val_idx,
        labels[train_val_idx],
        test_size=config.val_ratio,
        seed=config.seed,
    )
    return train_idx, val_idx, test_idx, "stratified_random"


def _compute_fit_sample_weight(
    labels: np.ndarray,
    class_weight_strategy: str,
) -> Tuple[np.ndarray | None, float | None]:
    strategy = class_weight_strategy.lower().strip()
    if strategy == "none":
        return None, None
    if strategy != "balanced":
        raise ValueError(f"Unknown class_weight_strategy: {class_weight_strategy}")

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) == 2:
        negative = counts[classes == 0][0] if np.any(classes == 0) else 0
        positive = counts[classes == 1][0] if np.any(classes == 1) else 0
        if positive > 0:
            return None, float(negative / positive) if negative > 0 else 1.0
        return None, None
    sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
    return sample_weight.astype(np.float32), None


def create_xgboost_estimator(
    config: CNVXGBoostConfig,
    num_classes: int,
    scale_pos_weight: float | None,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required for CNV baseline training. Install it with `pip install xgboost` "
            "or `pip install -r requirements.txt`."
        ) from exc

    params: Dict[str, Any] = {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "gamma": config.gamma,
        "random_state": config.seed,
        "n_jobs": config.n_jobs,
        "early_stopping_rounds": config.early_stopping_rounds if config.early_stopping_rounds > 0 else None,
        "tree_method": "hist",
    }

    if num_classes == 2:
        params.update({"objective": "binary:logistic", "eval_metric": "auc"})
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
    else:
        params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": num_classes})

    params = {key: value for key, value in params.items() if value is not None}
    return XGBClassifier(**params)


def _ensure_proba_shape(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities)
    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)
    return probs


def evaluate_split(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    if X.size == 0 or y.size == 0:
        return {
            "loss": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "auroc": None,
            "auprc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "brier_score": None,
            "confusion_matrix": [],
            "num_samples": 0,
        }

    probabilities = _ensure_proba_shape(estimator.predict_proba(X))
    probabilities = np.clip(probabilities, 1e-7, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(probabilities[np.arange(len(y)), y])))
    return compute_classification_metrics(y, probabilities, loss=loss, class_names=class_names)


def save_predictions(
    path: Path,
    cohort: pd.DataFrame,
    indices: Sequence[int],
    probabilities: np.ndarray,
    class_names: Dict[int, str],
) -> None:
    probs = _ensure_proba_shape(probabilities)
    split_df = cohort.iloc[list(indices)][["gene_id", "label_name", "label", "split"]].reset_index(drop=True).copy()
    split_df["prediction"] = probs.argmax(axis=1)
    for class_idx, class_name in class_names.items():
        split_df[f"prob_{class_name}"] = probs[:, class_idx]
    split_df.to_csv(path, index=False, encoding="utf-8-sig")


def train_xgboost_baseline(config: CNVXGBoostConfig) -> Dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_df, gene_id_col, feature_names, inferred_gene_label_col = load_gene_feature_table(
        config.gene_tsv,
        gene_id_col=config.gene_id_col,
        gene_label_col=config.gene_label_col,
    )
    config.gene_id_col = gene_id_col
    if config.gene_label_col is None:
        config.gene_label_col = inferred_gene_label_col

    cohort, class_names, cohort_stats = build_cohort_table(config.metadata_csv, gene_df, config)
    if cohort.empty:
        raise RuntimeError("No valid CNV samples remain after merging metadata CSV with gene TSV.")

    cohort, dedup_stats = deduplicate_gene_rows(cohort)
    if cohort.empty:
        raise RuntimeError("All CNV samples were dropped during deduplication.")

    train_idx, val_idx, test_idx, split_source = split_cohort(cohort, config)
    if not train_idx:
        raise RuntimeError("Training split is empty.")

    feature_matrix = cohort[feature_names].to_numpy(dtype=np.float32)
    labels = cohort["label"].to_numpy(dtype=np.int64)
    num_classes = len(class_names)
    selection_metric = resolve_selection_metric(config.selection_metric, num_classes)

    X_train = feature_matrix[train_idx]
    y_train = labels[train_idx]
    X_val = feature_matrix[val_idx] if val_idx else np.empty((0, feature_matrix.shape[1]), dtype=np.float32)
    y_val = labels[val_idx] if val_idx else np.empty((0,), dtype=np.int64)
    X_test = feature_matrix[test_idx] if test_idx else np.empty((0, feature_matrix.shape[1]), dtype=np.float32)
    y_test = labels[test_idx] if test_idx else np.empty((0,), dtype=np.int64)

    fit_sample_weight, scale_pos_weight = _compute_fit_sample_weight(y_train, config.class_weight_strategy)
    estimator = create_xgboost_estimator(config, num_classes=num_classes, scale_pos_weight=scale_pos_weight)

    fit_kwargs: Dict[str, Any] = {}
    if fit_sample_weight is not None:
        fit_kwargs["sample_weight"] = fit_sample_weight
    if len(X_val) > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    estimator.fit(X_train, y_train, **fit_kwargs)

    train_metrics = evaluate_split(estimator, X_train, y_train, class_names)
    val_metrics = evaluate_split(estimator, X_val, y_val, class_names)
    test_metrics = evaluate_split(estimator, X_test, y_test, class_names)
    selection_reference = val_metrics if val_metrics.get("num_samples", 0) else train_metrics
    selection_score, selection_metric_used = resolve_selection_score(selection_reference, selection_metric)

    model_path = output_dir / "best_model.json"
    estimator.save_model(model_path)
    with open(output_dir / "best_model.pkl", "wb") as f:
        pickle.dump(estimator, f)
    with open(output_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    if config.save_predictions:
        save_predictions(output_dir / "train_predictions.csv", cohort, train_idx, estimator.predict_proba(X_train), class_names)
        if len(X_val) > 0:
            save_predictions(output_dir / "val_predictions.csv", cohort, val_idx, estimator.predict_proba(X_val), class_names)
        if len(X_test) > 0:
            save_predictions(output_dir / "test_predictions.csv", cohort, test_idx, estimator.predict_proba(X_test), class_names)

    metrics = {
        "config": _jsonable_config(config),
        "class_names": class_names,
        "feature_dim": int(len(feature_names)),
        "split_source": split_source,
        "selection_metric": selection_metric,
        "selection_metric_used": selection_metric_used,
        "selection_score": _safe_float(selection_score),
        "cohort_stats": {
            **cohort_stats,
            **dedup_stats,
            "num_total": int(len(cohort)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "label_distribution": {
                class_names[int(label)]: int(count)
                for label, count in sorted(cohort["label"].value_counts().to_dict().items())
            },
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "epoch_logs": {
            "train": build_epoch_log(train_metrics),
            "val": build_epoch_log(val_metrics),
            "test": build_epoch_log(test_metrics),
        },
        "model_path": str(model_path),
        "best_iteration": int(getattr(estimator, "best_iteration", -1) or -1),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("CNV only XGBoost baseline")
    print(f"Output dir: {output_dir}")
    print(f"Split source: {split_source}")
    print(f"Samples: total={len(cohort)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"Selection metric: {selection_metric_used} = {_safe_float(selection_score)}")
    print(f"Validation AUROC: {val_metrics.get('auroc')}")
    if test_metrics.get("num_samples", 0):
        print(f"Test AUROC: {test_metrics.get('auroc')}")
    print("=" * 60)

    return metrics


def parse_args() -> CNVXGBoostConfig:
    parser = argparse.ArgumentParser(description="Train a CNV-only XGBoost baseline.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Metadata CSV path.")
    parser.add_argument("--gene-tsv", type=Path, required=True, help="CNV / gene TSV path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--label-col", type=str, default="样本类型", help="Metadata label column.")
    parser.add_argument("--metadata-gene-id-col", type=str, default="SampleID", help="Metadata gene ID column.")
    parser.add_argument("--split-col", type=str, default="CT_train_val_split", help="Metadata split column.")
    parser.add_argument("--gene-id-col", type=str, default=None, help="Gene TSV ID column. Defaults to the first column.")
    parser.add_argument("--gene-label-col", type=str, default=None, help="Gene TSV label column to drop from features.")
    parser.add_argument("--class-mode", type=str, choices=["multiclass", "binary"], default="binary")
    parser.add_argument("--binary-task", type=str, choices=["malignant_vs_rest", "abnormal_vs_normal", "malignant_vs_normal"], default="malignant_vs_normal")
    parser.add_argument("--selection-metric", type=str, choices=["auto", "accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "loss"], default="auto")
    parser.add_argument("--split-mode", type=str, choices=["train_val", "train_val_test"], default="train_val_test")
    parser.add_argument("--use-predefined-split", action="store_true", help="Use split labels from the metadata CSV when available.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when a validation split must be created.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test ratio when a test split must be created.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=600, help="XGBoost n_estimators.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate.")
    parser.add_argument("--max-depth", type=int, default=4, help="XGBoost max_depth.")
    parser.add_argument("--min-child-weight", type=float, default=2.0, help="XGBoost min_child_weight.")
    parser.add_argument("--subsample", type=float, default=0.8, help="XGBoost subsample.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="XGBoost colsample_bytree.")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="XGBoost reg_lambda.")
    parser.add_argument("--gamma", type=float, default=0.0, help="XGBoost gamma.")
    parser.add_argument("--early-stopping-rounds", type=int, default=30, help="XGBoost early stopping rounds.")
    parser.add_argument("--n-jobs", type=int, default=4, help="XGBoost n_jobs.")
    parser.add_argument("--class-weight-strategy", type=str, choices=["none", "balanced"], default="balanced")
    parser.add_argument("--no-save-predictions", action="store_true", help="Disable saving prediction CSV files.")

    args = parser.parse_args()
    return CNVXGBoostConfig(
        metadata_csv=args.metadata_csv,
        gene_tsv=args.gene_tsv,
        output_dir=args.output_dir,
        label_col=args.label_col,
        metadata_gene_id_col=args.metadata_gene_id_col,
        split_col=args.split_col,
        gene_id_col=args.gene_id_col,
        gene_label_col=args.gene_label_col,
        class_mode=args.class_mode,
        binary_task=args.binary_task,
        selection_metric=args.selection_metric,
        split_mode=args.split_mode,
        use_predefined_split=args.use_predefined_split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=args.n_jobs,
        class_weight_strategy=args.class_weight_strategy,
        save_predictions=not args.no_save_predictions,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    config = parse_args()
    train_xgboost_baseline(config)


if __name__ == "__main__":
    main()
