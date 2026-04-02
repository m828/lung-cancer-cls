from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lung_cancer_cls.cnv_xgboost import load_gene_feature_table, resolve_label_config
from lung_cancer_cls.dataset import INTRANET_LABEL_MAP, get_default_transforms, get_default_volume_transforms
from lung_cancer_cls.model import build_model
from lung_cancer_cls.text_clinical import (
    TextClinicalFeatureConfig,
    load_text_feature_table,
    prepare_text_feature_table,
)
from lung_cancer_cls.train import (
    build_epoch_log,
    compute_classification_metrics,
    resolve_selection_metric,
    resolve_selection_score,
    set_seed,
)
from lung_cancer_cls.training_components import build_class_weights, create_loss, create_optimizer, create_scheduler

VALID_MODALITIES = ("ct", "text", "cnv")
UNSPECIFIED_SPLITS = {"", "none", "nan", "null", "unspecified"}


@dataclass
class MultiModalTrainConfig:
    data_root: Path
    metadata_csv: Path
    output_dir: Path
    modalities: Tuple[str, ...] = ("ct", "text", "cnv")
    reference_manifest: Path | None = None
    ct_root: Path | None = None
    gene_tsv: Path | None = None
    text_feature_tsv: Path | None = None
    text_health_csv: Path | None = None
    text_disease_csv: Path | None = None
    bert_model_path: Path | None = None
    text_embedding_backend: str = "bert"
    text_hash_dim: int = 128
    text_batch_size: int = 8
    text_max_length: int = 128
    text_cache_tsv: Path | None = None
    label_col: str = "样本类型"
    metadata_sample_id_col: str = "SampleID"
    metadata_text_id_col: str = "record_id"
    split_col: str = "CT_train_val_split"
    ct_path_col: str = "CT_numpy_cloud路径"
    text_record_id_col: str = "record_id"
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
    cpu: bool = False
    ct_model: str = "resnet3d18"
    pretrained: bool = False
    use_3d_input: bool = True
    depth_size: int = 32
    volume_hw: int = 128
    image_size: int = 224
    aug_profile: str = "strong"
    ct_feature_dim: int = 128
    text_feature_dim: int = 256
    gene_hidden_dim: int = 256
    fusion_hidden_dim: int = 256
    dropout: float = 0.3
    epochs: int = 20
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    loss_name: str = "ce"
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    sampling_strategy: str = "weighted"
    class_weight_strategy: str = "effective_num"
    effective_num_beta: float = 0.999
    save_predictions: bool = True


@dataclass
class StudentKDConfig(MultiModalTrainConfig):
    teacher_run_dir: Path | None = None
    distillation_alpha: float = 0.5
    distillation_temperature: float = 4.0


def normalize_modalities(raw: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(raw, str):
        values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    else:
        values = [str(item).strip().lower() for item in raw if str(item).strip()]
    normalized: List[str] = []
    for value in values:
        if value not in VALID_MODALITIES:
            raise ValueError(f"Unknown modality: {value}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one modality is required.")
    return tuple(normalized)


def jsonable_dataclass(config: Any) -> Dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if np.isnan(value_f) or np.isinf(value_f):
        return None
    return value_f


def _format_metric(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def infer_run_family(modalities: Sequence[str], is_student: bool = False) -> str:
    normalized = normalize_modalities(modalities)
    if is_student:
        return "student_kd"
    if normalized == ("text",):
        return "text_only"
    if normalized == ("ct",):
        return "ct_only"
    if normalized == ("cnv",):
        return "cnv_only_mlp"
    if len(normalized) == 2 and set(normalized) == {"ct", "cnv"}:
        return "ct_cnv"
    if len(normalized) == 2 and set(normalized) == {"ct", "text"}:
        return "ct_text"
    if len(normalized) == 3 and set(normalized) == {"ct", "cnv", "text"}:
        return "ct_cnv_text"
    return "multimodal"


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


def load_or_prepare_text_table(
    config: MultiModalTrainConfig,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any], Path]:
    if config.text_feature_tsv is not None and Path(config.text_feature_tsv).exists():
        table_path = Path(config.text_feature_tsv)
        df, num_cols, bert_cols, meta = load_text_feature_table(table_path, record_id_col=config.text_record_id_col)
        return df, num_cols, bert_cols, meta, table_path

    cache_path = config.text_cache_tsv or (config.output_dir / "prepared_text_features.tsv")
    feature_config = TextClinicalFeatureConfig(
        output_tsv=cache_path,
        text_health_csv=config.text_health_csv,
        text_disease_csv=config.text_disease_csv,
        bert_model_path=config.bert_model_path,
        embedding_backend=config.text_embedding_backend,
        hash_dim=config.text_hash_dim,
        batch_size=config.text_batch_size,
        max_length=config.text_max_length,
        record_id_col=config.text_record_id_col,
        label_col=config.label_col,
        text_cache_tsv=cache_path if cache_path.exists() else None,
    )
    prepare_text_feature_table(feature_config)
    df, num_cols, bert_cols, meta = load_text_feature_table(cache_path, record_id_col=config.text_record_id_col)
    return df, num_cols, bert_cols, meta, cache_path


def build_multimodal_cohort(
    config: MultiModalTrainConfig,
    required_modalities: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[int, str], Dict[str, Any]]:
    modalities = normalize_modalities(required_modalities)
    df_meta = pd.read_csv(config.metadata_csv).fillna("")
    if df_meta.empty:
        raise RuntimeError(f"Metadata CSV is empty: {config.metadata_csv}")

    for col in [config.metadata_sample_id_col, config.label_col]:
        if col not in df_meta.columns:
            raise ValueError(f"Required metadata column not found: {col}")

    class_names, label_mapper = resolve_label_config(config.class_mode, config.binary_task)
    split_series = pd.Series([""] * len(df_meta), index=df_meta.index, dtype=object)
    if config.split_col in df_meta.columns:
        split_series = df_meta[config.split_col].map(_normalize_split)

    cohort = pd.DataFrame(
        {
            "sample_id": df_meta[config.metadata_sample_id_col].astype(str).str.strip(),
            "label_name": df_meta[config.label_col].astype(str).str.strip(),
            "split": split_series,
        }
    )
    if config.metadata_text_id_col in df_meta.columns:
        cohort["record_id"] = df_meta[config.metadata_text_id_col].astype(str).str.strip()
    else:
        cohort["record_id"] = ""
    if config.ct_path_col in df_meta.columns:
        cohort["ct_rel_path"] = df_meta[config.ct_path_col].astype(str).str.strip()
    else:
        cohort["ct_rel_path"] = ""

    cohort = cohort.loc[cohort["sample_id"] != ""].copy()
    cohort["label_3class"] = cohort["label_name"].map(INTRANET_LABEL_MAP)
    cohort = cohort.dropna(subset=["label_3class"]).copy()
    cohort["label_3class"] = cohort["label_3class"].astype(int)
    cohort["label"] = cohort["label_3class"].map(label_mapper)
    cohort = cohort.dropna(subset=["label"]).copy()
    cohort["label"] = cohort["label"].astype(int)

    feature_info: Dict[str, Any] = {
        "modalities": list(modalities),
        "gene_cols": [],
        "text_num_cols": [],
        "text_emb_cols": [],
        "text_feature_tsv": None,
    }
    stats: Dict[str, Any] = {
        "metadata_rows": int(len(df_meta)),
        "rows_after_label_filter": int(len(cohort)),
    }

    if "ct" in modalities:
        if config.ct_path_col not in df_meta.columns:
            raise ValueError(f"CT path column not found: {config.ct_path_col}")
        ct_root = config.ct_root or config.data_root
        cohort["ct_path"] = cohort["ct_rel_path"].map(
            lambda rel: str((ct_root / str(rel).replace("\\", "/").lstrip("/")).resolve()) if str(rel).strip() else ""
        )
        cohort["ct_exists"] = cohort["ct_path"].map(lambda path: bool(path) and Path(path).exists())
        stats["missing_ct_rows_dropped"] = int((~cohort["ct_exists"]).sum())
        cohort = cohort.loc[cohort["ct_exists"]].drop(columns=["ct_exists"]).copy()

    if "cnv" in modalities:
        if config.gene_tsv is None:
            raise ValueError("gene_tsv is required when cnv modality is enabled.")
        gene_df, gene_id_col, gene_cols, inferred_gene_label_col = load_gene_feature_table(
            config.gene_tsv,
            gene_id_col=config.gene_id_col,
            gene_label_col=config.gene_label_col,
        )
        if config.gene_label_col is None:
            config.gene_label_col = inferred_gene_label_col
        feature_info["gene_cols"] = list(gene_cols)
        feature_info["gene_id_col"] = gene_id_col
        cohort = cohort.merge(gene_df, left_on="sample_id", right_on=gene_id_col, how="inner")
        stats["rows_after_cnv_merge"] = int(len(cohort))
        if cohort.empty:
            raise RuntimeError("No aligned samples remain after merging metadata with gene TSV.")

    if "text" in modalities:
        if config.metadata_text_id_col not in df_meta.columns:
            raise ValueError(f"text record id column not found in metadata CSV: {config.metadata_text_id_col}")
        text_df, text_num_cols, text_emb_cols, text_meta, text_path = load_or_prepare_text_table(config)
        feature_info["text_num_cols"] = list(text_num_cols)
        feature_info["text_emb_cols"] = list(text_emb_cols)
        feature_info["text_meta"] = text_meta
        feature_info["text_feature_tsv"] = str(text_path)
        if not text_num_cols and not text_emb_cols:
            raise RuntimeError("Prepared text feature table contains no usable text features.")
        cohort = cohort.merge(
            text_df,
            left_on="record_id",
            right_on=config.text_record_id_col,
            how="inner",
        )
        stats["rows_after_text_merge"] = int(len(cohort))
        if cohort.empty:
            raise RuntimeError("No aligned samples remain after merging metadata with text features.")

    kept_rows: List[pd.Series] = []
    duplicate_rows = 0
    label_conflicts = 0
    split_conflicts = 0
    for _, group in cohort.groupby("sample_id", sort=False):
        duplicate_rows += max(0, len(group) - 1)
        if len(set(group["label"].tolist())) > 1:
            label_conflicts += 1
            continue
        split_values = {split for split in group["split"].tolist() if split not in UNSPECIFIED_SPLITS}
        if len(split_values) > 1:
            split_conflicts += 1
            continue
        row = group.iloc[0].copy()
        row["split"] = sorted(split_values)[0] if split_values else ""
        kept_rows.append(row)

    deduped = pd.DataFrame(kept_rows).reset_index(drop=True)
    if deduped.empty:
        raise RuntimeError("All aligned samples were dropped during deduplication.")

    stats.update(
        {
            "rows_before_dedup": int(len(cohort)),
            "rows_after_dedup": int(len(deduped)),
            "duplicate_rows_collapsed": int(duplicate_rows),
            "duplicate_label_conflicts_dropped": int(label_conflicts),
            "duplicate_split_conflicts_dropped": int(split_conflicts),
        }
    )
    return deduped, feature_info, class_names, stats


def split_cohort(
    cohort: pd.DataFrame,
    config: MultiModalTrainConfig,
) -> Tuple[List[int], List[int], List[int], str]:
    labels = cohort["label"].to_numpy(dtype=np.int64)
    all_indices = list(range(len(cohort)))

    if config.use_predefined_split:
        splits = cohort["split"].tolist()
        train_pool = [idx for idx, split in enumerate(splits) if split in {"train", ""}]
        val_idx = [idx for idx, split in enumerate(splits) if split == "val"]
        test_idx = [idx for idx, split in enumerate(splits) if split == "test"]

        if not train_pool and not val_idx and not test_idx:
            fallback = MultiModalTrainConfig(**{**asdict(config), "use_predefined_split": False})
            return split_cohort(cohort, fallback)

        if config.split_mode == "train_val":
            if not val_idx:
                train_idx, val_idx = _safe_split(train_pool, labels[train_pool], config.val_ratio, config.seed)
            else:
                train_idx = train_pool
            return train_idx, val_idx, [], "predefined"

        if not val_idx and train_pool:
            train_idx, val_idx = _safe_split(train_pool, labels[train_pool], config.val_ratio, config.seed)
        else:
            train_idx = train_pool

        if not test_idx:
            combined_pool = train_idx + val_idx
            train_plus_val_idx, test_idx = _safe_split(combined_pool, labels[combined_pool], config.test_ratio, config.seed)
            if not val_idx:
                train_idx, val_idx = _safe_split(train_plus_val_idx, labels[train_plus_val_idx], config.val_ratio, config.seed)
            else:
                train_idx = train_plus_val_idx
        return train_idx, val_idx, test_idx, "predefined"

    if config.split_mode == "train_val":
        train_idx, val_idx = _safe_split(all_indices, labels, config.val_ratio, config.seed)
        return train_idx, val_idx, [], "stratified_random"

    train_val_idx, test_idx = _safe_split(all_indices, labels, config.test_ratio, config.seed)
    train_idx, val_idx = _safe_split(train_val_idx, labels[train_val_idx], config.val_ratio, config.seed)
    return train_idx, val_idx, test_idx, "stratified_random"


class UnifiedMultiModalDataset(Dataset):
    def __init__(
        self,
        cohort: pd.DataFrame,
        indices: Sequence[int],
        feature_info: Dict[str, Any],
        modalities: Sequence[str],
        transform: Any = None,
        use_3d: bool = True,
        depth_size: int = 32,
        volume_hw: int = 128,
    ):
        subset = cohort.iloc[list(indices)].reset_index(drop=True).copy()
        self.modalities = normalize_modalities(modalities)
        self.sample_ids = subset["sample_id"].astype(str).tolist()
        self.record_ids = subset.get("record_id", pd.Series([""] * len(subset))).astype(str).tolist()
        self.label_names = subset["label_name"].astype(str).tolist()
        self.splits = subset["split"].astype(str).tolist()
        self.labels = subset["label"].astype(int).tolist()
        self.transform = transform
        self.use_3d = use_3d
        self.depth_size = depth_size
        self.volume_hw = volume_hw
        self.ct_paths = [Path(path) for path in subset["ct_path"].tolist()] if "ct" in self.modalities else []
        self.gene_cols = list(feature_info.get("gene_cols", []))
        self.text_num_cols = list(feature_info.get("text_num_cols", []))
        self.text_emb_cols = list(feature_info.get("text_emb_cols", []))
        self.gene_matrix = (
            subset[self.gene_cols].to_numpy(dtype=np.float32, copy=True)
            if self.gene_cols else np.zeros((len(subset), 0), dtype=np.float32)
        )
        self.text_num_matrix = (
            subset[self.text_num_cols].to_numpy(dtype=np.float32, copy=True)
            if self.text_num_cols else np.zeros((len(subset), 0), dtype=np.float32)
        )
        self.text_emb_matrix = (
            subset[self.text_emb_cols].to_numpy(dtype=np.float32, copy=True)
            if self.text_emb_cols else np.zeros((len(subset), 0), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def _load_ct_tensor(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if self.use_3d:
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim != 3:
                raise ValueError(f"Unsupported CT array shape for 3D mode: {arr.shape}, path={path}")
            arr = arr - arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr = arr / max_val
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.depth_size, self.volume_hw, self.volume_hw),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            tensor = (tensor - 0.5) / 0.5
            if self.transform is not None:
                tensor = self.transform(tensor)
            return tensor

        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported CT array shape for 2D mode: {arr.shape}, path={path}")
        arr = arr - arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        arr = (arr * 255.0).astype(np.uint8)
        image = Image.fromarray(arr, mode="L")
        if self.transform is not None:
            return self.transform(image)
        from torchvision import transforms
        return transforms.ToTensor()(image)

    def __getitem__(self, idx: int):
        inputs: Dict[str, torch.Tensor] = {}
        if "ct" in self.modalities:
            inputs["ct"] = self._load_ct_tensor(self.ct_paths[idx])
        if "cnv" in self.modalities:
            inputs["cnv"] = torch.from_numpy(self.gene_matrix[idx])
        if "text" in self.modalities:
            inputs["text_num"] = torch.from_numpy(self.text_num_matrix[idx])
            inputs["text_emb"] = torch.from_numpy(self.text_emb_matrix[idx])
        return inputs, self.labels[idx]


class TextClinicalEncoder(nn.Module):
    def __init__(
        self,
        num_dim: int,
        text_dim: int,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_dim = int(num_dim)
        self.text_dim = int(text_dim)
        branch_dim = max(32, output_dim // 2)
        self.output_dim = output_dim

        self.num_branch = None
        if self.num_dim > 0:
            hidden = max(128, min(512, self.num_dim * 2))
            self.num_branch = nn.Sequential(
                nn.Linear(self.num_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, branch_dim),
                nn.BatchNorm1d(branch_dim),
                nn.ReLU(inplace=True),
            )

        self.text_branch = None
        if self.text_dim > 0:
            hidden = max(256, min(768, max(self.text_dim // 2, branch_dim * 2)))
            self.text_branch = nn.Sequential(
                nn.Linear(self.text_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, branch_dim),
                nn.BatchNorm1d(branch_dim),
                nn.ReLU(inplace=True),
            )

        if self.num_branch is not None and self.text_branch is not None:
            self.attention = nn.Sequential(
                nn.Linear(branch_dim * 2, max(32, branch_dim // 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, branch_dim // 2), 2),
                nn.Softmax(dim=1),
            )
            self.fused_proj = nn.Sequential(
                nn.Linear(branch_dim * 2, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True),
            )
            self.single_num_proj = None
            self.single_text_proj = None
        else:
            self.attention = None
            self.fused_proj = None
            self.single_num_proj = nn.Linear(branch_dim, output_dim) if self.num_branch is not None else None
            self.single_text_proj = nn.Linear(branch_dim, output_dim) if self.text_branch is not None else None

    def forward(self, text_num: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        if self.num_branch is None and self.text_branch is None:
            raise RuntimeError("TextClinicalEncoder has neither numeric nor embedding branch.")

        num_feat = self.num_branch(text_num) if self.num_branch is not None else None
        text_feat = self.text_branch(text_emb) if self.text_branch is not None else None

        if num_feat is not None and text_feat is not None:
            concat = torch.cat([num_feat, text_feat], dim=1)
            weights = self.attention(concat)
            fused = torch.cat(
                [
                    num_feat * weights[:, 0:1],
                    text_feat * weights[:, 1:2],
                ],
                dim=1,
            )
            return self.fused_proj(fused)
        if num_feat is not None and self.single_num_proj is not None:
            return self.single_num_proj(num_feat)
        if text_feat is not None and self.single_text_proj is not None:
            return self.single_text_proj(text_feat)
        raise RuntimeError("Failed to compute text features.")


class CNVEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        mid_dim = max(hidden_dim, min(1024, max(128, input_dim // 2)))
        self.output_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlexibleMultiModalClassifier(nn.Module):
    def __init__(
        self,
        modalities: Sequence[str],
        num_classes: int,
        ct_model: str = "resnet3d18",
        pretrained: bool = False,
        ct_feature_dim: int = 128,
        text_num_dim: int = 0,
        text_emb_dim: int = 0,
        text_feature_dim: int = 256,
        gene_dim: int = 0,
        gene_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modalities = normalize_modalities(modalities)
        fusion_dim = 0

        if "ct" in self.modalities:
            self.ct_encoder = build_model(ct_model, num_classes=ct_feature_dim, pretrained=pretrained)
            self.ct_feature_dim = ct_feature_dim
            fusion_dim += ct_feature_dim
        else:
            self.ct_encoder = None
            self.ct_feature_dim = 0

        if "text" in self.modalities:
            self.text_encoder = TextClinicalEncoder(
                num_dim=text_num_dim,
                text_dim=text_emb_dim,
                output_dim=text_feature_dim,
                dropout=dropout,
            )
            self.text_feature_dim = self.text_encoder.output_dim
            fusion_dim += self.text_feature_dim
        else:
            self.text_encoder = None
            self.text_feature_dim = 0

        if "cnv" in self.modalities:
            self.gene_encoder = CNVEncoder(gene_dim, hidden_dim=gene_hidden_dim, dropout=dropout)
            self.gene_feature_dim = self.gene_encoder.output_dim
            fusion_dim += self.gene_feature_dim
        else:
            self.gene_encoder = None
            self.gene_feature_dim = 0

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features: List[torch.Tensor] = []
        if self.ct_encoder is not None:
            features.append(self.ct_encoder(inputs["ct"]))
        if self.text_encoder is not None:
            features.append(self.text_encoder(inputs["text_num"], inputs["text_emb"]))
        if self.gene_encoder is not None:
            features.append(self.gene_encoder(inputs["cnv"]))
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)


def move_inputs_to_device(
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: List[torch.Tensor] = []
    all_probabilities: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = move_inputs_to_device(inputs, device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_labels.append(labels.detach().cpu())
            all_probabilities.append(torch.softmax(logits, dim=1).detach().cpu())

    if total == 0:
        return {
            "loss": None,
            "accuracy": None,
            "balanced_accuracy": None,
            "auroc": None,
            "auprc": None,
            "mcc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "npv": None,
            "fpr": None,
            "fnr": None,
            "youden_j": None,
            "brier_score": None,
            "ece": None,
            "confusion_matrix": [],
            "num_samples": 0,
        }

    y_true_np = torch.cat(all_labels).numpy().astype(np.int64)
    probabilities_np = torch.cat(all_probabilities).numpy()
    average_loss = total_loss / max(total, 1)
    return compute_classification_metrics(
        y_true_np,
        probabilities_np,
        loss=average_loss,
        class_names=class_names,
    )


def train_epoch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scheduler_step_per_batch: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0

    for inputs, labels in loader:
        inputs = move_inputs_to_device(inputs, device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)

    return running_loss / max(seen, 1)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-4)
    return F.kl_div(
        F.log_softmax(student_logits / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1),
        reduction="batchmean",
    ) * (temp ** 2)


def train_epoch_student_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    temperature: float,
    scheduler: Any = None,
    scheduler_step_per_batch: bool = False,
) -> float:
    student.train()
    teacher.eval()
    running_loss = 0.0
    seen = 0

    for inputs, labels in loader:
        inputs = move_inputs_to_device(inputs, device)
        labels = labels.to(device)

        optimizer.zero_grad()
        student_logits = student(inputs)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        ce = criterion(student_logits, labels)
        kd = distillation_loss(student_logits, teacher_logits, temperature=temperature)
        loss = (1.0 - alpha) * ce + alpha * kd
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)

    return running_loss / max(seen, 1)


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = move_inputs_to_device(inputs, device)
            logits = model(inputs)
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)


def save_prediction_csv(
    path: Path,
    dataset: UnifiedMultiModalDataset,
    probabilities: np.ndarray,
    class_names: Dict[int, str],
) -> None:
    df = pd.DataFrame(
        {
            "sample_id": dataset.sample_ids,
            "record_id": dataset.record_ids,
            "label_name": dataset.label_names,
            "label": dataset.labels,
            "split": dataset.splits,
            "prediction": probabilities.argmax(axis=1) if probabilities.size else [],
        }
    )
    for class_idx, class_name in class_names.items():
        if probabilities.size:
            df[f"prob_{class_name}"] = probabilities[:, class_idx]
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_split_manifest(
    path: Path,
    cohort: pd.DataFrame,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    frames: List[pd.DataFrame] = []
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not indices:
            continue
        frame = cohort.iloc[list(indices)][["sample_id", "record_id", "label_name", "label"]].copy()
        frame["assigned_split"] = split_name
        frames.append(frame)
    manifest = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame(
        columns=["sample_id", "record_id", "label_name", "label", "assigned_split"]
    )
    manifest.to_csv(path, index=False, encoding="utf-8-sig")


def load_split_manifest_indices(
    cohort: pd.DataFrame,
    manifest_path: Path,
) -> Tuple[List[int], List[int], List[int]]:
    manifest = pd.read_csv(manifest_path).fillna("")
    if "sample_id" not in manifest.columns:
        raise ValueError(f"sample_id column not found in split manifest: {manifest_path}")
    split_col = "assigned_split" if "assigned_split" in manifest.columns else "split"
    split_map = {
        str(row["sample_id"]).strip(): _normalize_split(row[split_col])
        for _, row in manifest.iterrows()
        if str(row["sample_id"]).strip()
    }
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for idx, sample_id in enumerate(cohort["sample_id"].astype(str).tolist()):
        split = split_map.get(sample_id, "")
        if split == "train":
            train_idx.append(idx)
        elif split == "val":
            val_idx.append(idx)
        elif split == "test":
            test_idx.append(idx)
    return train_idx, val_idx, test_idx


def filter_cohort_to_manifest(
    cohort: pd.DataFrame,
    manifest_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path).fillna("")
    if "sample_id" not in manifest.columns:
        raise ValueError(f"sample_id column not found in split manifest: {manifest_path}")
    split_col = "assigned_split" if "assigned_split" in manifest.columns else "split"
    manifest = manifest.copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
    manifest[split_col] = manifest[split_col].map(_normalize_split)
    manifest = manifest.loc[
        manifest["sample_id"].ne("") & manifest[split_col].isin({"train", "val", "test"})
    ].drop_duplicates(subset=["sample_id"], keep="first")
    manifest["_manifest_order"] = np.arange(len(manifest), dtype=np.int64)

    filtered = cohort.merge(
        manifest[["sample_id", split_col, "_manifest_order"]],
        on="sample_id",
        how="inner",
    )
    if filtered.empty:
        raise RuntimeError(f"No overlapping samples remain after applying manifest: {manifest_path}")
    filtered["split"] = filtered[split_col].map(_normalize_split)
    filtered = filtered.sort_values("_manifest_order").drop(columns=[split_col, "_manifest_order"]).reset_index(drop=True)
    return filtered


def create_model_from_config(
    config: MultiModalTrainConfig,
    feature_info: Dict[str, Any],
    class_names: Dict[int, str],
) -> FlexibleMultiModalClassifier:
    return FlexibleMultiModalClassifier(
        modalities=config.modalities,
        num_classes=len(class_names),
        ct_model=config.ct_model,
        pretrained=config.pretrained,
        ct_feature_dim=config.ct_feature_dim,
        text_num_dim=len(feature_info.get("text_num_cols", [])),
        text_emb_dim=len(feature_info.get("text_emb_cols", [])),
        text_feature_dim=config.text_feature_dim,
        gene_dim=len(feature_info.get("gene_cols", [])),
        gene_hidden_dim=config.gene_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
    )


def prepare_dataloaders(
    config: MultiModalTrainConfig,
    cohort: pd.DataFrame,
    feature_info: Dict[str, Any],
    modalities: Sequence[str],
    indices: Tuple[List[int], List[int], List[int]],
) -> Tuple[UnifiedMultiModalDataset, UnifiedMultiModalDataset, UnifiedMultiModalDataset | None, DataLoader, DataLoader, DataLoader | None, DataLoader]:
    train_idx, val_idx, test_idx = indices
    train_tf, val_tf = get_default_transforms(None, config.image_size, aug_profile=config.aug_profile)
    train_vol_tf, val_vol_tf = get_default_volume_transforms(config.aug_profile)
    use_3d = bool(config.use_3d_input)

    dataset_modalities = normalize_modalities(modalities)
    train_ds = UnifiedMultiModalDataset(
        cohort,
        train_idx,
        feature_info,
        modalities=dataset_modalities,
        transform=train_vol_tf if use_3d else train_tf,
        use_3d=use_3d,
        depth_size=config.depth_size,
        volume_hw=config.volume_hw,
    )
    val_ds = UnifiedMultiModalDataset(
        cohort,
        val_idx,
        feature_info,
        modalities=dataset_modalities,
        transform=val_vol_tf if use_3d else val_tf,
        use_3d=use_3d,
        depth_size=config.depth_size,
        volume_hw=config.volume_hw,
    )
    test_ds = None
    if test_idx:
        test_ds = UnifiedMultiModalDataset(
            cohort,
            test_idx,
            feature_info,
            modalities=dataset_modalities,
            transform=val_vol_tf if use_3d else val_tf,
            use_3d=use_3d,
            depth_size=config.depth_size,
            volume_hw=config.volume_hw,
        )

    num_classes = int(max(train_ds.labels) + 1) if train_ds.labels else 0
    train_label_counts = [0 for _ in range(max(1, num_classes))]
    for label in train_ds.labels:
        train_label_counts[label] += 1

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    if config.sampling_strategy == "weighted" and train_ds.labels:
        per_class_weights = [1.0 / max(1, count) for count in train_label_counts]
        sample_weights = [per_class_weights[label] for label in train_ds.labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=config.num_workers,
        )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) if test_ds is not None else None
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, train_eval_loader


def train_multimodal_model(config: MultiModalTrainConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    config.modalities = normalize_modalities(config.modalities)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multimodal training")
    print(f"Modalities: {','.join(config.modalities)}")
    print(f"Metadata CSV: {config.metadata_csv}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 60)

    cohort, feature_info, class_names, cohort_stats = build_multimodal_cohort(config, config.modalities)
    if config.reference_manifest is not None:
        cohort = filter_cohort_to_manifest(cohort, config.reference_manifest)
        train_idx, val_idx, test_idx = load_split_manifest_indices(cohort, config.reference_manifest)
        split_source = "reference_manifest"
    else:
        train_idx, val_idx, test_idx, split_source = split_cohort(cohort, config)
    if not train_idx:
        raise RuntimeError("Training split is empty.")
    save_split_manifest(config.output_dir / "split_manifest.csv", cohort, train_idx, val_idx, test_idx)

    selection_metric = resolve_selection_metric(config.selection_metric, len(class_names))
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, train_eval_loader = prepare_dataloaders(
        config,
        cohort,
        feature_info,
        config.modalities,
        (train_idx, val_idx, test_idx),
    )
    train_label_counts = [0 for _ in range(len(class_names))]
    for label in train_ds.labels:
        train_label_counts[label] += 1

    model = create_model_from_config(config, feature_info, class_names).to(device)
    optimizer = create_optimizer(config.optimizer_name, model.parameters(), config.lr, config.weight_decay)
    criterion = create_loss(
        config.loss_name,
        label_smoothing=config.label_smoothing,
        focal_gamma=config.focal_gamma,
        class_weights=build_class_weights(
            train_label_counts,
            strategy=config.class_weight_strategy,
            effective_num_beta=config.effective_num_beta,
        ),
    ).to(device)
    scheduler, scheduler_step_per_batch = create_scheduler(
        config.scheduler_name,
        optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_score = -float("inf")
    best_epoch = 0
    best_val_metrics: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []

    print(f"Aligned samples: total={len(cohort)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} split_source={split_source}")
    print(f"Selection metric: {selection_metric}")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch_model(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )
        val_metrics = evaluate_model(model, val_loader, device, criterion, class_names)
        selection_score, used_metric = resolve_selection_score(val_metrics, selection_metric)

        if scheduler is not None and not scheduler_step_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(selection_score)
            else:
                scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_metric": selection_metric,
            "selection_metric_used": used_metric,
            "selection_score": _safe_float(selection_score),
        }
        epoch_log.update({f"val_{k}": v for k, v in build_epoch_log(val_metrics).items()})
        history.append(epoch_log)

        print(
            f"[Epoch {epoch}/{config.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={_format_metric(_safe_float(val_metrics.get('loss')))} "
            f"val_acc={_format_metric(_safe_float(val_metrics.get('accuracy')))} "
            f"val_auc={_format_metric(_safe_float(val_metrics.get('auroc')))} "
            f"val_bacc={_format_metric(_safe_float(val_metrics.get('balanced_accuracy')))} "
            f"val_f1={_format_metric(_safe_float(val_metrics.get('f1')))} "
            f"monitor({used_metric})={_format_metric(_safe_float(val_metrics.get(used_metric)))} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if best_val_metrics is None or selection_score > best_val_score:
            best_val_score = selection_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), config.output_dir / "best_model.pt")
            print(f"  保存最佳模型 (epoch {epoch})")

    model.load_state_dict(torch.load(config.output_dir / "best_model.pt", map_location=device))
    train_metrics = evaluate_model(model, train_eval_loader, device, criterion, class_names)
    val_metrics = evaluate_model(model, val_loader, device, criterion, class_names)
    test_metrics = evaluate_model(model, test_loader, device, criterion, class_names) if test_loader is not None else None

    if config.save_predictions:
        save_prediction_csv(config.output_dir / "train_predictions.csv", train_ds, predict_probabilities(model, train_eval_loader, device), class_names)
        save_prediction_csv(config.output_dir / "val_predictions.csv", val_ds, predict_probabilities(model, val_loader, device), class_names)
        if test_loader is not None and test_ds is not None:
            save_prediction_csv(config.output_dir / "test_predictions.csv", test_ds, predict_probabilities(model, test_loader, device), class_names)

    metrics = {
        "family": infer_run_family(config.modalities, is_student=False),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_names": class_names,
        "modalities": list(config.modalities),
        "modality_feature_dims": {
            "ct": int(model.ct_feature_dim),
            "text_num": int(len(feature_info.get("text_num_cols", []))),
            "text_emb": int(len(feature_info.get("text_emb_cols", []))),
            "text": int(model.text_feature_dim),
            "cnv": int(model.gene_feature_dim),
        },
        "split_source": split_source,
        "cohort_stats": {
            **cohort_stats,
            "num_total": int(len(cohort)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "label_distribution": {
                class_names[int(label)]: int(count)
                for label, count in sorted(cohort["label"].value_counts().to_dict().items())
            },
        },
        "history": history,
        "config": jsonable_dataclass(config),
    }
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Multimodal training complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val AUROC: {_safe_float((best_val_metrics or {}).get('auroc'))}")
    if test_metrics is not None:
        print(f"Test AUROC: {_safe_float(test_metrics.get('auroc'))}")
    print(f"Outputs saved to: {config.output_dir}")
    print("=" * 60)
    return metrics


CONFIG_PATH_KEYS = {
    "data_root",
    "metadata_csv",
    "output_dir",
    "reference_manifest",
    "ct_root",
    "gene_tsv",
    "text_feature_tsv",
    "text_health_csv",
    "text_disease_csv",
    "bert_model_path",
    "text_cache_tsv",
    "teacher_run_dir",
}


def config_from_dict(raw: Dict[str, Any], config_cls: type[MultiModalTrainConfig] = MultiModalTrainConfig) -> MultiModalTrainConfig:
    data = dict(raw or {})
    for key in CONFIG_PATH_KEYS:
        if key in data and data[key] not in {None, ""}:
            data[key] = Path(data[key])
        elif key in data and data[key] in {None, ""}:
            data[key] = None
    if "modalities" in data:
        data["modalities"] = normalize_modalities(data["modalities"])
    field_names = set(config_cls.__dataclass_fields__.keys())
    filtered = {key: value for key, value in data.items() if key in field_names}
    return config_cls(**filtered)


def inherit_paths_from_teacher(
    student_config: StudentKDConfig,
    teacher_config: MultiModalTrainConfig,
) -> StudentKDConfig:
    inherited = StudentKDConfig(**{**asdict(student_config)})
    for key in [
        "data_root",
        "metadata_csv",
        "reference_manifest",
        "ct_root",
        "gene_tsv",
        "text_feature_tsv",
        "text_health_csv",
        "text_disease_csv",
        "bert_model_path",
        "text_cache_tsv",
    ]:
        if getattr(inherited, key) is None:
            setattr(inherited, key, getattr(teacher_config, key))
    return inherited


def load_teacher_bundle(
    teacher_run_dir: Path,
) -> Tuple[Dict[str, Any], MultiModalTrainConfig, Tuple[str, ...], Path]:
    metrics_path = teacher_run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Teacher metrics not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        teacher_metrics = json.load(f)
    teacher_config = config_from_dict(teacher_metrics.get("config", {}), MultiModalTrainConfig)
    teacher_modalities = normalize_modalities(teacher_metrics.get("modalities", teacher_config.modalities))
    teacher_config.modalities = teacher_modalities
    return teacher_metrics, teacher_config, teacher_modalities, metrics_path


def train_student_kd(config: StudentKDConfig) -> Dict[str, Any]:
    if config.teacher_run_dir is None:
        raise ValueError("teacher_run_dir is required for student KD training.")

    set_seed(config.seed)
    config.modalities = normalize_modalities(config.modalities)
    if "ct" not in config.modalities or "cnv" in config.modalities:
        raise ValueError("Student modalities must be `ct` or `ct,text`.")

    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    teacher_metrics, teacher_config, teacher_modalities, _ = load_teacher_bundle(config.teacher_run_dir)
    config = inherit_paths_from_teacher(config, teacher_config)
    required_modalities = normalize_modalities(list(teacher_modalities) + list(config.modalities))

    print("=" * 60)
    print("Student KD training")
    print(f"Teacher dir: {config.teacher_run_dir}")
    print(f"Teacher modalities: {','.join(teacher_modalities)}")
    print(f"Student modalities: {','.join(config.modalities)}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 60)

    cohort, feature_info, class_names, cohort_stats = build_multimodal_cohort(config, required_modalities)
    teacher_manifest = config.teacher_run_dir / "split_manifest.csv"
    if config.reference_manifest is not None:
        cohort = filter_cohort_to_manifest(cohort, config.reference_manifest)
        train_idx, val_idx, test_idx = load_split_manifest_indices(cohort, config.reference_manifest)
        split_source = "reference_manifest"
    elif teacher_manifest.exists():
        cohort = filter_cohort_to_manifest(cohort, teacher_manifest)
        train_idx, val_idx, test_idx = load_split_manifest_indices(cohort, teacher_manifest)
        split_source = "teacher_manifest"
    else:
        train_idx, val_idx, test_idx, split_source = split_cohort(cohort, teacher_config)
        split_source = f"{split_source}_teacher_config"

    if not train_idx:
        raise RuntimeError("Training split is empty after aligning teacher/student cohort.")
    save_split_manifest(config.output_dir / "split_manifest.csv", cohort, train_idx, val_idx, test_idx)

    selection_metric = resolve_selection_metric(config.selection_metric, len(class_names))
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, train_eval_loader = prepare_dataloaders(
        config,
        cohort,
        feature_info,
        required_modalities,
        (train_idx, val_idx, test_idx),
    )
    train_label_counts = [0 for _ in range(len(class_names))]
    for label in train_ds.labels:
        train_label_counts[label] += 1

    teacher_model = create_model_from_config(teacher_config, feature_info, class_names).to(device)
    teacher_ckpt = config.teacher_run_dir / "best_model.pt"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")
    teacher_model.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher_model.eval()

    student_model = create_model_from_config(config, feature_info, class_names).to(device)
    optimizer = create_optimizer(config.optimizer_name, student_model.parameters(), config.lr, config.weight_decay)
    criterion = create_loss(
        config.loss_name,
        label_smoothing=config.label_smoothing,
        focal_gamma=config.focal_gamma,
        class_weights=build_class_weights(
            train_label_counts,
            strategy=config.class_weight_strategy,
            effective_num_beta=config.effective_num_beta,
        ),
    ).to(device)
    scheduler, scheduler_step_per_batch = create_scheduler(
        config.scheduler_name,
        optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_score = -float("inf")
    best_epoch = 0
    best_val_metrics: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []

    print(f"Aligned samples: total={len(cohort)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} split_source={split_source}")
    print(f"Selection metric: {selection_metric}")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch_student_kd(
            student_model,
            teacher_model,
            train_loader,
            device,
            criterion,
            optimizer,
            alpha=config.distillation_alpha,
            temperature=config.distillation_temperature,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )
        val_metrics = evaluate_model(student_model, val_loader, device, criterion, class_names)
        selection_score, used_metric = resolve_selection_score(val_metrics, selection_metric)

        if scheduler is not None and not scheduler_step_per_batch:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(selection_score)
            else:
                scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_metric": selection_metric,
            "selection_metric_used": used_metric,
            "selection_score": _safe_float(selection_score),
            "distillation_alpha": config.distillation_alpha,
            "distillation_temperature": config.distillation_temperature,
        }
        epoch_log.update({f"val_{k}": v for k, v in build_epoch_log(val_metrics).items()})
        history.append(epoch_log)

        print(
            f"[Epoch {epoch}/{config.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={_format_metric(_safe_float(val_metrics.get('loss')))} "
            f"val_acc={_format_metric(_safe_float(val_metrics.get('accuracy')))} "
            f"val_auc={_format_metric(_safe_float(val_metrics.get('auroc')))} "
            f"val_bacc={_format_metric(_safe_float(val_metrics.get('balanced_accuracy')))} "
            f"val_f1={_format_metric(_safe_float(val_metrics.get('f1')))} "
            f"monitor({used_metric})={_format_metric(_safe_float(val_metrics.get(used_metric)))} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if best_val_metrics is None or selection_score > best_val_score:
            best_val_score = selection_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(student_model.state_dict(), config.output_dir / "best_model.pt")
            print(f"  保存最佳学生模型 (epoch {epoch})")

    student_model.load_state_dict(torch.load(config.output_dir / "best_model.pt", map_location=device))
    train_metrics = evaluate_model(student_model, train_eval_loader, device, criterion, class_names)
    val_metrics = evaluate_model(student_model, val_loader, device, criterion, class_names)
    test_metrics = evaluate_model(student_model, test_loader, device, criterion, class_names) if test_loader is not None else None

    if config.save_predictions:
        save_prediction_csv(
            config.output_dir / "train_predictions.csv",
            train_ds,
            predict_probabilities(student_model, train_eval_loader, device),
            class_names,
        )
        save_prediction_csv(
            config.output_dir / "val_predictions.csv",
            val_ds,
            predict_probabilities(student_model, val_loader, device),
            class_names,
        )
        if test_loader is not None and test_ds is not None:
            save_prediction_csv(
                config.output_dir / "test_predictions.csv",
                test_ds,
                predict_probabilities(student_model, test_loader, device),
                class_names,
            )

    metrics = {
        "family": infer_run_family(config.modalities, is_student=True),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_names": class_names,
        "modalities": list(config.modalities),
        "teacher_modalities": list(teacher_modalities),
        "teacher_run_dir": str(config.teacher_run_dir),
        "teacher_family": teacher_metrics.get("family"),
        "distillation_alpha": config.distillation_alpha,
        "distillation_temperature": config.distillation_temperature,
        "modality_feature_dims": {
            "ct": int(student_model.ct_feature_dim),
            "text_num": int(len(feature_info.get("text_num_cols", []))),
            "text_emb": int(len(feature_info.get("text_emb_cols", []))),
            "text": int(student_model.text_feature_dim),
            "cnv": int(student_model.gene_feature_dim),
        },
        "split_source": split_source,
        "cohort_stats": {
            **cohort_stats,
            "num_total": int(len(cohort)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "num_test": int(len(test_idx)),
            "label_distribution": {
                class_names[int(label)]: int(count)
                for label, count in sorted(cohort["label"].value_counts().to_dict().items())
            },
        },
        "history": history,
        "config": jsonable_dataclass(config),
        "teacher_config": jsonable_dataclass(teacher_config),
    }
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Student KD training complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val AUROC: {_safe_float((best_val_metrics or {}).get('auroc'))}")
    if test_metrics is not None:
        print(f"Test AUROC: {_safe_float(test_metrics.get('auroc'))}")
    print(f"Outputs saved to: {config.output_dir}")
    print("=" * 60)
    return metrics


def add_common_cli_args(parser: argparse.ArgumentParser, require_core_paths: bool) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=Path, required=require_core_paths, default=None, help="Project data root.")
    parser.add_argument("--metadata-csv", type=Path, required=require_core_paths, default=None, help="Metadata CSV path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--modalities", type=str, required=False, default="ct,text,cnv", help="Comma-separated modalities among ct,text,cnv.")
    parser.add_argument("--reference-manifest", type=Path, default=None, help="Optional split_manifest.csv used to align the cohort and split with another run.")
    parser.add_argument("--ct-root", type=Path, default=None, help="CT .npy root directory.")
    parser.add_argument("--gene-tsv", type=Path, default=None, help="CNV / gene TSV path.")
    parser.add_argument("--text-feature-tsv", type=Path, default=None, help="Prepared text feature TSV path.")
    parser.add_argument("--text-health-csv", type=Path, default=None, help="Healthy / control text CSV path.")
    parser.add_argument("--text-disease-csv", type=Path, default=None, help="Disease text CSV path.")
    parser.add_argument("--bert-model-path", type=Path, default=None, help="Local BERT model path for text embedding extraction.")
    parser.add_argument("--text-embedding-backend", type=str, choices=["bert", "hash"], default="bert")
    parser.add_argument("--text-hash-dim", type=int, default=128, help="Hash embedding dim for smoke tests or non-BERT fallback.")
    parser.add_argument("--text-batch-size", type=int, default=8, help="Batch size used when computing BERT embeddings.")
    parser.add_argument("--text-max-length", type=int, default=128, help="Tokenizer max length for BERT embeddings.")
    parser.add_argument("--text-cache-tsv", type=Path, default=None, help="Reusable cached prepared text feature TSV path.")
    parser.add_argument("--label-col", type=str, default="样本类型", help="Metadata label column.")
    parser.add_argument("--metadata-sample-id-col", type=str, default="SampleID", help="Metadata sample ID column.")
    parser.add_argument("--metadata-text-id-col", type=str, default="record_id", help="Metadata text record ID column.")
    parser.add_argument("--split-col", type=str, default="CT_train_val_split", help="Metadata split column.")
    parser.add_argument("--ct-path-col", type=str, default="CT_numpy_cloud路径", help="Metadata CT relative path column.")
    parser.add_argument("--text-record-id-col", type=str, default="record_id", help="Prepared text feature record ID column.")
    parser.add_argument("--gene-id-col", type=str, default=None, help="Gene TSV ID column.")
    parser.add_argument("--gene-label-col", type=str, default=None, help="Gene TSV label column to drop from features.")
    parser.add_argument("--class-mode", type=str, choices=["multiclass", "binary"], default="binary")
    parser.add_argument("--binary-task", type=str, choices=["malignant_vs_rest", "abnormal_vs_normal", "malignant_vs_normal"], default="malignant_vs_normal")
    parser.add_argument("--selection-metric", type=str, choices=["auto", "accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "loss"], default="auto")
    parser.add_argument("--split-mode", type=str, choices=["train_val", "train_val_test"], default="train_val_test")
    parser.add_argument("--use-predefined-split", action="store_true", help="Use metadata split labels when available.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ct-model", type=str, default="resnet3d18", help="CT backbone name from the existing model registry.")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--disable-3d-input", action="store_true", help="Use 2D middle-slice CT input instead of 3D volume input.")
    parser.add_argument("--depth-size", type=int, default=32)
    parser.add_argument("--volume-hw", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--aug-profile", type=str, choices=["basic", "strong"], default="strong")
    parser.add_argument("--ct-feature-dim", type=int, default=128)
    parser.add_argument("--text-feature-dim", type=int, default=256)
    parser.add_argument("--gene-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "onecycle", "plateau"], default="cosine")
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--sampling-strategy", type=str, choices=["default", "weighted"], default="weighted")
    parser.add_argument("--class-weight-strategy", type=str, choices=["none", "inverse", "sqrt_inverse", "effective_num"], default="effective_num")
    parser.add_argument("--effective-num-beta", type=float, default=0.999)
    parser.add_argument("--no-save-predictions", action="store_true", help="Disable prediction CSV export.")
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train text / multimodal teacher models with CT, CNV, and text-clinical inputs.")
    return add_common_cli_args(parser, require_core_paths=True)


def build_student_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CT or CT+text student models with knowledge distillation from a multimodal teacher.")
    add_common_cli_args(parser, require_core_paths=False)
    parser.add_argument("--teacher-run-dir", type=Path, required=True, help="Teacher run directory containing best_model.pt and metrics.json.")
    parser.add_argument("--distillation-alpha", type=float, default=0.5, help="KD loss weight.")
    parser.add_argument("--distillation-temperature", type=float, default=4.0, help="KD temperature.")
    return parser


def parse_train_args() -> MultiModalTrainConfig:
    args = build_train_parser().parse_args()
    return MultiModalTrainConfig(
        data_root=args.data_root,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        modalities=normalize_modalities(args.modalities),
        reference_manifest=args.reference_manifest,
        ct_root=args.ct_root,
        gene_tsv=args.gene_tsv,
        text_feature_tsv=args.text_feature_tsv,
        text_health_csv=args.text_health_csv,
        text_disease_csv=args.text_disease_csv,
        bert_model_path=args.bert_model_path,
        text_embedding_backend=args.text_embedding_backend,
        text_hash_dim=args.text_hash_dim,
        text_batch_size=args.text_batch_size,
        text_max_length=args.text_max_length,
        text_cache_tsv=args.text_cache_tsv,
        label_col=args.label_col,
        metadata_sample_id_col=args.metadata_sample_id_col,
        metadata_text_id_col=args.metadata_text_id_col,
        split_col=args.split_col,
        ct_path_col=args.ct_path_col,
        text_record_id_col=args.text_record_id_col,
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
        cpu=args.cpu,
        ct_model=args.ct_model,
        pretrained=args.pretrained,
        use_3d_input=not args.disable_3d_input,
        depth_size=args.depth_size,
        volume_hw=args.volume_hw,
        image_size=args.image_size,
        aug_profile=args.aug_profile,
        ct_feature_dim=args.ct_feature_dim,
        text_feature_dim=args.text_feature_dim,
        gene_hidden_dim=args.gene_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        sampling_strategy=args.sampling_strategy,
        class_weight_strategy=args.class_weight_strategy,
        effective_num_beta=args.effective_num_beta,
        save_predictions=not args.no_save_predictions,
    )


def parse_student_args() -> StudentKDConfig:
    args = build_student_parser().parse_args()
    return StudentKDConfig(
        data_root=args.data_root,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        modalities=normalize_modalities(args.modalities),
        reference_manifest=args.reference_manifest,
        ct_root=args.ct_root,
        gene_tsv=args.gene_tsv,
        text_feature_tsv=args.text_feature_tsv,
        text_health_csv=args.text_health_csv,
        text_disease_csv=args.text_disease_csv,
        bert_model_path=args.bert_model_path,
        text_embedding_backend=args.text_embedding_backend,
        text_hash_dim=args.text_hash_dim,
        text_batch_size=args.text_batch_size,
        text_max_length=args.text_max_length,
        text_cache_tsv=args.text_cache_tsv,
        label_col=args.label_col,
        metadata_sample_id_col=args.metadata_sample_id_col,
        metadata_text_id_col=args.metadata_text_id_col,
        split_col=args.split_col,
        ct_path_col=args.ct_path_col,
        text_record_id_col=args.text_record_id_col,
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
        cpu=args.cpu,
        ct_model=args.ct_model,
        pretrained=args.pretrained,
        use_3d_input=not args.disable_3d_input,
        depth_size=args.depth_size,
        volume_hw=args.volume_hw,
        image_size=args.image_size,
        aug_profile=args.aug_profile,
        ct_feature_dim=args.ct_feature_dim,
        text_feature_dim=args.text_feature_dim,
        gene_hidden_dim=args.gene_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        sampling_strategy=args.sampling_strategy,
        class_weight_strategy=args.class_weight_strategy,
        effective_num_beta=args.effective_num_beta,
        save_predictions=not args.no_save_predictions,
        teacher_run_dir=args.teacher_run_dir,
        distillation_alpha=args.distillation_alpha,
        distillation_temperature=args.distillation_temperature,
    )


def main_train() -> None:
    config = parse_train_args()
    train_multimodal_model(config)


def main_student_kd() -> None:
    config = parse_student_args()
    train_student_kd(config)


def main() -> None:
    main_train()


if __name__ == "__main__":
    main()
