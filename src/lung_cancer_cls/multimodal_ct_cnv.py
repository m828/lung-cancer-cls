from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lung_cancer_cls.cnv_xgboost import load_gene_feature_table, resolve_label_config
from lung_cancer_cls.dataset import INTRANET_LABEL_MAP, get_default_transforms, get_default_volume_transforms
from lung_cancer_cls.model import build_model
from lung_cancer_cls.train import (
    build_epoch_log,
    compute_classification_metrics,
    resolve_selection_metric,
    resolve_selection_score,
    set_seed,
)
from lung_cancer_cls.training_components import build_class_weights, create_loss, create_optimizer, create_scheduler

UNSPECIFIED_SPLITS = {"", "none", "nan", "null", "unspecified"}


@dataclass
class CTCNVTrainConfig:
    data_root: Path
    metadata_csv: Path
    gene_tsv: Path
    output_dir: Path
    ct_root: Path | None = None
    label_col: str = "样本类型"
    metadata_gene_id_col: str = "SampleID"
    split_col: str = "CT_train_val_split"
    ct_path_col: str = "CT_numpy_cloud路径"
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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if np.isnan(value_f) or np.isinf(value_f):
        return None
    return value_f


def _format_metric(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


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


def build_ct_cnv_cohort(
    config: CTCNVTrainConfig,
) -> Tuple[pd.DataFrame, List[str], Dict[int, str], Dict[str, Any]]:
    gene_df, gene_id_col, feature_names, inferred_gene_label_col = load_gene_feature_table(
        config.gene_tsv,
        gene_id_col=config.gene_id_col,
        gene_label_col=config.gene_label_col,
    )
    config.gene_id_col = gene_id_col
    if config.gene_label_col is None:
        config.gene_label_col = inferred_gene_label_col

    df_meta = pd.read_csv(config.metadata_csv).fillna("")
    if df_meta.empty:
        raise RuntimeError(f"Metadata CSV is empty: {config.metadata_csv}")

    for col in [config.metadata_gene_id_col, config.label_col, config.ct_path_col]:
        if col not in df_meta.columns:
            raise ValueError(f"Required metadata column not found: {col}")

    ct_root = config.ct_root or config.data_root
    class_names, label_mapper = resolve_label_config(config.class_mode, config.binary_task)

    split_series = pd.Series([""] * len(df_meta), index=df_meta.index, dtype=object)
    if config.split_col in df_meta.columns:
        split_series = df_meta[config.split_col].map(_normalize_split)

    cohort = pd.DataFrame(
        {
            "gene_id": df_meta[config.metadata_gene_id_col].astype(str).str.strip(),
            "label_name": df_meta[config.label_col].astype(str).str.strip(),
            "ct_rel_path": df_meta[config.ct_path_col].astype(str).str.strip(),
            "split": split_series,
        }
    )
    cohort = cohort.loc[(cohort["gene_id"] != "") & (cohort["ct_rel_path"] != "")].copy()
    cohort["label_3class"] = cohort["label_name"].map(INTRANET_LABEL_MAP)
    cohort = cohort.dropna(subset=["label_3class"]).copy()
    cohort["label_3class"] = cohort["label_3class"].astype(int)
    cohort["label"] = cohort["label_3class"].map(label_mapper)
    cohort = cohort.dropna(subset=["label"]).copy()
    cohort["label"] = cohort["label"].astype(int)
    cohort["ct_path"] = cohort["ct_rel_path"].map(
        lambda rel: str((ct_root / rel.replace("\\", "/").lstrip("/")).resolve())
    )
    cohort["ct_exists"] = cohort["ct_path"].map(lambda path: Path(path).exists())
    missing_ct = int((~cohort["ct_exists"]).sum())
    cohort = cohort.loc[cohort["ct_exists"]].drop(columns=["ct_exists"]).copy()
    cohort = cohort.merge(gene_df, left_on="gene_id", right_on=gene_id_col, how="inner")
    if cohort.empty:
        raise RuntimeError("No aligned CT+CNV samples remain after merging metadata CSV with gene TSV.")

    kept_rows: List[pd.Series] = []
    duplicate_rows = 0
    label_conflicts = 0
    split_conflicts = 0
    ct_duplicates_collapsed = 0
    for _, group in cohort.groupby("gene_id", sort=False):
        duplicate_rows += max(0, len(group) - 1)
        if len(set(group["label"].tolist())) > 1:
            label_conflicts += 1
            continue
        split_values = {split for split in group["split"].tolist() if split not in UNSPECIFIED_SPLITS}
        if len(split_values) > 1:
            split_conflicts += 1
            continue
        ct_duplicates_collapsed += max(0, len(set(group["ct_path"].tolist())) - 1)
        row = group.iloc[0].copy()
        row["split"] = sorted(split_values)[0] if split_values else ""
        kept_rows.append(row)

    deduped = pd.DataFrame(kept_rows).reset_index(drop=True)
    if deduped.empty:
        raise RuntimeError("All aligned CT+CNV samples were dropped during deduplication.")

    stats = {
        "metadata_rows": int(len(df_meta)),
        "missing_ct_rows_dropped": missing_ct,
        "rows_after_alignment": int(len(cohort)),
        "rows_after_dedup": int(len(deduped)),
        "duplicate_rows_collapsed": int(duplicate_rows),
        "duplicate_gene_label_conflicts_dropped": int(label_conflicts),
        "duplicate_gene_split_conflicts_dropped": int(split_conflicts),
        "duplicate_ct_paths_collapsed": int(ct_duplicates_collapsed),
        "gene_id_col": gene_id_col,
        "gene_label_col": config.gene_label_col,
    }
    return deduped, feature_names, class_names, stats


def split_multimodal_cohort(
    cohort: pd.DataFrame,
    config: CTCNVTrainConfig,
) -> Tuple[List[int], List[int], List[int], str]:
    labels = cohort["label"].to_numpy(dtype=np.int64)
    all_indices = list(range(len(cohort)))

    if config.use_predefined_split:
        splits = cohort["split"].tolist()
        train_pool = [idx for idx, split in enumerate(splits) if split in {"train", ""}]
        val_idx = [idx for idx, split in enumerate(splits) if split == "val"]
        test_idx = [idx for idx, split in enumerate(splits) if split == "test"]

        if not train_pool and not val_idx and not test_idx:
            fallback = CTCNVTrainConfig(**{**asdict(config), "use_predefined_split": False})
            return split_multimodal_cohort(cohort, fallback)

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


class CTCNVDataset(Dataset):
    def __init__(
        self,
        cohort: pd.DataFrame,
        indices: Sequence[int],
        feature_names: Sequence[str],
        transform: Any = None,
        use_3d: bool = True,
        depth_size: int = 32,
        volume_hw: int = 128,
    ):
        subset = cohort.iloc[list(indices)].reset_index(drop=True).copy()
        self.sample_ids = subset["gene_id"].astype(str).tolist()
        self.label_names = subset["label_name"].astype(str).tolist()
        self.splits = subset["split"].astype(str).tolist()
        self.ct_paths = [Path(path) for path in subset["ct_path"].tolist()]
        self.labels = subset["label"].astype(int).tolist()
        self.gene_matrix = subset[list(feature_names)].to_numpy(dtype=np.float32, copy=True)
        self.transform = transform
        self.use_3d = use_3d
        self.depth_size = depth_size
        self.volume_hw = volume_hw

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
        ct_tensor = self._load_ct_tensor(self.ct_paths[idx])
        gene_tensor = torch.from_numpy(self.gene_matrix[idx])
        label = self.labels[idx]
        return ct_tensor, gene_tensor, label


class CTCNVFusionClassifier(nn.Module):
    def __init__(
        self,
        ct_model: str,
        num_classes: int,
        gene_dim: int,
        ct_feature_dim: int = 128,
        gene_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.3,
        pretrained: bool = False,
    ):
        super().__init__()
        self.ct_encoder = build_model(ct_model, num_classes=ct_feature_dim, pretrained=pretrained)
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_dim, gene_hidden_dim),
            nn.BatchNorm1d(gene_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gene_hidden_dim, gene_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(ct_feature_dim + gene_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, ct: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        ct_features = self.ct_encoder(ct)
        gene_features = self.gene_encoder(gene)
        fused = torch.cat([ct_features, gene_features], dim=1)
        return self.classifier(fused)


def evaluate_multimodal(
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
        for ct, gene, y in loader:
            ct = ct.to(device)
            gene = gene.to(device)
            y = y.to(device)
            logits = model(ct, gene)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_labels.append(y.detach().cpu())
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


def train_epoch_multimodal(
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

    for ct, gene, y in loader:
        ct = ct.to(device)
        gene = gene.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(ct, gene)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        running_loss += loss.item() * y.size(0)
        seen += y.size(0)

    return running_loss / max(seen, 1)


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for ct, gene, _ in loader:
            ct = ct.to(device)
            gene = gene.to(device)
            logits = model(ct, gene)
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def save_prediction_csv(
    path: Path,
    dataset: CTCNVDataset,
    probabilities: np.ndarray,
    class_names: Dict[int, str],
) -> None:
    df = pd.DataFrame(
        {
            "gene_id": dataset.sample_ids,
            "label_name": dataset.label_names,
            "label": dataset.labels,
            "split": dataset.splits,
            "prediction": probabilities.argmax(axis=1),
        }
    )
    for class_idx, class_name in class_names.items():
        df[f"prob_{class_name}"] = probabilities[:, class_idx]
    df.to_csv(path, index=False, encoding="utf-8-sig")


def train_ct_cnv_model(config: CTCNVTrainConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CT + CNV multimodal training")
    print(f"Metadata CSV: {config.metadata_csv}")
    print(f"Gene TSV: {config.gene_tsv}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 60)

    cohort, feature_names, class_names, cohort_stats = build_ct_cnv_cohort(config)
    train_idx, val_idx, test_idx, split_source = split_multimodal_cohort(cohort, config)
    if not train_idx:
        raise RuntimeError("Training split is empty.")

    num_classes = len(class_names)
    selection_metric = resolve_selection_metric(config.selection_metric, num_classes)
    print(
        f"Aligned samples: total={len(cohort)} train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)} split_source={split_source}"
    )

    train_tf, val_tf = get_default_transforms(None, config.image_size, aug_profile=config.aug_profile)
    train_vol_tf, val_vol_tf = get_default_volume_transforms(config.aug_profile)
    use_3d = bool(config.use_3d_input)

    train_ds = CTCNVDataset(
        cohort,
        train_idx,
        feature_names,
        transform=train_vol_tf if use_3d else train_tf,
        use_3d=use_3d,
        depth_size=config.depth_size,
        volume_hw=config.volume_hw,
    )
    val_ds = CTCNVDataset(
        cohort,
        val_idx,
        feature_names,
        transform=val_vol_tf if use_3d else val_tf,
        use_3d=use_3d,
        depth_size=config.depth_size,
        volume_hw=config.volume_hw,
    )
    test_ds = None
    if test_idx:
        test_ds = CTCNVDataset(
            cohort,
            test_idx,
            feature_names,
            transform=val_vol_tf if use_3d else val_tf,
            use_3d=use_3d,
            depth_size=config.depth_size,
            volume_hw=config.volume_hw,
        )

    train_label_counts = [0 for _ in range(num_classes)]
    for label in train_ds.labels:
        train_label_counts[label] += 1

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    if config.sampling_strategy == "weighted":
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
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = CTCNVFusionClassifier(
        ct_model=config.ct_model,
        num_classes=num_classes,
        gene_dim=len(feature_names),
        ct_feature_dim=config.ct_feature_dim,
        gene_hidden_dim=config.gene_hidden_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        dropout=config.dropout,
        pretrained=config.pretrained,
    ).to(device)
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

    print(f"CT model: {config.ct_model} (pretrained={config.pretrained})")
    print(f"Gene feature dim: {len(feature_names)}")
    print(f"Selection metric: {selection_metric}")
    print("-" * 60)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch_multimodal(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
        )
        val_metrics = evaluate_multimodal(model, val_loader, device, criterion, class_names)
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
    train_metrics = evaluate_multimodal(model, train_eval_loader, device, criterion, class_names)
    val_metrics = evaluate_multimodal(model, val_loader, device, criterion, class_names)
    test_metrics = evaluate_multimodal(model, test_loader, device, criterion, class_names) if test_loader is not None else None

    if config.save_predictions:
        save_prediction_csv(config.output_dir / "train_predictions.csv", train_ds, predict_probabilities(model, train_eval_loader, device), class_names)
        save_prediction_csv(config.output_dir / "val_predictions.csv", val_ds, predict_probabilities(model, val_loader, device), class_names)
        if test_loader is not None and test_ds is not None:
            save_prediction_csv(config.output_dir / "test_predictions.csv", test_ds, predict_probabilities(model, test_loader, device), class_names)

    metrics = {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_names": class_names,
        "feature_dim": int(len(feature_names)),
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
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("CT + CNV multimodal training complete")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val AUROC: {_safe_float((best_val_metrics or {}).get('auroc'))}")
    if test_metrics is not None:
        print(f"Test AUROC: {_safe_float(test_metrics.get('auroc'))}")
    print(f"Outputs saved to: {config.output_dir}")
    print("=" * 60)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CT + CNV multimodal fusion baseline.")
    parser.add_argument("--data-root", type=Path, required=True, help="Project data root.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Metadata CSV path.")
    parser.add_argument("--gene-tsv", type=Path, required=True, help="CNV / gene TSV path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--ct-root", type=Path, default=None, help="CT .npy root directory.")
    parser.add_argument("--label-col", type=str, default="样本类型", help="Metadata label column.")
    parser.add_argument("--metadata-gene-id-col", type=str, default="SampleID", help="Metadata sample / gene id column.")
    parser.add_argument("--split-col", type=str, default="CT_train_val_split", help="Metadata split column.")
    parser.add_argument("--ct-path-col", type=str, default="CT_numpy_cloud路径", help="Metadata CT path column.")
    parser.add_argument("--gene-id-col", type=str, default=None, help="Gene TSV ID column.")
    parser.add_argument("--gene-label-col", type=str, default=None, help="Gene TSV label column to drop from features.")
    parser.add_argument("--class-mode", type=str, choices=["multiclass", "binary"], default="binary")
    parser.add_argument("--binary-task", type=str, choices=["malignant_vs_rest", "abnormal_vs_normal", "malignant_vs_normal", "benign_vs_malignant"], default="malignant_vs_normal")
    parser.add_argument("--selection-metric", type=str, choices=["auto", "accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "loss"], default="auto")
    parser.add_argument("--split-mode", type=str, choices=["train_val", "train_val_test"], default="train_val_test")
    parser.add_argument("--use-predefined-split", action="store_true", help="Use metadata split labels when available.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ct-model", type=str, default="resnet3d18", help="CT backbone name from the existing model registry.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when supported by the CT backbone.")
    parser.add_argument("--disable-3d-input", action="store_true", help="Use 2D middle-slice CT input instead of 3D volume input.")
    parser.add_argument("--depth-size", type=int, default=32)
    parser.add_argument("--volume-hw", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--aug-profile", type=str, choices=["basic", "strong"], default="strong")
    parser.add_argument("--ct-feature-dim", type=int, default=128)
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


def parse_args() -> CTCNVTrainConfig:
    args = build_parser().parse_args()
    return CTCNVTrainConfig(
        data_root=args.data_root,
        metadata_csv=args.metadata_csv,
        gene_tsv=args.gene_tsv,
        output_dir=args.output_dir,
        ct_root=args.ct_root,
        label_col=args.label_col,
        metadata_gene_id_col=args.metadata_gene_id_col,
        split_col=args.split_col,
        ct_path_col=args.ct_path_col,
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


def main() -> None:
    config = parse_args()
    train_ct_cnv_model(config)


if __name__ == "__main__":
    main()
