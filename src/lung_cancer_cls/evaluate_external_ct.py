from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from lung_cancer_cls.dataset import (
    DatasetType,
    IntranetCTDataset,
    Sample,
    create_dataset,
    get_default_transforms,
    get_default_volume_transforms,
)
from lung_cancer_cls.model import build_model
from lung_cancer_cls.multimodal_teacher_student import (
    MultiModalTrainConfig,
    create_model_from_config,
    jsonable_dataclass,
    move_inputs_to_device,
    normalize_modalities,
    config_from_dict,
)
from lung_cancer_cls.train import compute_classification_metrics, remap_samples_by_class_mode


@dataclass
class ExternalCTEvalConfig:
    run_dir: Path
    output_dir: Path
    data_root: Path
    intranet_source: str = "bundle"
    bundle_nm_path: Path | None = None
    bundle_bn_path: Path | None = None
    bundle_mt_path: Path | None = None
    class_mode: str | None = None
    binary_task: str | None = None
    batch_size: int = 8
    num_workers: int = 2
    cpu: bool = False


def _load_class_names(metrics: Dict[str, Any]) -> Dict[int, str]:
    raw = metrics.get("class_names", {})
    if not isinstance(raw, dict):
        return {}
    return {int(k): str(v) for k, v in raw.items()}


def _load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_binary_setup(eval_config: ExternalCTEvalConfig, run_metrics: Dict[str, Any]) -> Tuple[str, str]:
    config = run_metrics.get("config", {})
    class_mode = eval_config.class_mode or str(config.get("class_mode") or "binary")
    binary_task = eval_config.binary_task or str(config.get("binary_task") or "malignant_vs_normal")
    return class_mode, binary_task


def _build_bundle_dataset(
    eval_config: ExternalCTEvalConfig,
    use_3d_input: bool,
    depth_size: int,
    volume_hw: int,
    image_size: int,
    aug_profile: str,
    class_mode: str,
    binary_task: str,
) -> Tuple[IntranetCTDataset, Dict[int, str], List[Sample]]:
    dataset = create_dataset(
        DatasetType.INTRANET_CT,
        eval_config.data_root,
        intranet_source=eval_config.intranet_source,
        bundle_nm_path=eval_config.bundle_nm_path,
        bundle_bn_path=eval_config.bundle_bn_path,
        bundle_mt_path=eval_config.bundle_mt_path,
        use_3d=use_3d_input,
        depth_size=depth_size,
        volume_hw=volume_hw,
    )
    samples, class_names = remap_samples_by_class_mode(dataset.get_samples(), class_mode=class_mode, binary_task=binary_task)
    if use_3d_input:
        _, val_transform = get_default_volume_transforms(aug_profile)
    else:
        _, val_transform = get_default_transforms(DatasetType.INTRANET_CT, image_size=image_size, aug_profile=aug_profile)
    eval_dataset = IntranetCTDataset(
        samples=samples,
        transform=val_transform,
        use_3d=use_3d_input,
        depth_size=depth_size,
        volume_hw=volume_hw,
    )
    return eval_dataset, class_names, samples


def _load_multimodal_ct_model(run_metrics: Dict[str, Any], run_dir: Path, device: torch.device) -> nn.Module:
    config = config_from_dict(run_metrics.get("config", {}), MultiModalTrainConfig)
    modalities = normalize_modalities(run_metrics.get("modalities") or config.modalities)
    if modalities != ("ct",):
        raise ValueError(f"External CT bundle evaluation only supports ct-only runs, got modalities={modalities}")
    config.modalities = modalities
    class_names = _load_class_names(run_metrics)
    feature_info = {"gene_cols": [], "text_num_cols": [], "text_emb_cols": []}
    model = create_model_from_config(config, feature_info, class_names).to(device)
    ckpt_path = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def _load_plain_ct_model(run_metrics: Dict[str, Any], run_dir: Path, device: torch.device) -> nn.Module:
    config = run_metrics.get("config", {})
    class_names = _load_class_names(run_metrics)
    model_name = str(config.get("model") or config.get("ct_model") or "resnet3d18")
    model = build_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    ckpt_path = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    model.eval()
    return model


def _predict_ct_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_multimodal_ct: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    total = 0
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if is_multimodal_ct:
                logits = model(move_inputs_to_device({"ct": x}, device))
            else:
                logits = model(x)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            labels_all.append(y.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
    probabilities = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, 0), dtype=np.float32)
    labels = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,), dtype=np.int64)
    avg_loss = float(total_loss / max(total, 1)) if total > 0 else None
    return labels, probabilities if avg_loss is not None else probabilities


def _save_external_predictions(
    path: Path,
    samples: Sequence[Sample],
    probabilities: np.ndarray,
    class_names: Dict[int, str],
) -> None:
    rows: List[Dict[str, Any]] = []
    predictions = probabilities.argmax(axis=1) if probabilities.size else np.asarray([], dtype=np.int64)
    for idx, sample in enumerate(samples):
        row = {
            "sample_index": idx,
            "source_path": str(sample.image_path),
            "label": int(sample.label),
            "prediction": int(predictions[idx]) if idx < len(predictions) else None,
        }
        if sample.metadata:
            for key in ["bundle_path", "bundle_index", "split"]:
                if key in sample.metadata:
                    row[key] = sample.metadata[key]
        for class_idx, class_name in class_names.items():
            if probabilities.size:
                row[f"prob_{class_name}"] = float(probabilities[idx, class_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def evaluate_external_ct_bundle(eval_config: ExternalCTEvalConfig) -> Dict[str, Any]:
    run_metrics = _load_run_metrics(eval_config.run_dir)
    run_family = str(run_metrics.get("family") or "unknown")
    class_mode, binary_task = _resolve_binary_setup(eval_config, run_metrics)
    config = run_metrics.get("config", {})
    use_3d_input = bool(config.get("use_3d_input", True))
    depth_size = int(config.get("depth_size", 32))
    volume_hw = int(config.get("volume_hw", 128))
    image_size = int(config.get("image_size", 224))
    aug_profile = str(config.get("aug_profile") or "strong")

    device = torch.device("cuda" if torch.cuda.is_available() and not eval_config.cpu else "cpu")
    eval_config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset, discovered_class_names, samples = _build_bundle_dataset(
        eval_config=eval_config,
        use_3d_input=use_3d_input,
        depth_size=depth_size,
        volume_hw=volume_hw,
        image_size=image_size,
        aug_profile=aug_profile,
        class_mode=class_mode,
        binary_task=binary_task,
    )
    class_names = _load_class_names(run_metrics) or discovered_class_names
    loader = DataLoader(dataset, batch_size=eval_config.batch_size, shuffle=False, num_workers=eval_config.num_workers)

    is_multimodal_ct = "modalities" in config or run_family == "student_kd"
    if is_multimodal_ct:
        model = _load_multimodal_ct_model(run_metrics, eval_config.run_dir, device)
    else:
        model = _load_plain_ct_model(run_metrics, eval_config.run_dir, device)

    labels, probabilities = _predict_ct_model(model, loader, device, is_multimodal_ct=is_multimodal_ct)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.size == 0:
        raise RuntimeError("No probabilities were produced during external bundle evaluation.")
    loss = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-7, 1.0))))
    external_metrics = compute_classification_metrics(labels, probabilities, loss=loss, class_names=class_names)

    predictions_path = eval_config.output_dir / "external_bundle_predictions.csv"
    metrics_path = eval_config.output_dir / "external_bundle_metrics.json"
    _save_external_predictions(predictions_path, samples, probabilities, class_names)

    payload = {
        "source_run_dir": str(eval_config.run_dir),
        "source_family": run_family,
        "class_mode": class_mode,
        "binary_task": binary_task,
        "external_dataset": "intranet_bundle",
        "class_names": class_names,
        "num_samples": int(len(samples)),
        "metrics": external_metrics,
        "config": {
            **jsonable_dataclass(eval_config),
            "use_3d_input": use_3d_input,
            "depth_size": depth_size,
            "volume_hw": volume_hw,
            "image_size": image_size,
            "aug_profile": aug_profile,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"External bundle evaluation complete for {eval_config.run_dir.name}")
    print(f"AUROC: {external_metrics.get('auroc')}")
    print(f"Balanced accuracy: {external_metrics.get('balanced_accuracy')}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print("=" * 60)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained CT-only model on the bundle-based external CT dataset.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Trained run directory containing metrics.json and best_model.pt.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write external evaluation outputs.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory used by the bundle dataset loader.")
    parser.add_argument("--bundle-nm-path", type=Path, default=None, help="Path to NM_all.npy or directory.")
    parser.add_argument("--bundle-bn-path", type=Path, default=None, help="Path to BN_all.npy or directory.")
    parser.add_argument("--bundle-mt-path", type=Path, default=None, help="Path to MT_all.npy or directory.")
    parser.add_argument("--class-mode", type=str, choices=["multiclass", "binary"], default=None)
    parser.add_argument("--binary-task", type=str, choices=["malignant_vs_rest", "abnormal_vs_normal", "malignant_vs_normal", "benign_vs_malignant"], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser


def parse_args() -> ExternalCTEvalConfig:
    args = build_parser().parse_args()
    return ExternalCTEvalConfig(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        data_root=args.data_root,
        bundle_nm_path=args.bundle_nm_path,
        bundle_bn_path=args.bundle_bn_path,
        bundle_mt_path=args.bundle_mt_path,
        class_mode=args.class_mode,
        binary_task=args.binary_task,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cpu=args.cpu,
    )


def main() -> None:
    evaluate_external_ct_bundle(parse_args())


if __name__ == "__main__":
    main()
