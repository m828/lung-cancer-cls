from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class LIDCCropConfig:
    hu_min: float = -1000.0
    hu_max: float = 400.0
    target_depth: int = 32
    target_hw: int = 128
    context_scale: float = 1.5
    min_size_xy: int = 32
    min_size_z: int = 8
    z_margin_mm: float = 1.5


def load_dicom_series(series_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not dicom_names:
        raise RuntimeError(f"No DICOM files found under {series_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # [D,H,W]

    z_coords = []
    for z_idx in range(volume.shape[0]):
        point = image.TransformIndexToPhysicalPoint((0, 0, z_idx))
        z_coords.append(float(point[2]))
    return volume, np.asarray(z_coords, dtype=np.float32)


def normalize_hu(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def find_z_index_range(
    z_coords: np.ndarray,
    z_min_mm: float,
    z_max_mm: float,
    z_center_mm: float,
    min_size_z: int,
    z_margin_mm: float,
) -> tuple[int, int]:
    if z_coords.ndim != 1 or len(z_coords) == 0:
        raise ValueError("z_coords must be a non-empty 1D array")

    low = min(z_min_mm, z_max_mm) - z_margin_mm
    high = max(z_min_mm, z_max_mm) + z_margin_mm
    mask = ((z_coords >= low) & (z_coords <= high)) | ((z_coords <= low) & (z_coords >= high))
    candidate_idx = np.where(mask)[0]

    if candidate_idx.size > 0:
        start = int(candidate_idx.min())
        end = int(candidate_idx.max()) + 1
    else:
        center_idx = int(np.argmin(np.abs(z_coords - float(z_center_mm))))
        half = max(1, int(np.ceil(min_size_z / 2.0)))
        start = center_idx - half
        end = center_idx + half

    if end - start < min_size_z:
        extra = int(np.ceil((min_size_z - (end - start)) / 2.0))
        start -= extra
        end += extra

    start = max(0, start)
    end = min(len(z_coords), end)
    if end - start < min_size_z:
        if start == 0:
            end = min(len(z_coords), min_size_z)
        elif end == len(z_coords):
            start = max(0, len(z_coords) - min_size_z)
    return int(start), int(end)


def compute_crop_bounds(
    row: Dict[str, Any] | pd.Series,
    volume_shape: Tuple[int, int, int],
    z_coords: np.ndarray,
    config: LIDCCropConfig,
) -> tuple[slice, slice, slice]:
    depth, height, width = volume_shape

    x_min = float(row["x_min"])
    x_max = float(row["x_max"])
    y_min = float(row["y_min"])
    y_max = float(row["y_max"])
    z_min = float(row["z_min"])
    z_max = float(row["z_max"])
    z_center = float(row["z_center"])

    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    width_px = max(1.0, x_max - x_min + 1.0)
    height_px = max(1.0, y_max - y_min + 1.0)
    crop_hw = int(np.ceil(max(width_px, height_px) * config.context_scale))
    crop_hw = max(crop_hw, int(config.min_size_xy))
    half_hw = crop_hw / 2.0

    x0 = _safe_int(np.floor(center_x - half_hw))
    x1 = _safe_int(np.ceil(center_x + half_hw))
    y0 = _safe_int(np.floor(center_y - half_hw))
    y1 = _safe_int(np.ceil(center_y + half_hw))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)

    if x1 - x0 < config.min_size_xy:
        deficit = config.min_size_xy - (x1 - x0)
        left = deficit // 2
        right = deficit - left
        x0 = max(0, x0 - left)
        x1 = min(width, x1 + right)
    if y1 - y0 < config.min_size_xy:
        deficit = config.min_size_xy - (y1 - y0)
        top = deficit // 2
        bottom = deficit - top
        y0 = max(0, y0 - top)
        y1 = min(height, y1 + bottom)

    z0, z1 = find_z_index_range(
        z_coords,
        z_min_mm=z_min,
        z_max_mm=z_max,
        z_center_mm=z_center,
        min_size_z=int(config.min_size_z),
        z_margin_mm=float(config.z_margin_mm),
    )
    z0 = max(0, z0)
    z1 = min(depth, z1)
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def resize_volume(volume: np.ndarray, target_depth: int, target_hw: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume.shape}")
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(target_depth, target_hw, target_hw),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def crop_and_preprocess_volume(
    volume: np.ndarray,
    z_coords: np.ndarray,
    row: Dict[str, Any] | pd.Series,
    config: LIDCCropConfig,
) -> np.ndarray:
    crop_slices = compute_crop_bounds(row, volume.shape, z_coords, config)
    cropped = volume[crop_slices]
    normalized = normalize_hu(cropped, config.hu_min, config.hu_max)
    return resize_volume(normalized, config.target_depth, config.target_hw)


def sanitize_sample_id(sample_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in sample_id)


def load_split_lookup(split_manifest_csv: Path | None, split_fold: int | None) -> Dict[str, Dict[str, Any]]:
    if split_manifest_csv is None:
        return {}

    split_df = pd.read_csv(split_manifest_csv)
    if "sample_id" not in split_df.columns or "split" not in split_df.columns:
        raise RuntimeError("split_manifest.csv must include sample_id and split columns")

    if "fold" in split_df.columns:
        if split_fold is None:
            sample_to_folds = split_df.groupby("sample_id")["fold"].nunique()
            if (sample_to_folds > 1).any():
                raise RuntimeError(
                    "split_manifest.csv contains multiple folds per sample_id. "
                    "Please pass --split-fold to select one fold."
                )
        else:
            split_df = split_df.loc[split_df["fold"] == split_fold].copy()
    elif split_fold is not None:
        raise RuntimeError("Requested split_fold but split_manifest.csv has no fold column")

    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in split_df.iterrows():
        sample_id = str(row["sample_id"])
        record: Dict[str, Any] = {"sample_id": sample_id, "split": str(row["split"])}
        for column in ["fold", "patient_id", "nodule_id", "class_name", "label"]:
            if column in split_df.columns and not pd.isna(row[column]):
                record[column] = row[column]
        mapping[sample_id] = record
    return mapping


def resolve_series_dir(row: Dict[str, Any] | pd.Series, input_root: Path) -> Path:
    candidates: List[Path] = []

    for key in ["source_path", "patient_dir", "xml_path"]:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        candidate = Path(str(value))
        if key == "xml_path":
            candidate = candidate.parent
        candidates.append(candidate)

    patient_id = str(row.get("patient_id", "")).strip()
    if patient_id:
        candidates.append(input_root / patient_id)
        candidates.append(input_root / "LIDC-IDRI" / patient_id)
        candidates.append(input_root / "LIDC-IDRI" / "LIDC-IDRI" / patient_id)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve DICOM series directory for sample_id={row.get('sample_id')} "
        f"from source_path/patient_dir/xml_path under input_root={input_root}"
    )


def write_processed_manifests(
    output_root: Path,
    processed_rows: List[Dict[str, Any]],
    split_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Path]:
    processed_df = pd.DataFrame(processed_rows).sort_values(["class_name", "sample_id"]).reset_index(drop=True)
    processed_manifest_path = output_root / "processed_manifest.csv"
    processed_df.to_csv(processed_manifest_path, index=False)

    outputs = {"processed_manifest_csv": processed_manifest_path}
    if split_lookup:
        merged_rows: List[Dict[str, Any]] = []
        for row in processed_rows:
            split_meta = split_lookup.get(str(row["sample_id"]))
            if split_meta is None:
                continue
            merged_rows.append({**row, **split_meta})
        merged_df = pd.DataFrame(merged_rows).sort_values(
            ["fold", "split", "class_name", "sample_id"]
        ).reset_index(drop=True)
        split_out = output_root / "processed_split_manifest.csv"
        merged_df.to_csv(split_out, index=False)
        outputs["processed_split_manifest_csv"] = split_out
    return outputs


def preprocess_lidc_consensus_manifest(
    input_root: Path,
    manifest_csv: Path,
    output_root: Path,
    config: LIDCCropConfig,
    split_manifest_csv: Path | None = None,
    split_fold: int | None = None,
    limit: int | None = None,
) -> Dict[str, Path]:
    manifest_df = pd.read_csv(manifest_csv)
    if manifest_df.empty:
        raise RuntimeError(f"Manifest is empty: {manifest_csv}")

    required_columns = {
        "sample_id",
        "patient_id",
        "nodule_id",
        "class_name",
        "label",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "z_center",
    }
    missing = sorted(required_columns - set(manifest_df.columns))
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {missing}")

    split_lookup = load_split_lookup(split_manifest_csv, split_fold)
    if split_lookup:
        manifest_df = manifest_df.loc[manifest_df["sample_id"].astype(str).isin(split_lookup.keys())].copy()

    manifest_df = manifest_df.sort_values(["source_path", "sample_id"]).reset_index(drop=True)
    if limit is not None:
        manifest_df = manifest_df.head(limit).copy()
    if manifest_df.empty:
        raise RuntimeError("No manifest rows remain after split/fold filtering")

    output_root.mkdir(parents=True, exist_ok=True)
    processed_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}

    cached_series_dir: Path | None = None
    cached_volume: np.ndarray | None = None
    cached_z_coords: np.ndarray | None = None

    for _, row in manifest_df.iterrows():
        sample_id = str(row["sample_id"])
        class_name = str(row["class_name"])
        try:
            series_dir = resolve_series_dir(row, input_root=input_root)
            if cached_series_dir != series_dir:
                cached_volume, cached_z_coords = load_dicom_series(series_dir)
                cached_series_dir = series_dir
            assert cached_volume is not None
            assert cached_z_coords is not None

            crop = crop_and_preprocess_volume(cached_volume, cached_z_coords, row, config)
            output_stem = sanitize_sample_id(sample_id)
            class_dir = output_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = class_dir / f"{output_stem}.npy"
            np.save(out_path, crop)

            processed_rows.append(
                {
                    "sample_id": sample_id,
                    "output_stem": output_stem,
                    "patient_id": str(row["patient_id"]),
                    "nodule_id": str(row["nodule_id"]),
                    "class_name": class_name,
                    "label": int(row["label"]),
                    "malignancy_score": float(row["malignancy_score"]) if "malignancy_score" in row else None,
                    "source_path": str(series_dir),
                    "output_path": str(out_path),
                    "relative_path": str(out_path.relative_to(output_root)),
                }
            )
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        except Exception as exc:  # pragma: no cover - exercised in real preprocessing
            failed_rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": str(row.get("patient_id", "")),
                    "class_name": class_name,
                    "source_path": str(row.get("source_path", "")),
                    "error": str(exc),
                }
            )

    if not processed_rows:
        raise RuntimeError("Preprocessing failed for every sample; inspect failures.csv for details.")

    outputs = write_processed_manifests(output_root, processed_rows, split_lookup)

    failures_path = output_root / "preprocess_failures.csv"
    pd.DataFrame(failed_rows).to_csv(failures_path, index=False)
    outputs["failures_csv"] = failures_path

    summary = {
        "input_root": str(input_root),
        "manifest_csv": str(manifest_csv),
        "split_manifest_csv": str(split_manifest_csv) if split_manifest_csv is not None else None,
        "split_fold": split_fold,
        "num_manifest_rows": int(len(manifest_df)),
        "num_processed": int(len(processed_rows)),
        "num_failed": int(len(failed_rows)),
        "class_counts": class_counts,
        "crop_config": asdict(config),
    }
    summary_path = output_root / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["summary_json"] = summary_path
    return outputs
