from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


LABEL_TO_CLASS_DIR: Dict[str, str] = {
    "健康对照": "normal",
    "良性结节": "benign",
    "肺癌": "malignant",
    "normal": "normal",
    "benign": "benign",
    "malignant": "malignant",
}

LOCALIZER_TOKENS = (
    "localizer",
    "scout",
    "topogram",
    "定位",
    "定位片",
    "剂量",
    "dose",
)


@dataclass
class IntranetCTPreprocessConfig:
    output_root: Path
    manifest_out: Path
    qc_csv: Path
    summary_json: Path
    source_csv: Path | None = None
    input_roots: Tuple[str, ...] = ()
    dicom_path_col: str = "CT dicom路径"
    npy_path_col: str = "CT_numpy路径"
    npy_cloud_path_col: str = "CT_numpy_cloud路径"
    label_col: str = "样本类型"
    split_col: str = "CT_train_val_split"
    sample_id_col: str = "SampleID"
    root_split_mode: str = "blank"
    train_ratio: float = 0.8
    seed: int = 42
    target_depth: int = 128
    target_hw: int = 256
    hu_min: float = -1000.0
    hu_max: float = 400.0
    min_slices: int = 32
    min_hw: int = 128
    max_slice_thickness_mm: float | None = 5.0
    max_z_spacing_mm: float | None = 5.0
    max_xy_spacing_mm: float | None = 2.5
    scan_only: bool = False
    series_mode: str = "best"
    cloud_prefix: str = ""
    limit: int | None = None
    overwrite: bool = False


@dataclass
class CaseInput:
    case_id: str
    dicom_root: Path
    label_name: str
    split: str = ""
    source: str = "root"
    row_data: Dict[str, Any] | None = None


@dataclass
class SeriesRecord:
    case_id: str
    label_name: str
    source: str
    series_dir: Path
    series_id: str
    num_slices: int
    rows: int
    columns: int
    spacing_x: float | None
    spacing_y: float | None
    spacing_z: float | None
    slice_thickness: float | None
    modality: str
    body_part: str
    series_description: str
    protocol_name: str
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    study_date: str
    manufacturer: str
    scanner_model: str
    eligible: bool
    exclude_reasons: str


def sanitize_filename(value: str, fallback: str = "case") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", str(value)).strip("._-")
    return cleaned or fallback


def _safe_float(value: Any) -> float | None:
    try:
        value_f = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if np.isnan(value_f) or np.isinf(value_f):
        return None
    return value_f


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "pandasnan"} else text


def _normalize_split(value: Any) -> str:
    split = _safe_text(value).lower()
    if split in {"valid", "validation", "dev"}:
        return "val"
    return split if split in {"train", "val", "test"} else ""


def _parse_input_root_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH for --input-root, got: {spec}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if label not in LABEL_TO_CLASS_DIR:
        raise ValueError(f"Unsupported label for --input-root: {label}")
    return label, Path(raw_path.strip())


def _has_direct_dicom_series(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        import SimpleITK as sitk

        return bool(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or [])
    except Exception:
        return False


def _case_id_from_row(row: pd.Series, sample_id_col: str, fallback_path: Path, row_idx: int) -> str:
    if sample_id_col in row.index:
        sample_id = _safe_text(row.get(sample_id_col))
        if sample_id:
            return sample_id
    path_name = sanitize_filename(fallback_path.name, fallback=f"row_{row_idx:06d}")
    return path_name if path_name else f"row_{row_idx:06d}"


def build_case_inputs(config: IntranetCTPreprocessConfig) -> List[CaseInput]:
    cases: List[CaseInput] = []

    if config.source_csv is not None:
        df = pd.read_csv(config.source_csv).fillna("")
        for required in [config.dicom_path_col, config.label_col]:
            if required not in df.columns:
                raise ValueError(f"Required source CSV column not found: {required}")
        for row_idx, row in df.iterrows():
            label_name = _safe_text(row.get(config.label_col))
            if label_name not in LABEL_TO_CLASS_DIR:
                continue
            dicom_path_text = _safe_text(row.get(config.dicom_path_col))
            if not dicom_path_text:
                continue
            dicom_root = Path(dicom_path_text)
            cases.append(
                CaseInput(
                    case_id=_case_id_from_row(row, config.sample_id_col, dicom_root, row_idx),
                    dicom_root=dicom_root,
                    label_name=label_name,
                    split=_normalize_split(row.get(config.split_col)) if config.split_col in row.index else "",
                    source="csv",
                    row_data={str(k): row[k] for k in row.index},
                )
            )

    root_cases: List[CaseInput] = []
    for spec in config.input_roots:
        label_name, input_root = _parse_input_root_spec(spec)
        if input_root.is_dir():
            if _has_direct_dicom_series(input_root):
                candidate_dirs = [input_root]
            else:
                children = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
                candidate_dirs = children or [input_root]
        else:
            candidate_dirs = [input_root]
        for idx, case_dir in enumerate(candidate_dirs):
            root_cases.append(
                CaseInput(
                    case_id=sanitize_filename(case_dir.name, fallback=f"root_{idx:06d}"),
                    dicom_root=case_dir,
                    label_name=label_name,
                    split="",
                    source="root",
                    row_data=None,
                )
            )

    _assign_root_splits(root_cases, config)
    cases.extend(root_cases)

    if config.limit is not None:
        cases = cases[: config.limit]
    return cases


def _assign_root_splits(cases: Sequence[CaseInput], config: IntranetCTPreprocessConfig) -> None:
    mode = config.root_split_mode.lower().strip()
    if mode == "blank":
        return
    if mode == "train":
        for case in cases:
            case.split = "train"
        return
    if mode not in {"train_val", "train_val_test"}:
        raise ValueError(f"Unknown root_split_mode: {config.root_split_mode}")

    rng = random.Random(config.seed)
    by_label: Dict[str, List[CaseInput]] = {}
    for case in cases:
        by_label.setdefault(case.label_name, []).append(case)

    for label_cases in by_label.values():
        shuffled = list(label_cases)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(round(n_total * config.train_ratio))
        n_train = min(max(n_train, 1 if n_total > 1 else n_total), n_total)
        if mode == "train_val":
            for idx, case in enumerate(shuffled):
                case.split = "train" if idx < n_train else "val"
            continue

        remaining = n_total - n_train
        n_val = remaining // 2
        for idx, case in enumerate(shuffled):
            if idx < n_train:
                case.split = "train"
            elif idx < n_train + n_val:
                case.split = "val"
            else:
                case.split = "test"


def _iter_dirs(root: Path) -> Iterable[Path]:
    if root.is_dir():
        yield root
        for path in root.rglob("*"):
            if path.is_dir():
                yield path


def _get_reader_metadata(reader: Any, idx: int, tag: str) -> str:
    try:
        if reader.HasMetaDataKey(idx, tag):
            return reader.GetMetaData(idx, tag).strip()
    except Exception:
        return ""
    return ""


def _metadata_for_series(case: CaseInput, series_dir: Path, series_id: str, config: IntranetCTPreprocessConfig) -> SeriesRecord:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(str(series_dir), series_id)
    if not filenames:
        raise RuntimeError(f"No DICOM files found for series_id={series_id} under {series_dir}")
    reader.SetFileNames(filenames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    size = image.GetSize()  # [X, Y, Z]
    spacing = image.GetSpacing()

    modality = _get_reader_metadata(reader, 0, "0008|0060")
    body_part = _get_reader_metadata(reader, 0, "0018|0015")
    series_description = _get_reader_metadata(reader, 0, "0008|103e")
    protocol_name = _get_reader_metadata(reader, 0, "0018|1030")
    slice_thickness = _safe_float(_get_reader_metadata(reader, 0, "0018|0050"))
    patient_id = _get_reader_metadata(reader, 0, "0010|0020")
    study_uid = _get_reader_metadata(reader, 0, "0020|000d")
    series_uid = _get_reader_metadata(reader, 0, "0020|000e") or series_id
    study_date = _get_reader_metadata(reader, 0, "0008|0020")
    manufacturer = _get_reader_metadata(reader, 0, "0008|0070")
    scanner_model = _get_reader_metadata(reader, 0, "0008|1090")

    rows = int(size[1]) if len(size) >= 2 else 0
    columns = int(size[0]) if len(size) >= 1 else 0
    num_slices = int(size[2]) if len(size) >= 3 else len(filenames)
    spacing_x = float(spacing[0]) if len(spacing) >= 1 else None
    spacing_y = float(spacing[1]) if len(spacing) >= 2 else None
    spacing_z = float(spacing[2]) if len(spacing) >= 3 else None
    eligible, reasons = assess_series_quality(
        modality=modality,
        rows=rows,
        columns=columns,
        num_slices=num_slices,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        slice_thickness=slice_thickness,
        series_description=series_description,
        protocol_name=protocol_name,
        config=config,
    )

    return SeriesRecord(
        case_id=case.case_id,
        label_name=case.label_name,
        source=case.source,
        series_dir=series_dir,
        series_id=series_id,
        num_slices=num_slices,
        rows=rows,
        columns=columns,
        spacing_x=spacing_x,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        slice_thickness=slice_thickness,
        modality=modality,
        body_part=body_part,
        series_description=series_description,
        protocol_name=protocol_name,
        patient_id=patient_id,
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        study_date=study_date,
        manufacturer=manufacturer,
        scanner_model=scanner_model,
        eligible=eligible,
        exclude_reasons=";".join(reasons),
    )


def assess_series_quality(
    modality: str,
    rows: int,
    columns: int,
    num_slices: int,
    spacing_x: float | None,
    spacing_y: float | None,
    spacing_z: float | None,
    slice_thickness: float | None,
    series_description: str,
    protocol_name: str,
    config: IntranetCTPreprocessConfig,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if modality and modality.upper() != "CT":
        reasons.append(f"non_ct:{modality}")
    if num_slices < config.min_slices:
        reasons.append(f"few_slices:{num_slices}<min{config.min_slices}")
    if rows < config.min_hw or columns < config.min_hw:
        reasons.append(f"small_hw:{rows}x{columns}<min{config.min_hw}")
    if config.max_slice_thickness_mm is not None and slice_thickness is not None:
        if slice_thickness > config.max_slice_thickness_mm:
            reasons.append(f"thick_slice:{slice_thickness:g}>{config.max_slice_thickness_mm:g}")
    if config.max_z_spacing_mm is not None and spacing_z is not None:
        if spacing_z > config.max_z_spacing_mm:
            reasons.append(f"large_z_spacing:{spacing_z:g}>{config.max_z_spacing_mm:g}")
    if config.max_xy_spacing_mm is not None:
        if spacing_x is not None and spacing_x > config.max_xy_spacing_mm:
            reasons.append(f"large_x_spacing:{spacing_x:g}>{config.max_xy_spacing_mm:g}")
        if spacing_y is not None and spacing_y > config.max_xy_spacing_mm:
            reasons.append(f"large_y_spacing:{spacing_y:g}>{config.max_xy_spacing_mm:g}")

    series_text = f"{series_description} {protocol_name}".lower()
    if any(token in series_text for token in LOCALIZER_TOKENS):
        reasons.append("localizer_or_non_diagnostic_series")
    return not reasons, reasons


def discover_case_series(case: CaseInput, config: IntranetCTPreprocessConfig) -> List[SeriesRecord]:
    import SimpleITK as sitk

    if not case.dicom_root.exists():
        return []
    records: List[SeriesRecord] = []
    for series_dir in _iter_dirs(case.dicom_root):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_dir)) or []
        except Exception:
            continue
        for series_id in series_ids:
            try:
                records.append(_metadata_for_series(case, series_dir, series_id, config))
            except Exception:
                continue
    return records


def _series_preference(record: SeriesRecord) -> Tuple[int, int, float, float, int]:
    text = f"{record.body_part} {record.series_description} {record.protocol_name}".lower()
    clinical_hint = 0
    if any(token in text for token in ["lung", "chest", "thorax", "肺", "胸"]):
        clinical_hint += 100_000
    if any(token in text for token in ["thin", "1mm", "1.0", "薄"]):
        clinical_hint += 50_000
    z_spacing = record.spacing_z if record.spacing_z is not None else 999.0
    slice_thickness = record.slice_thickness if record.slice_thickness is not None else 999.0
    return (
        1 if record.eligible else 0,
        clinical_hint + record.num_slices,
        -float(z_spacing),
        -float(slice_thickness),
        record.rows * record.columns,
    )


def select_series(records: Sequence[SeriesRecord], series_mode: str = "best") -> List[SeriesRecord]:
    if not records:
        return []
    mode = series_mode.lower().strip()
    if mode == "all":
        return [record for record in records if record.eligible]
    if mode != "best":
        raise ValueError(f"Unknown series_mode: {series_mode}")
    eligible = [record for record in records if record.eligible]
    candidates = eligible or list(records)
    return [max(candidates, key=_series_preference)]


def load_dicom_series_volume(series_dir: Path, series_id: str) -> np.ndarray:
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    filenames = reader.GetGDCMSeriesFileNames(str(series_dir), series_id)
    if not filenames:
        raise RuntimeError(f"No DICOM files found for series_id={series_id} under {series_dir}")
    reader.SetFileNames(filenames)
    image = reader.Execute()
    return sitk.GetArrayFromImage(image).astype(np.float32)


def preprocess_volume(
    volume: np.ndarray,
    target_depth: int,
    target_hw: int,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    if volume.ndim == 2:
        volume = volume[None, ...]
    elif volume.ndim != 3:
        raise ValueError(f"Unsupported DICOM volume shape: {volume.shape}")

    clipped = np.clip(volume.astype(np.float32), hu_min, hu_max)
    normalized = (clipped - hu_min) / max(hu_max - hu_min, 1e-6)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(int(target_depth), int(target_hw), int(target_hw)),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _hash_record(record: SeriesRecord) -> str:
    raw = f"{record.series_dir.as_posix()}::{record.series_id}::{record.case_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _class_dir(label_name: str) -> str:
    if label_name not in LABEL_TO_CLASS_DIR:
        raise ValueError(f"Unsupported label: {label_name}")
    return LABEL_TO_CLASS_DIR[label_name]


def _output_path_for_record(record: SeriesRecord, config: IntranetCTPreprocessConfig, ordinal: int) -> Path:
    class_dir = _class_dir(record.label_name)
    uid_tail = sanitize_filename(record.series_instance_uid[-16:] if record.series_instance_uid else record.series_id)
    stem = sanitize_filename(f"{record.case_id}_{uid_tail}_{_hash_record(record)}")
    if ordinal > 0:
        stem = f"{stem}_{ordinal:02d}"
    return config.output_root / class_dir / f"{stem}.npy"


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _cloud_path(relative_path: str, cloud_prefix: str) -> str:
    prefix = cloud_prefix.strip().replace("\\", "/").strip("/")
    return f"{prefix}/{relative_path}" if prefix else relative_path


def _series_record_to_qc_row(case: CaseInput, record: SeriesRecord, selected: bool) -> Dict[str, Any]:
    row = asdict(record)
    row["series_dir"] = str(record.series_dir)
    row["dicom_root"] = str(case.dicom_root)
    row["case_split"] = case.split
    row["selected"] = bool(selected)
    return row


def _manifest_row(case: CaseInput, record: SeriesRecord, out_path: Path, config: IntranetCTPreprocessConfig) -> Dict[str, Any]:
    row = dict(case.row_data or {})
    rel_path = _relative_or_name(out_path, config.output_root)
    row[config.dicom_path_col] = str(case.dicom_root)
    row[config.npy_path_col] = str(out_path)
    row[config.npy_cloud_path_col] = _cloud_path(rel_path, config.cloud_prefix)
    row[config.label_col] = case.label_name
    row[config.split_col] = case.split
    row.setdefault(config.sample_id_col, case.case_id)
    row.update(
        {
            "preprocess_source": case.source,
            "preprocess_case_id": case.case_id,
            "preprocess_series_dir": str(record.series_dir),
            "preprocess_series_id": record.series_id,
            "preprocess_series_instance_uid": record.series_instance_uid,
            "preprocess_num_slices": record.num_slices,
            "preprocess_rows": record.rows,
            "preprocess_columns": record.columns,
            "preprocess_spacing_x": record.spacing_x,
            "preprocess_spacing_y": record.spacing_y,
            "preprocess_spacing_z": record.spacing_z,
            "preprocess_slice_thickness": record.slice_thickness,
            "preprocess_target_shape": f"{config.target_depth},{config.target_hw},{config.target_hw}",
            "preprocess_hu_window": f"{config.hu_min},{config.hu_max}",
        }
    )
    return row


def process_intranet_ct(config: IntranetCTPreprocessConfig) -> Dict[str, Any]:
    cases = build_case_inputs(config)
    if not cases:
        raise RuntimeError("No input cases were found from --source-csv or --input-root.")

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    config.qc_csv.parent.mkdir(parents=True, exist_ok=True)
    config.summary_json.parent.mkdir(parents=True, exist_ok=True)

    qc_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    skipped_cases: List[Dict[str, Any]] = []
    converted = 0

    for case_idx, case in enumerate(cases, start=1):
        print(f"[{case_idx}/{len(cases)}] {case.label_name} {case.case_id}: {case.dicom_root}")
        records = discover_case_series(case, config)
        selected_records = select_series(records, config.series_mode)
        selected_ids = {(record.series_dir, record.series_id) for record in selected_records}

        for record in records:
            qc_rows.append(_series_record_to_qc_row(case, record, (record.series_dir, record.series_id) in selected_ids))

        if not selected_records:
            skipped_cases.append(
                {
                    "case_id": case.case_id,
                    "dicom_root": str(case.dicom_root),
                    "label_name": case.label_name,
                    "reason": "no_eligible_series" if records else "no_dicom_series",
                }
            )
            continue

        for ordinal, record in enumerate(selected_records):
            if not record.eligible:
                skipped_cases.append(
                    {
                        "case_id": case.case_id,
                        "dicom_root": str(case.dicom_root),
                        "label_name": case.label_name,
                        "series_dir": str(record.series_dir),
                        "series_id": record.series_id,
                        "reason": f"selected_ineligible:{record.exclude_reasons}",
                    }
                )
                continue

            out_path = _output_path_for_record(record, config, ordinal)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not config.scan_only:
                if out_path.exists() and not config.overwrite:
                    raise FileExistsError(f"Output exists. Use --overwrite to replace: {out_path}")
                volume = load_dicom_series_volume(record.series_dir, record.series_id)
                processed = preprocess_volume(
                    volume,
                    target_depth=config.target_depth,
                    target_hw=config.target_hw,
                    hu_min=config.hu_min,
                    hu_max=config.hu_max,
                )
                np.save(out_path, processed)
            manifest_rows.append(_manifest_row(case, record, out_path, config))
            converted += 1

    qc_df = pd.DataFrame(qc_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    skipped_df = pd.DataFrame(skipped_cases)

    qc_df.to_csv(config.qc_csv, index=False, encoding="utf-8-sig")
    manifest_df.to_csv(config.manifest_out, index=False, encoding="utf-8-sig")
    skipped_path = config.summary_json.with_name(config.summary_json.stem + "_skipped_cases.csv")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    label_distribution = (
        manifest_df[config.label_col].value_counts().to_dict()
        if not manifest_df.empty and config.label_col in manifest_df.columns
        else {}
    )
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_cases": int(len(cases)),
        "num_qc_series": int(len(qc_df)),
        "num_manifest_rows": int(len(manifest_df)),
        "num_skipped_cases": int(len(skipped_df)),
        "scan_only": bool(config.scan_only),
        "converted_or_planned": int(converted),
        "label_distribution": {str(key): int(value) for key, value in label_distribution.items()},
        "qc_csv": str(config.qc_csv),
        "manifest_csv": str(config.manifest_out),
        "skipped_cases_csv": str(skipped_path),
    }
    config.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Intranet CT DICOM preprocessing complete")
    print(f"Cases: {len(cases)}")
    print(f"QC series: {len(qc_df)}")
    print(f"Manifest rows: {len(manifest_df)}")
    print(f"Skipped cases: {len(skipped_df)}")
    print(f"QC CSV: {config.qc_csv}")
    print(f"Manifest CSV: {config.manifest_out}")
    print(f"Summary: {config.summary_json}")
    print("=" * 72)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan intranet CT DICOM series, export QC, and convert selected series to fixed-size .npy files."
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for generated .npy files.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Output training manifest CSV.")
    parser.add_argument("--qc-csv", type=Path, required=True, help="Output series-level QC CSV.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Output preprocessing summary JSON.")
    parser.add_argument("--source-csv", type=Path, default=None, help="Existing multimodal CT CSV to rebuild from DICOM paths.")
    parser.add_argument(
        "--input-root",
        action="append",
        default=[],
        help="Additional DICOM case root in LABEL=PATH form, e.g. 良性结节=Z:\\良性患者500例. Can be repeated.",
    )
    parser.add_argument("--dicom-path-col", type=str, default="CT dicom路径")
    parser.add_argument("--npy-path-col", type=str, default="CT_numpy路径")
    parser.add_argument("--npy-cloud-path-col", type=str, default="CT_numpy_cloud路径")
    parser.add_argument("--label-col", type=str, default="样本类型")
    parser.add_argument("--split-col", type=str, default="CT_train_val_split")
    parser.add_argument("--sample-id-col", type=str, default="SampleID")
    parser.add_argument("--root-split-mode", type=str, choices=["blank", "train", "train_val", "train_val_test"], default="blank")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-depth", type=int, default=128)
    parser.add_argument("--target-hw", type=int, default=256)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=400.0)
    parser.add_argument("--min-slices", type=int, default=32)
    parser.add_argument("--min-hw", type=int, default=128)
    parser.add_argument("--max-slice-thickness-mm", type=float, default=5.0)
    parser.add_argument("--max-z-spacing-mm", type=float, default=5.0)
    parser.add_argument("--max-xy-spacing-mm", type=float, default=2.5)
    parser.add_argument("--series-mode", type=str, choices=["best", "all"], default="best")
    parser.add_argument("--cloud-prefix", type=str, default="", help="Optional prefix prepended to relative .npy paths in the manifest.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit on input cases.")
    parser.add_argument("--scan-only", action="store_true", help="Only write QC/manifest plan; do not save .npy files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated .npy files.")
    return parser


def parse_args() -> IntranetCTPreprocessConfig:
    args = build_parser().parse_args()
    return IntranetCTPreprocessConfig(
        output_root=args.output_root,
        manifest_out=args.manifest_out,
        qc_csv=args.qc_csv,
        summary_json=args.summary_json,
        source_csv=args.source_csv,
        input_roots=tuple(args.input_root or ()),
        dicom_path_col=args.dicom_path_col,
        npy_path_col=args.npy_path_col,
        npy_cloud_path_col=args.npy_cloud_path_col,
        label_col=args.label_col,
        split_col=args.split_col,
        sample_id_col=args.sample_id_col,
        root_split_mode=args.root_split_mode,
        train_ratio=args.train_ratio,
        seed=args.seed,
        target_depth=args.target_depth,
        target_hw=args.target_hw,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        min_slices=args.min_slices,
        min_hw=args.min_hw,
        max_slice_thickness_mm=args.max_slice_thickness_mm,
        max_z_spacing_mm=args.max_z_spacing_mm,
        max_xy_spacing_mm=args.max_xy_spacing_mm,
        scan_only=args.scan_only,
        series_mode=args.series_mode,
        cloud_prefix=args.cloud_prefix,
        limit=args.limit,
        overwrite=args.overwrite,
    )


def main() -> None:
    process_intranet_ct(parse_args())


if __name__ == "__main__":
    main()
