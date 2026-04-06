from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


PATIENT_ID_RE = re.compile(r"(LIDC-IDRI-\d{4})", flags=re.IGNORECASE)


@dataclass
class LIDCSplitConfig:
    input_root: Path
    output_dir: Path
    metadata_csv: Path | None = None
    metadata_source: str = "auto"
    task: str = "benign_vs_malignant"
    label_policy: str = "score12_vs_score45"
    split_scheme: str = "patient_kfold"
    n_splits: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    patient_col: str | None = None
    nodule_col: str | None = None
    malignancy_col: str | None = None
    path_col: str | None = None
    annotation_policy: str = "consensus"
    consensus_min_readers: int = 1
    xy_tolerance_px: float = 15.0
    z_tolerance_mm: float = 3.0


@dataclass
class XMLAnnotation:
    patient_id: str
    xml_path: Path
    reader_id: int
    nodule_id: str
    malignancy_score: float
    x_center: float
    y_center: float
    z_center: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    approx_diameter_xy: float
    roi_count: int
    point_count: int


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_first_matching_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        match = normalized.get(normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def infer_patient_id_from_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    match = PATIENT_ID_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def resolve_patient_root(input_root: Path) -> Path:
    direct_root = input_root / "LIDC-IDRI"
    if direct_root.exists():
        nested_root = direct_root / "LIDC-IDRI"
        if nested_root.exists():
            return nested_root
        return direct_root
    return input_root


def discover_patient_dirs(patient_root: Path) -> Dict[str, Path]:
    patient_dirs: Dict[str, Path] = {}
    if not patient_root.exists():
        return patient_dirs
    for path in patient_root.iterdir():
        if path.is_dir() and PATIENT_ID_RE.fullmatch(path.name):
            patient_dirs[path.name.upper()] = path
    return patient_dirs


def detect_metadata_columns(
    df: pd.DataFrame,
    patient_col: str | None = None,
    nodule_col: str | None = None,
    malignancy_col: str | None = None,
    path_col: str | None = None,
) -> Dict[str, Any]:
    columns = list(df.columns)
    patient_candidates = [
        "patient_id",
        "patientid",
        "subject_id",
        "subjectid",
        "case_id",
        "caseid",
        "scan_id",
        "scanid",
        "subject id",
        "patient",
    ]
    nodule_candidates = [
        "nodule_id",
        "noduleid",
        "annotation_id",
        "annotationid",
        "cluster_id",
        "clusterid",
        "lesion_id",
        "lesionid",
        "nod_id",
    ]
    path_candidates = [
        "path",
        "series_path",
        "image_path",
        "dicom_path",
        "scan_path",
        "filepath",
        "file_path",
        "directory",
        "folder",
        "patient_path",
    ]
    single_malignancy_candidates = [
        "malignancy",
        "malignancy_score",
        "malignancy_mean",
        "mean_malignancy",
        "avg_malignancy",
        "malignancy_avg",
        "malignancyrating",
    ]

    detected_patient = patient_col or find_first_matching_column(columns, patient_candidates)
    detected_nodule = nodule_col or find_first_matching_column(columns, nodule_candidates)
    detected_path = path_col or find_first_matching_column(columns, path_candidates)

    if malignancy_col is not None:
        malignancy_cols = [malignancy_col]
    else:
        single = find_first_matching_column(columns, single_malignancy_candidates)
        if single is not None:
            malignancy_cols = [single]
        else:
            malignancy_cols = [
                col
                for col in columns
                if normalize_column_name(col).startswith("malignancy")
            ]

    return {
        "patient_col": detected_patient,
        "nodule_col": detected_nodule,
        "path_col": detected_path,
        "malignancy_cols": malignancy_cols,
    }


def parse_numeric_scores(values: Iterable[Any]) -> List[float]:
    scores: List[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if pd.isna(value):
            continue
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    return scores


def score_to_binary_label(score: float, label_policy: str) -> tuple[int, str] | None:
    policy = label_policy.lower().strip()
    if policy == "score12_vs_score45":
        if score <= 2.0:
            return 0, "benign"
        if score >= 4.0:
            return 1, "malignant"
        return None
    if policy == "score123_vs_score45":
        if score <= 3.0:
            return 0, "benign"
        if score >= 4.0:
            return 1, "malignant"
        return None
    if policy == "score12_vs_score345":
        if score <= 2.0:
            return 0, "benign"
        if score >= 3.0:
            return 1, "malignant"
        return None
    raise ValueError(f"Unsupported label_policy: {label_policy}")


def build_manifest_from_metadata(config: LIDCSplitConfig) -> pd.DataFrame:
    if config.metadata_csv is None or not config.metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {config.metadata_csv}")

    df = pd.read_csv(config.metadata_csv)
    if df.empty:
        raise RuntimeError("metadata.csv is empty")

    detected = detect_metadata_columns(
        df,
        patient_col=config.patient_col,
        nodule_col=config.nodule_col,
        malignancy_col=config.malignancy_col,
        path_col=config.path_col,
    )
    if not detected["malignancy_cols"]:
        raise RuntimeError(
            "Unable to detect malignancy columns from metadata.csv. "
            "Use --malignancy-col to specify one explicitly."
        )

    patient_root = resolve_patient_root(config.input_root)
    patient_dirs = discover_patient_dirs(patient_root)

    records: List[Dict[str, Any]] = []
    dropped_missing_patient = 0
    dropped_missing_score = 0
    dropped_by_policy = 0

    for row_idx, row in df.iterrows():
        patient_id = None
        if detected["patient_col"] is not None:
            patient_id = infer_patient_id_from_text(row.get(detected["patient_col"]))
        if patient_id is None and detected["path_col"] is not None:
            patient_id = infer_patient_id_from_text(row.get(detected["path_col"]))
        if patient_id is None:
            for value in row.tolist():
                patient_id = infer_patient_id_from_text(value)
                if patient_id is not None:
                    break

        if patient_id is None:
            dropped_missing_patient += 1
            continue

        scores = parse_numeric_scores(row.get(col) for col in detected["malignancy_cols"])
        if not scores:
            dropped_missing_score += 1
            continue
        malignancy_score = float(np.mean(scores))

        label_info = score_to_binary_label(malignancy_score, config.label_policy)
        if label_info is None:
            dropped_by_policy += 1
            continue
        label, class_name = label_info

        nodule_raw = row.get(detected["nodule_col"]) if detected["nodule_col"] is not None else None
        nodule_id = str(nodule_raw).strip() if nodule_raw is not None and not pd.isna(nodule_raw) else f"row_{row_idx:06d}"
        nodule_id = re.sub(r"\s+", "_", nodule_id)
        sample_id = f"{patient_id}__{nodule_id}"

        source_path = None
        if detected["path_col"] is not None:
            value = row.get(detected["path_col"])
            if value is not None and not pd.isna(value):
                source_path = str(value)

        patient_dir = patient_dirs.get(patient_id)
        records.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "nodule_id": nodule_id,
                "label": int(label),
                "class_name": class_name,
                "malignancy_score": malignancy_score,
                "score_source": "metadata",
                "source_path": source_path or "",
                "patient_dir": str(patient_dir) if patient_dir is not None else "",
                "metadata_row": int(row_idx),
            }
        )

    if not records:
        raise RuntimeError(
            "No valid LIDC samples were built from metadata.csv. "
            "Check column detection or pass explicit --patient-col / --malignancy-col."
        )

    manifest = pd.DataFrame(records).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    manifest.attrs["dropped_missing_patient"] = dropped_missing_patient
    manifest.attrs["dropped_missing_score"] = dropped_missing_score
    manifest.attrs["dropped_by_policy"] = dropped_by_policy
    manifest.attrs["column_detection"] = detected
    return manifest


def _xml_text(element: ET.Element | None, path: str, ns: Dict[str, str]) -> str | None:
    if element is None:
        return None
    child = element.find(path, ns)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def find_xml_files(patient_root: Path) -> List[Path]:
    return sorted(path for path in patient_root.rglob("*.xml") if path.is_file())


def _findall_variants(element: ET.Element, variants: Sequence[str], ns: Dict[str, str]) -> List[ET.Element]:
    for variant in variants:
        nodes = element.findall(variant, ns)
        if nodes:
            return nodes
    return []


def _parse_xml_annotation(
    patient_id: str,
    xml_path: Path,
    reader_idx: int,
    nodule: ET.Element,
    ns: Dict[str, str],
) -> XMLAnnotation | None:
    nodule_id = _xml_text(nodule, "nih:noduleID", ns) or _xml_text(nodule, "noduleID", ns)
    chars = nodule.find("nih:characteristics", ns) or nodule.find("characteristics")
    malignancy_text = _xml_text(chars, "nih:malignancy", ns) or _xml_text(chars, "malignancy", ns)
    if malignancy_text is None:
        return None
    try:
        malignancy_score = float(malignancy_text)
    except ValueError:
        return None

    roi_nodes = _findall_variants(nodule, [".//nih:roi", ".//roi"], ns)
    if not roi_nodes:
        return None

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    roi_count = 0

    for roi in roi_nodes:
        z_text = _xml_text(roi, "nih:imageZposition", ns) or _xml_text(roi, "imageZposition", ns)
        if z_text is None:
            continue
        try:
            z_value = float(z_text)
        except ValueError:
            continue

        edge_maps = _findall_variants(roi, [".//nih:edgeMap", ".//edgeMap"], ns)
        roi_points = 0
        for edge in edge_maps:
            x_text = _xml_text(edge, "nih:xCoord", ns) or _xml_text(edge, "xCoord", ns)
            y_text = _xml_text(edge, "nih:yCoord", ns) or _xml_text(edge, "yCoord", ns)
            if x_text is None or y_text is None:
                continue
            try:
                xs.append(float(x_text))
                ys.append(float(y_text))
                zs.append(z_value)
                roi_points += 1
            except ValueError:
                continue
        if roi_points > 0:
            roi_count += 1

    if not xs or not ys or not zs:
        return None

    x_min = float(min(xs))
    x_max = float(max(xs))
    y_min = float(min(ys))
    y_max = float(max(ys))
    z_min = float(min(zs))
    z_max = float(max(zs))
    approx_diameter_xy = float(max(x_max - x_min, y_max - y_min))
    derived_nodule_id = re.sub(r"\s+", "_", nodule_id or f"reader{reader_idx}_nodule")

    return XMLAnnotation(
        patient_id=patient_id,
        xml_path=xml_path,
        reader_id=reader_idx,
        nodule_id=derived_nodule_id,
        malignancy_score=malignancy_score,
        x_center=float(np.mean(xs)),
        y_center=float(np.mean(ys)),
        z_center=float(np.mean(zs)),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        approx_diameter_xy=approx_diameter_xy,
        roi_count=int(roi_count),
        point_count=int(len(xs)),
    )


def _annotations_match(
    left: XMLAnnotation,
    right: XMLAnnotation,
    xy_tolerance_px: float,
    z_tolerance_mm: float,
) -> bool:
    if left.patient_id != right.patient_id:
        return False
    if left.xml_path != right.xml_path:
        return False
    if left.reader_id == right.reader_id:
        return False

    left_z_min = left.z_min - z_tolerance_mm
    left_z_max = left.z_max + z_tolerance_mm
    right_z_min = right.z_min - z_tolerance_mm
    right_z_max = right.z_max + z_tolerance_mm
    z_overlaps = min(left_z_max, right_z_max) >= max(left_z_min, right_z_min)
    if not z_overlaps:
        return False

    xy_distance = float(np.hypot(left.x_center - right.x_center, left.y_center - right.y_center))
    adaptive_xy_tol = max(
        xy_tolerance_px,
        0.5 * max(left.approx_diameter_xy, right.approx_diameter_xy),
    )
    return xy_distance <= adaptive_xy_tol


def _cluster_series_annotations(
    annotations: Sequence[XMLAnnotation],
    xy_tolerance_px: float,
    z_tolerance_mm: float,
) -> List[List[XMLAnnotation]]:
    if not annotations:
        return []

    parent = list(range(len(annotations)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left_idx: int, right_idx: int) -> None:
        left_root = find(left_idx)
        right_root = find(right_idx)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_idx in range(len(annotations)):
        for right_idx in range(left_idx + 1, len(annotations)):
            if _annotations_match(
                annotations[left_idx],
                annotations[right_idx],
                xy_tolerance_px=xy_tolerance_px,
                z_tolerance_mm=z_tolerance_mm,
            ):
                union(left_idx, right_idx)

    clusters: Dict[int, List[XMLAnnotation]] = {}
    for idx, annotation in enumerate(annotations):
        clusters.setdefault(find(idx), []).append(annotation)
    return list(clusters.values())


def _build_consensus_record(
    cluster: Sequence[XMLAnnotation],
    cluster_idx: int,
    label_policy: str,
    consensus_min_readers: int,
) -> Dict[str, Any] | None:
    unique_readers = sorted({annotation.reader_id for annotation in cluster})
    if len(unique_readers) < consensus_min_readers:
        return None

    malignancy_score = float(np.mean([annotation.malignancy_score for annotation in cluster]))
    label_info = score_to_binary_label(malignancy_score, label_policy)
    if label_info is None:
        return None
    label, class_name = label_info

    patient_id = cluster[0].patient_id
    xml_path = cluster[0].xml_path
    series_token = re.sub(r"[^A-Za-z0-9._-]+", "_", xml_path.parent.name or xml_path.stem)
    nodule_id = f"{series_token}__cluster_{cluster_idx:04d}"
    sample_id = f"{patient_id}__{nodule_id}"
    scores = [annotation.malignancy_score for annotation in cluster]

    return {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "nodule_id": nodule_id,
        "label": int(label),
        "class_name": class_name,
        "malignancy_score": malignancy_score,
        "score_source": "xml_consensus",
        "source_path": "",
        "patient_dir": str(xml_path.parents[2]) if len(xml_path.parents) >= 3 else "",
        "metadata_row": -1,
        "xml_path": str(xml_path),
        "reader_id": -1,
        "z_center": float(np.mean([annotation.z_center for annotation in cluster])),
        "z_min": float(min(annotation.z_min for annotation in cluster)),
        "z_max": float(max(annotation.z_max for annotation in cluster)),
        "x_center": float(np.mean([annotation.x_center for annotation in cluster])),
        "y_center": float(np.mean([annotation.y_center for annotation in cluster])),
        "x_min": float(min(annotation.x_min for annotation in cluster)),
        "x_max": float(max(annotation.x_max for annotation in cluster)),
        "y_min": float(min(annotation.y_min for annotation in cluster)),
        "y_max": float(max(annotation.y_max for annotation in cluster)),
        "num_readers": int(len(unique_readers)),
        "annotation_count": int(len(cluster)),
        "reader_ids": json.dumps(unique_readers),
        "reader_nodule_ids": json.dumps([annotation.nodule_id for annotation in cluster]),
        "malignancy_scores": json.dumps(scores),
    }


def build_manifest_from_xml(config: LIDCSplitConfig) -> pd.DataFrame:
    patient_root = resolve_patient_root(config.input_root)
    xml_files = find_xml_files(patient_root)
    if not xml_files:
        raise RuntimeError(f"No XML annotation files found under {patient_root}")

    ns = {"nih": "http://www.nih.gov"}
    records: List[Dict[str, Any]] = []
    dropped_by_policy = 0
    dropped_missing_geometry = 0
    dropped_by_consensus_filter = 0
    annotation_policy = config.annotation_policy.lower().strip()
    if annotation_policy not in {"reader", "consensus"}:
        raise ValueError(f"Unsupported annotation_policy: {config.annotation_policy}")

    for xml_path in xml_files:
        patient_id = infer_patient_id_from_text(xml_path.as_posix())
        if patient_id is None:
            continue

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()

        reading_sessions = root.findall(".//nih:readingSession", ns)
        if not reading_sessions:
            reading_sessions = root.findall(".//readingSession")

        series_annotations: List[XMLAnnotation] = []
        for reader_idx, session in enumerate(reading_sessions):
            nodules = session.findall(".//nih:unblindedReadNodule", ns)
            if not nodules:
                nodules = session.findall(".//unblindedReadNodule")

            for nodule in nodules:
                annotation = _parse_xml_annotation(patient_id, xml_path, reader_idx, nodule, ns)
                if annotation is None:
                    dropped_missing_geometry += 1
                    continue
                series_annotations.append(annotation)

        if annotation_policy == "reader":
            for annotation in series_annotations:
                label_info = score_to_binary_label(annotation.malignancy_score, config.label_policy)
                if label_info is None:
                    dropped_by_policy += 1
                    continue
                label, class_name = label_info
                annotation_id = f"{patient_id}__reader{annotation.reader_id}__{annotation.nodule_id}"
                records.append(
                    {
                        "sample_id": annotation_id,
                        "patient_id": patient_id,
                        "nodule_id": annotation.nodule_id,
                        "label": int(label),
                        "class_name": class_name,
                        "malignancy_score": annotation.malignancy_score,
                        "score_source": "xml_reader_annotation",
                        "source_path": str(xml_path.parent),
                        "patient_dir": str(patient_root / patient_id),
                        "metadata_row": -1,
                        "xml_path": str(xml_path),
                        "reader_id": int(annotation.reader_id),
                        "z_center": annotation.z_center,
                        "z_min": annotation.z_min,
                        "z_max": annotation.z_max,
                        "x_center": annotation.x_center,
                        "y_center": annotation.y_center,
                        "x_min": annotation.x_min,
                        "x_max": annotation.x_max,
                        "y_min": annotation.y_min,
                        "y_max": annotation.y_max,
                        "num_readers": 1,
                        "annotation_count": 1,
                        "reader_ids": json.dumps([annotation.reader_id]),
                        "reader_nodule_ids": json.dumps([annotation.nodule_id]),
                        "malignancy_scores": json.dumps([annotation.malignancy_score]),
                    }
                )
            continue

        clusters = _cluster_series_annotations(
            series_annotations,
            xy_tolerance_px=config.xy_tolerance_px,
            z_tolerance_mm=config.z_tolerance_mm,
        )
        for cluster_idx, cluster in enumerate(clusters):
            record = _build_consensus_record(
                cluster,
                cluster_idx=cluster_idx,
                label_policy=config.label_policy,
                consensus_min_readers=config.consensus_min_readers,
            )
            if record is None:
                if len({annotation.reader_id for annotation in cluster}) < config.consensus_min_readers:
                    dropped_by_consensus_filter += 1
                else:
                    dropped_by_policy += 1
                continue
            record["patient_dir"] = str(patient_root / patient_id)
            record["source_path"] = str(xml_path.parent)
            records.append(record)

    if not records:
        raise RuntimeError(
            "No valid nodules were extracted from XML. "
            "If metadata.csv exists, prefer --metadata-source csv or auto."
        )

    manifest = pd.DataFrame(records)
    manifest.attrs["dropped_missing_patient"] = 0
    manifest.attrs["dropped_missing_score"] = dropped_missing_geometry
    manifest.attrs["dropped_by_policy"] = dropped_by_policy
    manifest.attrs["column_detection"] = {
        "source": "xml_consensus" if annotation_policy == "consensus" else "xml_reader",
        "annotation_policy": annotation_policy,
        "consensus_min_readers": int(config.consensus_min_readers),
        "xy_tolerance_px": float(config.xy_tolerance_px),
        "z_tolerance_mm": float(config.z_tolerance_mm),
        "dropped_by_consensus_filter": int(dropped_by_consensus_filter),
    }
    return manifest


def load_lidc_manifest(config: LIDCSplitConfig) -> pd.DataFrame:
    source = config.metadata_source.lower().strip()
    if source == "csv":
        return build_manifest_from_metadata(config)
    if source == "xml":
        return build_manifest_from_xml(config)
    if source != "auto":
        raise ValueError(f"Unsupported metadata_source: {config.metadata_source}")

    if config.metadata_csv is not None and config.metadata_csv.exists():
        try:
            return build_manifest_from_metadata(config)
        except Exception:
            return build_manifest_from_xml(config)
    return build_manifest_from_xml(config)


def _group_majority_labels(labels: Sequence[int], groups: Sequence[str]) -> Dict[str, int]:
    group_to_labels: Dict[str, List[int]] = {}
    for label, group in zip(labels, groups):
        group_to_labels.setdefault(str(group), []).append(int(label))
    result: Dict[str, int] = {}
    for group, group_labels in group_to_labels.items():
        values, counts = np.unique(group_labels, return_counts=True)
        result[group] = int(values[np.argmax(counts)])
    return result


def _group_shuffle_split(
    groups: Sequence[str],
    labels: Sequence[int],
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    group_array = np.asarray(groups)
    label_array = np.asarray(labels)
    indices = np.arange(len(group_array))

    try:
        from sklearn.model_selection import StratifiedGroupShuffleSplit

        splitter = StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(indices, y=label_array, groups=group_array))
        return train_idx, test_idx
    except Exception:
        unique_groups = list(dict.fromkeys(group_array.tolist()))
        majority = _group_majority_labels(label_array.tolist(), group_array.tolist())
        group_labels = [majority[group] for group in unique_groups]
        if len(set(group_labels)) < 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            return next(splitter.split(indices, groups=group_array))

        from sklearn.model_selection import train_test_split

        train_groups, test_groups = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=group_labels,
        )
        train_set = set(train_groups)
        train_idx = np.array([idx for idx, group in enumerate(group_array.tolist()) if group in train_set], dtype=int)
        test_idx = np.array([idx for idx, group in enumerate(group_array.tolist()) if group not in train_set], dtype=int)
        return train_idx, test_idx


def build_patient_holdout_assignments(
    manifest: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    groups = manifest["patient_id"].astype(str).to_numpy()
    labels = manifest["label"].astype(int).to_numpy()
    all_idx = np.arange(len(manifest))

    remaining_ratio = 1.0 - train_ratio
    if val_ratio >= remaining_ratio:
        raise ValueError("val_ratio must be smaller than (1 - train_ratio) for patient_holdout")

    train_idx, heldout_idx = _group_shuffle_split(groups, labels, test_size=remaining_ratio, seed=seed)
    if len(heldout_idx) == 0:
        split = np.array(["train" for _ in all_idx], dtype=object)
    else:
        heldout_groups = groups[heldout_idx]
        heldout_labels = labels[heldout_idx]
        val_fraction_within_heldout = val_ratio / remaining_ratio
        test_ratio_within_heldout = max(0.01, min(0.99, 1.0 - val_fraction_within_heldout))
        val_sub_idx, test_sub_idx = _group_shuffle_split(
            heldout_groups,
            heldout_labels,
            test_size=test_ratio_within_heldout,
            seed=seed + 17,
        )
        split = np.array(["" for _ in all_idx], dtype=object)
        split[train_idx] = "train"
        split[heldout_idx[val_sub_idx]] = "val"
        split[heldout_idx[test_sub_idx]] = "test"
        split[split == ""] = "val"

    assignments = manifest.copy()
    assignments["fold"] = 0
    assignments["split"] = split
    return assignments


def build_patient_kfold_assignments(
    manifest: pd.DataFrame,
    n_splits: int,
    val_ratio: float,
    seed: int,
) -> pd.DataFrame:
    from sklearn.model_selection import GroupKFold

    groups = manifest["patient_id"].astype(str).to_numpy()
    labels = manifest["label"].astype(int).to_numpy()
    indices = np.arange(len(manifest))
    unique_patients = np.unique(groups)
    if len(unique_patients) < n_splits:
        raise ValueError(
            f"patient_kfold requires at least {n_splits} unique patients, got {len(unique_patients)}"
        )

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_splits = list(splitter.split(indices, y=labels, groups=groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        outer_splits = list(splitter.split(indices, groups=groups))

    parts: List[pd.DataFrame] = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_splits):
        train_val_groups = groups[train_val_idx]
        train_val_labels = labels[train_val_idx]
        if len(np.unique(train_val_groups)) <= 1:
            inner_train_idx = train_val_idx
            inner_val_idx = np.array([], dtype=int)
        else:
            inner_rel_train, inner_rel_val = _group_shuffle_split(
                train_val_groups,
                train_val_labels,
                test_size=max(0.01, min(0.5, val_ratio)),
                seed=seed + fold_idx + 1,
            )
            inner_train_idx = train_val_idx[inner_rel_train]
            inner_val_idx = train_val_idx[inner_rel_val]

        split = np.array(["" for _ in indices], dtype=object)
        split[inner_train_idx] = "train"
        split[inner_val_idx] = "val"
        split[test_idx] = "test"
        split[split == ""] = "train"

        fold_df = manifest.copy()
        fold_df["fold"] = int(fold_idx)
        fold_df["split"] = split
        parts.append(fold_df)

    return pd.concat(parts, ignore_index=True)


def build_split_assignments(config: LIDCSplitConfig, manifest: pd.DataFrame) -> pd.DataFrame:
    scheme = config.split_scheme.lower().strip()
    if scheme == "patient_holdout":
        return build_patient_holdout_assignments(
            manifest,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
    if scheme == "patient_kfold":
        return build_patient_kfold_assignments(
            manifest,
            n_splits=config.n_splits,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
    raise ValueError(f"Unsupported split_scheme: {config.split_scheme}")


def build_patient_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for patient_id, group in manifest.groupby("patient_id"):
        counts = group["class_name"].value_counts().to_dict()
        rows.append(
            {
                "patient_id": patient_id,
                "num_samples": int(len(group)),
                "num_benign": int(counts.get("benign", 0)),
                "num_malignant": int(counts.get("malignant", 0)),
                "has_mixed_labels": bool(len(set(group["label"].tolist())) > 1),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def build_summary_json(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    config: LIDCSplitConfig,
) -> Dict[str, Any]:
    patient_summary = build_patient_summary(manifest)
    per_fold: Dict[str, Any] = {}
    for fold, group in assignments.groupby("fold"):
        fold_key = str(int(fold))
        per_fold[fold_key] = {}
        for split_name, split_df in group.groupby("split"):
            counts = split_df["class_name"].value_counts().to_dict()
            patients = split_df["patient_id"].nunique()
            per_fold[fold_key][split_name] = {
                "samples": int(len(split_df)),
                "patients": int(patients),
                "benign": int(counts.get("benign", 0)),
                "malignant": int(counts.get("malignant", 0)),
            }

    return {
        "task": config.task,
        "label_policy": config.label_policy,
        "annotation_policy": config.annotation_policy,
        "consensus_min_readers": int(config.consensus_min_readers),
        "xy_tolerance_px": float(config.xy_tolerance_px),
        "z_tolerance_mm": float(config.z_tolerance_mm),
        "split_scheme": config.split_scheme,
        "n_splits": int(config.n_splits),
        "train_ratio": float(config.train_ratio),
        "val_ratio": float(config.val_ratio),
        "seed": int(config.seed),
        "num_samples": int(len(manifest)),
        "num_patients": int(manifest["patient_id"].nunique()),
        "num_mixed_label_patients": int(patient_summary["has_mixed_labels"].sum()),
        "class_distribution": {
            class_name: int(count)
            for class_name, count in manifest["class_name"].value_counts().to_dict().items()
        },
        "metadata_debug": {
            "dropped_missing_patient": int(manifest.attrs.get("dropped_missing_patient", 0)),
            "dropped_missing_score": int(manifest.attrs.get("dropped_missing_score", 0)),
            "dropped_by_policy": int(manifest.attrs.get("dropped_by_policy", 0)),
            "column_detection": manifest.attrs.get("column_detection", {}),
        },
        "per_fold": per_fold,
    }


def write_lidc_split_artifacts(
    config: LIDCSplitConfig,
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    patient_summary = build_patient_summary(manifest)
    summary = build_summary_json(manifest, assignments, config)

    manifest_path = config.output_dir / "nodule_manifest.csv"
    assignments_path = config.output_dir / "split_manifest.csv"
    patient_summary_path = config.output_dir / "patient_summary.csv"
    summary_json_path = config.output_dir / "summary.json"

    manifest.sort_values(["patient_id", "sample_id"]).to_csv(manifest_path, index=False)
    assignments.sort_values(["fold", "split", "patient_id", "sample_id"]).to_csv(assignments_path, index=False)
    patient_summary.to_csv(patient_summary_path, index=False)
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "manifest_csv": manifest_path,
        "split_manifest_csv": assignments_path,
        "patient_summary_csv": patient_summary_path,
        "summary_json": summary_json_path,
    }


def build_and_write_lidc_splits(config: LIDCSplitConfig) -> Dict[str, Path]:
    manifest = load_lidc_manifest(config)
    assignments = build_split_assignments(config, manifest)
    return write_lidc_split_artifacts(config, manifest, assignments)
