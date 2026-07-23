"""Identity-safe utilities for the privileged-genomic attribution suite.

The functions in this module deliberately avoid coupling to a training model.
They enforce the invariants shared by launchers, training extensions, analyses,
and tests: string-preserving sample IDs, split-local permutations, deterministic
weight shuffling, atomic checkpoint writes, and paired prediction alignment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


class AttributionAuditError(RuntimeError):
    """Raised when an attribution experiment would violate an audit invariant."""


def sample_id(value: Any) -> str:
    """Return a lossless, whitespace-trimmed string sample identifier."""

    result = str(value).strip()
    if not result:
        raise AttributionAuditError("empty sample_id is not allowed")
    return result


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically save a torch checkpoint while keeping torch an optional import."""

    import torch  # Imported lazily so audit-only scripts do not require torch.

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def normalize_split(value: Any) -> str:
    raw = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val", "dev": "val"}
    result = aliases.get(raw, raw)
    if result not in {"train", "val", "test"}:
        raise AttributionAuditError(f"unsupported split value: {value!r}")
    return result


def load_split_manifest(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise AttributionAuditError(f"empty split manifest: {path}")
    split_col = "assigned_split" if "assigned_split" in rows[0] else "split"
    if "sample_id" not in rows[0] or split_col not in rows[0]:
        raise AttributionAuditError(f"manifest requires sample_id and split columns: {path}")
    mapping: dict[str, str] = {}
    for row in rows:
        sid = sample_id(row["sample_id"])
        if sid in mapping:
            raise AttributionAuditError(f"duplicate sample_id in manifest: {sid}")
        mapping[sid] = normalize_split(row[split_col])
    assert_disjoint_split_mapping(mapping)
    return mapping


def assert_disjoint_split_mapping(mapping: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for raw_sid, raw_split in mapping.items():
        sid = sample_id(raw_sid)
        split = normalize_split(raw_split)
        previous = seen.get(sid)
        if previous is not None and previous != split:
            raise AttributionAuditError(f"sample_id crosses splits: {sid} ({previous}, {split})")
        seen[sid] = split


def assert_same_identity_set(
    expected: Iterable[Any],
    observed: Iterable[Any],
    *,
    context: str,
) -> None:
    expected_ids = {sample_id(value) for value in expected}
    observed_ids = {sample_id(value) for value in observed}
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if missing or extra:
        raise AttributionAuditError(
            f"{context}: identity mismatch; missing={missing[:20]}, extra={extra[:20]}"
        )


def assert_unique(values: Iterable[Any], *, context: str) -> list[str]:
    normalized = [sample_id(value) for value in values]
    duplicates = sorted(value for value, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise AttributionAuditError(f"{context}: duplicate sample IDs: {duplicates[:20]}")
    return normalized


def _derangement_indices(size: int, rng: random.Random) -> list[int]:
    """Return a deterministic no-fixed-point permutation for a split."""

    if size < 2:
        raise AttributionAuditError("CNV permutation requires at least two samples per split")
    indices = list(range(size))
    for _ in range(128):
        candidate = indices.copy()
        rng.shuffle(candidate)
        if all(target != donor for target, donor in enumerate(candidate)):
            return candidate
    shift = rng.randrange(1, size)
    return indices[shift:] + indices[:shift]


def split_local_donor_mapping(
    ids: Sequence[Any],
    splits: Sequence[Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Create a whole-row donor mapping independently within each split."""

    if len(ids) != len(splits):
        raise AttributionAuditError("ids and splits must have the same length")
    normalized_ids = assert_unique(ids, context="CNV permutation targets")
    normalized_splits = [normalize_split(value) for value in splits]
    by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, split in enumerate(normalized_splits):
        by_split[split].append(index)

    rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(("train", "val", "test")):
        positions = by_split[split]
        if not positions:
            continue
        rng = random.Random(int(seed) + split_index * 1_000_003)
        local_donors = _derangement_indices(len(positions), rng)
        for local_target, local_donor in enumerate(local_donors):
            target_pos = positions[local_target]
            donor_pos = positions[local_donor]
            rows.append(
                {
                    "split": split,
                    "target_index": target_pos,
                    "donor_index": donor_pos,
                    "sample_id": normalized_ids[target_pos],
                    "donor_sample_id": normalized_ids[donor_pos],
                    "is_fixed_point": int(target_pos == donor_pos),
                    "permutation_seed": int(seed),
                }
            )
    rows.sort(key=lambda row: int(row["target_index"]))
    if len(rows) != len(normalized_ids):
        raise AttributionAuditError("CNV permutation mapping does not cover every sample")
    if any(int(row["is_fixed_point"]) for row in rows):
        raise AttributionAuditError("CNV permutation unexpectedly contains a fixed point")
    for row in rows:
        donor_split = normalized_splits[int(row["donor_index"])]
        if donor_split != row["split"]:
            raise AttributionAuditError(f"CNV donor crosses split for {row['sample_id']}")
    return rows


def permute_feature_rows(
    feature_rows: Sequence[Sequence[float]],
    mapping_rows: Sequence[Mapping[str, Any]],
) -> list[list[float]]:
    """Apply a donor mapping to complete feature rows, never to columns."""

    snapshot = [list(row) for row in feature_rows]
    if len(snapshot) != len(mapping_rows):
        raise AttributionAuditError("feature row count does not match permutation mapping")
    output: list[list[float]] = [[] for _ in snapshot]
    for row in mapping_rows:
        target = int(row["target_index"])
        donor = int(row["donor_index"])
        output[target] = list(snapshot[donor])
    widths = {len(row) for row in output}
    if len(widths) != 1:
        raise AttributionAuditError("permuted CNV rows have inconsistent feature widths")
    return output


def shuffled_weight_mapping(
    ids: Sequence[Any],
    weights: Sequence[float],
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Permute confidence weights among training identities only.

    The returned mapping preserves the exact weight multiset. Logits, labels,
    and iteration order are external to this function and remain untouched.
    """

    if len(ids) != len(weights):
        raise AttributionAuditError("ids and weights must have the same length")
    normalized_ids = assert_unique(ids, context="confidence shuffle")
    original = [float(value) for value in weights]
    if len(original) < 2:
        raise AttributionAuditError("confidence shuffle requires at least two training samples")
    rng = random.Random(int(seed))
    donor_indices = _derangement_indices(len(original), rng)
    lookup: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for target_index, donor_index in enumerate(donor_indices):
        sid = normalized_ids[target_index]
        shuffled = original[donor_index]
        lookup[sid] = shuffled
        rows.append(
            {
                "sample_id": sid,
                "donor_sample_id": normalized_ids[donor_index],
                "original_weight": original[target_index],
                "shuffled_weight": shuffled,
                "changed": int(target_index != donor_index),
                "confidence_shuffle_seed": int(seed),
            }
        )
    if sorted(original) != sorted(lookup.values()):
        raise AttributionAuditError("shuffled-confidence changed the weight distribution")
    return lookup, rows


def build_kd_weights(
    ids: Sequence[Any],
    confidences: Sequence[float],
    *,
    mode: str,
    floor: float,
    maximum: float,
    shuffle_seed: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Build audited uniform, confidence, or shuffled-confidence weights."""

    normalized_ids = assert_unique(ids, context="KD weights")
    clipped = [max(float(floor), min(float(maximum), float(value))) for value in confidences]
    if mode == "uniform":
        values = [1.0 for _ in normalized_ids]
        return dict(zip(normalized_ids, values)), [
            {"sample_id": sid, "original_weight": 1.0, "effective_weight": 1.0, "mode": mode}
            for sid in normalized_ids
        ]
    if mode == "confidence":
        return dict(zip(normalized_ids, clipped)), [
            {"sample_id": sid, "original_weight": weight, "effective_weight": weight, "mode": mode}
            for sid, weight in zip(normalized_ids, clipped)
        ]
    if mode == "shuffled_confidence":
        if shuffle_seed is None:
            raise AttributionAuditError("shuffle_seed is required for shuffled_confidence")
        lookup, rows = shuffled_weight_mapping(normalized_ids, clipped, shuffle_seed)
        for row in rows:
            row["effective_weight"] = row["shuffled_weight"]
            row["mode"] = mode
        return lookup, rows
    raise AttributionAuditError(f"unsupported attribution KD weight mode: {mode}")


def normalized_weighted_sum(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return sum(values * weights) / sum(weights), matching the KD objective."""

    if len(values) != len(weights) or not values:
        raise AttributionAuditError("values and weights must be non-empty and aligned")
    denominator = sum(float(value) for value in weights)
    if denominator <= 0:
        raise AttributionAuditError("KD weight denominator must be positive")
    return sum(float(value) * float(weight) for value, weight in zip(values, weights)) / denominator


def combine_hard_and_kd_loss(ce_loss: float, kd_per_sample: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    """Reference scalar form of CE + alpha * normalized weighted KD."""

    return float(ce_loss) + float(alpha) * normalized_weighted_sum(kd_per_sample, weights)


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    label: int
    probabilities: tuple[float, ...]
    prediction: int | None = None


def _probability_columns(fields: Sequence[str]) -> list[str]:
    numeric = sorted(
        (field for field in fields if field.startswith("prob_") and field[5:].isdigit()),
        key=lambda field: int(field[5:]),
    )
    if numeric:
        return numeric
    named_order = ["prob_normal", "prob_benign", "prob_malignant"]
    named = [field for field in named_order if field in fields]
    if named:
        return named
    if "prob_malignant" in fields:
        return ["prob_normal", "prob_malignant"] if "prob_normal" in fields else ["prob_malignant"]
    raise AttributionAuditError(f"prediction file has no recognized probability columns: {fields}")


def load_prediction_csv(path: Path) -> dict[str, Prediction]:
    rows = read_csv_rows(path)
    if not rows:
        raise AttributionAuditError(f"empty prediction file: {path}")
    fields = list(rows[0])
    id_col = "sample_id" if "sample_id" in fields else ("record_id" if "record_id" in fields else "")
    if not id_col or "label" not in fields:
        raise AttributionAuditError(f"prediction file requires sample_id/record_id and label: {path}")
    prob_cols = _probability_columns(fields)
    output: dict[str, Prediction] = {}
    for row in rows:
        sid = sample_id(row[id_col])
        if sid in output:
            raise AttributionAuditError(f"duplicate prediction sample_id in {path}: {sid}")
        probabilities = tuple(float(row[col]) for col in prob_cols if col in row and row[col] != "")
        if len(probabilities) == 1:
            probabilities = (1.0 - probabilities[0], probabilities[0])
        prediction = int(row["prediction"]) if row.get("prediction", "") != "" else None
        output[sid] = Prediction(sid, int(float(row["label"])), probabilities, prediction)
    return output


def align_prediction_maps(
    named_predictions: Mapping[str, Mapping[str, Prediction]],
    *,
    require_identical: bool = True,
) -> tuple[list[str], dict[str, list[Prediction]]]:
    if not named_predictions:
        raise AttributionAuditError("no prediction maps supplied")
    sets = {name: set(records) for name, records in named_predictions.items()}
    common = set.intersection(*sets.values())
    if require_identical:
        reference_name, reference = next(iter(sets.items()))
        for name, identities in sets.items():
            if identities != reference:
                missing = sorted(reference - identities)
                extra = sorted(identities - reference)
                raise AttributionAuditError(
                    f"prediction identity mismatch {reference_name} vs {name}: "
                    f"missing={missing[:20]}, extra={extra[:20]}"
                )
    ids = sorted(common)
    if not ids:
        raise AttributionAuditError("prediction maps have no common identities")
    aligned = {name: [records[sid] for sid in ids] for name, records in named_predictions.items()}
    for index, sid in enumerate(ids):
        labels = {records[index].label for records in aligned.values()}
        if len(labels) != 1:
            raise AttributionAuditError(f"label mismatch for sample_id={sid}: {sorted(labels)}")
    return ids, aligned


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5, ece_bins: int = 10) -> dict[str, float]:
    if len(labels) != len(probabilities) or not labels:
        raise AttributionAuditError("binary metric inputs must be non-empty and aligned")
    y = [int(value) for value in labels]
    p = [min(1.0, max(0.0, float(value))) for value in probabilities]
    predictions = [int(value >= threshold) for value in p]
    tp = sum(label == 1 and pred == 1 for label, pred in zip(y, predictions))
    tn = sum(label == 0 and pred == 0 for label, pred in zip(y, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(y, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(y, predictions))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    bacc = (sensitivity + specificity) / 2.0
    positives = [value for value, label in zip(p, y) if label == 1]
    negatives = [value for value, label in zip(p, y) if label == 0]
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            concordance += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    auroc = concordance / (len(positives) * len(negatives)) if positives and negatives else float("nan")
    ece = 0.0
    for bin_index in range(int(ece_bins)):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [idx for idx, value in enumerate(p) if lower <= value < upper or (bin_index == ece_bins - 1 and value == 1.0)]
        if members:
            mean_probability = statistics.mean(p[idx] for idx in members)
            observed_rate = statistics.mean(y[idx] for idx in members)
            ece += len(members) / len(y) * abs(mean_probability - observed_rate)
    brier = statistics.mean((probability - label) ** 2 for probability, label in zip(p, y))
    accuracy = (tp + tn) / len(y)
    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "balanced_accuracy": bacc,
        "f1": f1,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "ece": ece,
        "brier_score": brier,
    }


def select_binary_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Select the validation threshold used by the existing R3 analyses."""

    best_threshold = 0.5
    best_score = -float("inf")
    for step in range(1, 100):
        threshold = step / 100.0
        metrics = binary_metrics(labels, probabilities, threshold)
        score = 0.5 * metrics["balanced_accuracy"] + 0.5 * metrics["f1"] + 0.25 * metrics["sensitivity"]
        if score > best_score or (math.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = threshold
    return best_threshold


def checkpoint_score(name: str, metrics: Mapping[str, Any]) -> float:
    aliases = {
        "val_auroc": "auroc",
        "val_f1": "f1",
        "val_bacc": "balanced_accuracy",
        "val_loss": "loss",
    }
    if name == "composite" or name == "val_composite":
        required = ["auroc", "balanced_accuracy", "f1", "recall", "ece"]
        missing = [key for key in required if metrics.get(key) is None]
        if missing:
            raise AttributionAuditError(f"composite checkpoint missing validation metrics: {missing}")
        return (
            float(metrics["auroc"])
            + 0.5 * float(metrics["balanced_accuracy"])
            + 0.5 * float(metrics["f1"])
            + 0.25 * float(metrics["recall"])
            - 0.25 * float(metrics["ece"])
        )
    metric = aliases.get(name, name)
    if metrics.get(metric) is None:
        raise AttributionAuditError(f"checkpoint metric unavailable: {name}")
    value = float(metrics[metric])
    return -value if metric == "loss" else value


def checkpoint_filename(criterion: str) -> str:
    """Return the stable filename for a validation-selected checkpoint."""

    allowed = {"val_loss", "val_auroc", "val_f1", "val_bacc", "val_composite"}
    if criterion not in allowed:
        raise AttributionAuditError(f"unsupported checkpoint criterion: {criterion}")
    return f"best_{criterion}.pt"


def checkpoint_payload(
    *,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any] | None,
    epoch: int,
    criterion: str,
    val_metrics: Mapping[str, Any],
    threshold: float | None,
    configuration: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state_dict": dict(model_state),
        "optimizer_state_dict": dict(optimizer_state),
        "scheduler_state_dict": dict(scheduler_state) if scheduler_state is not None else None,
        "epoch": int(epoch),
        "checkpoint_criterion": criterion,
        "validation_score": checkpoint_score(criterion, val_metrics),
        "val_metrics": dict(val_metrics),
        "validation_threshold": threshold,
        "configuration": dict(configuration),
        "test_metrics_used_for_selection": False,
    }
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    return payload


def multiclass_confusion(labels: Sequence[int], predictions: Sequence[int], class_count: int) -> list[list[int]]:
    if len(labels) != len(predictions):
        raise AttributionAuditError("multiclass labels and predictions are not aligned")
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(labels, predictions):
        if not 0 <= int(label) < class_count or not 0 <= int(prediction) < class_count:
            raise AttributionAuditError(f"class index outside [0, {class_count}): {label}, {prediction}")
        matrix[int(label)][int(prediction)] += 1
    return matrix


def row_normalize_confusion(matrix: Sequence[Sequence[int]]) -> list[list[float]]:
    output: list[list[float]] = []
    for row in matrix:
        denominator = sum(int(value) for value in row)
        output.append([float(value) / denominator if denominator else 0.0 for value in row])
    return output


def multiclass_class_metrics(matrix: Sequence[Sequence[int]]) -> list[dict[str, float]]:
    class_count = len(matrix)
    output: list[dict[str, float]] = []
    for class_index in range(class_count):
        tp = float(matrix[class_index][class_index])
        support = float(sum(matrix[class_index]))
        predicted = float(sum(matrix[row][class_index] for row in range(class_count)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.append(
            {
                "class_index": float(class_index),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return output


def mean_and_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    return statistics.mean(finite), statistics.stdev(finite) if len(finite) > 1 else 0.0


def partition_teacher_correction_groups(
    labels: Sequence[int],
    ct_text_predictions: Sequence[int],
    full_teacher_predictions: Sequence[int],
) -> list[str]:
    if not (len(labels) == len(ct_text_predictions) == len(full_teacher_predictions)):
        raise AttributionAuditError("teacher correction group inputs are not aligned")
    groups: list[str] = []
    for label, ct_prediction, full_prediction in zip(labels, ct_text_predictions, full_teacher_predictions):
        ct_correct = int(ct_prediction) == int(label)
        full_correct = int(full_prediction) == int(label)
        if not ct_correct and full_correct:
            groups.append("A")
        elif ct_correct and full_correct:
            groups.append("B")
        elif ct_correct and not full_correct:
            groups.append("C")
        else:
            groups.append("D")
    if len(groups) != len(labels) or any(group not in {"A", "B", "C", "D"} for group in groups):
        raise AttributionAuditError("teacher correction groups do not form a complete partition")
    return groups


def capture_environment(prefixes: Sequence[str] = ("CUDA", "SLURM", "SMOKE", "RUN_MODE", "SEEDS", "OUTPUT_ROOT")) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if any(key == prefix or key.startswith(f"{prefix}_") for prefix in prefixes)
    }


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.mean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }
