"""Shared analysis primitives for the 0542 attribution experiments."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.attribution_audit import (  # noqa: E402
    AttributionAuditError,
    Prediction,
    align_prediction_maps,
    binary_metrics,
    load_prediction_csv,
    mean_and_sample_sd,
    select_binary_threshold,
    write_csv_rows,
)


BINARY_METRICS = (
    "auroc",
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "ece",
    "brier_score",
)


@dataclass
class RunPredictions:
    arm: str
    seed: int
    run_dir: Path
    val: dict[str, Prediction]
    test: dict[str, Prediction]
    threshold: float
    metrics: dict[str, float]


def parse_seed(value: str | Path) -> int | None:
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", str(value), re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"seed(\d+)", str(value), re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_seed_list(raw: str | Sequence[int]) -> list[int]:
    """Parse a comma/space separated seed list into stable unique integers."""

    if isinstance(raw, str):
        values = raw.replace(",", " ").split()
    else:
        values = [str(value) for value in raw]
    seeds: list[int] = []
    for value in values:
        seed = int(value)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def _metrics_candidates_following_links(arm_dir: Path) -> list[Path]:
    """Find run metrics while following only explicit experiment-arm symlinks.

    ``Path.rglob`` does not recurse into directory symlinks on the supported
    Python versions. The suite deliberately reuses controls through symlinks,
    so each encountered link is resolved and searched explicitly. Resolved
    metric paths are deduplicated to avoid counting one run twice.
    """

    if not arm_dir.exists() and not arm_dir.is_symlink():
        return []
    search_roots = [arm_dir.resolve() if arm_dir.is_symlink() else arm_dir]
    if arm_dir.is_dir() and not arm_dir.is_symlink():
        for candidate in arm_dir.rglob("*"):
            if candidate.is_symlink() and candidate.resolve().is_dir():
                search_roots.append(candidate.resolve())
    metrics: dict[Path, Path] = {}
    for search_root in search_roots:
        direct = search_root / "metrics.json"
        if direct.is_file():
            metrics[direct.resolve()] = direct.resolve()
        for path in search_root.rglob("metrics.json") if search_root.is_dir() else []:
            if path.is_file():
                metrics[path.resolve()] = path.resolve()
    return sorted(metrics.values())


def positive_probability(record: Prediction) -> float:
    if len(record.probabilities) != 2:
        raise AttributionAuditError(f"binary analysis received {len(record.probabilities)} classes")
    return float(record.probabilities[1])


def load_run_predictions(arm: str, seed: int, run_dir: Path) -> RunPredictions:
    val = load_prediction_csv(run_dir / "val_predictions.csv")
    test = load_prediction_csv(run_dir / "test_predictions.csv")
    val_ids = sorted(val)
    val_labels = [val[sid].label for sid in val_ids]
    val_probs = [positive_probability(val[sid]) for sid in val_ids]
    threshold = select_binary_threshold(val_labels, val_probs)
    test_ids = sorted(test)
    test_labels = [test[sid].label for sid in test_ids]
    test_probs = [positive_probability(test[sid]) for sid in test_ids]
    metrics = binary_metrics(test_labels, test_probs, threshold)
    return RunPredictions(arm, seed, run_dir, val, test, threshold, metrics)


def discover_arm_runs(
    root: Path,
    arms: Sequence[str],
    *,
    run_mode: str | None = None,
) -> tuple[dict[str, dict[int, RunPredictions]], list[dict[str, Any]]]:
    """Discover one prediction run per arm and seed for the requested mode.

    Legacy synchronized results have no run-mode marker and remain eligible as
    read-only controls. New smoke and full outputs are never mixed.
    """

    if run_mode not in {None, "smoke", "full"}:
        raise ValueError(f"unsupported run mode: {run_mode}")
    output: dict[str, dict[int, RunPredictions]] = {arm: {} for arm in arms}
    audit: list[dict[str, Any]] = []
    for arm in arms:
        arm_dir = root / arm
        if not arm_dir.is_dir():
            audit.append({"arm": arm, "seed": "", "status": "MISSING_ARM", "run_dir": str(arm_dir), "reason": "directory absent"})
            continue
        candidates = _metrics_candidates_following_links(arm_dir)
        for metrics_path in candidates:
            run_dir = metrics_path.parent
            metrics_json = read_json(metrics_path)
            seed = parse_seed(run_dir.name)
            if seed is None:
                seed_value = (metrics_json.get("config") or {}).get("seed")
                seed = int(seed_value) if seed_value is not None else None
            if seed is None:
                audit.append({"arm": arm, "seed": "", "status": "INVALID", "run_dir": str(run_dir), "reason": "seed unavailable"})
                continue
            complete_marker = read_json(run_dir / "run_complete.json")
            mode = complete_marker.get("run_mode") or (metrics_json.get("cached_kd_config") or {}).get("run_mode") or "legacy"
            if mode == "full_candidate":
                mode = "full"
            path_modes = {part for part in run_dir.parts if part in {"smoke", "full"}}
            if path_modes and not complete_marker:
                audit.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "status": "INCOMPLETE",
                        "run_dir": str(run_dir),
                        "reason": "0542 run_mode path lacks run_complete.json",
                    }
                )
                continue
            if complete_marker and complete_marker.get("status") not in {None, "complete"}:
                audit.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "status": "INCOMPLETE",
                        "run_dir": str(run_dir),
                        "reason": f"run status={complete_marker.get('status')}",
                    }
                )
                continue
            if run_mode is not None and mode not in {run_mode, "legacy"}:
                audit.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "status": f"EXCLUDED_{str(mode).upper()}",
                        "run_dir": str(run_dir),
                        "reason": f"requested run_mode={run_mode}",
                    }
                )
                continue
            try:
                run = load_run_predictions(arm, seed, run_dir)
            except (AttributionAuditError, FileNotFoundError, ValueError) as exc:
                audit.append({"arm": arm, "seed": seed, "status": "MISSING", "run_dir": str(run_dir), "reason": str(exc)})
                continue
            if seed in output[arm]:
                raise AttributionAuditError(f"multiple complete runs for {arm} seed{seed}")
            output[arm][seed] = run
            audit.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "status": "complete",
                    "run_dir": str(run_dir),
                    "reason": "",
                    "threshold": run.threshold,
                    "test_rows": len(run.test),
                }
            )
    return output, audit


def expected_run_completeness(
    runs: Mapping[str, Mapping[int, RunPredictions]],
    arms: Sequence[str],
    expected_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], bool]:
    """Return explicit arm/seed completion rows and an all-complete flag."""

    rows: list[dict[str, Any]] = []
    complete = True
    for arm in arms:
        for seed in expected_seeds:
            run = runs.get(arm, {}).get(int(seed))
            if run is None:
                complete = False
                rows.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "status": "MISSING",
                        "run_dir": "",
                        "reason": "required arm/seed output unavailable",
                        "threshold": "",
                        "test_rows": "",
                    }
                )
            else:
                rows.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "status": "complete",
                        "run_dir": str(run.run_dir),
                        "reason": "",
                        "threshold": run.threshold,
                        "test_rows": len(run.test),
                    }
                )
    return rows, complete


def seed_metric_rows(runs: Mapping[str, Mapping[int, RunPredictions]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, arm_runs in runs.items():
        for seed, run in sorted(arm_runs.items()):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "validation_threshold": run.threshold,
                    "test_identity_count": len(run.test),
                    "run_dir": str(run.run_dir),
                    **run.metrics,
                }
            )
    return rows


def summary_rows(seed_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    arms = sorted({str(row["arm"]) for row in seed_rows})
    output: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [row for row in seed_rows if row["arm"] == arm]
        record: dict[str, Any] = {"arm": arm, "n_seeds": len(arm_rows)}
        for metric in BINARY_METRICS:
            mean, sd = mean_and_sample_sd([float(row[metric]) for row in arm_rows if row.get(metric) not in {None, ""}])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        output.append(record)
    return output


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_on_indices(records: Sequence[Prediction], indices: Sequence[int], threshold: float) -> dict[str, float]:
    labels = [records[index].label for index in indices]
    probabilities = [positive_probability(records[index]) for index in indices]
    return binary_metrics(labels, probabilities, threshold)


def paired_contrast_bootstrap(
    runs: Mapping[str, Mapping[int, RunPredictions]],
    coefficients: Mapping[str, float],
    *,
    iterations: int,
    random_seed: int,
    effect_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bootstrap a linear model contrast without stacking seed x identity rows."""

    required_arms = [arm for arm, coefficient in coefficients.items() if coefficient != 0]
    common_seeds = sorted(set.intersection(*(set(runs.get(arm, {})) for arm in required_arms)))
    alignment_rows: list[dict[str, Any]] = []
    if not common_seeds:
        return [
            {
                "effect": effect_name,
                "metric": metric,
                "point_estimate": "",
                "ci95_low": "",
                "ci95_high": "",
                "ci_crosses_zero": "",
                "n_seeds": 0,
                "n_bootstrap": 0,
                "status": "MISSING",
            }
            for metric in BINARY_METRICS
        ], alignment_rows

    aligned_by_seed: dict[int, dict[str, list[Prediction]]] = {}
    ids_by_seed: dict[int, list[str]] = {}
    for seed in common_seeds:
        prediction_maps = {arm: runs[arm][seed].test for arm in required_arms}
        ids, aligned = align_prediction_maps(prediction_maps, require_identical=True)
        aligned_by_seed[seed] = aligned
        ids_by_seed[seed] = ids
        alignment_rows.append(
            {
                "effect": effect_name,
                "seed": seed,
                "identity_count": len(ids),
                "identity_hash": __import__("hashlib").sha256("\n".join(ids).encode("utf-8")).hexdigest(),
                "status": "OK",
            }
        )

    observed_by_metric: dict[str, list[float]] = {metric: [] for metric in BINARY_METRICS}
    full_indices: dict[int, list[int]] = {seed: list(range(len(ids_by_seed[seed]))) for seed in common_seeds}
    for seed in common_seeds:
        metric_by_arm = {
            arm: _metric_on_indices(aligned_by_seed[seed][arm], full_indices[seed], runs[arm][seed].threshold)
            for arm in required_arms
        }
        for metric in BINARY_METRICS:
            observed_by_metric[metric].append(
                sum(float(coefficients[arm]) * metric_by_arm[arm][metric] for arm in required_arms)
            )

    rng = random.Random(int(random_seed))
    boot_values: dict[str, list[float]] = {metric: [] for metric in BINARY_METRICS}
    for _ in range(max(1, int(iterations))):
        seed_contrasts: dict[str, list[float]] = {metric: [] for metric in BINARY_METRICS}
        for seed in common_seeds:
            count = len(ids_by_seed[seed])
            indices = [rng.randrange(count) for _ in range(count)]
            metric_by_arm = {
                arm: _metric_on_indices(aligned_by_seed[seed][arm], indices, runs[arm][seed].threshold)
                for arm in required_arms
            }
            for metric in BINARY_METRICS:
                seed_contrasts[metric].append(
                    sum(float(coefficients[arm]) * metric_by_arm[arm][metric] for arm in required_arms)
                )
        for metric in BINARY_METRICS:
            values = [value for value in seed_contrasts[metric] if math.isfinite(value)]
            if values:
                boot_values[metric].append(statistics.mean(values))

    rows: list[dict[str, Any]] = []
    for metric in BINARY_METRICS:
        observed = [value for value in observed_by_metric[metric] if math.isfinite(value)]
        point = statistics.mean(observed) if observed else float("nan")
        lower = percentile(boot_values[metric], 0.025)
        upper = percentile(boot_values[metric], 0.975)
        rows.append(
            {
                "effect": effect_name,
                "metric": metric,
                "point_estimate": point,
                "ci95_low": lower,
                "ci95_high": upper,
                "ci_crosses_zero": bool(lower <= 0 <= upper),
                "n_seeds": len(common_seeds),
                "identity_count_by_seed": ";".join(f"{seed}:{len(ids_by_seed[seed])}" for seed in common_seeds),
                "n_bootstrap": max(1, int(iterations)),
                "status": "complete",
            }
        )
    return rows, alignment_rows


def write_markdown_table(path: Path, title: str, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else str(value))
        body.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n".join([header, separator, *body]) + "\n", encoding="utf-8")


def standard_metric_fields() -> list[str]:
    return ["arm", "seed", "validation_threshold", "test_identity_count", *BINARY_METRICS, "run_dir"]


def standard_summary_fields() -> list[str]:
    fields = ["arm", "n_seeds"]
    for metric in BINARY_METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_sd"])
    return fields


def write_standard_outputs(
    output_dir: Path,
    seed_filename: str,
    summary_filename: str,
    runs: Mapping[str, Mapping[int, RunPredictions]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = seed_metric_rows(runs)
    summaries = summary_rows(seed_rows)
    write_csv_rows(output_dir / seed_filename, seed_rows, standard_metric_fields())
    write_csv_rows(output_dir / summary_filename, summaries, standard_summary_fields())
    return seed_rows, summaries
