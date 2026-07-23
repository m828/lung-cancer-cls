from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.utils.attribution_audit import AttributionAuditError, Prediction, align_prediction_maps
from experiments.analysis.attribution_common import discover_arm_runs, expected_run_completeness


def prediction(sid: str, label: int, probability: float) -> Prediction:
    return Prediction(sid, label, (1.0 - probability, probability), int(probability >= 0.5))


def test_prediction_alignment_uses_sample_id_not_row_order():
    first = {"A": prediction("A", 0, 0.1), "B": prediction("B", 1, 0.8)}
    second = {"B": prediction("B", 1, 0.7), "A": prediction("A", 0, 0.2)}
    ids, aligned = align_prediction_maps({"first": first, "second": second})
    assert ids == ["A", "B"]
    assert [record.sample_id for record in aligned["second"]] == ids


def test_prediction_alignment_rejects_missing_identity_and_label_mismatch():
    first = {"A": prediction("A", 0, 0.1), "B": prediction("B", 1, 0.8)}
    with pytest.raises(AttributionAuditError):
        align_prediction_maps({"first": first, "missing": {"A": prediction("A", 0, 0.2)}})
    with pytest.raises(AttributionAuditError):
        align_prediction_maps({"first": first, "wrong": {"A": prediction("A", 1, 0.2), "B": prediction("B", 1, 0.7)}})


def test_run_discovery_follows_reused_arm_symlink_and_marks_expected_seeds(tmp_path: Path):
    source = tmp_path / "source" / "full" / "seed42"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text(json.dumps({"config": {"seed": 42}}), encoding="utf-8")
    (source / "run_complete.json").write_text(json.dumps({"run_mode": "full"}), encoding="utf-8")
    for split in ("val", "test"):
        with (source / f"{split}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "prob_0", "prob_1"])
            writer.writeheader()
            writer.writerows(
                [
                    {"sample_id": "A", "label": 0, "prob_0": 0.9, "prob_1": 0.1},
                    {"sample_id": "B", "label": 1, "prob_0": 0.1, "prob_1": 0.9},
                ]
            )
    root = tmp_path / "analysis"
    root.mkdir()
    (root / "reused").symlink_to(tmp_path / "source", target_is_directory=True)
    runs, _ = discover_arm_runs(root, ["reused"], run_mode="full")
    rows, complete = expected_run_completeness(runs, ["reused"], [42, 43])
    assert 42 in runs["reused"]
    assert complete is False
    assert [row["status"] for row in rows] == ["complete", "MISSING"]
