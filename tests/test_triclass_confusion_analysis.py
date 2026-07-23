from __future__ import annotations

import pytest

from experiments.utils.attribution_audit import multiclass_class_metrics, multiclass_confusion, row_normalize_confusion


def test_triclass_confusion_and_class_metrics_are_correct():
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 2, 2, 0]
    matrix = multiclass_confusion(labels, predictions, 3)
    assert matrix == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    normalized = row_normalize_confusion(matrix)
    assert all(sum(row) == pytest.approx(1.0) for row in normalized)
    metrics = multiclass_class_metrics(matrix)
    assert [row["recall"] for row in metrics] == pytest.approx([0.5, 0.5, 0.5])
    assert [row["precision"] for row in metrics] == pytest.approx([0.5, 0.5, 0.5])


def test_mean_and_std_aggregation_is_per_seed_not_stacked():
    seed_matrices = [
        row_normalize_confusion([[2, 0, 0], [0, 1, 1], [0, 0, 2]]),
        row_normalize_confusion([[1, 1, 0], [0, 2, 0], [1, 0, 1]]),
    ]
    mean_normal_recall = sum(matrix[0][0] for matrix in seed_matrices) / len(seed_matrices)
    assert mean_normal_recall == pytest.approx(0.75)
