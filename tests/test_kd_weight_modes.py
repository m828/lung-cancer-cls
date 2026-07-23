from __future__ import annotations

import math

from experiments.utils.attribution_audit import build_kd_weights, combine_hard_and_kd_loss, normalized_weighted_sum


def test_uniform_weights_are_exactly_one():
    lookup, rows = build_kd_weights(["A", "B", "C"], [0.6, 0.8, 0.9], mode="uniform", floor=0.05, maximum=1.0)
    assert lookup == {"A": 1.0, "B": 1.0, "C": 1.0}
    assert all(row["effective_weight"] == 1.0 for row in rows)


def test_confidence_weights_follow_untempered_confidence_and_clipping():
    lookup, _ = build_kd_weights(["A", "B", "C"], [0.4, 0.75, 1.2], mode="confidence", floor=0.5, maximum=1.0)
    assert lookup == {"A": 0.5, "B": 0.75, "C": 1.0}


def test_shuffled_confidence_preserves_distribution_but_changes_assignment():
    ids = [f"S{index}" for index in range(8)]
    confidence = [0.51 + index * 0.05 for index in range(8)]
    lookup, rows = build_kd_weights(ids, confidence, mode="shuffled_confidence", floor=0.05, maximum=1.0, shuffle_seed=10042)
    assert sorted(lookup.values()) == sorted(confidence)
    assert all(row["sample_id"] != row["donor_sample_id"] for row in rows)
    assert any(not math.isclose(lookup[sid], value) for sid, value in zip(ids, confidence))


def test_kd_batch_denominator_and_hard_label_ce_are_preserved():
    kd = normalized_weighted_sum([1.0, 3.0], [1.0, 3.0])
    assert math.isclose(kd, 2.5)
    total = combine_hard_and_kd_loss(0.7, [1.0, 3.0], [1.0, 3.0], alpha=0.1)
    assert math.isclose(total, 0.7 + 0.1 * 2.5)

