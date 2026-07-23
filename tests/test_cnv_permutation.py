from __future__ import annotations

from collections import Counter

from experiments.utils.attribution_audit import permute_feature_rows, split_local_donor_mapping


def test_cnv_permutation_is_whole_row_split_local_and_reproducible():
    ids = [f"T{i}" for i in range(4)] + [f"V{i}" for i in range(3)] + [f"E{i}" for i in range(3)]
    splits = ["train"] * 4 + ["val"] * 3 + ["test"] * 3
    features = [[float(index), float(index + 100)] for index in range(len(ids))]
    first = split_local_donor_mapping(ids, splits, seed=20042)
    second = split_local_donor_mapping(ids, splits, seed=20042)
    assert first == second
    assert all(row["sample_id"] != row["donor_sample_id"] for row in first)
    split_by_id = dict(zip(ids, splits))
    assert all(split_by_id[row["sample_id"]] == split_by_id[row["donor_sample_id"]] for row in first)
    permuted = permute_feature_rows(features, first)
    assert Counter(tuple(row) for row in permuted) == Counter(tuple(row) for row in features)
    assert all(tuple(permuted[int(row["target_index"])]) == tuple(features[int(row["donor_index"])]) for row in first)


def test_cnv_permutation_seed_changes_mapping_without_labels():
    ids = [f"S{i}" for i in range(12)]
    splits = ["train"] * 4 + ["val"] * 4 + ["test"] * 4
    first = split_local_donor_mapping(ids, splits, seed=1)
    second = split_local_donor_mapping(ids, splits, seed=2)
    assert [row["donor_sample_id"] for row in first] != [row["donor_sample_id"] for row in second]

