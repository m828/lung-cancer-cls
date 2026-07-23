from experiments.utils.attribution_audit import partition_teacher_correction_groups


def test_teacher_correction_groups_are_mutually_exclusive_and_exhaustive():
    labels = [1, 1, 0, 0]
    ct_text = [0, 1, 0, 1]
    full = [1, 1, 1, 1]
    groups = partition_teacher_correction_groups(labels, ct_text, full)
    assert groups == ["A", "B", "C", "D"]
    assert len(groups) == len(labels)
    assert set(groups) == {"A", "B", "C", "D"}

