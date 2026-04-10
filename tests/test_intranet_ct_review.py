from pathlib import Path

import pandas as pd
import pytest

from lung_cancer_cls import intranet_ct_review as review
from lung_cancer_cls.intranet_ct_review import (
    apply_review_flags_csv,
    apply_review_flags_dataframe,
    build_review_flags_dataframe,
)


def _manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "preprocess_case_id": "case_keep",
                "preprocess_series_instance_uid": "uid_keep",
                "preprocess_num_slices": 96,
                "preprocess_slice_thickness": 1.0,
                "preprocess_spacing_z": 1.0,
                "CT_train_val_split": "train",
            },
            {
                "preprocess_case_id": "case_thick_slice",
                "preprocess_series_instance_uid": "uid_thick_slice",
                "preprocess_num_slices": 80,
                "preprocess_slice_thickness": 5.0,
                "preprocess_spacing_z": 1.25,
                "CT_train_val_split": "train",
            },
            {
                "preprocess_case_id": "case_thick_spacing",
                "preprocess_series_instance_uid": "uid_thick_spacing",
                "preprocess_num_slices": 72,
                "preprocess_slice_thickness": 3.0,
                "preprocess_spacing_z": 5.0,
                "CT_train_val_split": "val",
            },
        ]
    )


def test_build_review_flags_marks_exact_5mm_as_pending_review():
    flags_df = build_review_flags_dataframe(_manifest_df())
    rows = flags_df.set_index("preprocess_case_id")

    assert rows.loc["case_keep", "qc_bucket"] == "standard"
    assert rows.loc["case_keep", "review_status"] == "auto_pass"
    assert rows.loc["case_keep", "use_for_training"] == 1
    assert rows.loc["case_thick_slice", "qc_bucket"] == "thick5_borderline"
    assert rows.loc["case_thick_slice", "review_status"] == "pending"
    assert rows.loc["case_thick_slice", "use_for_training"] == 0
    assert rows.loc["case_thick_spacing", "qc_bucket"] == "thick5_borderline"
    assert rows.loc["case_thick_spacing", "use_for_training"] == 0


def test_apply_review_flags_keeps_only_training_approved_rows():
    manifest_df = _manifest_df()
    flags_df = build_review_flags_dataframe(manifest_df)
    flags_df.loc[
        flags_df["preprocess_case_id"] == "case_thick_slice",
        ["review_status", "use_for_training", "review_note"],
    ] = ["manual_pass", 1, "looks acceptable"]

    filtered_df, merged_df = apply_review_flags_dataframe(manifest_df, flags_df)

    assert list(filtered_df["preprocess_case_id"]) == ["case_keep", "case_thick_slice"]
    assert list(merged_df["review_status"]) == ["auto_pass", "manual_pass", "pending"]


def test_apply_review_flags_rejects_duplicate_flag_keys():
    manifest_df = _manifest_df()
    flags_df = build_review_flags_dataframe(manifest_df)
    flags_df = pd.concat([flags_df, flags_df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate review keys"):
        apply_review_flags_dataframe(manifest_df, flags_df)


def test_apply_review_flags_csv_writes_filtered_manifest_and_audit(monkeypatch):
    manifest_df = _manifest_df()
    flags_df = build_review_flags_dataframe(manifest_df)
    flags_df.loc[
        flags_df["preprocess_case_id"] == "case_thick_slice",
        ["review_status", "use_for_training"],
    ] = ["manual_pass", 1]

    reads = {
        "manifest.csv": manifest_df,
        "flags.csv": flags_df,
    }
    writes: dict[str, pd.DataFrame] = {}

    def fake_read_csv(path, *args, **kwargs):
        return reads[str(path)].copy()

    def fake_to_csv(self, path, index=False, encoding=None):
        writes[str(path)] = self.copy()

    monkeypatch.setattr(review.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv, raising=False)

    filtered_df, merged_df = apply_review_flags_csv(
        manifest_csv=Path("manifest.csv"),
        flags_csv=Path("flags.csv"),
        output_csv=Path("manifest_reviewed.csv"),
        audit_csv=Path("manifest_with_review.csv"),
    )

    assert "manifest_reviewed.csv" in writes
    assert "manifest_with_review.csv" in writes
    assert list(filtered_df["preprocess_case_id"]) == ["case_keep", "case_thick_slice"]
    assert list(merged_df["use_for_training"]) == [1, 1, 0]
