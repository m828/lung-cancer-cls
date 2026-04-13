from pathlib import Path

import pandas as pd

from lung_cancer_cls.intranet_ct_manifest_merge import merge_intranet_manifests


def test_merge_intranet_manifests_rewrites_relative_paths_from_abs_paths():
    ct_root = Path("/data")
    base_df = pd.DataFrame(
        [
            {
                "SampleID": "old_1",
                "样本类型": "健康对照",
                "CT_train_val_split": "train",
                "CT_numpy路径": "/data/CT1500/normal/old_1.npy",
                "CT_numpy_cloud路径": "legacy/should_be_replaced.npy",
            }
        ]
    )
    append_df = pd.DataFrame(
        [
            {
                "SampleID": "new_1",
                "样本类型": "良性结节",
                "CT_train_val_split": "val",
                "CT_numpy路径": "/data/rebuild500/benign/new_1.npy",
            }
        ]
    )

    merged = merge_intranet_manifests(base_df, append_df, ct_root=ct_root)

    assert list(merged["CT_numpy_cloud路径"]) == [
        "CT1500/normal/old_1.npy",
        "rebuild500/benign/new_1.npy",
    ]


def test_merge_intranet_manifests_keeps_later_duplicate_sample_id():
    base_df = pd.DataFrame(
        [
            {
                "SampleID": "same_id",
                "样本类型": "健康对照",
                "CT_numpy_cloud路径": "normal/old.npy",
            }
        ]
    )
    append_df = pd.DataFrame(
        [
            {
                "SampleID": "same_id",
                "样本类型": "良性结节",
                "CT_numpy_cloud路径": "benign/new.npy",
            }
        ]
    )

    merged = merge_intranet_manifests(base_df, append_df)

    assert len(merged) == 1
    assert merged.loc[0, "样本类型"] == "良性结节"
    assert merged.loc[0, "CT_numpy_cloud路径"] == "benign/new.npy"
