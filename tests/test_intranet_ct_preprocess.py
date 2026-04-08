from pathlib import Path

import numpy as np
import pandas as pd

from lung_cancer_cls import intranet_ct_preprocess as preprocess
from lung_cancer_cls.intranet_ct_preprocess import (
    CaseInput,
    IntranetCTPreprocessConfig,
    SeriesRecord,
    _assign_root_splits,
    build_case_inputs,
    preprocess_volume,
    select_series,
)


def _cfg(tmp_path: Path, **kwargs) -> IntranetCTPreprocessConfig:
    base = {
        "output_root": tmp_path / "npy",
        "manifest_out": tmp_path / "manifest.csv",
        "qc_csv": tmp_path / "qc.csv",
        "summary_json": tmp_path / "summary.json",
    }
    base.update(kwargs)
    return IntranetCTPreprocessConfig(**base)


def _record(case_id: str, *, eligible: bool, num_slices: int, z: float, desc: str = "") -> SeriesRecord:
    return SeriesRecord(
        case_id=case_id,
        label_name="良性结节",
        source="root",
        series_dir=Path(f"/tmp/{case_id}"),
        series_id=f"series_{case_id}",
        num_slices=num_slices,
        rows=512,
        columns=512,
        spacing_x=0.8,
        spacing_y=0.8,
        spacing_z=z,
        slice_thickness=z,
        modality="CT",
        body_part="CHEST",
        series_description=desc,
        protocol_name="",
        patient_id=case_id,
        study_instance_uid="study",
        series_instance_uid=f"uid_{case_id}",
        study_date="",
        manufacturer="",
        scanner_model="",
        eligible=eligible,
        exclude_reasons="" if eligible else "few_slices",
    )


def test_preprocess_volume_resizes_and_windows_hu():
    volume = np.linspace(-1200, 600, num=4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    out = preprocess_volume(volume, target_depth=3, target_hw=5, hu_min=-1000, hu_max=400)

    assert out.shape == (3, 5, 5)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_build_case_inputs_from_source_csv(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "SampleID": "S1",
                "CT dicom路径": "case1",
                "样本类型": "肺癌",
                "CT_train_val_split": "valid",
            },
            {
                "SampleID": "S2",
                "CT dicom路径": "case2",
                "样本类型": "其他",
                "CT_train_val_split": "train",
            },
        ]
    )
    monkeypatch.setattr(preprocess.pd, "read_csv", lambda _: df)

    cases = build_case_inputs(_cfg(Path("."), source_csv=Path("source.csv")))

    assert len(cases) == 1
    assert cases[0].case_id == "S1"
    assert cases[0].label_name == "肺癌"
    assert cases[0].split == "val"
    assert cases[0].source == "csv"


def test_source_data_root_rebases_old_csv_path_by_label():
    cfg = _cfg(Path("."), source_data_root=Path("/home/apulis-dev/userdata/Data"))
    rebased = preprocess._rebase_source_dicom_root(
        r"Z:\CT数据 20251120\健康对照_find1mm_fix1124\肺窗1mm标准\DJ20211120B0452",
        "健康对照",
        cfg,
    )

    assert rebased == Path("/home/apulis-dev/userdata/Data/健康对照_原始/健康对照/DJ20211120B0452")


def test_source_path_map_replaces_windows_prefix():
    cfg = _cfg(
        Path("."),
        source_path_maps=(
            r"Z:\CT数据 20251120\良性结节+肺癌_find1mm_fix1124\肺窗近1mm=/userdata/Data/良性结节+肺癌_原始/良性结节+肺癌",
        ),
    )
    rebased = preprocess._rebase_source_dicom_root(
        r"Z:\CT数据 20251120\良性结节+肺癌_find1mm_fix1124\肺窗近1mm\DJ20210315B0196",
        "肺癌",
        cfg,
    )

    assert rebased == Path("/userdata/Data/良性结节+肺癌_原始/良性结节+肺癌/DJ20210315B0196")


def test_root_split_assignment_train_val_test():
    cases = [
        CaseInput(
            case_id=f"case_{idx}",
            dicom_root=Path(f"case_{idx}"),
            label_name="良性结节",
        )
        for idx in range(5)
    ]
    _assign_root_splits(
        cases,
        _cfg(
            Path("."),
            root_split_mode="train_val_test",
            train_ratio=0.6,
            seed=1,
        ),
    )

    assert len(cases) == 5
    assert {case.label_name for case in cases} == {"良性结节"}
    assert {case.split for case in cases} <= {"train", "val", "test"}
    assert sum(1 for case in cases if case.split == "train") == 3


def test_select_series_prefers_eligible_lung_thin_series():
    thick = _record("A", eligible=True, num_slices=80, z=5.0, desc="CHEST")
    thin_lung = _record("B", eligible=True, num_slices=70, z=1.0, desc="Lung thin 1mm")
    ineligible = _record("C", eligible=False, num_slices=200, z=1.0, desc="Lung thin 1mm")

    selected = select_series([thick, thin_lung, ineligible], series_mode="best")

    assert selected == [thin_lung]


def test_select_series_all_returns_only_eligible():
    eligible = _record("A", eligible=True, num_slices=80, z=2.0)
    ineligible = _record("B", eligible=False, num_slices=100, z=1.0)

    assert select_series([eligible, ineligible], series_mode="all") == [eligible]
