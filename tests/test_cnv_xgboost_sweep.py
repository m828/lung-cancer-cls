from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.cnv_xgboost import CNVXGBoostConfig
from lung_cancer_cls.cnv_xgboost_sweep import CNV_SWEEP_PRESETS, CNVXGBoostSweepConfig, resolve_sweep_grid, run_sweep


def test_cnv_sweep_presets_are_resolvable():
    assert resolve_sweep_grid("fast", None, int, "max_depth_grid") == CNV_SWEEP_PRESETS["fast"]["max_depth_grid"]
    assert resolve_sweep_grid("formal", None, float, "learning_rate_grid") == CNV_SWEEP_PRESETS["formal"]["learning_rate_grid"]


def test_cnv_xgboost_sweep_generates_leaderboard(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("xgboost")

    gene_tsv = tmp_path / "gene.tsv"
    meta_csv = tmp_path / "meta.csv"

    gene_rows = []
    meta_rows = []
    for idx in range(12):
        sample_id = f"S{idx}"
        label_name = "肺癌" if idx % 2 else "健康对照"
        gene_rows.append(
            {
                "gene_id": sample_id,
                "gene_label": label_name,
                "f1": idx * 0.1,
                "f2": idx * 0.2,
                "f3": (idx % 3) * 0.3,
            }
        )
        split = "train" if idx < 8 else ("val" if idx < 10 else "test")
        meta_rows.append({"SampleID": sample_id, "样本类型": label_name, "CT_train_val_split": split})

    pd.DataFrame(gene_rows).to_csv(gene_tsv, sep="\t", index=False)
    pd.DataFrame(meta_rows).to_csv(meta_csv, index=False)

    base_config = CNVXGBoostConfig(
        metadata_csv=meta_csv,
        gene_tsv=gene_tsv,
        output_dir=tmp_path / "sweep",
        use_predefined_split=True,
        split_mode="train_val_test",
        n_estimators=40,
        early_stopping_rounds=10,
        n_jobs=1,
        save_predictions=False,
    )
    sweep_config = CNVXGBoostSweepConfig(
        base_config=base_config,
        output_dir=tmp_path / "sweep",
        seeds=[42, 52],
        max_depth_grid=[3],
        min_child_weight_grid=[1.0],
        subsample_grid=[0.8],
        colsample_bytree_grid=[0.8],
        learning_rate_grid=[0.1],
        reg_lambda_grid=[1.0],
        gamma_grid=[0.0],
        save_predictions=False,
    )
    summary = run_sweep(sweep_config)

    assert summary["num_runs"] == 2
    assert (tmp_path / "sweep" / "leaderboard.csv").exists()
    assert (tmp_path / "sweep" / "summary.json").exists()
