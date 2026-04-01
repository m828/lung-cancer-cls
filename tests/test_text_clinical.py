from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lung_cancer_cls.text_clinical import TextClinicalFeatureConfig, prepare_text_feature_table


def test_prepare_text_feature_table_hash_smoke(tmp_path: Path):
    health_csv = tmp_path / "health.csv"
    disease_csv = tmp_path / "disease.csv"
    output_tsv = tmp_path / "prepared.tsv"

    pd.DataFrame(
        [
            {
                "record_id": "R0",
                "样本类型": "健康对照",
                "入院记录": "无明显异常",
                "主诉": "体检发现",
                "性别": "女",
                "年龄": 52,
                "<SEP2>肿瘤标志物": 0.1,
            }
        ]
    ).to_csv(health_csv, index=False)
    pd.DataFrame(
        [
            {
                "record_id": "R1",
                "样本类型": "肺癌",
                "入院记录": "右肺结节",
                "主诉": "咳嗽一周",
                "性别": "男",
                "年龄": 64,
                "<SEP2>肿瘤标志物": 2.3,
            }
        ]
    ).to_csv(disease_csv, index=False)

    df, meta = prepare_text_feature_table(
        TextClinicalFeatureConfig(
            output_tsv=output_tsv,
            text_health_csv=health_csv,
            text_disease_csv=disease_csv,
            embedding_backend="hash",
            hash_dim=8,
        )
    )

    assert list(df["record_id"]) == ["R0", "R1"]
    assert any(col.startswith("num__") for col in df.columns)
    assert len([col for col in df.columns if col.startswith("bert_")]) == 8
    assert meta["embedding_backend"] == "hash"
    assert (output_tsv.with_suffix(output_tsv.suffix + ".meta.json")).exists()
    saved_meta = json.loads((output_tsv.with_suffix(output_tsv.suffix + ".meta.json")).read_text(encoding="utf-8"))
    assert saved_meta["record_id_col"] == "record_id"
