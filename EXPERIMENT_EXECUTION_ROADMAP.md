# 项目实验总览与执行路线

这份文档用于把当前项目里已经可用的三条实验线整理成一张总路线图，减少在多个 `md` 和多个脚本之间来回切换的混乱。

适用范围：

- `1500` 例内网多模态数据二分类
- `1500` 例内网多模态数据三分类
- `LIDC-IDRI` 公开数据二分类

当前代码检查后的统一结论：

- 内网 `CT` 预处理底座已经具备，可继续沿用 [`INTRANET_CT_PREPROCESS.md`](./INTRANET_CT_PREPROCESS.md)
- 内网二分类主线已经具备，可继续沿用 [`BINARY_CT_STAGE_NOTES.md`](./BINARY_CT_STAGE_NOTES.md)
- 文本、多模态 teacher/student/KD 主线已经具备，推荐统一走 [`TEXT_MULTIMODAL_KD_STAGE.md`](./TEXT_MULTIMODAL_KD_STAGE.md)
- `LIDC-IDRI` 二分类公开 benchmark 已具备，推荐统一走 [`LIDC_IDRI_BENCHMARK_NOTES.md`](./LIDC_IDRI_BENCHMARK_NOTES.md) 和 [`LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md`](./LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md)

这份总览文档不替代上面的详细文档，而是负责回答四个问题：

1. 现在每条线应该用哪个脚本。
2. 哪些步骤是主线，哪些只是补充验证。
3. 哪些实验应该先做，哪些应该等主线稳定后再做。
4. 后续结果应该如何命名，方便横向对比。

## 1. 三条主线

### 1.1 主线 A：1500 例内网多模态二分类

定位：

- 当前论文主线
- 当前最成熟、最应该优先稳定的路线
- 主要回答“稀缺基因信息能否通过 teacher-student 迁移到低成本 `CT` 或 `CT+文本` 模型”

推荐任务定义：

- `--class-mode binary`
- `--binary-task malignant_vs_normal`

### 1.2 主线 B：1500 例内网多模态三分类

定位：

- 补充验证线
- 用来证明方法不是只对一个二分类任务有效
- 不是当前最优先的论文主结论，但非常适合作为补充实验

推荐任务定义：

- `--class-mode multiclass`
- 标签为 `normal / benign / malignant`

### 1.3 主线 C：LIDC-IDRI 公开数据二分类

定位：

- `CT` 表征迁移验证线
- 公开 benchmark 补充线
- 不等价于内网主任务，但可以验证 student 学到的 `CT` 表征是否更可迁移

推荐任务定义：

- `--class-mode binary`
- `--binary-task benign_vs_malignant`
- 当前主线标签折叠：`S1、S2 -> 良性`，`S4、S5 -> 恶性`，`S3 -> 丢弃`

## 2. 统一脚本分工

| 场景 | 推荐主入口 | 作用 | 详细文档 |
|---|---|---|---|
| 内网 `CT` 预处理 | `prepare_intranet_ct_npy.py` | DICOM 转 `.npy`、生成 manifest、做 QC | [`INTRANET_CT_PREPROCESS.md`](./INTRANET_CT_PREPROCESS.md) |
| 内网 `CT only` 训练 | `train.py` | 单模态 `CT` 二分类或三分类 | [`BINARY_CT_STAGE_NOTES.md`](./BINARY_CT_STAGE_NOTES.md) |
| 内网文本特征准备 | `prepare_text_features.py` | 生成文本与结构化临床特征表 | [`TEXT_MULTIMODAL_KD_STAGE.md`](./TEXT_MULTIMODAL_KD_STAGE.md) |
| 内网多模态 teacher / baseline | `train_multimodal.py` | `Text only`、`CT+Text`、`CT+CNV`、`CT+CNV+Text` | [`TEXT_MULTIMODAL_KD_STAGE.md`](./TEXT_MULTIMODAL_KD_STAGE.md) |
| 内网 student + KD | `train_student_kd.py` | `CT` student 或 `CT+Text` student | [`TEXT_MULTIMODAL_KD_STAGE.md`](./TEXT_MULTIMODAL_KD_STAGE.md) |
| `CNV only` baseline | `train_cnv_xgboost.py` / `train_cnv_xgboost_sweep.py` | `XGBoost` 单模态 `CNV` 对照 | [`CNV_XGBOOST_BASELINE.md`](./CNV_XGBOOST_BASELINE.md) |
| `LIDC-IDRI` manifest 构建 | `build_lidc_idri_split_manifest.py` | 原始 DICOM + XML 生成 patient-wise / consensus split | [`LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md`](./LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md) |
| `LIDC-IDRI` 3D 结节裁剪 | `prepare_lidc_idri_consensus_3d.py` | 生成 `64 x 64 x 64` nodule crop | [`LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md`](./LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md) |
| `LIDC-IDRI` 训练 | `train.py` | clean baseline 和 student-init 迁移 | [`LIDC_IDRI_BENCHMARK_NOTES.md`](./LIDC_IDRI_BENCHMARK_NOTES.md) |
| 结果表导出 | `export_experiment_table.py` | 汇总多个 run 的 `metrics.json` | [`TEXT_MULTIMODAL_KD_STAGE.md`](./TEXT_MULTIMODAL_KD_STAGE.md) |
| 可视化导出 | `export_experiment_plots.py` | 导出 ROC、PR、校准、混淆矩阵等图 | [`BINARY_CT_STAGE_NOTES.md`](./BINARY_CT_STAGE_NOTES.md) |

统一建议：

- 内网多模态实验尽量统一走 `prepare_text_features.py + train_multimodal.py + train_student_kd.py`
- 老脚本如 `train_ct_cnv.py` 仍可用，但现在更推荐把它当历史兼容入口，而不是新的主入口
- 内网 `CT only` 和 `LIDC-IDRI` 仍统一走 `train.py`

## 3. 命名建议

为了后续不乱，建议统一用下面这套命名方式：

- `mvn`：`malignant_vs_normal`
- `mc`：`multiclass`
- `tvt`：`train_val_test`
- `sc`：same cohort，表示用 `--reference-manifest` 对齐到 teacher
- `fold0` 到 `fold4`：`LIDC-IDRI` 第几折

推荐示例：

- `outputs/ct_only_mvn_tvt`
- `outputs/text_only_mvn_tvt`
- `outputs/ct_text_mvn_sc_tvt`
- `outputs/ct_cnv_text_teacher_mvn_tvt`
- `outputs/ct_student_kd_mvn_tvt`
- `outputs/ct_text_student_kd_mvn_tvt`
- `outputs/ct_only_mc_tvt`
- `outputs/ct_cnv_text_teacher_mc_tvt`
- `outputs/lidc_bvm_resnet3d18_fold0`
- `outputs/lidc_bvm_resnet3d18_student_init_fold0`

## 4. 1500 例内网数据公共底座

二分类和三分类共用这一层，不建议分开维护两套预处理。

### 4.1 CT 数据底座

详细见：

- [`INTRANET_CT_PREPROCESS.md`](./INTRANET_CT_PREPROCESS.md)

主入口：

```bash
python prepare_intranet_ct_npy.py \
  --source-csv <OLD_MULTI_MODAL_CSV> \
  --source-data-root <SOURCE_DATA_ROOT> \
  --output-root <NPY_OUTPUT_ROOT> \
  --manifest-out outputs/intranet_rebuild_manifest.csv \
  --qc-csv outputs/intranet_rebuild_qc.csv \
  --summary-json outputs/intranet_rebuild_summary.json \
  --overwrite
```

这一步完成后，后续内网训练常用的几个核心输入会固定下来：

- `<NPY_OUTPUT_ROOT>`
- `<METADATA_CSV>`
- `<CT_ROOT>`

通常可以理解为：

- `<METADATA_CSV> = outputs/intranet_rebuild_manifest.csv`
- `<CT_ROOT> = <NPY_OUTPUT_ROOT>`

### 4.2 文本特征底座

主入口：

```bash
python prepare_text_features.py \
  --output-tsv outputs/text_features_v1.tsv \
  --text-health-csv <TEXT_HEALTH_CSV> \
  --text-disease-csv <TEXT_DISEASE_CSV> \
  --bert-model-path <LOCAL_BERT_DIR> \
  --embedding-backend bert \
  --batch-size 8 \
  --max-length 128
```

后续统一使用：

- `<TEXT_TSV> = outputs/text_features_v1.tsv`

### 4.3 CNV 底座

如果基因表已经准备好，后续统一使用：

- `<GENE_TSV>`

`CNV only` 的详细对照实验见：

- [`CNV_XGBOOST_BASELINE.md`](./CNV_XGBOOST_BASELINE.md)

## 5. 主线 A：1500 例二分类执行顺序

这一条线建议继续作为当前论文主线。

### A0. CT only 二分类基线

详细见：

- [`BINARY_CT_STAGE_NOTES.md`](./BINARY_CT_STAGE_NOTES.md)

推荐入口：

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_mvn_tvt \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --use-3d-input \
  --model resnet3d18 \
  --depth-size 128 \
  --volume-hw 256 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc
```

### A1. CNV only 二分类对照

这条线是补充单模态对照，不是主线，但建议保留。

```bash
python train_cnv_xgboost.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_only_mvn_tvt \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc
```

如果要做更稳的 `CNV only`：

```bash
python train_cnv_xgboost_sweep.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_only_mvn_formal \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc \
  --preset formal
```

### A2. Text only 二分类

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --output-dir outputs/text_only_mvn_tvt \
  --modalities text \
  --text-feature-tsv <TEXT_TSV> \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc
```

### A3. CT + CNV 二分类

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_mvn_tvt \
  --modalities ct,cnv \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc
```

### A4. CT + Text 二分类

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --text-feature-tsv <TEXT_TSV> \
  --output-dir outputs/ct_text_mvn_tvt \
  --modalities ct,text \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc
```

### A5. CT + CNV + Text teacher

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --text-feature-tsv <TEXT_TSV> \
  --output-dir outputs/ct_cnv_text_teacher_mvn_tvt \
  --modalities ct,cnv,text \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric auroc
```

### A6. same-cohort 普通 baseline

这一步非常重要，用来做“普通模型 vs KD student”的公平比较。

teacher 跑完后，使用：

- `outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv`

作为统一参考划分。

`CT only same-cohort`：

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_mvn_sc_tvt \
  --modalities ct \
  --reference-manifest outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --selection-metric auroc
```

`CT + Text same-cohort`：

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --text-feature-tsv <TEXT_TSV> \
  --output-dir outputs/ct_text_mvn_sc_tvt \
  --modalities ct,text \
  --reference-manifest outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --selection-metric auroc
```

### A7. CT student + KD

```bash
python train_student_kd.py \
  --output-dir outputs/ct_student_kd_mvn_tvt \
  --modalities ct \
  --teacher-run-dir outputs/ct_cnv_text_teacher_mvn_tvt \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --split-mode train_val_test \
  --selection-metric auroc \
  --distillation-alpha 0.5 \
  --distillation-temperature 4.0
```

### A8. CT + Text student + KD

```bash
python train_student_kd.py \
  --output-dir outputs/ct_text_student_kd_mvn_tvt \
  --modalities ct,text \
  --text-feature-tsv <TEXT_TSV> \
  --teacher-run-dir outputs/ct_cnv_text_teacher_mvn_tvt \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --split-mode train_val_test \
  --selection-metric auroc \
  --distillation-alpha 0.5 \
  --distillation-temperature 4.0
```

### A9. 结果导出

统一结果表：

```bash
python export_experiment_table.py \
  --run ct_only=outputs/ct_only_mvn_tvt \
  --run cnv_only=outputs/cnv_only_mvn_tvt \
  --run text_only=outputs/text_only_mvn_tvt \
  --run ct_cnv=outputs/ct_cnv_mvn_tvt \
  --run ct_text=outputs/ct_text_mvn_tvt \
  --run teacher=outputs/ct_cnv_text_teacher_mvn_tvt \
  --run ct_only_sc=outputs/ct_only_mvn_sc_tvt \
  --run ct_text_sc=outputs/ct_text_mvn_sc_tvt \
  --run ct_student_kd=outputs/ct_student_kd_mvn_tvt \
  --run ct_text_student_kd=outputs/ct_text_student_kd_mvn_tvt \
  --output-dir outputs/table_mvn_tvt
```

单个 run 可视化：

```bash
python export_experiment_plots.py \
  --run-dir outputs/ct_text_student_kd_mvn_tvt \
  --output-dir outputs/ct_text_student_kd_mvn_tvt/plots \
  --split test
```

## 6. 主线 B：1500 例三分类执行顺序

这条线建议作为“补充验证线”，用于证明方法不是只对一个二分类标签折叠有效。

统一原则：

- 预处理仍然使用内网同一套 `CT`、文本、`CNV` 底座
- 与二分类尽量保持相同的骨干、输入和训练口径
- 三分类更推荐 `--selection-metric balanced_accuracy`，也可以使用 `auto`

### B0. CT only 三分类基线

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_mc_tvt \
  --class-mode multiclass \
  --use-3d-input \
  --model resnet3d18 \
  --depth-size 128 \
  --volume-hw 256 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric balanced_accuracy
```

### B1. CNV only 三分类对照

可选，不是必须，但保留会更完整。

```bash
python train_cnv_xgboost.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_only_mc_tvt \
  --class-mode multiclass \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric balanced_accuracy
```

### B2. Text only 三分类

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --output-dir outputs/text_only_mc_tvt \
  --modalities text \
  --text-feature-tsv <TEXT_TSV> \
  --class-mode multiclass \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric balanced_accuracy
```

### B3. CT + Text 三分类

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --text-feature-tsv <TEXT_TSV> \
  --output-dir outputs/ct_text_mc_tvt \
  --modalities ct,text \
  --class-mode multiclass \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric balanced_accuracy
```

### B4. CT + CNV + Text 三分类 teacher

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --text-feature-tsv <TEXT_TSV> \
  --output-dir outputs/ct_cnv_text_teacher_mc_tvt \
  --modalities ct,cnv,text \
  --class-mode multiclass \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --split-mode train_val_test \
  --use-predefined-split \
  --selection-metric balanced_accuracy
```

### B5. same-cohort 普通 baseline

`CT only same-cohort`：

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_mc_sc_tvt \
  --modalities ct \
  --reference-manifest outputs/ct_cnv_text_teacher_mc_tvt/split_manifest.csv \
  --class-mode multiclass \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --selection-metric balanced_accuracy
```

`CT + Text same-cohort`：

```bash
python train_multimodal.py \
  --data-root <CT_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --text-feature-tsv <TEXT_TSV> \
  --output-dir outputs/ct_text_mc_sc_tvt \
  --modalities ct,text \
  --reference-manifest outputs/ct_cnv_text_teacher_mc_tvt/split_manifest.csv \
  --class-mode multiclass \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --selection-metric balanced_accuracy
```

### B6. CT student + KD 三分类

```bash
python train_student_kd.py \
  --output-dir outputs/ct_student_kd_mc_tvt \
  --modalities ct \
  --teacher-run-dir outputs/ct_cnv_text_teacher_mc_tvt \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --class-mode multiclass \
  --split-mode train_val_test \
  --selection-metric balanced_accuracy \
  --distillation-alpha 0.5 \
  --distillation-temperature 4.0
```

### B7. CT + Text student + KD 三分类

```bash
python train_student_kd.py \
  --output-dir outputs/ct_text_student_kd_mc_tvt \
  --modalities ct,text \
  --text-feature-tsv <TEXT_TSV> \
  --teacher-run-dir outputs/ct_cnv_text_teacher_mc_tvt \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --class-mode multiclass \
  --split-mode train_val_test \
  --selection-metric balanced_accuracy \
  --distillation-alpha 0.5 \
  --distillation-temperature 4.0
```

### B8. 结果导出

```bash
python export_experiment_table.py \
  --run ct_only=outputs/ct_only_mc_tvt \
  --run text_only=outputs/text_only_mc_tvt \
  --run ct_text=outputs/ct_text_mc_tvt \
  --run teacher=outputs/ct_cnv_text_teacher_mc_tvt \
  --run ct_only_sc=outputs/ct_only_mc_sc_tvt \
  --run ct_text_sc=outputs/ct_text_mc_sc_tvt \
  --run ct_student_kd=outputs/ct_student_kd_mc_tvt \
  --run ct_text_student_kd=outputs/ct_text_student_kd_mc_tvt \
  --output-dir outputs/table_mc_tvt
```

## 7. 主线 C：LIDC-IDRI 二分类执行顺序

当前这条线已经可以完整走通，建议继续按 fixed fold 的方式做正式公开验证。

详细见：

- [`LIDC_IDRI_BENCHMARK_NOTES.md`](./LIDC_IDRI_BENCHMARK_NOTES.md)
- [`LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md`](./LIDC_IDRI_CONSENSUS_3D_WORKFLOW.md)

### C0. 构建 consensus split manifest

```bash
python build_lidc_idri_split_manifest.py \
  --input-root /workspace/data-lung/LIDC-IDRI \
  --output-dir outputs/lidc_bvm_manifest_consensus \
  --metadata-source auto \
  --annotation-policy consensus \
  --consensus-min-readers 2 \
  --xy-tolerance-px 15 \
  --z-tolerance-mm 3 \
  --label-policy score12_vs_score45 \
  --split-scheme patient_kfold \
  --n-splits 5 \
  --val-ratio 0.1 \
  --seed 42
```

### C1. 裁剪 fold 级 3D nodule crop

当前主线尺寸：

- `64 x 64 x 64`

```bash
python prepare_lidc_idri_consensus_3d.py \
  --input-root /workspace/data-lung/LIDC-IDRI \
  --manifest-csv outputs/lidc_bvm_manifest_consensus/nodule_manifest.csv \
  --split-manifest-csv outputs/lidc_bvm_manifest_consensus/split_manifest.csv \
  --split-fold 0 \
  --output-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --depth-size 64 \
  --volume-hw 64 \
  --context-scale 1.5 \
  --min-size-xy 32 \
  --min-size-z 8
```

### C2. clean baseline

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --output-dir outputs/lidc_bvm_resnet3d18_fold0 \
  --model resnet3d18 \
  --pretrained \
  --use-3d-input \
  --depth-size 64 \
  --volume-hw 64 \
  --class-mode binary \
  --binary-task benign_vs_malignant \
  --use-predefined-split \
  --split-manifest-csv /workspace/data-lung/lidc_idri_consensus_3d_fold0/processed_split_manifest.csv \
  --split-fold 0 \
  --split-mode train_val_test \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --selection-metric auroc
```

### C3. student-init 迁移对照

这一步验证的是 student 学到的 `CT` 表征是否更可迁移，不是内网主任务的同任务外部验证。

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --output-dir outputs/lidc_bvm_resnet3d18_student_init_fold0 \
  --model resnet3d18 \
  --use-3d-input \
  --depth-size 64 \
  --volume-hw 64 \
  --class-mode binary \
  --binary-task benign_vs_malignant \
  --use-predefined-split \
  --split-manifest-csv /workspace/data-lung/lidc_idri_consensus_3d_fold0/processed_split_manifest.csv \
  --split-fold 0 \
  --split-mode train_val_test \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --selection-metric auroc \
  --init-checkpoint outputs/ct_student_kd_mvn_tvt/best_model.pt \
  --init-checkpoint-prefix ct_encoder.
```

### C4. fold 汇总

`LIDC-IDRI` 不建议只看单折。

推荐：

- 跑完 `fold0` 到 `fold4`
- baseline 和 student-init 使用完全相同的 fold
- 最后汇总 `mean ± std`

## 8. 推荐实际执行顺序

如果从今天开始按一份文档走，最推荐的顺序是：

1. 先稳定主线 A 的二分类 formal run。
2. 在主线 A 中优先拿到 teacher、same-cohort baseline、student KD 的公平对照。
3. 再补主线 B 的三分类，验证增益是否仍然存在。
4. 再做主线 C 的 `LIDC-IDRI clean baseline`。
5. 最后做 `LIDC-IDRI student-init`，验证 `CT` 表征迁移。

换句话说，优先级是：

- 先把“基因信息能否迁移到无基因 student”这件事在内网真实数据上证明清楚
- 再把“这种 student 学到的 `CT` 表征是否能迁移到公开数据”这件事补强

## 9. 每条线的主要结论应该怎么比较

### 9.1 二分类主线最关键的比较

- `CT only same-cohort` vs `CT student + KD`
- `CT + Text same-cohort` vs `CT + Text student + KD`
- `CT + CNV + Text teacher` vs `CT only`

### 9.2 三分类补充线最关键的比较

- `CT only same-cohort` vs `CT student + KD`
- `CT + Text same-cohort` vs `CT + Text student + KD`
- `teacher` 是否仍然保持最强或接近最强

### 9.3 LIDC-IDRI 公开验证线最关键的比较

- `clean baseline` vs `student-init`

## 10. 当前阶段最建议看的指标

二分类正式报告建议至少看：

- `AUROC`
- `AUPRC`
- `accuracy`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `F1`
- `MCC`

三分类正式报告建议至少看：

- `accuracy`
- `balanced_accuracy`
- `macro F1`
- 每类 `precision / recall / F1`

## 11. 一句话总结

当前最不容易乱的做法是：

- 内网真实数据：统一用 `prepare_intranet_ct_npy.py + prepare_text_features.py + train_multimodal.py + train_student_kd.py`
- 内网 `CT only` 和 `LIDC-IDRI`：统一用 `train.py`
- 二分类作为当前论文主线
- 三分类作为补充验证
- `LIDC-IDRI` 作为公开迁移验证

后面只要新结果继续沿着这份文档的命名和阶段往下补，横向比较就会非常直观。
