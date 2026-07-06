# Huawei Split Compare Audit

生成时间：2026-07-06

## 结论

当前工作区已经新增独立对照实验框架，但本机未挂载真实内网数据，因此没有伪造华为 8:2 或 7:1:2 的实际 ID manifest。

阻断项：

- 缺失 CT CSV：`/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_文本划分0205_修复.csv`
- 缺失 CT root：`/home/apulis-dev/userdata/Data/CT1500`
- 缺失 gene TSV：`/home/apulis-dev/userdata/Data/Gene/FDEM_CNV_merge_pcc.tsv`
- 未发现可直接使用的真实文本特征 TSV 或华为 `dataset_cache/processed_df.csv + features.npz`
- 新增训练包装器在三模态模式下会提前阻断缺失文本特征的启动，避免假装完成三模态训练。

因此：split 生成脚本、完整性检查脚本、训练启动脚本均已准备好；实际样本 ID、重叠分析和性能表需要在内网数据挂载后运行生成。

## 代码审计

### 华为原始代码

来源：`/workspace/MultiModel_Code_2(1).zip`

关键文件：

- `MultiModel_Code_2/datasets/eval_dataset.py`
- `MultiModel_Code_2/datasets/ct_base_mm_dataset.py`

确认点：

- 原始 CT 主队列来自 CSV。
- `CT_train_val_split` 是原始约 8:2 split 的核心列。
- `_load_csv_data()` 会过滤标签无效、CT 路径为空、CT 文件不存在的样本。
- `split_predefined()` 使用 CSV 中的 `train` 作为 train，但会把 CSV 中的 `test` 再按标签拆成 val/test 两半。这一点和“CT train=1396, test=374”的原始 CSV 口径不同，后续比较时必须说明。
- 本实验框架中的 `huawei_8_2/split_manifest.csv` 会严格保留 CSV 原始 `train/test` 标签，不重新随机 8:2。

### 当前改造代码

来源：`/workspace/multimodal_unpacked/MultiModel_Code_2`

关键文件：

- `datasets/eval_dataset.py`
- `datasets/ct_base_mm_dataset.py`
- `train_text.py`
- `train_gene_multimodal.py`
- `train_mm_ensemble.py`

确认点：

- `split_train_val_test()` 使用 CT 全队列按标签分层生成 7:1:2。
- `split_5_fold()` 使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`。
- 当前 5-fold 口径是 train/val CV，`test_idx` 为空占位。
- 文本和基因应在 CT split 内过滤可用样本，不能独立重划分。

### lung-cancer-cls

来源：`/workspace/lung-cancer-cls`

确认点：

- 旧入口 `train_ft_multis.py` 调用 `data_process2.split_dataset_by_label(dataset, train_ratio=0.8)`，会自行随机 8:2，不适合本次严谨对照。
- 新入口 `train_multimodal.py` 包装 `src/lung_cancer_cls/multimodal_teacher_student.py`。
- `train_multimodal.py` 支持 `--reference-manifest`，可按外部 `split_manifest.csv` 固定 train/val/test。
- 该入口对三模态使用 inner merge，因此 `modalities=ct,text,cnv` 时实际训练/测试样本是 CT split 内三模态可用交集。

## Split Manifest 设计

新增目录：

- `splits/huawei_8_2/`
- `splits/split_7_1_2/`
- `splits/split_5_fold/`

每个 manifest 包含：

- `sample_id`
- `patient_id`
- `label`
- `label_name`
- `split`
- `fold_idx`
- `ct_path`
- `ct_rel_path`
- `text_id`
- `gene_id`
- `has_text`
- `has_gene`
- `row_index`
- `source_split`

约束：

- CT 是主队列。
- 文本 train/val/test 只从对应 CT split 中 `has_text=True` 的样本得到。
- 基因 train/val/test 只从对应 CT split 中 `has_gene=True` 的样本得到。
- 最终三模态样本是当前 CT split 内 `has_text=True and has_gene=True` 的交集。
- 不允许 text/gene 独立随机划分。
- 不允许跨 split 补样本。

## 样本数

当前本机无法生成真实样本数。用户提供的背景目标值如下，尚未在本机复核：

| split | CT train | CT val | CT test | text train | text val | text test | gene train | gene val | gene test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Huawei 原始 8:2 | 1396 | - | 374 | 1020 | - | 283 | 1017 | - | 281 |
| 7:1:2 示例 | 约 1237/1238 | 约 176 | 约 356/357 | 待生成 | 待生成 | 待生成 | 待生成 | 待生成 | 待生成 |

内网运行后，`splits/*/split_stats.json` 和 `reports/integrity_report.json` 会给出实际计数。

## 标签分布

当前本机无法生成真实标签分布。内网运行后由以下文件给出：

- `splits/huawei_8_2/split_stats.json`
- `splits/split_7_1_2/split_stats.json`
- `splits/split_5_fold/split_stats.json`
- `reports/integrity_report.json`

标签映射固定为：

- `健康对照 -> 0`
- `良性结节 -> 1`
- `肺癌 -> 2`

## Test Set 重叠分析

当前本机无法计算真实 overlap。内网运行：

```bash
python experiments/huawei_split_compare/check_split_integrity.py
```

会输出：

- 原始 8:2 test 数量
- 7:1:2 test 数量
- overlap 数量
- overlap ratio
- 只在 8:2 test 中的样本
- 只在 7:1:2 test 中的样本

## 性能对比表

当前未在工作区中发现真实三模态 93% 日志，也没有本次对照训练结果。

| 实验 | acc | macro-F1 | AUC | 备注 |
|---|---:|---:|---:|---|
| 华为原始 8:2 | 待日志提取 | 待日志提取 | 待日志提取 | 需确认是否最终三模态交集口径 |
| lung-cancer-cls + Huawei 8:2 | 待训练 | 待训练 | 待训练 | 使用 `huawei_8_2/split_manifest.csv` |
| 华为改造 7:1:2 | 待日志提取 | 待日志提取 | 待日志提取 | 需确认 seed/fold/交集样本 |
| lung-cancer-cls + 7:1:2 | 待训练 | 待训练 | 待训练 | 使用 `split_7_1_2/split_manifest.csv` |

## 当前可执行命令

如果有华为文本缓存，先导出 text feature TSV：

```bash
python experiments/huawei_split_compare/export_huawei_text_features.py \
  --text_cache_dir /path/to/huawei/dataset_cache \
  --output_tsv experiments/huawei_split_compare/text_features/huawei_text_features.tsv
```

生成 split manifest：

```bash
python experiments/huawei_split_compare/generate_split_manifests.py
```

检查 split 完整性：

```bash
python experiments/huawei_split_compare/check_split_integrity.py
```

运行 Huawei 原始 8:2 对照：

```bash
bash experiments/huawei_split_compare/run_lungcls_huawei_8_2_seed42.sh
```

运行 7:1:2 对照：

```bash
bash experiments/huawei_split_compare/run_lungcls_7_1_2_seed42.sh
```

## 不能直接下结论的点

- 原始 93% 是否来自最终三模态交集 test set，当前没有日志和真实 split ID 无法确认。
- 原始华为 `split_predefined()` 是否在该 93% 实验中把 CSV test 再拆为 val/test，当前需要结合原始启动命令或日志确认。
- `lung-cancer-cls` 的 text feature 输入需要明确：推荐使用华为文本缓存导出的 TSV，并将 `--metadata_text_id_col` 与 `--text_record_id_col` 设为同一 ID 列。
