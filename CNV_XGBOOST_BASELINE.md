# CNV Only XGBoost 基线说明

## 1. 目标

这一阶段先把 `CNV only` 的二分类基线搭起来，并和当前的 `CT only` 主线保持一致的任务定义与指标口径。

当前最推荐的任务定义：

- 主任务：`恶性(肺癌) vs 健康`
- 参数对应：
  - `--class-mode binary`
  - `--binary-task malignant_vs_normal`

## 2. 和旧多模态代码的对齐关系

这个脚本直接按 [data_process2.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/data_process2.py) 的字段组织来兼容：

- `metadata csv` 默认读取：
  - `SampleID`
  - `样本类型`
  - `CT_train_val_split`
- `gene tsv` 默认处理：
  - 第一列作为 `gene id`
  - 第二列视为旧标签列并从特征中剔除
  - 其余列作为 `CNV` 特征

如果你的 `tsv` 列结构不同，可以通过：

- `--gene-id-col`
- `--gene-label-col`

显式指定。

## 3. 当前新增内容

新增入口：

- [train_cnv_xgboost.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/train_cnv_xgboost.py)

核心实现：

- [cnv_xgboost.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/src/lung_cancer_cls/cnv_xgboost.py)

当前能力：

- 支持 `multiclass / binary`
- 支持 `malignant_vs_normal / malignant_vs_rest / abnormal_vs_normal`
- 支持优先读取 `CT_train_val_split`
- 如果 `csv` 只有 `train/test`，会继续从 `train` 里切 `val`
- 如果没有可用预定义划分，会退回到分层随机划分
- 自动对重复 `SampleID` 去重
- 如果同一 `SampleID` 存在标签冲突或 `train/test` 冲突，会直接丢弃
- 输出与 `CT only` 尽量一致的二分类指标

## 4. 输出文件

运行后会在 `output-dir` 下生成：

- `best_model.json`
- `best_model.pkl`
- `feature_columns.json`
- `metrics.json`
- `train_predictions.csv`
- `val_predictions.csv`
- `test_predictions.csv`

`metrics.json` 中重点看：

- `auroc`
- `auprc`
- `balanced_accuracy`
- `precision`
- `recall / sensitivity`
- `specificity`
- `f1`
- `brier_score`
- `confusion_matrix`

当前最推荐的主指标仍然是：

- `AUROC`

## 5. 推荐命令

### 5.1 主推荐命令

```bash
python train_cnv_xgboost.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_xgboost_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --split-mode train_val_test \
  --use-predefined-split \
  --val-ratio 0.2 \
  --n-estimators 600 \
  --learning-rate 0.05 \
  --max-depth 4 \
  --min-child-weight 2 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-lambda 1.0 \
  --gamma 0.0 \
  --early-stopping-rounds 30 \
  --class-weight-strategy balanced
```

### 5.2 如果没有可用的预定义划分

```bash
python train_cnv_xgboost.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_xgboost_mvn_randomsplit \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --split-mode train_val_test \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --n-estimators 600 \
  --learning-rate 0.05 \
  --max-depth 4 \
  --min-child-weight 2 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-lambda 1.0 \
  --gamma 0.0 \
  --early-stopping-rounds 30 \
  --class-weight-strategy balanced
```

### 5.3 轻量 sweep / 多 seed

当你怀疑单次 `AUROC` 低于数据的正常水平时，优先做轻量重复实验，而不是直接上大规模搜索。

新增入口：

- `train_cnv_xgboost_sweep.py`
- 默认支持两档 preset：
  - `fast`
  - `formal`

推荐顺序：

1. 先跑 `fast`，确认数据、split 和指标口径没有问题
2. 再跑 `formal`，作为当前阶段更正式的 `CNV only` 收口结果

示例：

```bash
python train_cnv_xgboost_sweep.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_xgboost_sweep_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --split-mode train_val_test \
  --use-predefined-split \
  --preset fast
```

```bash
python train_cnv_xgboost_sweep.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_xgboost_sweep_mvn_formal \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --split-mode train_val_test \
  --use-predefined-split \
  --preset formal
```

输出重点看：

- `leaderboard.csv`
- `leaderboard.md`
- `summary.json`
- 每个 run 下的 `metrics.json`

`metrics.json` 里现在会额外记录：

- `best_iteration`
- `used_boosting_rounds`
- `best_score`
- `top_features`

`summary.json` 里会记录：

- 本次使用的 `preset`
- 实际计划运行数
- 最优 run 的关键指标

## 6. 当前建议

`CNV only` 这一步不要先追复杂模型，先把下面这些做稳：

- 任务定义固定为 `恶性 vs 健康`
- 尽量和 `CT only` 用同一套 split
- 指标以 `AUROC + sensitivity + specificity + balanced_accuracy` 为主
- 先拿到可信、可复现的基线，再进入教师网络

后续顺序仍然是：

- `CT only`
- `CNV only`
- `CT + CNV` 教师网络
- `CT` 学生网络
- `KD` 消融

## 7. 注意事项

- 这个脚本依赖 `xgboost`，已加入 [requirements.txt](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/requirements.txt)。
- 当前实现的是“可直接跑的 baseline 脚手架”，不是已经完成参数搜索的最终结果。
- 如果后续发现 `gene tsv` 的第二列并不是旧标签列，需要显式传 `--gene-label-col`，或者按实际表结构调整默认行为。
