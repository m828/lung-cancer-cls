# LIDC-IDRI Consensus 3D 工作流

这份文档记录当前项目中更干净、更接近文献设定的 `LIDC-IDRI` 原始数据处理与训练流程。

## 1. 目标

基于原始 `LIDC-IDRI` DICOM + XML 数据，构建一个更接近文献常见设定的 `良性 vs 恶性` 公开 benchmark：

1. 先生成病人级 `split_manifest.csv`
2. 再把 XML 中多个 reader 的标注聚合成 `consensus nodule`
3. 再从原始 DICOM 中裁出 3D 结节块
4. 最后在固定 fold 上训练，而不是在 `train.py` 里再次随机切分

## 2. 当前使用的标签折叠规则

当前这条主线实际使用的是：

- `S1、S2 -> 良性`
- `S4、S5 -> 恶性`
- `S3 -> 丢弃`

对应参数是：

- `--label-policy score12_vs_score45`

这是当前更推荐的主线，因为它更接近文献里常见的 `clear-cut benign vs malignant` 设定。

项目中也支持其它折叠方式，例如：

- `score123_vs_score45`
  即 `S1、S2、S3 -> 良性`，`S4、S5 -> 恶性`
- `score12_vs_score345`
  即 `S1、S2 -> 良性`，`S3、S4、S5 -> 恶性`

但这两个都不是当前这轮实验实际使用的规则。

## 3. 第一步：生成 Consensus Split Manifest

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

预期输出：

- `nodule_manifest.csv`
- `split_manifest.csv`
- `patient_summary.csv`
- `summary.json`

建议先检查：

- 打开 `summary.json`
- 确认 `annotation_policy=consensus`
- 确认类别分布合理
- 确认 mixed-label patient 比旧的 reader-level 版本明显下降

## 4. 第二步：裁剪 Consensus 3D 结节块

当前主线推荐尺寸先用：

- `64 x 64 x 64`

这比之前的 `32 x 128 x 128` 更接近 `LIDC-IDRI` 3D 结节分类文献里常见的立方体输入设定。

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

预期输出：

- `benign/*.npy`
- `malignant/*.npy`
- `processed_manifest.csv`
- `processed_split_manifest.csv`
- `preprocess_summary.json`
- `preprocess_failures.csv`

建议先检查：

- `processed_split_manifest.csv` 是否只包含当前 fold
- 输出文件名是否与 `output_stem` 对应
- `preprocess_failures.csv` 是否为空，或只有极少数可解释失败

## 5. 第三步：训练公开数据 clean baseline

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

说明：

- `split_manifest_csv` 现在会让 `LIDCIDRIDataset` 直接挂上固定的 `train/val/test` 元数据
- 当 `lidc_idri` 同时传入 `split_manifest_csv` 时，`train.py` 会自动启用 `use_predefined_split`
- 这是当前更适合论文级可复现性的训练方式

## 6. 第四步：Student 初始化迁移对照

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
  --init-checkpoint outputs/ct_student_kd_v1_tvt/best_model.pt \
  --init-checkpoint-prefix ct_encoder.
```

同一 fold 下，重点比较：

- `AUROC`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `MCC`

## 7. 推荐汇报方式

不要只报一折。更推荐：

1. 跑完整 `5` 折
2. 导出每一折结果
3. 汇总成 `mean ± std`

论文叙事里更稳妥的写法是：

- 内部真实数据验证多模态 teacher-student 方法是否成立
- `LIDC-IDRI` 用于验证 CT 表征迁移和公开 benchmark 表现
- 不把它写成“与内部主任务完全同定义的外部验证”

## 8. 实验记录

### 2026-04-06：Fold 0 Clean Baseline

实验设置：

- manifest：`outputs/lidc_bvm_manifest_consensus`
- 标签规则：`S1、S2 -> 良性；S4、S5 -> 恶性；S3 丢弃`
- crop size：`64 x 64 x 64`
- model：`resnet3d18`
- task：`benign_vs_malignant`
- split：`patient-wise fold 0`
- checkpoint metric：`AUROC`

预处理结果：

- `num_manifest_rows = 590`
- `num_processed = 590`
- `num_failed = 0`
- 类别数：`benign = 364`，`malignant = 226`

训练划分结果：

- 训练时实际发现样本数：`579`
- train：`421`
- val：`54`
- test：`104`

最佳验证 checkpoint：

- epoch：`11`
- `val AUROC = 0.9971`
- `val AUPRC = 0.9955`
- `val ACC = 0.9630`
- `val BACC = 0.9500`
- `val sensitivity = 0.9000`
- `val specificity = 1.0000`
- `val MCC = 0.9220`

测试集结果：

- `test AUROC = 0.9590`
- `test AUPRC = 0.9085`
- `test ACC = 0.9135`
- `test BACC = 0.9207`
- `test sensitivity = 0.9429`
- `test specificity = 0.8986`
- `test precision = 0.8250`
- `test F1 = 0.8800`
- `test MCC = 0.8172`
- confusion matrix：`[[62, 7], [2, 33]]`

结果解读：

- 这是一组不错的第一折 clean baseline，已经足够支持后续继续跑 `student-init`
- 这组结果在测试集上偏高召回，`sensitivity` 很高，`precision` 相对略低
- 当前验证集和测试集结果都支持继续用 `AUROC` 作为 checkpoint 保留指标

训练后发现的重要数据问题：

- `processed_manifest.csv` 有 `590` 行
- 磁盘上实际 `.npy` 文件只有 `579` 个
- `output_stem.nunique() = 579`
- 说明有 `11` 个样本在预处理导出阶段发生了文件名碰撞并被覆盖

当前结论：

- 这次 `fold 0` 结果仍然可以作为阶段性 baseline 参考
- 但在正式做多折汇总或 `student-init` 严格对比前，建议先修复预处理导出的唯一命名问题
