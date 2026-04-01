# 文本模态、多模态教师网络与蒸馏说明

## 1. 这轮整合完成了什么

这轮把 `C:\Users\a1823.MMY\Desktop\output\test_all3.py` 里的旧文本训练思路正式并入项目主线，补成了可复用、可组合、可蒸馏的版本。

当前已经支持：

- `Text only` 训练
- `CT + CNV` 训练
- `CT + Text` 训练
- `CT + CNV + Text` 训练
- `CT` 学生网络蒸馏
- `CT + Text` 学生网络蒸馏
- 统一结果表导出
- 阶段记录与推荐命令

主入口脚本：

- `prepare_text_features.py`
- `train_multimodal.py`
- `train_student_kd.py`
- `export_experiment_table.py`

## 2. `test_all3.py` 在项目中的对应关系

旧脚本里的“文本模态”其实不是纯自由文本，而是：

- 结构化临床数值特征
- 多列病历文本拼接后的 `BERT CLS embedding`
- 数值分支和文本分支的注意力融合

项目里现在的对应实现是：

- `src/lung_cancer_cls/text_clinical.py`
- `src/lung_cancer_cls/multimodal_teacher_student.py`

因此这里更准确的名字是：

- `TextClinical` 模态

也就是说，这里的 `text` 实际上表示：

- `clinical numeric + text embedding`

## 3. 当前推荐的脚本使用方式

如果只是跑旧的 `CT + CNV` 参考线，仍然可以继续用：

- `train_ct_cnv.py`

但如果你现在要做：

- 文本单模态
- `CT + Text`
- `CT + CNV + Text`
- teacher / student / KD

更推荐统一切到：

- `train_multimodal.py`
- `train_student_kd.py`

这样所有模态组合都走同一套：

- 对齐逻辑
- split 逻辑
- 结果保存格式
- `metrics.json`
- `split_manifest.csv`
- 结果表导出

## 4. 文本特征准备

如果你已经有准备好的文本特征表：

- 直接传 `--text-feature-tsv`

如果你只有原始健康/疾病文本 CSV：

- 先跑 `prepare_text_features.py`

推荐命令：

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

输出：

- `text_features_v1.tsv`
- `text_features_v1.tsv.meta.json`

## 5. 推荐实验顺序

当前更实用的顺序不是先把单模态抠到极致，而是先把整条 teacher/student 链路搭稳：

1. `Text only`
2. `CT + CNV`
3. `CT + Text`
4. `CT + CNV + Text` teacher
5. `CT` student + KD
6. `CT + Text` student + KD

这样做的目的不是证明每个单模态都最强，而是先确认：

- 文本是否提供增益
- `CNV` 是否和文本一起形成互补
- 全模态 teacher 是否稳定优于 `CT only`
- 学生网络能否继承多模态能力

## 6. 推荐命令模板

### 6.1 Text only

```bash
python train_multimodal.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --output-dir outputs/text_only_v1 \
  --modalities text \
  --text-feature-tsv outputs/text_features_v1.tsv \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --epochs 30 \
  --batch-size 16 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 6.2 CT + CNV

```bash
python train_multimodal.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_mm_v1 \
  --modalities ct,cnv \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 6.3 CT + Text

```bash
python train_multimodal.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --text-feature-tsv outputs/text_features_v1.tsv \
  --output-dir outputs/ct_text_v1 \
  --modalities ct,text \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 6.4 CT + CNV + Text Teacher

```bash
python train_multimodal.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --text-feature-tsv outputs/text_features_v1.tsv \
  --output-dir outputs/ct_cnv_text_teacher_v1 \
  --modalities ct,cnv,text \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 6.5 CT Student + KD

```bash
python train_student_kd.py \
  --output-dir outputs/ct_student_kd_v1 \
  --modalities ct \
  --teacher-run-dir outputs/ct_cnv_text_teacher_v1 \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --distillation-alpha 0.5 \
  --distillation-temperature 4.0
```

### 6.6 CT + Text Student + KD

```bash
python train_student_kd.py \
  --output-dir outputs/ct_text_student_kd_v1 \
  --modalities ct,text \
  --text-feature-tsv outputs/text_features_v1.tsv \
  --teacher-run-dir outputs/ct_cnv_text_teacher_v1 \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --distillation-alpha 0.5 \
  --distillation-temperature 4.0
```

## 7. 为什么现在推荐这样做

当前更应该验证的问题是：

- `Text` 是否能补 `CT`
- `CT + CNV + Text` 是否稳定优于 `CT`
- `CT student` 或 `CT + Text student` 能否继承全模态 teacher 的排序能力

不推荐一上来就把文本或 `CNV` 单模态各自抠到极限，因为那样很容易花大量时间在局部 1 个点上，却没有回答真正决定后续蒸馏价值的核心问题：

- 多模态 teacher 到底有没有显著增益

## 8. 结果文件与对齐方式

这套实现会额外保存：

- `metrics.json`
- `best_model.pt`
- `split_manifest.csv`
- `train_predictions.csv / val_predictions.csv / test_predictions.csv`

其中 `split_manifest.csv` 很关键：

- teacher 保存自己的实际训练划分
- student 会优先复用 teacher 的这个划分
- 这样可以避免 teacher/student 因随机切分不同而无法公平对比

## 9. 推荐的结果表导出

```bash
python export_experiment_table.py \
  --run text_only=outputs/text_only_v1 \
  --run ct_cnv=outputs/ct_cnv_mm_v1 \
  --run ct_text=outputs/ct_text_v1 \
  --run teacher=outputs/ct_cnv_text_teacher_v1 \
  --run student_ct=outputs/ct_student_kd_v1 \
  --run student_ct_text=outputs/ct_text_student_kd_v1 \
  --output-dir outputs/text_teacher_student_summary
```

现在导表会自动识别这些 family：

- `text_only`
- `ct_cnv`
- `ct_text`
- `ct_cnv_text`
- `student_kd`

## 10. 这轮实现的本地验证

本地已完成的最小验证：

- `text_clinical.py` 生成哈希文本特征 smoke test
- `Text only` 训练 smoke test
- `CT + CNV + Text` teacher 训练 smoke test
- `CT student + KD` smoke test
- `experiment_table` 对新 family 的识别 smoke test

注意：

- 这里的 smoke 只验证训练闭环和保存逻辑
- 正式实验仍然以你的真实数据和固定 split 为准
