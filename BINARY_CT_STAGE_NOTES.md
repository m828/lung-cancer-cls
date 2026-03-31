# 二分类阶段记录与后续计划

## 1. 当前共识

基于目前数据情况，项目主线从“三分类”切换到“二分类”更合理。

当前优先任务定义为：

- 主任务：`恶性(肺癌) vs 健康`
- 主数据来源：`csv` 对应的多模态队列
- 当前优先模态：`CT only`
- 当前不作为主线的模态：`Text`
- 后续多模态方向：`CT + CNV` 教师网络，`CT` 学生网络

这样做的原因：

- 良性样本数量少且质量不稳，会持续拖累三分类结果。
- 文本目前存在时序错位、术后复查等噪声，不适合先当核心增益点。
- 与 `Evaluation and integration of cell-free DNA signatures for detection of lung cancer` 这类二分类工作更容易衔接。

## 2. 本阶段已经完成什么

本阶段主要完成了统一训练框架在 `CT only + 二分类` 方向上的基础能力补齐。

### 2.1 任务层面

在统一训练框架中新增了二分类折叠方式：

- `malignant_vs_normal`

该模式会：

- 仅保留 `normal(0)` 和 `malignant(2)` 样本
- 自动剔除 `benign(1)` 样本
- 映射为二分类标签：`normal=0, malignant=1`

代码位置：

- `src/lung_cancer_cls/train.py`

### 2.2 评估层面

原来训练主流程更偏向只看 `acc`。

本阶段补充了二分类更合适的评估输出，`metrics.json` 中现在会写入：

- `accuracy`
- `balanced_accuracy`
- `auroc`
- `auprc`
- `precision`
- `recall`
- `sensitivity`
- `specificity`
- `f1`
- `brier_score`
- `confusion_matrix`

其中当前最推荐的选模指标为：

- `AUROC`

### 2.3 选模层面

新增参数：

- `--selection-metric`

支持：

- `auto`
- `accuracy`
- `balanced_accuracy`
- `auroc`
- `auprc`
- `f1`
- `loss`

推荐设置：

- 二分类：`--selection-metric auroc`
- 若用 `auto`，二分类默认也会走 `auroc`

### 2.4 入口层面

项目正式训练入口保持不变：

- 根目录 `train.py`

也就是说，后续实验仍然按 README 的方式运行：

```bash
python train.py ...
```

## 3. 本阶段新增了什么

本阶段新增/确认的内容如下。

### 3.1 新增功能

- 二分类标签折叠：`malignant_vs_normal`
- 选模指标参数：`--selection-metric`
- 二分类完整评估指标输出
- 根目录 `train.py` 仍作为正式入口

### 3.2 当前建议的评估原则

以后不再把 `acc` 作为主判断标准。

当前建议：

- 选模主指标：`AUROC`
- 同时观察：`balanced_accuracy`、`sensitivity`、`specificity`、`f1`
- 输出保留：`confusion_matrix`

## 4. 这一阶段还没做什么

以下内容还没有开始或还没有完成，不应混淆为“已经做完”。

- `CNV only` 的 `XGBoost` 复现
- 公平对齐的 `CT/CNV` 同病人集合构建
- 教师网络 `CT + CNV`
- 学生网络 `CT + KD`
- 文本重新清洗后的重新接入
- 与经典/近两年模型的正式对比实验表

## 5. 下一阶段继续做什么

后续建议严格按下面顺序推进。

### Step 1. 先把 CT only 做稳

目标：

- 在 `恶性 vs 健康` 任务上拿到稳定、可信、可复现实验结果
- 确定后续学生网络的 CT 主干

推荐模型优先级：

1. `resnet3d18`
2. `densenet3d_121`
3. `attention3d_cnn`
4. `swin3d_tiny`

### Step 2. 复现 CNV only

目标：

- 用 `XGBoost` 或同类树模型复现 `CNV only`
- 得到后续教师网络的单模态基线

要求：

- 尽量和 CT 侧共用同一批病人、同一套 split

### Step 3. 教师网络

主版本：

- 输入：`CT + CNV`

扩展版本：

- 输入：`CT + CNV + Text`

但文本版本不作为当前第一优先级。

### Step 4. 学生网络

主版本：

- 输入：`CT`

关键对照：

- `CT only without KD`
- `CT only with KD`

真正要证明的是：

- 经过带基因信息的教师指导后，`CT student` 是否优于普通 `CT only`

## 6. 推荐的训练/验证策略

### 6.1 推荐的任务设置

统一使用：

```bash
--class-mode binary
--binary-task malignant_vs_normal
--selection-metric auroc
```

### 6.2 推荐的划分原则

最推荐：

- 固定病人级 `train/val/test`

如果当前 `csv` 已有稳定划分列并且可信：

- 使用 `--use-predefined-split`

如果当前 `csv` 只有 `train/test` 没有 `val`：

- 建议后续补一列病人级 `val`
- 或者先使用固定病人级重新生成 `train/val/test`

不建议：

- 扫描级随机拆分
- 不同模态各用不同 split

## 7. 当前阶段推荐的完整训练参数

下面给出当前阶段最推荐的 `CT only` 首轮实验参数。

### 7.1 共用参数模板

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --use-3d-input \
  --depth-size 32 \
  --volume-hw 128 \
  --aug-profile strong \
  --loss ce \
  --label-smoothing 0.05 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

说明：

- 如果显存不足，优先减小 `batch-size`
- 如果 `csv` 中没有可用的 `val` 划分，先不要强依赖 `--use-predefined-split`

### 7.2 推荐实验一：ResNet3D18

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_resnet3d18_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --model resnet3d18 \
  --use-3d-input \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --loss ce \
  --label-smoothing 0.05 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 7.3 推荐实验二：DenseNet3D121

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_densenet3d121_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --model densenet3d_121 \
  --use-3d-input \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 40 \
  --batch-size 6 \
  --lr 3e-4 \
  --aug-profile strong \
  --loss ce \
  --label-smoothing 0.05 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 7.4 推荐实验三：Attention3DCNN

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_attention3dcnn_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --model attention3d_cnn \
  --use-3d-input \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --loss ce \
  --label-smoothing 0.05 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 7.5 推荐实验四：Swin3D-Tiny

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --output-dir outputs/ct_only_swin3d_tiny_mvn \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --model swin3d_tiny \
  --use-3d-input \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 40 \
  --batch-size 4 \
  --lr 2e-4 \
  --aug-profile strong \
  --loss ce \
  --label-smoothing 0.05 \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

## 8. 当前阶段推荐的结果记录方式

每个实验目录建议至少保留：

- `best_model.pt`
- `metrics.json`
- 一份手动整理的结果表

建议人工汇总这些字段：

- 模型名
- 任务定义
- 样本数
- 训练/验证/测试划分方式
- `val AUROC`
- `test AUROC`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `f1`

## 9. 当前阶段一句话目标

先把 `CT only` 在 `恶性 vs 健康` 任务上做稳、做对、做可复现，再进入 `CNV only -> Teacher -> Student`。

## 10. 暂定论文题目

当前可以先暂定两个版本。

中文题目：

- `基于特权基因信息指导的肺癌CT二分类教师-学生框架`

英文题目：

- `Privileged Genomic Guidance for CT-based Binary Lung Cancer Detection: A Teacher-Student Framework`

如果后续主实验最终更突出“蒸馏”而不是“多模态融合”，英文标题还可以进一步收敛成：

- `Gene-Guided Distillation for CT-based Binary Lung Cancer Detection`

## 11. 本轮补充完成什么

本轮在上一阶段文档基础上，又补了下面这些内容：

- 在 [README.md](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/README.md) 中新增了阶段文档入口
- 新增了 `CNV only` 的独立训练入口 [train_cnv_xgboost.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/train_cnv_xgboost.py)
- 新增了核心实现 [cnv_xgboost.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/src/lung_cancer_cls/cnv_xgboost.py)
- 新增了说明文档 [CNV_XGBOOST_BASELINE.md](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/CNV_XGBOOST_BASELINE.md)
- 在 [requirements.txt](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/requirements.txt) 中补充了 `xgboost`
- 新增了基础测试 [test_cnv_xgboost.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/tests/test_cnv_xgboost.py)

这一轮完成的是“脚手架与口径统一”，不是“CNV only 最终结果已经跑完”。

当前已经具备的 `CNV only` 能力包括：

- 兼容旧 `data_process2.py` 使用过的 `SampleID / 样本类型 / CT_train_val_split`
- 默认将 `gene tsv` 第一列视为 `gene id`，第二列从特征中剔除
- 支持 `malignant_vs_normal`
- 支持重复 `SampleID` 去重和冲突样本丢弃
- 输出和 `CT only` 对齐的二分类评估指标

## 12. 接下来最实际的顺序

后面建议按这个顺序往前推：

1. 先真正跑通 `CT only` 首轮实验
2. 再用 [train_cnv_xgboost.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/train_cnv_xgboost.py) 跑 `CNV only`
3. 固定公平对齐集合
4. 再做 `CT + CNV` 教师网络
5. 最后做 `CT student` 和 `KD` 对照

## 13. 本轮交流后的最新判断

结合目前已经跑出的结果与新增脚手架，当前更合理的推进方式是：

- `CT only` 不再长时间为单模态 1 个点死扣
- `CNV only` 仍值得再做一轮更规范的验证，因为当前 `AUROC≈0.89` 还不能直接判断是否已经到达该数据的正常水平
- 当前主任务应转向 `CT + CNV` 教师网络，尽快验证基因信息是否真的给 `CT` 带来稳定增益
- 只有在 `CT + CNV > CT only` 成立后，才值得继续推进 `KD`

当前阶段最重要的问题不再是：

- `CNV only` 能不能再多抠 1 个点

而是：

- `CNV` 对 `CT` 是否存在稳定、可复现的互补增益

## 14. 本轮新增的实用工具

本轮在仓库中又补了两类更实用的能力。

### 14.1 CNV sweep 收口工具

新增入口：

- [train_cnv_xgboost_sweep.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/train_cnv_xgboost_sweep.py)

核心实现：

- [cnv_xgboost_sweep.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/src/lung_cancer_cls/cnv_xgboost_sweep.py)

当前具备：

- 支持多 `seed`
- 支持轻量参数网格
- 支持 `fast / formal` 两档推荐 preset
- 自动输出 `leaderboard.csv / leaderboard.md / summary.json`

### 14.2 结果表导出工具

新增入口：

- [export_experiment_table.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/export_experiment_table.py)

核心实现：

- [experiment_table.py](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/src/lung_cancer_cls/experiment_table.py)

用途：

- 将 `CT only / CNV only / CT + CNV` 的 `metrics.json` 统一整理成一份结果表
- 自动输出 `csv + markdown`
- 方便后续写教师网络消融表、阶段汇总表和论文实验表

### 14.3 CT + CNV 教师网络说明文档

新增说明：

- [CT_CNV_MULTIMODAL_V1.md](/c:/Users/a1823.MMY/Desktop/output/lung-cancer-cls/CT_CNV_MULTIMODAL_V1.md)

其中已补充：

- `CT + CNV` v1 的结构说明
- 当前最推荐的 teacher 消融组合
- 结果表导出命令

## 15. 当前更推荐的 CNV sweep 策略

当前不建议一上来就做很宽的搜索。

更推荐两阶段：

### Step 1. 先跑 `fast`

目的：

- 检查 `gene tsv` 列结构
- 检查 split 是否对齐
- 检查 `best_iteration` 是否明显过小
- 快速确认当前 `CNV only` 大致量级

推荐命令：

```bash
python train_cnv_xgboost_sweep.py \
  --metadata-csv <METADATA_CSV> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/cnv_xgboost_sweep_mvn_fast \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --split-mode train_val_test \
  --use-predefined-split \
  --preset fast
```

### Step 2. 再跑 `formal`

目的：

- 得到当前阶段更正式的 `CNV only` 收口结果
- 观察多 seed 后的均值与波动
- 为后续教师网络提供更可信的单模态对照

推荐命令：

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

当前重点看：

- `leaderboard.csv`
- `leaderboard.md`
- `summary.json`
- 每个 run 的 `best_iteration / used_boosting_rounds / top_features`

## 16. 当前更推荐的 Teacher 消融方案

当前阶段不建议一下子把 teacher 分成很多复杂版本。

最实用的首批 teacher 组合是：

1. `CT only` 参考线
2. `CT + CNV` 主版本：`resnet3d18`
3. `CT + CNV` 轻量对照：`attention3d_cnn`
4. `CT + CNV` 2D 对照：`resnet3d18 + --disable-3d-input`

推荐命名：

- `outputs/ct_only_resnet3d18_mvn_formal`
- `outputs/ct_cnv_resnet3d18_mvn_v1`
- `outputs/ct_cnv_attention3d_mvn_v1`
- `outputs/ct_cnv_resnet3d18_2d_mvn_v1`

### 16.1 Teacher 主版本

```bash
python train_ct_cnv.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_resnet3d18_mvn_v1 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --optimizer adamw \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 16.2 Teacher 轻量对照

```bash
python train_ct_cnv.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_attention3d_mvn_v1 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model attention3d_cnn \
  --depth-size 32 \
  --volume-hw 128 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --optimizer adamw \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

### 16.3 Teacher 2D 对照

```bash
python train_ct_cnv.py \
  --data-root <DATA_ROOT> \
  --metadata-csv <METADATA_CSV> \
  --ct-root <CT_ROOT> \
  --gene-tsv <GENE_TSV> \
  --output-dir outputs/ct_cnv_resnet3d18_2d_mvn_v1 \
  --class-mode binary \
  --binary-task malignant_vs_normal \
  --selection-metric auroc \
  --ct-model resnet3d18 \
  --disable-3d-input \
  --image-size 224 \
  --epochs 30 \
  --batch-size 8 \
  --lr 3e-4 \
  --aug-profile strong \
  --optimizer adamw \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --use-predefined-split
```

## 17. 当前阶段更推荐的结果导出方式

当你跑完第一批参考线与 teacher 消融后，建议不要手工抄表，直接导出统一结果表：

```bash
python export_experiment_table.py \
  --run ct_only=outputs/ct_only_resnet3d18_mvn_formal \
  --run cnv_formal=outputs/cnv_xgboost_sweep_mvn_formal/seed42_md3_mcw1_ss0.8_cs0.8_lr0.03_l21_gm0 \
  --run teacher_main=outputs/ct_cnv_resnet3d18_mvn_v1 \
  --run teacher_light=outputs/ct_cnv_attention3d_mvn_v1 \
  --run teacher_2d=outputs/ct_cnv_resnet3d18_2d_mvn_v1 \
  --output-dir outputs/stage_binary_summary
```

导出后会得到：

- `experiment_table.csv`
- `experiment_table.md`
- `summary.json`

更推荐的使用方式是：

- `csv` 用于后续继续补列、作图和对比
- `markdown` 用于直接粘进阶段记录、周报或实验备忘

## 18. 当前阶段一句话更新版目标

先把 `CNV only` 通过 `fast -> formal` 跑出可信区间，再用最小而清晰的一批 teacher 消融验证 `CT + CNV` 是否真实优于 `CT only`，最后才决定是否进入 `KD`。
