# 层次化多任务分类阶段记录

## 1. 背景与动机

### 1.1 Subregion-Unet 调研

项目 README 中提到了参考 Subregion-Unet 思想的方向。经调研发现：

- Subregion-Unet 是**分割任务**，需要像素级结节 mask 标注
- 内网 CT 数据为 3D `.npy` 格式，现有 `DataGenerator` 仅支持 2D PIL 图像
- **无法直接在内网 3D CT 数据上运行**

### 1.2 现有级联方案的问题

用户已有的级联分类方案（`train_ft.py` + `eval_cot_0416.py`）：

- Stage 1: 二分类 `0 vs (1+2)` → 健康 vs 非健康
- Stage 2: 二分类 `1 vs 2` → 良性 vs 恶性（仅 Stage 1 判为非健康时触发）
- 准确率 **89.6%**，优于直接三分类的 **~86%**

级联方案的局限：

- **误差传播**：Stage 1 的错误会直接级联到 Stage 2，无法纠正
- 两个模型独立训练，无法共享梯度信号
- Stage 2 仅在 Stage 1 判为非健康时才运行，健康样本的特征无法帮助 Stage 2

### 1.3 核心问题：良性类别

- 良性样本仅约 **100 例**，远少于健康和恶性
- 良性在临床上是活检确认的可疑病例，影像特征与恶性高度相似
- 直接三分类时良性类别混淆严重

## 2. 方案设计：层次化多任务分类

受 Subregion-Unet "类别分组" 思想启发，提出**层次化多任务学习**方案。

### 2.1 核心思路

共享 3D backbone，同时训练三个分类任务：

| 任务 | 分组 | 说明 |
|------|------|------|
| Task A | `0` vs `(1,2)` | 健康 vs 非健康（所有样本参与） |
| Task B | `0, 1, 2` | 标准三分类（所有样本参与，**主输出**） |
| Task C | `1` vs `2` | 良性 vs 恶性（**仅非健康样本参与**） |

与级联方案的关键区别：

- 三个任务**并行训练**，共享 backbone 梯度
- Task A 和 Task C 的梯度信号**同时**反传到 backbone
- 推理时可选择直接用 Task B 输出，或用 Task A + Task C 级联

### 2.2 损失函数

```
L = w_abnormal * L_A + w_main * L_B + w_bm * L_C
```

- `L_A`: 健康 vs 非健康交叉熵
- `L_B`: 三分类交叉熵（支持 class_weights 和 label_smoothing）
- `L_C`: 良性 vs 恶性交叉熵（仅对非健康样本计算）

## 3. 已实现内容

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `train_hierarchical.py` | 完整训练脚本，支持层次化多任务训练和评估 |

### 3.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/lung_cancer_cls/model.py` | 新增 `HierarchicalMultiTaskClassifier` 类 |
| `src/lung_cancer_cls/training_components.py` | 新增 `HierarchicalMultiTaskLoss`、`ClassSpecificFocalLoss`，更新 `create_loss()` |

### 3.3 模型结构

`HierarchicalMultiTaskClassifier`:

- 共享 backbone（支持 resnet3d18, densenet3d_121 / densenet121, swin3d_tiny 等）
- `backbone_proj`: AdaptiveAvgPool3d → Flatten → Linear → ReLU → Dropout
- `head_abnormal`: Linear → ReLU → Dropout → Linear(→2) — Task A
- `head_main`: Linear → ReLU → Dropout → Linear(→num_classes) — Task B
- `head_benign_malignant`: Linear → ReLU → Dropout → Linear(→2) — Task C
- 训练时返回 `(logits_main, logits_abnormal, logits_bm)`
- 评估时仅返回 `logits_main`

### 3.4 Benign-Malignant Mixup 数据增强 (`--bm-mixup`)

- 仅对良性(label=1)和恶性(label=2)样本进行 Mixup
- 使用 Beta(α,α) 分布采样混合比例 λ
- 混合后使用 soft target，而不是硬指定为良性：Head B 三分类目标为 `[0, λ, 1-λ]`，Head C 良恶性目标为 `[λ, 1-λ]`，Head A 始终为非健康
- 与层次化多任务损失叠加使用，避免把混合样本全部推向良性造成标签偏置

### 3.5 Class-Specific Focal Loss (`--focal-gammas`)

- 为不同类别设置不同的 focal gamma 值
- 良性类别可使用更大的 gamma（如 3.0），更聚焦难样本
- 健康类别使用较小的 gamma（如 2.0）
- 可单独使用 `class_specific_focal` 损失，也可通过 `hierarchical_class_specific_focal` 与多任务组合

### 3.6 混淆感知微调 (`--ca-ft`)

- 先正常训练 `--ca-ft-start` 个 epoch
- 在验证集上分析混淆矩阵，计算每个类别的错误率
- 为错误率高的类别生成更高的训练权重
- 用加权损失继续微调剩余 epoch

## 4. 完整训练脚本

以下是可直接复制使用的训练命令。请根据实际数据路径替换 `<CT_ROOT>` 和 `<METADATA_CSV>`。

### 4.1 公共参数（所有脚本共用）

```bash
# === 请替换为实际路径 ===
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"
OUTPUT_BASE="outputs"
```

### 4.2 基础层次化多任务训练

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_resnet3d18_base" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --aug-profile strong \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --effective-num-beta 0.999 \
    --label-smoothing 0.05 \
    --hidden-dim 256 \
    --dropout 0.3 \
    --pretrained \
    --weight-abnormal 1.0 \
    --weight-main 1.0 \
    --weight-benign-malignant 1.0 \
    --selection-metric balanced_accuracy \
    --seed 42
```

### 4.3 层次化多任务 + BM-Mixup

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_resnet3d18_bmmixup" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --aug-profile strong \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --effective-num-beta 0.999 \
    --label-smoothing 0.05 \
    --hidden-dim 256 \
    --dropout 0.3 \
    --pretrained \
    --weight-abnormal 1.0 \
    --weight-main 1.0 \
    --weight-benign-malignant 1.0 \
    --bm-mixup \
    --bm-mixup-alpha 0.4 \
    --selection-metric balanced_accuracy \
    --seed 42
```

### 4.4 层次化多任务 + 混淆感知微调

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_resnet3d18_caft" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --aug-profile strong \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --effective-num-beta 0.999 \
    --label-smoothing 0.05 \
    --hidden-dim 256 \
    --dropout 0.3 \
    --pretrained \
    --weight-abnormal 1.0 \
    --weight-main 1.0 \
    --weight-benign-malignant 1.0 \
    --ca-ft \
    --ca-ft-start 24 \
    --ca-ft-boost 2.0 \
    --selection-metric balanced_accuracy \
    --seed 42
```

### 4.5 层次化多任务 + Class-Specific Focal Loss

良性类别 gamma=3.0（更聚焦难样本），健康和恶性 gamma=2.0（标准）。

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_resnet3d18_csfocal" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --aug-profile strong \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --effective-num-beta 0.999 \
    --label-smoothing 0.0 \
    --hidden-dim 256 \
    --dropout 0.3 \
    --pretrained \
    --weight-abnormal 1.0 \
    --weight-main 1.0 \
    --weight-benign-malignant 1.0 \
    --focal-gammas 2.0 3.0 2.0 \
    --selection-metric balanced_accuracy \
    --seed 42
```

### 4.6 层次化多任务 + BM-Mixup + 混淆感知微调（组合使用）

先用 Mixup 训练 24 个 epoch，再切换到混淆感知微调。

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_resnet3d18_mixup_caft" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 50 \
    --batch-size 4 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --aug-profile strong \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --effective-num-beta 0.999 \
    --label-smoothing 0.05 \
    --hidden-dim 256 \
    --dropout 0.3 \
    --pretrained \
    --weight-abnormal 1.0 \
    --weight-main 1.0 \
    --weight-benign-malignant 1.0 \
    --bm-mixup \
    --bm-mixup-alpha 0.4 \
    --ca-ft \
    --ca-ft-start 30 \
    --ca-ft-boost 2.0 \
    --selection-metric balanced_accuracy \
    --seed 42
```

### 4.7 级联评估模式

训练完成后，用 Head A + Head C 做级联推理（替代仅用主 head）。

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_resnet3d18_cascade" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --scheduler cosine \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --pretrained \
    --weight-abnormal 1.0 \
    --weight-main 1.0 \
    --weight-benign-malignant 1.0 \
    --eval-mode cascade \
    --selection-metric balanced_accuracy \
    --seed 42
```

### 4.8 DenseNet121 / DenseNet3D-121 backbone（推荐对照）

可以把上述脚本中的 `--backbone resnet3d18` 替换为 DenseNet121 3D 版本：

```bash
    --backbone densenet121 \
    --hidden-dim 512 \
```

`densenet121` 是 `densenet3d_121` 的别名，二者等价。该 backbone 依赖 MONAI；如内网环境缺少 MONAI，需要先安装 `monai`。层次化模型会保留 MONAI DenseNet 的 relu/pool/flatten 特征汇聚层，仅移除最后的 Linear 分类层，使三任务 head 接收 1024 维 DenseNet 特征。

建议先跑一个稳定版对照：不加 Mixup、不加强 class weight，降低辅助任务权重，确认主三分类能正常收敛后再逐步加 benign 优化策略。

```bash
python3 train_hierarchical.py \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/hierarchical_densenet121_stable" \
    --backbone densenet121 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --epochs 40 \
    --batch-size 2 \
    --lr 1e-4 \
    --scheduler cosine \
    --aug-profile basic \
    --class-weight-strategy none \
    --selection-metric accuracy \
    --weight-main 1.0 \
    --weight-abnormal 0.3 \
    --weight-benign-malignant 0.5
```

### 4.9 实验对照：直接三分类 baseline

用于与层次化方案对比，直接三分类无多任务。

```bash
CT_ROOT="/path/to/ct_data"
METADATA_CSV="/path/to/metadata.csv"

# 需要使用 src/lung_cancer_cls/train.py 中的标准训练流程
python3 -m lung_cancer_cls.train \
    --dataset-type intranet_ct \
    --data-root "$CT_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --ct-root "$CT_ROOT" \
    --output-dir "${OUTPUT_BASE}/baseline_resnet3d18_3class" \
    --backbone resnet3d18 \
    --use-3d-input \
    --depth-size 128 \
    --volume-hw 256 \
    --class-mode multiclass \
    --split-mode train_val_test \
    --use-predefined-split \
    --epochs 40 \
    --batch-size 4 \
    --lr 3e-4 \
    --scheduler cosine \
    --sampling-strategy weighted \
    --class-weight-strategy effective_num \
    --pretrained \
    --selection-metric balanced_accuracy \
    --seed 42
```

## 5. 其他已提出但未实现的方法

### 5.1 Confusion-Aware Fine-tuning — ✅ 已实现

参见 3.6 节和 4.4 节。

### 5.2 Class-Specific Focal Loss — ✅ 已实现

参见 3.5 节和 4.5 节。

### 5.3 Prototypical Networks / Metric Learning — 未实现

- 在 backbone 特征空间上学习类别的原型表示
- 推理时计算与各原型的距离进行分类
- 对少样本类别（良性）效果好于直接分类

### 5.4 Test-Time Augmentation (TTA) — 未实现

- 推理时对同一输入做多次随机增强（翻转、旋转等）
- 对多次预测取平均
- 无训练成本，直接提升推理精度

## 6. 参数调优建议

| 参数 | 建议范围 | 说明 |
|------|----------|------|
| `--weight-abnormal` | 0.5 ~ 2.0 | Task A 权重，过高会压制其他任务 |
| `--weight-main` | 1.0 ~ 2.0 | Task B 权重，主任务建议不低于 1.0 |
| `--weight-benign-malignant` | 1.0 ~ 3.0 | Task C 权重，可适当调高以关注良性 |
| `--bm-mixup-alpha` | 0.2 ~ 0.8 | 越小越接近硬标签，越大越平滑 |
| `--ca-ft-start` | epochs*0.5 ~ epochs*0.7 | 太早则模型未充分学习，太晚则微调不够 |
| `--ca-ft-boost` | 1.0 ~ 3.0 | 过高会导致过拟合混淆类别 |
| `--focal-gammas` | 与类别数一致 | 良性建议 3.0，其他 2.0 |

## 7. 当前状态与后续步骤

### 已完成

- [x] 调研 Subregion-Unet 在内网数据上的可行性
- [x] 设计层次化多任务方案
- [x] 实现 `HierarchicalMultiTaskClassifier` 模型
- [x] 实现 `HierarchicalMultiTaskLoss` 损失函数
- [x] 实现 `train_hierarchical.py` 训练脚本
- [x] 实现 Benign-Malignant Mixup 数据增强
- [x] 实现 Class-Specific Focal Loss
- [x] 实现 Confusion-Aware Fine-tuning
- [x] 语法验证通过

### 待完成

- [ ] 在内网环境中实际训练运行
- [ ] 调参：多任务权重比例、Mixup alpha、CA-FT boost
- [ ] 与现有级联方案（89.6%）对比
- [ ] 尝试其他 backbone（densenet3d_121, swin3d_tiny）
- [ ] 尝试 Prototypical Networks / Metric Learning
- [ ] 尝试 Test-Time Augmentation (TTA)
