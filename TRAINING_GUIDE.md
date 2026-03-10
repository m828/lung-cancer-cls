# 肺癌 CT 三分类统一训练框架 - 使用指南

## 概述

项目已重构为清晰的结构，**只有数据集加载部分不同，其他部分完全统一**，方便对比和查看：

```
统一训练流程
├── 数据划分：统一的 80-10-10 分层抽样
├── 模型训练：统一的训练循环
├── 验证：统一的评估逻辑
└── 对比：支持两个数据集的直接对比

不同的只有：
├── IQ-OTH/NCCD：直接加载 2D 图像
└── LUNA16：提取切片后加载 2D 图像
```

## 项目结构

```
lung-cancer-cls/
├── src/lung_cancer_cls/
│   ├── dataset.py          # 统一的数据集接口
│   │   ├── BaseCTDataset     # 基类
│   │   ├── IQOTHNCCDDataset  # IQ-OTH/NCCD 实现
│   │   └── LUNA16Dataset     # LUNA16 实现
│   ├── model.py            # 模型定义（保持不变）
│   └── train.py            # 统一的训练框架
│       ├── TrainConfig       # 配置类
│       ├── train_model       # 统一训练函数
│       └── stratified_split  # 统一数据划分
├── train.py                # 统一训练入口
├── train_iqothnccd.py       # IQ-OTH/NCCD 快捷入口
├── train_luna16.py          # LUNA16 快捷入口
├── prepare_luna16_slices.py # LUNA16 切片提取
├── requirements.txt          # 依赖
└── TRAINING_GUIDE.md        # 本文档
```

## 快速开始

### 方式一：使用快捷脚本（最简单）

**训练 IQ-OTH/NCCD**
```bash
cd /workspace/lung-cancer-cls
python train_iqothnccd.py
```

**训练 LUNA16**
```bash
cd /workspace/lung-cancer-cls
python train_luna16.py
```

### 方式二：使用统一训练脚本（灵活）

**训练 IQ-OTH/NCCD**
```bash
python train.py \
  --dataset-type iqothnccd \
  --data-root "/workspace/data-lung/IQ-OTHNCCD Lung Cancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset" \
  --output-dir outputs/iqothnccd_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 50
```

**训练 LUNA16**
```bash
# 1. 先提取切片
python prepare_luna16_slices.py \
  --luna16-root /workspace/data-lung/luna16 \
  --output-dir /workspace/data-lung/luna16/extracted_slices \
  --subsets 0,1,2 \
  --slices-per-series 20

# 2. 再训练
python train.py \
  --dataset-type luna16 \
  --data-root /workspace/data-lung/luna16/extracted_slices \
  --output-dir outputs/luna16_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 30
```

## 统一参数说明

两个数据集使用**完全相同**的训练参数：

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--dataset-type` | 数据集类型：`iqothnccd` 或 `luna16` | 必填 |
| `--data-root` | 数据集根目录 | 必填 |
| `--output-dir` | 输出目录 | 必填 |
| `--image-size` | 输入图像尺寸 | 224 |
| `--epochs` | 训练轮数 | 10 |
| `--batch-size` | 批次大小 | 16 |
| `--lr` | 学习率 | 1e-3 |
| `--weight-decay` | 权重衰减 | 1e-4 |
| `--train-ratio` | 训练集比例（剩余平分验证/测试） | 0.8 |
| `--model` | 模型：`simple` 或 `resnet18` | simple |
| `--pretrained` | 使用 ImageNet 预训练（仅 resnet18） | False |
| `--cpu` | 强制使用 CPU | False |
| `--seed` | 随机种子 | 42 |

## 统一的数据划分

**两个数据集使用完全相同的划分逻辑：**

```python
# 1. 80% 训练集
# 2. 20% 临时集 -> 10% 验证集 + 10% 测试集
# 3. 每层都使用分层抽样，保证类别比例一致
```

这个逻辑在 `train.py` 的 `stratified_split()` 函数中统一实现。

## 统一的输出格式

两个数据集的输出格式完全相同，方便对比：

```
outputs/
├── iqothnccd_resnet18/
│   ├── best_model.pt        # 最佳模型权重
│   └── metrics.json         # 训练指标
└── luna16_resnet18/
    ├── best_model.pt
    └── metrics.json
```

**metrics.json 结构：**
```json
{
  "dataset_type": "IQ_OTHNCCD",
  "best_val_acc": 0.982,
  "test_acc": 0.975,
  "test_loss": 0.085,
  "history": [...],
  "config": {...}
}
```

## 数据集说明

### IQ-OTH/NCCD

- **特点**：已标注正常/良性/恶性三类的 2D 图像
- **目录结构**：
  ```
  root/
  ├── Normal cases/
  ├── Benign cases/
  └── Malignant cases/
  ```
- **标签**：0=normal, 1=benign, 2=malignant

### LUNA16

- **特点**：3D CT 扫描，需先提取 2D 切片
- **目录结构**：
  ```
  root/
  ├── subset0/ ... subset9/  # 3D CT 数据
  ├── annotations.csv       # 结节标注
  └── extracted_slices/     # 2D 切片（我们生成）
      ├── normal/
      ├── benign/
      └── malignant/
  ```
- **标签策略**：
  - 0 (normal)：无结节的切片
  - 1 (benign)：小结节（直径 < 6mm）的切片
  - 2 (malignant)：大结节（直径 >= 6mm）的切片

## 对比训练示例

要对比两个数据集的性能，使用完全相同的训练参数：

```bash
# 1. 训练 IQ-OTH/NCCD
python train.py \
  --dataset-type iqothnccd \
  --data-root "/path/to/iqothnccd" \
  --output-dir outputs/compare_iqothnccd \
  --model resnet18 --pretrained --epochs 50 \
  --batch-size 32 --image-size 224

# 2. 训练 LUNA16（使用相同的参数）
python train.py \
  --dataset-type luna16 \
  --data-root /path/to/luna16/extracted_slices \
  --output-dir outputs/compare_luna16 \
  --model resnet18 --pretrained --epochs 50 \
  --batch-size 32 --image-size 224  # 完全相同的参数
```

## 常见问题

**Q: 为什么重构为统一框架？**

A: 为了方便对比两个数据集的性能，只有数据加载部分不同，其他部分完全一致。

**Q: LUNA16 没有良恶性标注怎么办？**

A: 我们使用结节直径作为近似标签：<6mm 为良性，>=6mm 为恶性。这是研究中常用的近似策略。

**Q: 可以自定义数据增强吗？**

A: 可以！在 `dataset.py` 的 `get_default_transforms()` 函数中修改，或在训练脚本中传入自定义 transform。

**Q: 如何添加新的数据集？**

A: 继承 `BaseCTDataset` 基类，实现 `discover()` 和 `get_samples()` 方法即可！

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/lung_cancer_cls/dataset.py` | 数据集定义：基类 + IQ-OTH/NCCD + LUNA16 |
| `src/lung_cancer_cls/train.py` | 统一训练框架：所有数据集共用 |
| `train.py` | 统一训练入口 |
| `train_iqothnccd.py` | IQ-OTH/NCCD 快捷训练 |
| `train_luna16.py` | LUNA16 快捷训练（含自动提取） |
| `prepare_luna16_slices.py` | LUNA16 切片提取工具 |
| `TRAINING_GUIDE.md` | 本文档 |
