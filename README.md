# lung-cancer-cls

一个可直接运行的 **肺癌 CT 三分类统一训练框架**，支持 IQ-OTH/NCCD、LUNA16 和内网 CT（`intranet_ct`）三个数据来源，确保数据划分、模型训练及验证的方式一致，方便对比和查看结果。

## 1. 目标

- 统一训练框架：数据划分、训练循环、验证完全一致
- 支持三个数据源：IQ-OTH/NCCD（2D 图像）、LUNA16（3D CT 切片）、内网 CT（CSV 索引 + .npy）
- 先在公开数据集 IQ-OTH/NCCD 上跑通，再扩展到 LUNA16
- 输出可迁移到内网数据的训练骨架

## 2. 项目结构

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
├── train_iqothnccd.py       # IQ-OTH/NCCD 快捷训练
├── train_luna16.py          # LUNA16 快捷训练
├── prepare_luna16_slices.py # LUNA16 切片提取
├── requirements.txt          # 依赖
└── TRAINING_GUIDE.md        # 统一训练框架使用说明
```

## 3. 安装

```bash
pip install -r requirements.txt
```

## 4. 快速开始

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

  ============================================================
  数据集统计:
    正常 (normal): 5289
    良性 (benign): 19
    恶性 (malignant): 32
    总计: 5340 张切片
    输出目录: /workspace/data-lung/luna16/extracted_slices
  ============================================================
  样本列表已保存到: /workspace/data-lung/luna16/extracted_slices/samples.csv

# 2. 再训练
python train.py \
  --dataset-type luna16 \
  --data-root /workspace/data-lung/luna16/extracted_slices \
  --output-dir outputs/luna16_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 30
```

## 5. 统一参数说明

公开数据与内网数据使用同一套训练入口，核心参数可保持一致：

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--dataset-type` | 数据集类型：`iqothnccd` / `luna16` / `intranet_ct` | 必填 |
| `--data-root` | 数据集根目录 | 必填 |
| `--output-dir` | 输出目录 | 必填 |
| `--image-size` | 输入图像尺寸 | 224 |
| `--epochs` | 训练轮数 | 10 |
| `--batch-size` | 批次大小 | 16 |
| `--lr` | 学习率 | 1e-3 |
| `--weight-decay` | 权重衰减 | 1e-4 |
| `--train-ratio` | 训练集比例（剩余平分验证/测试） | 0.8 |
| `--model` | 模型：`simple`/`resnet18`/`resnet18_se`/`resnet18_cbam`/`efficientnet_b0`/`convnext_tiny`/`resnet3d18` | simple |
| `--pretrained` | 使用预训练（对 2D ImageNet 与 3D Kinetics400 模型生效） | False |
| `--aug-profile` | 数据增强：`basic` / `strong` | basic |
| `--loss` | 损失：`ce` / `focal` | ce |
| `--label-smoothing` | CE 标签平滑 | 0.0 |
| `--focal-gamma` | Focal Loss gamma | 2.0 |
| `--optimizer` | 优化器：`adamw` / `sgd` | adamw |
| `--scheduler` | 调度器：`none` / `cosine` / `onecycle` / `plateau` | none |
| `--sampling-strategy` | 采样策略：`default` / `weighted` | default |
| `--class-weight-strategy` | 类别权重：`none`/`inverse`/`sqrt_inverse`/`effective_num` | none |
| `--effective-num-beta` | `effective_num` 权重的 beta | 0.999 |
| `--cpu` | 强制使用 CPU | False |
| `--seed` | 随机种子 | 42 |

### 内网 CT（intranet_ct）额外参数

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--metadata-csv` | 内网索引表 CSV 路径 | `data-root/多模态统一检索表_CT本地路径_CT划分.csv` |
| `--ct-root` | CT `.npy` 根目录 | `data-root` |
| `--use-predefined-split` | 使用 CSV 中的 `train/val/test` 划分 | False |
| `--use-3d-input` | 启用 3D 体输入（仅内网 `.npy`） | False |
| `--depth-size` | 3D 输入重采样深度 | 32 |

示例：

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root /home/apulis-dev/userdata \
  --metadata-csv /home/apulis-dev/userdata/mmy/ct/多模态统一检索表_CT本地路径_CT划分.csv \
  --ct-root /home/apulis-dev/userdata/Data/CT1500 \
  --use-predefined-split \
  --output-dir outputs/intranet_ct_resnet18 \
  --model resnet18 \
  --epochs 30
```

## 6. 训练结果

### IQ-OTH/NCCD 训练结果（ResNet18，5 轮）

```
数据集类型: IQ_OTHNCCD
数据根目录: /workspace/data-lung/IQ-OTHNCCD Lung Cancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset
输出目录: outputs/iqothnccd_resnet18

找到 1097 个样本
类别分布:
  normal (label=0): 416
  benign (label=1): 120
  malignant (label=2): 561

数据划分: train=876, val=110, test=111

模型: resnet18 (pretrained=True)

开始训练（共 5 轮）
------------------------------------------------------------
[Epoch 1/5] train_loss=0.5174 val_loss=2.1090 val_acc=0.2818
[Epoch 2/5] train_loss=0.3060 val_loss=2.5552 val_acc=0.6000
[Epoch 3/5] train_loss=0.3961 val_loss=5.2217 val_acc=0.4091
[Epoch 4/5] train_loss=0.3154 val_loss=0.2561 val_acc=0.8818
[Epoch 5/5] train_loss=0.2934 val_loss=5.5950 val_acc=0.5364

在测试集上评估最佳模型...
测试结果: loss=0.3048, acc=0.9009

最佳验证准确率: 0.8818
测试准确率: 0.9009
```

## 7. 与旧代码关系

仓库原有多模态脚本保留；本次重构为统一训练框架，确保不同数据集的训练过程一致，方便对比和分析。

## 8. 迁移到内网建议

1. 保持训练脚本 `train.py` 不变，替换 `--data-root` 指向内网 CT 数据。
2. 先对齐标签映射到 `normal/benign/malignant`。
3. 如需提升精度，优先尝试：
   - 更强 backbone（如 EfficientNet/ConvNeXt）
   - 更长训练与学习率调度
   - class-balanced sampler / focal loss
   - 2.5D 或 3D 输入（按你们内网 CT 数据形态升级）

## 9. 模块化增强（用于组合调用与消融）

### 模型（`--model`）

- `simple`：轻量 CNN baseline
- `resnet18`：经典基线
- `resnet18_se`：ResNet18 + SE 通道注意力
- `resnet18_cbam`：ResNet18 + CBAM（通道+空间注意力）
- `efficientnet_b0`：小样本常见高性价比 backbone
- `convnext_tiny`：近两年常用强基线
- `resnet3d18`：3D CT 体数据模型（`[B,1,D,H,W]`）

### 数据增强（`--aug-profile`）

- `basic`：原始轻增强
- `strong`：`RandomResizedCrop + Affine + Blur + Sharpness + RandomErasing`

### 损失函数（`--loss`）

- `ce`：交叉熵（支持 `--label-smoothing`）
- `focal`：Focal Loss（支持 `--focal-gamma`）

### 优化器与学习率调度

- 优化器 `--optimizer`：`adamw` / `sgd`
- 调度器 `--scheduler`：`none` / `cosine` / `onecycle` / `plateau`

### 类别不平衡专项（LUNA16 / 内网重点推荐）

- 采样侧：`--sampling-strategy weighted`
  - 使用 `WeightedRandomSampler` 对少数类（如良性）进行过采样。
- 损失侧：`--class-weight-strategy`
  - `inverse`：按类别频次反比加权
  - `sqrt_inverse`：按频次开方反比加权（更稳）
  - `effective_num`：Class-Balanced Loss 常用权重，配合 `--effective-num-beta`（默认 `0.999`）
- 与损失结合：
  - `--loss ce --class-weight-strategy effective_num`
  - 或 `--loss focal --focal-gamma 2.0 --class-weight-strategy inverse`

建议从这两组起步：
1. **稳健方案**：`weighted sampler + CE(label smoothing=0.05) + effective_num`
2. **强化少数类召回**：`weighted sampler + focal(gamma=2.0) + inverse`

### 3D 输入（内网 `.npy`）

- `--use-3d-input`：启用 3D 模式
- `--depth-size`：重采样深度（默认 `32`）
- 注意：当前 3D 模式仅支持 `intranet_ct`；IQ-OTH/NCCD 和当前 LUNA16 切片流程仍为 2D。

### 组合示例

```bash
# 2D：ResNet18 + CBAM + 强增强 + Focal + Cosine
python train.py \
  --dataset-type intranet_ct \
  --data-root /path/to/root \
  --output-dir outputs/ablation_res18_cbam \
  --model resnet18_cbam \
  --aug-profile strong \
  --loss focal --focal-gamma 2.0 \
  --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --scheduler cosine

# 2D：ConvNeXt-Tiny + CE(label smoothing) + OneCycle
python train.py \
  --dataset-type iqothnccd \
  --data-root /path/to/iqoth \
  --output-dir outputs/iqoth_convnext \
  --model convnext_tiny --pretrained \
  --aug-profile strong \
  --loss ce --label-smoothing 0.1 \
  --optimizer adamw --lr 1e-3 \
  --scheduler onecycle

# 3D：ResNet3D18 + 内网体数据
python train.py \
  --dataset-type intranet_ct \
  --data-root /path/to/root \
  --metadata-csv /path/to/meta.csv \
  --ct-root /path/to/ct_npy \
  --output-dir outputs/intranet_resnet3d18 \
  --model resnet3d18 --pretrained \
  --use-3d-input --depth-size 32 \
  --optimizer sgd --lr 5e-3 --weight-decay 1e-4 \
  --scheduler cosine

# 不平衡重点：良性样本较少时（例如 1500 例中良性 ~100）
python train.py \
  --dataset-type intranet_ct \
  --data-root /path/to/root \
  --metadata-csv /path/to/meta.csv \
  --ct-root /path/to/ct_npy \
  --output-dir outputs/intranet_imbalance_recipe \
  --model resnet18_cbam --pretrained \
  --aug-profile strong \
  --sampling-strategy weighted \
  --loss ce --label-smoothing 0.05 \
  --class-weight-strategy effective_num --effective-num-beta 0.999 \
  --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --scheduler cosine
```
