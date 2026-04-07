# lung-cancer-cls

## 阶段文档

- [二分类阶段记录与后续计划](./BINARY_CT_STAGE_NOTES.md)
- [CNV Only XGBoost 基线说明](./CNV_XGBOOST_BASELINE.md)
- [CT + CNV 多模态 V1 说明](./CT_CNV_MULTIMODAL_V1.md)
- [文本模态、多模态教师网络与蒸馏说明](./TEXT_MULTIMODAL_KD_STAGE.md)
- [临床动机、论文定位与审稿人应对笔记](./CLINICAL_POSITIONING_AND_PAPER_NOTES.md)
- [LIDC-IDRI 基线、优化点与 student 验证路线](./LIDC_IDRI_BENCHMARK_NOTES.md)
- [内网 CT DICOM 质控与 NPY 重建说明](./INTRANET_CT_PREPROCESS.md)

当前实验阶段新增实用脚本：

- `train_multimodal.py`
- `train_student_kd.py`
- `evaluate_bundle_ct.py`
- `export_experiment_plots.py`
- `prepare_intranet_ct_npy.py`

一个可直接运行的 **肺癌 CT 三分类统一训练框架**，支持 IQ-OTH/NCCD、LUNA16、LIDC-IDRI 和内网 CT（`intranet_ct`）四个数据来源，确保数据划分、模型训练及验证的方式一致，方便对比和查看结果。

## 1. 目标

- 统一训练框架：数据划分、训练循环、验证完全一致
- 支持四个数据源：IQ-OTH/NCCD（2D 图像）、LUNA16（2D 切片）、LIDC-IDRI（2D 切片）、内网 CT（CSV 索引 + .npy）
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
├── train_lidc_idri.py       # LIDC-IDRI 快捷训练
├── prepare_lidc_idri_3d.py  # LIDC-IDRI 3D 预处理
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

**训练 LIDC-IDRI**
```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_slices \
  --output-dir outputs/lidc_idri_resnet18 \
  --model resnet18 \
  --pretrained \
  --group-split-mode nodule \
  --split-mode train_val_test \
  --epochs 30

# 若已有独立测试集，仅做 train/val 8:2
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_slices \
  --output-dir outputs/lidc_idri_train_val_only \
  --model resnet18 \
  --group-split-mode nodule \
  --split-mode train_val \
  --epochs 30

# LIDC-IDRI 3D 流程（先预处理再训练）
python prepare_lidc_idri_3d.py \
  --input-root /workspace/data-lung/lidc_idri_raw \
  --output-root /workspace/data-lung/lidc_idri_3d_npy \
  --depth-size 32 \
  --image-size 128

python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_3d_npy \
  --output-dir outputs/lidc_idri_swin3d \
  --model swin3d_tiny --pretrained \
  --group-split-mode nodule \
  --split-mode train_val \
  --epochs 30
```

如果要更贴近 `LIDC-IDRI` 文献中的主流 benchmark，当前更推荐优先尝试：

```bash
python build_lidc_idri_split_manifest.py \
  --input-root /workspace/data-lung/LIDC-IDRI \
  --output-dir outputs/lidc_bvm_manifest \
  --metadata-source auto \
  --label-policy score12_vs_score45 \
  --split-scheme patient_kfold \
  --n-splits 5 \
  --val-ratio 0.1 \
  --seed 42
```

先用这条命令从原始 `LIDC-IDRI` + `metadata.csv` 生成文献常见的 `1-2 vs 4-5`、patient-wise split manifest，再基于整理好的 `3D .npy` 目录跑下面的训练命令。
当 `metadata.csv` 里没有 malignancy 标签时，脚本会默认回退到 XML，并按 reader 标注的空间位置聚合成 `consensus nodule`，而不是把每个 reader annotation 都直接当成独立样本。

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_3d_npy \
  --output-dir outputs/lidc_bvm_resnet3d18 \
  --model resnet3d18 \
  --pretrained \
  --use-3d-input \
  --group-split-mode nodule \
  --class-mode binary \
  --binary-task benign_vs_malignant \
  --split-mode train_val_test \
  --epochs 40 \
  --batch-size 8 \
  --lr 3e-4 \
  --scheduler cosine \
  --sampling-strategy weighted \
  --class-weight-strategy effective_num \
  --selection-metric auroc
```

如果你想验证内部蒸馏得到的 `CT student` 是否对公开数据迁移有帮助，也可以在同一设置下继续微调：

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_3d_npy \
  --output-dir outputs/lidc_bvm_resnet3d18_student_init \
  --model resnet3d18 \
  --use-3d-input \
  --group-split-mode nodule \
  --class-mode binary \
  --binary-task benign_vs_malignant \
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

更完整的原因分析、文献设定和推荐实验顺序，见：

- [LIDC_IDRI_BENCHMARK_NOTES.md](./LIDC_IDRI_BENCHMARK_NOTES.md)

## 5. 统一参数说明

公开数据与内网数据使用同一套训练入口，核心参数可保持一致：

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--dataset-type` | 数据集类型：`iqothnccd` / `luna16` / `lidc_idri` / `intranet_ct` | 必填 |
| `--data-root` | 数据集根目录 | 必填 |
| `--output-dir` | 输出目录 | 必填 |
| `--image-size` | 输入图像尺寸 | 224 |
| `--epochs` | 训练轮数 | 10 |
| `--batch-size` | 批次大小 | 16 |
| `--lr` | 学习率 | 1e-3 |
| `--weight-decay` | 权重衰减 | 1e-4 |
| `--train-ratio` | 训练集比例（剩余平分验证/测试） | 0.8 |
| `--split-mode` | 划分模式：`train_val_test`（80/10/10）或 `train_val`（8/2，无测试集） | train_val_test |
| `--model` | 模型：`simple`/`resnet18`/`resnet18_se`/`resnet18_cbam`/`efficientnet_b0`/`convnext_tiny`/`resnet3d18`/`mc3_18`/`r2plus1d_18`/`swin3d_tiny`/`densenet3d`/`attention3d_cnn` | simple |
| `--pretrained` | 使用预训练（对 2D ImageNet 与 3D Kinetics400 模型生效） | False |
| `--aug-profile` | 数据增强：`basic` / `strong` | basic |
| `--loss` | 损失：`ce` / `focal` | ce |
| `--label-smoothing` | CE 标签平滑 | 0.0 |
| `--focal-gamma` | Focal Loss gamma | 2.0 |
| `--mask-loss-weight` | `mask_aware` 中掩膜分支 CE 权重 | 0.5 |
| `--consistency-weight` | `mask_aware` 中一致性 KL 权重 | 0.1 |
| `--use-mask-guided-input` | 使用 mask 先过滤输入，减少背景噪声 | False |
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
| `--intranet-source` | 内网数据来源：`csv` / `bundle` / `both` | csv |
| `--bundle-nm-path` | processed 正常样本 NPY 路径 | `/home/apulis-dev/userdata/processed/NM_all.npy` |
| `--bundle-bn-path` | processed 良性样本 NPY 路径 | `/home/apulis-dev/userdata/processed/BN_all.npy` |
| `--bundle-mt-path` | processed 恶性样本 NPY 路径 | `/home/apulis-dev/userdata/processed/MT_all.npy` |
| `--two-stage-bundle-to-csv` | 两阶段：先 bundle 预训练再 csv 微调 | False |
| `--finetune-epochs` | 两阶段微调轮数 | 10 |
| `--finetune-lr` | 两阶段微调学习率 | 1e-4 |
| `--mask-txt` | mask-aware 样本列表（每行：`mask_path image_path [label]`） | None |
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
  --split-mode train_val \
  --epochs 30

# 方式1：仅使用 processed/NM_all.npy、BN_all.npy、MT_all.npy 训练
python train.py \
  --dataset-type intranet_ct \
  --data-root /home/apulis-dev/userdata \
  --intranet-source bundle \
  --bundle-nm-path /home/apulis-dev/userdata/processed/NM_all.npy \
  --bundle-bn-path /home/apulis-dev/userdata/processed/BN_all.npy \
  --bundle-mt-path /home/apulis-dev/userdata/processed/MT_all.npy \
  --output-dir outputs/intranet_bundle_only \
  --model resnet18 \
  --split-mode train_val

# 方式2：bundle + CSV 索引数据联合训练
python train.py \
  --dataset-type intranet_ct \
  --data-root /home/apulis-dev/userdata \
  --intranet-source both \
  --metadata-csv /home/apulis-dev/userdata/mmy/ct/多模态统一检索表_CT本地路径_CT划分.csv \
  --ct-root /home/apulis-dev/userdata/Data/CT1500 \
  --output-dir outputs/intranet_bundle_plus_csv \
  --model resnet18_cbam \
  --split-mode train_val

# 方式3：先 bundle 预训练，再用 CSV 数据微调
python train.py \
  --dataset-type intranet_ct \
  --data-root /home/apulis-dev/userdata \
  --two-stage-bundle-to-csv \
  --metadata-csv /home/apulis-dev/userdata/mmy/ct/多模态统一检索表_CT本地路径_CT划分.csv \
  --ct-root /home/apulis-dev/userdata/Data/CT1500 \
  --output-dir outputs/intranet_two_stage \
  --model resnet18 \
  --epochs 30 \
  --finetune-epochs 10 \
  --finetune-lr 1e-4
```

## 6. 训练结果

### IQ-OTH/NCCD 训练结果（ResNet18，50 轮）

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

最佳验证准确率: 1.0
测试准确率: 0.982
```

### LUNA16 训练结果（ResNet18，50 轮）

```
模型: resnet18 (pretrained=True)

最佳验证准确率: 0.991
测试准确率: 0.989
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
- `mc3_18`：3D ResNet 变体（Kinetics 预训练可用）
- `r2plus1d_18`：R(2+1)D 3D 时空解耦卷积
- `swin3d_tiny`：Video Swin Transformer（3D Transformer 强基线）
- `densenet3d`：轻量 3D DenseNet（参数高效）
- `attention3d_cnn`：Attention 3D CNN（SE 注意力）

### Subregion-Unet 思想整合（Mask-aware）

- 新增 `DataGenerator`（`src/lung_cancer_cls/dataset.py`）支持按 txt 读取 `mask + image (+label)`。
- 新增 `mask_aware` 损失（`src/lung_cancer_cls/training_components.py`）：
  - `CE(full)` + `mask_loss_weight * CE(masked)` + `consistency_weight * KL(masked || full)`
- 可选 `--use-mask-guided-input`，先对输入做 `x * mask`，进一步抑制背景噪声。

示例（结节区域关注训练）：

```bash
python train.py \
  --dataset-type intranet_ct \
  --data-root /path/to/root \
  --mask-txt /path/to/mask_image_list.txt \
  --output-dir outputs/mask_aware_swin3d \
  --model swin3d_tiny \
  --split-mode train_val \
  --loss mask_aware \
  --mask-loss-weight 0.6 \
  --consistency-weight 0.1 \
  --use-mask-guided-input \
  --optimizer adamw --lr 3e-4 --weight-decay 1e-4 \
  --scheduler cosine
```

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

## LIDC-IDRI Consensus 3D Workflow

When the raw `LIDC-IDRI` download only provides XML annotations rather than a clean malignancy label table, the recommended workflow is:

1. Build a patient-wise benign-vs-malignant split manifest from the raw download.
2. Crop 3D nodule volumes from raw DICOM using the consensus manifest.
3. Train with the exported `processed_split_manifest.csv` and a fixed `fold`.

Recommended commands:

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

```bash
python prepare_lidc_idri_consensus_3d.py \
  --input-root /workspace/data-lung/LIDC-IDRI \
  --manifest-csv outputs/lidc_bvm_manifest_consensus/nodule_manifest.csv \
  --split-manifest-csv outputs/lidc_bvm_manifest_consensus/split_manifest.csv \
  --split-fold 0 \
  --output-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --depth-size 32 \
  --volume-hw 128 \
  --context-scale 1.5 \
  --min-size-xy 32 \
  --min-size-z 8
```

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root /workspace/data-lung/lidc_idri_consensus_3d_fold0 \
  --output-dir outputs/lidc_bvm_resnet3d18_fold0 \
  --model resnet3d18 \
  --pretrained \
  --use-3d-input \
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

If you want to test student initialization on the same public benchmark, add:

```bash
  --init-checkpoint outputs/ct_student_kd_v1_tvt/best_model.pt \
  --init-checkpoint-prefix ct_encoder.
```
