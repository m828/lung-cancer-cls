# LUNA16 数据集使用指南

本指南说明如何在现有代码基础上，同时支持 IQ-OTH/NCCD 2D 数据集和 LUNA16 数据集进行健康/良恶性三分类任务。

## 项目结构更新

```
lung-cancer-cls/
├── src/lung_cancer_cls/
│   ├── dataset.py          # 更新：支持 IQ-OTH 和 LUNA16 两种数据集
│   ├── model.py            # 保持不变
│   └── train.py            # 更新：支持数据集类型选择
├── train_ct.py             # IQ-OTH/NCCD 训练入口
├── train_luna16.py         # LUNA16 训练入口（新增）
├── prepare_luna16_slices.py   # LUNA16 切片提取脚本（新增）
├── download_lidc_idri.py      # LIDC-IDRI 下载脚本
└── requirements.txt        # 更新：添加 SimpleITK 和 pandas
```

## 数据集说明

### IQ-OTH/NCCD 数据集（原始支持）
- 2D 图像，已标注正常/良性/恶性三类
- 结构：`root/{Normal,Benign,Malignant}/*.jpg`
- 使用 `train_ct.py` 训练

### LUNA16 数据集（新增支持）
- 3D CT 扫描数据（MHD/RAW 格式）
- 包含肺结节检测标注，但无直接的良恶性标注
- 需要先提取 2D 切片并为切片分配标签
- 使用 `train_luna16.py` 或 `train.py --dataset-type luna16` 训练

**标签分配策略**：
- 正常 (normal, 0)：切片不含结节
- 良性 (benign, 1)：切片仅含小结节（直径 < 6mm）
- 恶性 (malignant, 2)：切片含大结节（直径 >= 6mm）

注：这是简化策略，实际 LUNA16 没有良恶性标注，以上基于结节大小做近似。

## 快速开始

### 方式一：使用 LUNA16 专用脚本（推荐）

#### 1. 首先提取 2D 切片

```bash
cd /workspace/lung-cancer-cls

# 安装依赖（如果还没安装）
pip install -r requirements.txt

# 提取前5个子集的切片（平衡速度和数据量）
python prepare_luna16_slices.py \
  --luna16-root /workspace/data-lung/luna16 \
  --output-dir /workspace/data-lung/luna16/extracted_slices \
  --subsets 0,1,2,3,4 \
  --slices-per-series 20
```

#### 2. 然后训练

```bash
# 训练 ResNet18 模型
python train_luna16.py \
  --data-root /workspace/data-lung/luna16/extracted_slices \
  --output-dir outputs/luna16_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 50 \
  --batch-size 32
```

### 方式二：一键提取并训练

```bash
# 自动提取切片并训练（如果尚未提取）
python train_luna16.py \
  --data-root /workspace/data-lung/luna16 \
  --output-dir outputs/luna16_auto \
  --model resnet18 \
  --pretrained \
  --epochs 30 \
  --auto-extract
```

### 方式三：使用通用训练脚本

```bash
# 确保已提取切片到 /workspace/data-lung/luna16/extracted_slices

PYTHONPATH=src python train_ct.py \
  --data-root /workspace/data-lung/luna16/extracted_slices \
  --output-dir outputs/luna16_generic \
  --dataset-type luna16 \
  --model resnet18 \
  --pretrained \
  --epochs 50
```

## 使用原有 IQ-OTH/NCCD 数据集

```bash
# 原有方式保持不变
PYTHONPATH=src python train_ct.py \
  --data-root "/workspace/data-lung/IQ-OTHNCCD Lung Cancer/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset" \
  --output-dir outputs/iqothnccd_resnet18 \
  --model resnet18 \
  --pretrained \
  --epochs 50
```

## 主要文件说明

### 1. `prepare_luna16_slices.py`

LUNA16 数据预处理脚本，用于：
- 读取 MHD/RAW 格式的 3D CT 数据
- 应用肺部窗宽窗位标准化
- 根据 annotations.csv 中的结节标注为切片分配标签
- 保存为 PNG 格式的 2D 切片

参数说明：
- `--luna16-root`: LUNA16 原始数据集根目录
- `--output-dir`: 切片输出目录
- `--subsets`: 使用的子集（0-9），如 "0,1,2"
- `--slices-per-series`: 每个序列提取的切片数

### 2. `train_luna16.py`

LUNA16 专用训练脚本：
- 可以自动检测并提取切片
- 使用 ResNet18 或简单 CNN
- 支持与 IQ-OTH/NCCD 相同的 80-10-10 数据划分

### 3. `src/lung_cancer_cls/dataset.py` 更新内容

新增功能：
- `discover_luna16_samples()`: LUNA16 样本发现函数
- `extract_and_discover_luna16_slices()`: 自动提取切片并发现样本
- `LUNACTDataset`: LUNA16 专用 Dataset 类
- `load_mhd_image()`: 加载 MHD/RAW 数据

### 4. `src/lung_cancer_cls/train.py` 更新内容

新增参数：
- `--dataset-type`: 选择数据集类型（`iqothnccd` 或 `luna16`）
- 根据数据集类型应用不同的数据增强和归一化

## 数据预处理细节

### CT 窗宽窗位

对于肺部 CT，使用以下窗宽窗位：
- 窗宽 (Window Width): 1500 HU
- 窗位 (Window Level): -600 HU
- 范围: [-1200, 300] HU

这可以最佳显示肺部组织。

### 标签策略

LUNA16 没有直接的良恶性标注，我们使用结节直径作为近似：
- < 6mm: 良性概率高
- >= 6mm: 恶性概率高

实际研究中，可以结合：
- 结节形态特征
- 生长速率
- 临床信息

## 常见问题

### Q: LUNA16 子集 0-9 有什么区别？

A: LUNA16 将数据分为 10 个子集用于交叉验证。训练时可以：
- 使用 subset 0-4: 训练（约 44GB）
- 使用 subset 5-9: 验证和测试

### Q: 切片提取需要多长时间？

A: 取决于子集数量：
- 1个子集: 约 10-15 分钟
- 5个子集: 约 1-2 小时
- 10个子集: 约 3-4 小时

### Q: 如何加速训练？

A: 可以：
- 使用更小的 image-size（如 128）
- 减少 slices-per-series（如 10）
- 使用更少的子集进行快速原型开发

### Q: 模型架构有什么建议？

A: 对于 LUNA16 分类：
- 快速基线: SimpleCTClassifier
- 较好性能: ResNet18 (推荐)
- 更好性能: ResNet34/50
- 3D 模型: 可以扩展为 3D CNN（需要修改数据加载）

## 进阶：自定义标签策略

如果需要更精确的良恶性分类，可以：

1. 结合原始 LIDC-IDRI 中的 XML 标注（包含医生的主观评分）
2. 使用结节的以下特征：
   - 毛刺征 (spiculation)
   - 分叶征 (lobulation)
   - 钙化 (calcification)
   - 内部结构
3. 结合临床数据（如果有）

## 相关资源

- [LUNA16 挑战赛官网](https://luna16.grand-challenge.org/)
- [LIDC-IDRI 数据集](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
- [SimpleITK 文档](https://simpleitk.readthedocs.io/)
- [肺结节诊断相关研究](https://pubmed.ncbi.nlm.nih.gov/)
