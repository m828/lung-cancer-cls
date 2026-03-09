# lung-cancer-cls

一个可直接运行的 **肺癌 CT 三分类基线仓库**，用于先提升 CT 模态精度，后续再并入文本/基因多模态。

## 1. 目标

- 先把 CT-only 分类流程标准化（normal / benign / malignant）
- 在公开数据集 IQ-OTH/NCCD 上先跑通
- 输出可迁移到内网数据的训练骨架

## 2. 数据集

Kaggle: `IQ-OTHNCCD Lung Cancer Dataset`

下载后解压，保证目录中包含类别名（大小写不敏感）：

- `normal` / `healthy`
- `benign`
- `malignant` / `cancer`

脚本会递归扫描目录，只要路径任一层级包含上述类别名即可自动识别标签。

## 3. 安装

```bash
pip install -r requirements.txt
```

## 4. 训练

```bash
PYTHONPATH=src python train_ct.py \
  --data-root /path/to/iqothnccd \
  --output-dir outputs/ct_baseline \
  --epochs 10 \
  --batch-size 16
```

产物：

- `outputs/ct_baseline/best_model.pt`
- `outputs/ct_baseline/metrics.json`

## 5. 迁移到内网建议

1. 保持训练脚本 `train_ct.py` 不变，替换 `--data-root` 指向内网 CT 数据。
2. 先对齐标签映射到 `normal/benign/malignant`。
3. 如需提升精度，优先尝试：
   - 更强 backbone（如 EfficientNet/ConvNeXt）
   - 更长训练与学习率调度
   - class-balanced sampler / focal loss
   - 2.5D 或 3D 输入（按你们内网 CT 数据形态升级）

## 6. 与旧代码关系

仓库原有多模态脚本保留；本次新增了清晰的 CT-only 训练入口，作为后续多模态整合前的稳定基线。
