# Huawei Split Compare

独立对照实验目录，用于比较：

- `lung-cancer-cls + Huawei 原始 CSV 8:2 split`
- `lung-cancer-cls + 当前 7:1:2 split`

不修改仓库原有训练代码，训练入口使用 `../../train_multimodal.py --reference-manifest`。

## 1. 可选：导出华为文本特征

如果有华为 `dataset_cache/processed_df.csv + features.npz`：

```bash
python export_huawei_text_features.py \
  --text_cache_dir /path/to/huawei/dataset_cache \
  --output_tsv text_features/huawei_text_features.tsv
```

## 2. 生成固定 split

```bash
python generate_split_manifests.py
```

默认读取：

- `/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_文本划分0205_修复.csv`
- `/home/apulis-dev/userdata/Data/CT1500`
- `/home/apulis-dev/userdata/Data/Gene/FDEM_CNV_merge_pcc.tsv`

## 3. 防泄漏检查

```bash
python check_split_integrity.py
```

## 4. 启动实验

Huawei 原始 8:2：

```bash
bash run_lungcls_huawei_8_2_seed42.sh
```

7:1:2：

```bash
bash run_lungcls_7_1_2_seed42.sh
```

多 seed：

```bash
bash run_lungcls_huawei_8_2_multiseed.sh
bash run_lungcls_7_1_2_multiseed.sh
```

可通过环境变量覆盖数据路径：

```bash
CT_CSV=/path/to/ct.csv \
CT_ROOT=/path/to/CT1500 \
GENE_TSV=/path/to/gene.tsv \
TEXT_FEATURE_TSV=/path/to/huawei_text_features.tsv \
bash run_lungcls_7_1_2_seed42.sh
```
