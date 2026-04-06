# LIDC-IDRI 二分类基线、常见问题与 student 验证路线

## 1. 先说结论

如果当前项目在 `LIDC-IDRI` 上二分类和三分类都不高，这并不奇怪。

最可能的原因不是单一模型太弱，而是下面几个因素叠加：

- 当前项目对 `LIDC-IDRI` 的公开数据支持更像通用分类入口，而不是文献里常见的“结节中心 ROI / 3D patch”专用 benchmark。
- 当前默认划分是样本级随机划分，不是病人级或结节级划分。
- 当前三分类 `normal / benign / malignant` 并不是 `LIDC-IDRI` 最常见的论文设定。
- `LIDC-IDRI` 的“恶性”标签通常来自放射科医师 1-5 分恶性评分，本身就有主观性和噪声。

所以：

- 三分类结果不高，往往是任务定义和预处理方式的问题，不一定是模型本身不行。
- 二分类想和高质量论文对齐，最推荐优先做 `benign_vs_malignant`。

## 2. 当前项目里最可能拖低性能的点

### 2.1 标签定义和论文常用设定不一致

当前项目支持的公开数据标签是：

- `normal`
- `benign`
- `malignant`

但 `LIDC-IDRI` 论文里最常见的主任务不是三分类，而是：

- `benign vs malignant`

而且很多工作会进一步做更严格筛选：

- 恶性评分 `1-2` 视为 benign
- 恶性评分 `4-5` 视为 malignant
- 评分 `3` 直接去掉，避免不确定样本污染标签

这和现在直接把 `normal / benign / malignant` 一起训是两种难度完全不同的任务。

### 2.2 当前 split 是样本级，不是病人级

当前 `train.py` 里的 `stratified_split` 是按样本索引直接分层随机切分，没有病人级或结节级 grouping：

- `src/lung_cancer_cls/train.py`

如果你的 `LIDC-IDRI` 数据是切片级 PNG，或者同一结节导出了多个相邻 patch，那么样本级划分会带来两个问题：

- 如果同一结节的相邻切片同时落进 train/val/test，会让结果偏乐观
- 如果切片噪声很大、视角差异大，也会让结果很不稳定

更规范的做法通常是：

- patient-wise split
- 或至少 nodule-wise split

### 2.3 当前入口更像“整张图分类”，而不是“结节 ROI 分类”

很多 `LIDC-IDRI` 高质量论文不是拿整张 slice 去分类，而是先做：

- 结节定位
- ROI crop
- 3D cube / multi-view patch
- 可选的 segmentation 或 radiologist attribute supervision

而当前项目的公开数据入口默认是：

- 读取 `normal / benign / malignant` 文件夹里的 2D 图像或 3D `.npy`
- 直接进分类模型

如果你的输入是整张 slice，结节只占很小一块区域，模型就容易把大部分容量浪费在背景上。

### 2.4 `LIDC-IDRI` 本身标签就不是病理金标准

`LIDC-IDRI` 原始数据库中的恶性相关标签主要来自放射科医师评分，而不是统一病理金标准。

这意味着：

- 中间分值样本天然更模糊
- 文献里很多高分结果本身就依赖“只保留 clear-cut 样本”

所以直接拿不同论文的数值横比，很容易失真。

## 3. 已发布论文里常见的训练方式

这里先看几个比较有代表性的方向。

### 3.1 原始数据和标签噪声

`LIDC-IDRI` 原始论文说明，结节由多位放射科医师标注，并带有恶性等主观评分，因此本身存在观察者间差异：

- Armato et al., 2011, `The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans`
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3041807/

这决定了：

- 评分 `3` 的结节往往不适合硬塞进 binary benchmark
- 直接拿全量样本做三分类通常不会特别高

### 3.2 典型 binary 设定

很多代表性工作会用：

- 评分 `1-2` = benign
- 评分 `4-5` = malignant
- 去掉评分 `3`

例如：

- HSCNN 使用 `40 x 40 x 40` 的 3D ROI，并把恶性评分 `1` 和 `2` 当 benign，`4` 和 `5` 当 malignant，报告 `AUC = 0.856`
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6623975/

- 弱监督 3D CNN 工作同样采用 `1-2 vs 4-5`，去掉 `3`，使用 8 张相邻切片和局部 patch，报告 `AUC = 0.91`
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8370883/

- NoduleX 在更“清晰”的子集上把 AUC 做到 `0.99`，但它本身就是更容易的 clear-cut 设定，不能直接当作所有 LIDC 工作的普遍水平
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6006355/

### 3.3 文献里常见的训练细节

反复出现的高质量做法有这些：

- 3D patch / nodule-centered crop，而不是整张切片
- 去掉不确定标签，例如 malignancy score = 3
- patient-wise 或 nodule-wise split
- 多视角输入，或者 3D 体块输入
- 融合 radiologist semantic attributes、segmentation 或 radiomics
- 使用 5-fold cross-validation，而不只是单次随机划分

## 4. 当前项目最值得优先优化的地方

按优先级排序，我会建议先做这几件事。

### 4.1 先切到 `benign_vs_malignant`

这次已经把 `train.py` 的 binary task 扩展到：

- `benign_vs_malignant`

这样你可以先跑更接近论文 benchmark 的二分类，而不是继续拿 `malignant_vs_rest` 去跟文献硬比。

### 4.2 优先用 3D，而不是 2D 整图

如果你手头已经有对齐好的 3D `.npy` 结节体块：

- 优先用 `resnet3d18`
- 或 `densenet3d_121`

如果只有 2D slice，建议至少保证：

- 输入是结节中心 crop
- 不是整张胸部 CT slice

### 4.3 先做一个“干净 baseline”，不要一开始追求最花哨

最推荐的第一组 baseline：

1. `resnet3d18`
2. `class_mode=binary`
3. `binary_task=benign_vs_malignant`
4. `split_mode=train_val_test`
5. `selection_metric=auroc`
6. `sampling_strategy=weighted`
7. `class_weight_strategy=effective_num`

### 4.4 真正严谨的话，要改成病人级或结节级 split

当前项目现在已经补了 `group split` 支持：

- `--group-split-mode auto`
- `--group-split-mode patient`
- `--group-split-mode nodule`

对 `LIDC-IDRI` 更推荐显式使用：

- `--group-split-mode nodule`

这样可以尽量避免同一结节的多张切片或多份派生样本同时落进 train / val / test。

如果后面你希望把 `LIDC-IDRI` 结果作为正式论文主结果，而不是辅助 benchmark，仍然建议继续加强：

- 明确写出 `group_id` 生成规则
- 尽量基于患者或结节原始 ID，而不是只靠文件名启发式推断

## 5. student 模型如何在 LIDC-IDRI 上验证

这里要把路线讲清楚。

### 5.1 不建议的理解

不要把它理解成：

- “直接拿内部 teacher/student 在 LIDC 上 zero-shot 评估”

这通常没有可比性，因为：

- 内部数据和 LIDC 的任务定义不同
- 内部 teacher 依赖 `CT + text + CNV`
- LIDC 只有 `CT`

### 5.2 更合理的验证方式

更合理的是把 `LIDC-IDRI` 作为一个公开迁移 benchmark：

1. 普通 `CT baseline` 从 ImageNet / Kinetics 预训练开始微调
2. `student-initialized CT model` 从你内部蒸馏得到的 student 权重开始微调
3. 两者在同一个 `LIDC-IDRI` split 上做比较

如果第 2 种 consistently 更好，才说明：

- 蒸馏出来的 CT 表征不仅在内部数据有效
- 在公开数据集上也具备更好的迁移性

### 5.3 当前项目已经补的支持

这次顺手补了两点：

1. `train.py` 现在支持 `--init-checkpoint`
2. `train.py` 现在支持 `--init-checkpoint-prefix`

这意味着你可以把内部 `student_kd` checkpoint 里的 `ct_encoder.` 前缀权重拿出来，作为公开 LIDC 训练的初始化权重。

## 6. 当前最推荐的实验顺序

### 6.1 先做公开数据 clean baseline

先跑：

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

先固定文献常见的 `1-2 -> benign`、`4-5 -> malignant`、`drop score=3` 和 patient-wise split，再基于这份 manifest 去做结节裁剪、3D 预处理和训练。
当前脚本默认优先生成 `consensus nodule` 级别样本；如果只有原始 XML 而没有更干净的结节级标签表，它会按 reader 标注的空间邻近关系先聚合，再做 malignancy 平均和标签折叠。

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root <LIDC_3D_ROOT> \
  --output-dir outputs/lidc_bvm_resnet3d18 \
  --model resnet3d18 \
  --pretrained \
  --use-3d-input \
  --group-split-mode nodule \
  --depth-size 32 \
  --volume-hw 128 \
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

### 6.2 再做 student 初始化迁移

如果内部 `student` 用的也是同一类 CT backbone，例如 `resnet3d18`，再跑：

```bash
python train.py \
  --dataset-type lidc_idri \
  --data-root <LIDC_3D_ROOT> \
  --output-dir outputs/lidc_bvm_resnet3d18_student_init \
  --model resnet3d18 \
  --use-3d-input \
  --group-split-mode nodule \
  --depth-size 32 \
  --volume-hw 128 \
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

然后重点比较：

- `val/test AUROC`
- `balanced_accuracy`
- `sensitivity / specificity`
- `MCC`

## 7. 怎样看“达到一般水平”

更稳妥的标准不是只盯一个 accuracy，而是：

1. 任务定义和文献一致
2. split 方式尽量接近文献
3. AUROC 达到文献中“非 clear-cut、非过度筛选”工作的常见区间
4. student 初始化相比普通 baseline 有稳定增益

对 `LIDC-IDRI` 来说，更重要的是 benchmark 设定是否干净，而不是只看一个最高数字。

## 8. 当前最推荐的一句话判断

如果现在 `LIDC-IDRI` 结果不高，最应该先怀疑的是：

- 任务定义不对齐
- 输入不是结节 ROI / 3D patch
- split 不够严谨

而不是先怀疑 student 思路本身无效。
