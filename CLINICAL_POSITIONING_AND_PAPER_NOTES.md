# 临床动机、论文定位与审稿人应对笔记

## 1. 当前工作的临床出发点

这条路线的核心，不是单纯追求“多模态越多越好”，而是解决一个临床上很实际的问题：

- `CT` 是相对低成本、覆盖面广、最容易部署的模态
- `文本` 代表病史、症状、检查结论等临床上下文，现实里通常比基因更容易获取
- `CNV / 基因` 代表更接近肿瘤生物学本质的信息，但成本高、覆盖率低、并不是每位患者都能拿到

因此，多模态教师网络的意义，不是让最终部署也依赖所有模态，而是：

- 利用少量高信息密度的 `CT + 文本 + CNV` 配对患者训练 `teacher`
- 再把 `teacher` 的知识蒸馏给更可部署的 `CT` 或 `CT + 文本` 学生网络
- 让没有基因检测的患者，也能间接受益于基因模态带来的信息

这是一种典型的 `privileged information -> deployable model` 路线。

## 2. 为什么“基因 + 蒸馏”是临床驱动的，而不是技术堆砌

可以把这项工作概括成一句话：

> 用少量做过基因检测的患者，去帮助大量没有基因检测的患者。

如果按这个思路写论文，那么：

- `CNV` 的价值在于提供高价值但稀缺的肿瘤信息
- `teacher` 的价值在于整合这种稀缺信息
- `student` 的价值在于把这种能力迁移到低成本模态上

因此，蒸馏不是为了让实验更复杂，而是为了降低临床部署门槛。

## 3. “健康 vs 恶性”是否有意义

### 3.1 可以成立的部分

`健康 vs 恶性` 不是没有意义，它至少有三层合理性：

1. 它可以作为当前三模态配对数据受限场景下的第一阶段验证任务。
2. 它可以验证 `teacher -> student` 的知识迁移路线是否成立。
3. 它可以对应“肿瘤风险识别 / high-risk triage / screening enrichment”这类更偏筛查和分流的场景。

也就是说，这个任务更适合被表述为：

- `lung cancer risk stratification`
- `screening enrichment`
- `malignancy-oriented detection`

而不是直接等价成“肺结节良恶性鉴别”。

### 3.2 审稿人可能提出的质疑

审稿人很可能会问：

- 现实临床更难的是 `良恶性` 鉴别，为什么你做的是 `健康 vs 恶性`？
- 这个任务是否过于容易？
- 这样的结果是否高估了模型能力？

这些质疑是合理的，不能回避。

### 3.3 最稳妥的回应方式

建议正面承认限制，但把问题定位讲清楚：

1. 当前三模态配对数据中，`良性` 样本过少且严重不均衡，直接做 `良恶性` 主任务会导致结论不稳定。
2. 当前工作首先关注的是“多模态高价值信息能否通过蒸馏迁移到可部署模型”，因此先在 `malignant vs normal` 上做方法学验证。
3. 这不是最终临床问题的终点，而是受数据约束下的阶段性验证任务。
4. 更接近真实临床决策的 `良恶性鉴别` 将作为后续扩展方向。

一句话版本：

> 本研究的主创新点是稀缺多模态信息向可部署模型的知识迁移；`健康 vs 恶性` 是受三模态样本约束下的阶段性验证任务，而非试图替代最终的良恶性鉴别临床问题。

## 4. 论文里更推荐的主线表述

### 4.1 不建议的写法

- “我们做了三模态融合，所以精度更高”
- “我们有基因数据，所以把它加进模型里”
- “健康 vs 恶性效果很好，因此临床价值很高”

这些表述都容易被审稿人认为问题定义不够强。

### 4.2 更建议的写法

更推荐把论文主线写成：

1. 临床上，高价值模态存在可及性差异，基因检测并非所有患者都具备。
2. 因此需要一种方法，把少量配对多模态患者中的信息迁移给可部署模型。
3. 我们构建了 `CT + 文本 + CNV teacher` 与 `CT / CT + 文本 student` 的蒸馏框架。
4. 我们关心的核心问题不是“teacher 是否最高分”，而是“student 是否在不依赖基因的情况下仍获得可验证增益”。

这样之后，文章的临床意义就变成：

- 减少对基因检测的依赖
- 提升普通影像模型的能力上限
- 让低成本模态承接高价值模态的知识

## 5. 可以直接放进论文里的贡献点

建议把 contribution 写成下面这种结构：

1. 提出一个面向临床可部署性的多模态蒸馏框架，将稀缺高价值模态的信息迁移到低成本输入模型。
2. 构建 `CT + 文本 + CNV` 教师网络，以及 `CT` 和 `CT + 文本` 学生网络，在同 cohort、同 split 下进行严格比较。
3. 证明在不依赖基因检测的情况下，学生模型能够继承教师网络的部分多模态能力，并在内部与外部数据上获得优于普通 baseline 的表现。

## 6. 当前任务的风险点与应对

### 6.1 风险一：任务可能被认为过容易

应对：

- 主动承认 `健康 vs 恶性` 不是最终最难临床任务
- 强调当前重点是验证蒸馏路线而不是定义最终诊断终局任务
- 增加外部验证，降低“只是在简单任务上过拟合”的质疑

### 6.2 风险二：文本模态可能存在标签泄漏

应对：

- 对文本做去泄漏消融
- 去掉 `诊断 / 病理 / 出院记录 / 影像结论` 等高风险字段后再报告结果
- 保留“全字段”和“去泄漏字段”两套表，讨论它们差异

### 6.3 风险三：`良性` 样本缺失导致主任务不够接近真实临床场景

应对：

- 明确写成数据限制
- 补 `bundle` 外部 `CT` 验证
- 后续补充 teacher 未见过的 `CT + 文本` 新数据
- 条件允许时再扩展到 `良恶性` 或 `恶性 vs 非恶性`

## 7. 当前最推荐的实验叙事顺序

### 7.1 方法验证主线

1. `CT + CNV + Text teacher`
2. `CT only baseline`，同 cohort、同 split
3. `CT + Text baseline`，同 cohort、同 split
4. `CT student + KD`
5. `CT + Text student + KD`

这里最关键的问题是：

- `CT student + KD` 是否优于普通 `CT baseline`
- `CT + Text student + KD` 是否优于普通 `CT + Text baseline`

### 7.2 泛化验证主线

1. 在内部 `train_val_test` 上做正式比较
2. 用 `bundle` 做 `CT baseline` 与 `CT student + KD` 的外部验证
3. 后续补 teacher 未见过的 `CT + 文本` 数据，再做 `CT + Text baseline` 与 `CT + Text student + KD` 的外部比较

## 8. 推荐的论文措辞

### 8.1 题目方向

可参考下面这种方向，而不是把题目写死在“良恶性鉴别”上：

- `Multimodal Teacher-Student Distillation for Clinically Deployable Lung Cancer Risk Stratification`
- `Transferring Genomic Knowledge to CT-Based Student Models for Lung Cancer Classification`
- `A Clinically Deployable Teacher-Student Framework Integrating CT, Clinical Text, and CNV for Lung Cancer Assessment`

### 8.2 摘要里的关键一句

可以直接沿用下面这种结构：

> Because genomic profiling is informative but not routinely available for all patients, we treat CNV as privileged information during training and distill multimodal knowledge into CT-based or CT-plus-text student models for practical deployment.

### 8.3 讨论部分的限制说明

建议明确写：

- 当前主任务为 `malignant vs normal`
- 这是受三模态配对数据中良性样本稀缺所限
- 后续将扩展到更具临床决策价值的 `benign vs malignant` 或 `malignant vs non-malignant` 任务

## 9. 对应实验与脚本入口

当前这部分工作对应的正式实验入口已经在项目中具备：

- `train_multimodal.py`
- `train_student_kd.py`
- `evaluate_bundle_ct.py`
- `export_experiment_plots.py`

更具体的推荐命令，统一以这些记录为准：

- [BINARY_CT_STAGE_NOTES.md](./BINARY_CT_STAGE_NOTES.md)
- [TEXT_MULTIMODAL_KD_STAGE.md](./TEXT_MULTIMODAL_KD_STAGE.md)

## 10. 当前最推荐的一句话定位

当前最稳的项目定位是：

> 这项工作不是在回答“最难的肺结节良恶性鉴别是否已经解决”，而是在回答“少量包含基因信息的患者，能否帮助我们训练出对大多数无基因患者也可部署的更强模型”。
