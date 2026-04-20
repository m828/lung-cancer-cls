# Student KD Sweep Stage Notes

## 1. 这轮阶段做了什么

这轮工作的重点，不再是单独补 teacher 或单独补 baseline，而是把下面三件事一起补齐：

- 把 `student KD` 从“单一 logits 蒸馏”扩展为“多种蒸馏方式可组合”
- 把一批重复命令整合成“批量实验脚本 + 正式配置”
- 把二分类 / 三分类、方法 sweep / backbone sweep 的正式运行入口整理出来

这轮之后，`student` 相关实验已经具备下面几类能力：

- 单条命令运行某一种蒸馏方式
- 在同一 student 上组合多种蒸馏方式
- 一次性批量跑多组实验
- 某个实验失败后自动跳过，不中断整批
- 跑完后自动生成汇总表和总览图

## 2. 当前 student KD 支持什么

当前 `student KD` 已支持以下蒸馏方式：

- `logits`
- `fused`
- `ct`
- `text`
- `hint`
- `relation`
- `attention`

其中：

- `logits`：经典 soft target / KL distillation
- `fused`：对 student / teacher 的融合隐藏表示做对齐
- `ct`：对齐 `CT` 分支特征
- `text`：对齐 `TextClinical` 分支特征
- `hint`：对齐融合前的输入级特征
- `relation`：对齐 batch 内样本关系结构
- `attention`：对齐特征能量分布

别名也已支持：

- `feature -> fused`
- `ct_feature -> ct`
- `text_feature -> text`
- `rkd -> relation`
- `at -> attention`

## 3. 这轮新增的代码入口

### 3.1 训练主逻辑

主要改动文件：

- `src/lung_cancer_cls/multimodal_teacher_student.py`

新增内容包括：

- 多蒸馏方法配置解析
- student / teacher 隐藏特征暴露
- 多蒸馏损失组合
- method-level 权重控制
- `hint / relation / attention` 蒸馏

### 3.2 批量实验入口

新增文件：

- `run_experiment_batch.py`
- `src/lung_cancer_cls/batch_experiment_runner.py`
- `tests/test_batch_experiment_runner.py`

这套入口负责：

- 读取 JSON 配置
- 顺序执行实验
- 失败自动跳过
- 汇总 `batch_results.csv`
- 输出 `batch_summary.json`
- 生成表格和总览图

## 4. 这轮新增的正式配置

### 4.1 蒸馏方法 sweep

- `configs/student_kd_distill_method_sweep_mvn_formal.json`
- `configs/student_kd_distill_method_sweep_mc_formal.json`

用途：

- 在固定 teacher、固定 split、固定 student 主干的前提下，只比较不同蒸馏方法组合

### 4.2 backbone sweep

- `configs/student_kd_backbone_sweep_mvn_formal.json`
- `configs/student_kd_backbone_sweep_mc_formal.json`

用途：

- 在固定 teacher、固定 split、固定蒸馏组合的前提下，只比较不同 `CT backbone`

### 4.3 辅助配置

- `configs/intranet_ct_mc_sampler_ablation_resnet3d18.json`
- `configs/student_kd_distill_method_sweep_template.json`

说明：

- `template` 只是模板，不是正式配置
- 正式跑实验优先使用 `*_formal.json`

## 5. 当前推荐执行顺序

### 第一优先级：二分类蒸馏方法 sweep

最值得优先回答的问题是：

- 在当前最稳的二分类场景里，蒸馏增强到底有没有真实增益？

因此建议先跑：

- `configs/student_kd_distill_method_sweep_mvn_formal.json`

而且优先先看：

- `CT student`

推荐第一批：

```bash
python run_experiment_batch.py \
  --config configs/student_kd_distill_method_sweep_mvn_formal.json \
  --only ct_student_kd_mvn_logits_repro,ct_student_kd_mvn_logits_fused,ct_student_kd_mvn_logits_ct,ct_student_kd_mvn_logits_hint_relation,ct_student_kd_mvn_full_combo
```

### 第二优先级：二分类 backbone sweep

当蒸馏方法有初步结论后，再去看：

- 哪个 `CT backbone` 更适合承接 teacher 知识

对应配置：

- `configs/student_kd_backbone_sweep_mvn_formal.json`

### 第三优先级：CT+Text student

如果 `CT student` 这条线已经显示明确正信号，再去看：

- `CT+Text student` 的蒸馏方式问题
- `CT+Text student` 的 backbone 问题

### 第四优先级：三分类

三分类目前主要作用是：

- 验证方法是否能从二分类推广到更难的标签空间

但它不是当前最优先的收益点。

## 6. 当前最重要的问题共识

### 6.1 teacher 强不是问题

当前 teacher 强，本身并不奇怪。

真正的问题是：

- `CT student` 能不能稳定靠近 teacher
- `CT+Text student` 能不能不再明显退化

### 6.2 三分类低分不是指标错了

当前三分类里，`balanced_accuracy` 低，不是因为指标太苛刻，而是因为它准确暴露了一个真实问题：

- 模型大多学会了 `normal vs abnormal`
- 但没有稳定学会 `benign vs malignant`

### 6.3 恶性样本少不是主因

现有分析更支持下面这个判断：

- 恶性样本数本身不是主因
- 问题更像是 `weighted sampler + effective_num` 的双重纠偏，加剧了 `benign` 的吸附效应
- 同时 student/CT-only 当前结构本身也未必是最优实现

## 7. 当前仍未闭环的地方

以下内容需要明确标记为“还没闭环”：

- 最新 sweep 还没有真正开跑
- 这轮新增代码还没有重新推到 GitHub
- 本机环境缺少 `sklearn`，完整 `pytest` 未跑透
- `densenet3d_121` 依赖 `MONAI`，如果服务器没装，这一支会在 batch 里失败并被跳过

## 8. 后续最建议补的内容

如果当前这批 sweep 跑完，后续最值得继续推进的是：

1. 明确哪种蒸馏方式在二分类上真正有收益
2. 明确哪种 backbone 最适合承接 teacher
3. 继续修正 `CT+Text student` 的退化问题
4. 再决定是否值得继续投入更大规模三分类实验

## 9. 推荐和哪些文档一起看

如果要理解这轮工作在项目全局中的位置，建议一起对照：

- `TEXT_MULTIMODAL_KD_STAGE.md`
- `BINARY_CT_STAGE_NOTES.md`
- `LIDC_IDRI_BENCHMARK_NOTES.md`
- `INTRANET_CT_PREPROCESS.md`
