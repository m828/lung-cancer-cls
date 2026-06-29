# CT+Text Student KD 系统性优化设计报告

## 目标

在不改模型结构、不改训练主代码的前提下，围绕三条约束完成 CT+Text Student KD 的系统性优化实验设计：

1. 仅使用 CLI 与训练策略参数；
2. 严格保持 CT+Text supervised strict 为 baseline；
3. 复用严格同分割（1019/652/163/204）与 strict-no-leakage pipeline；
4. 不覆盖现有 `outputs0531` / `outputs0532` 结果；
5. 先做小规模验证，不直接跑全量搜索。

## 实验框架

实验总入口：

- `scripts/run_kd_optimization_suite.sh`

固定参数（全部 experiment 下共享）：

- `ct_model = densenet3d_121`
- `modalities = ct,text`
- `split_mode = train_val_test`
- `strict_no_leakage = true`
- `disable_text_numeric_features = true`
- `reference_manifest` 为 `1019/652/163/204` 的统一 split
- `CLASS-MODE = binary / malignant_vs_normal`

输出目录结构：

- `outputs0533_kd_optimization_suite/alpha_sweep/`
- `outputs0533_kd_optimization_suite/temperature_sweep/`
- `outputs0533_kd_optimization_suite/optimizer_sweep/`
- `outputs0533_kd_optimization_suite/batch_size_sweep/`
- `outputs0533_kd_optimization_suite/light_combo_variants/`
- `outputs0533_kd_optimization_suite/calibration_kd/`
- `outputs0533_kd_optimization_suite/confidence_weighted_kd/`

每个阶段会在目录内写入 `logs/` 和 `scripts_used/`，并把 baseline/S1 对照结果链接到各 stage 中。

## 优化实验路线（按实验阶段）

### A. KD 强度（必须）

- KD alpha：`[0.05, 0.1, 0.2, 0.3, 0.5]`
- Temperature：`[2, 4, 6, 8]`
- 核心对比：`logits-only` 下的 trade-off
- 比较对象：AUROC / BAcc / F1 / Recall / ECE / Brier

### B. 优化器与训练策略（必须）

尝试组合：

- Optimizer：
  - `AdamW`（baseline）
  - `AdamW + cosine`
  - `AdamW + warmup + cosine`（通过 wrapper 注入）
  - `SGD + momentum`
- Learning rate：`1e-4 / 3e-4 / 5e-4`
- weight decay：`1e-5 / 1e-4`

### C. Batch size 与稳定性（必须）

- `batch_size ∈ {1,2,4}`
- 梯度累积模拟更大 batch：`accumulation_steps ∈ {2,4}`
- 主要判断：
  - 小 batch 是否改善泛化；
  - 模拟大 batch 是否更稳定。

### D. KD 方法结构优化（重点）

1. logits-only（已有 baseline）
2. `logits,fused`（light-combo）
3. `calibration-aware KD`：
   - teacher logits softening（温度缩放）；
4. `confidence-weighted KD`：
   - soft 模式（`conf^gamma`）
   - hard 模式（阈值筛选）

## 自动分析与统计（不训练）

分析脚本：

- `experiments/analysis/analyze_kd_optimization_suite.py`

需要输出/校验：

- 每个 run：AUROC、BAcc、F1、Recall、Specificity、ECE、Brier、NLL；
- split 合规检查（`1019/652/163/204`、`strict_no_leakage`、`disable_text_numeric_features`）；
- 与 S0 / S1 / best teacher 的成对 sample-level bootstrap（测试集对齐）；
- 阈值优化：
  - best F1
  - best BAcc
  - Youden index
- 校准：
  - val 上拟合温度缩放（probability-logit 标度）；
  - test 上应用后对比 ECE / NLL。

## 统计决策规则（用于最终判断）

1. 每个 candidate 与 baseline 的指标提升以 **4-seed mean ± std** 为主；
2. 与基线的差异以成对 bootstrap 95% CI + p-value 说明显著性；
3. 以 Recall 不下降、ECE 改善或不变为硬约束；
4. 当多个配置近似时优先选择：
   - 更高 AUROC；
   - 同时更低 ECE；
   - 训练复杂度更低（如无 warmup、较小 batch）;
5. 结论以“稳定可复现”为前提，不将单点提升当作有效结论。

## 运行说明（不启动训练的检查方式）

仅分析时可直接运行：

```bash
python3 experiments/analysis/analyze_kd_optimization_suite.py \
  --root outputs0533_kd_optimization_suite \
  --baseline-root /home/mmy/code/outputs0531_gene_privileged_ablation \
  --teacher-root /home/mmy/code/outputs0531_teacher_homogeneous_gene_test
```
