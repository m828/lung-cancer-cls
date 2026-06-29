# CT+Text Student KD 最优候选（待结果填充）

> 以下结果为分析框架模板。执行 `analyze_kd_optimization_suite.py` 后，依据 `kd_suite_candidate_summary.md` 与 `kd_suite_bootstrap_vs_references.csv` 自动更新结论。

## 当前状态

- 实验套件已实现（`outputs0533_kd_optimization_suite`），尚未在本次交付中执行训练；
- 现阶段结论依赖 `run` 完成后的分析输出。

## 评估口径

- 指标：AUROC、BAcc、F1、Recall、Specificity、ECE、Brier、NLL；
- 对比基线：`S0_supervised`、`S1_logits_alpha02_T4`、`best teacher (T1)`；
- 置信区间和显著性：sample-level paired bootstrap（204 对齐样本）；
- 校准：验证集阈值扫描 + 温度缩放。

## 判定准则

一个配置被判定为“候选优于基线”需满足：

1. 与 `S0_supervised` 对比：AUROC/BAcc/F1 方向不劣且 Recall 不下降；
2. 与 `S1_logits_alpha02_T4` 对比：有至少一个关键指标显著改善；
3. 在 bootstrap 上主要指标 95% CI 尽量不穿 0；
4. ECE 与 NLL 不恶化（或明显改善）；
5. 训练策略可复现且不依赖单一 seed。

## 待填充字段

- best_run（stage / run_name / seedset）
- primary 指标（mean±std）
- 与 S0/S1/T1 的 Δ（mean, 95% CI, p-value）
- 最优阈值策略（best F1 / best BAcc / Youden）
- 校准前后 ECE/NLL 对比
- 最终结论：是否存在稳定超越、是否仅在 Calibration 上有收益

## 当前建议

在当前结果尚未生成前，优先执行：

1. `alpha_sweep + mini`（或 `smoke`）进行快速验证；
2. `optimizer_sweep + mini`；
3. `batch_size_sweep + mini`；
4. `calibration_kd` 与 `confidence_weighted_kd` 的小样本验证。

若 `outputs0533_kd_optimization_suite` 中四种对齐 seed 均存在，则再执行 full。
