# CT+Text Student KD 失败分析（未执行训练前的风险检查清单）

本文件用于在实验后记录未达标结论与根因判断。当前先输出框架，待 `outputs0533_kd_optimization_suite` 分析产物补充。

## 常见失败模式

### 1) 仅 Recall 提升但 AUROC/F1 不提升

- 说明：模型偏向正类而未提升整体可分性；
- 验证：检查 Recall 与 BAcc/F1 的 bootstrap Δ 与 CI；
- 应对：调低 `alpha`、降低 teacher 过拟合权重、在阈值上使用 best F1 而非 0.5。

### 2) ECE 改善但 AUROC 未明显提升

- 说明：学生主要在后验概率标定上更好而非排名能力提升；
- 验证：对比 calibration 前后 ECE/NLL，结合 AUROC Δ；
- 应对：优先采纳 `calibration-aware KD` / 训练后温度缩放，而非继续加大 KD 强度。

### 3) optimizer 切换后波动增大

- 说明：训练动态不稳（lr/wd/scheduler 不匹配）；
- 验证：查看 batch/seed 方差与 bootstrap n>0 样本显著性；
- 应对：固定 `lr=3e-4`、`wd=1e-4` 起步，优先 `AdamW + cosine`。

### 4) 小 batch 优于大 batch 或反之

- 说明：梯度估计噪声与泛化偏置主导；
- 验证：比较 batch sweep 与等效累积（accumulation）结果；
- 应对：若小 batch 稳定但抖动大，用 `accumulation_steps` 做成本同等的平衡点。

### 5) 配置中出现 hint/非严格条件

- 说明：pipeline 被污染或与约束冲突；
- 验证：检查 run flags（strict/no-leakage、disable-text-numeric、distill_methods 无 hint、split）；
- 应对：剔除违反约束的 run，不纳入 candidate 统计。

## 诊断优先级

1. **约束完整性**（首先）：split、严格分离、backbone；
2. **配置稳定性**（次）：是否含 forbidden `hint`；
3. **指标一致性**：是否有 bootstrap 显著性；
4. **性能一致性**：4 seed 平均是否方向统一；
5. **部署可行性**：是否需要额外标定后处理。

## 结论模板（执行后替换）

- 结论 A：存在稳定超越 → 给出具体推荐配置；
- 结论 B：只在 calibration 上有收益 → 保留温度缩放后处理，不主推 KD；
- 结论 C：无显著收益 → 回退到 supervised baseline 并保留教师分析结论用于后续特征源选择。
