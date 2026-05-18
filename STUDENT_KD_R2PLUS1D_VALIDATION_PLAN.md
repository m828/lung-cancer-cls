# Student KD R2Plus1D Validation Plan

## 1. 当前判断

当前最值得推进的候选主结果是：

- `ct_text_student_kd_mvn_r2plus1d_18_full_combo`

已有结果显示它在 aligned 1019 例二分类任务上明显优于 `resnet3d18` 版 CT+Text student，并且优于 0417 的 `ct_text_sc`。但这仍然是单 split、单 seed 上的最优点，不能直接作为最终结论使用。

下一步目标不是继续扩大 sweep，而是回答一个更窄的问题：

- 这个 R2Plus1D + CT+Text + full combo 的提升是否稳定？
- 提升主要来自 backbone、CT+Text 输入，还是 full-combo 蒸馏？
- 是否存在 split 不一致、样本重复、验证集选择或个别样本驱动的问题？

## 2. 验证对象和比较对象

父结果：

- `outputs/ct_text_student_kd_mvn_r2plus1d_18_full_combo`

主要比较对象：

- `outputs/ct_text_student_kd_mvn_resnet3d18_full_combo`
- `outputs/ct_student_kd_mvn_r2plus1d_18_full_combo`
- `outputs/ct_text_student_kd_mvn_r2plus1d_18_logits_only`
- 旧结果中的 `ct_text_mvn_sc_tvt`

固定条件：

- cohort 固定为 teacher split 下的 1019 例。
- `reference_manifest` 固定为 `outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv`。
- class mode 固定为 binary。
- binary task 固定为 `malignant_vs_normal`。
- selection metric 固定为 AUROC。

## 3. 最小实验切片

| slice | 目的 | 预期用途 |
| --- | --- | --- |
| `ct_text_student_kd_mvn_r2plus1d_18_full_combo_seed43` | 主结果 seed 复跑 | 稳定性 |
| `ct_text_student_kd_mvn_r2plus1d_18_full_combo_seed44` | 主结果 seed 复跑 | 稳定性 |
| `ct_text_student_kd_mvn_r2plus1d_18_full_combo_seed45` | 主结果 seed 复跑 | 稳定性 |
| `ct_text_student_kd_mvn_r2plus1d_18_logits_only` | 去掉 feature/relation/attention 蒸馏 | 判断 full combo 是否必要 |
| `ct_student_kd_mvn_r2plus1d_18_full_combo` | 去掉 TextClinical 输入 | 判断 CT+Text 是否必要 |
| `ct_text_student_kd_mvn_resnet3d18_full_combo` | 换回 ResNet3D18 backbone | 判断 R2Plus1D 是否必要 |

其中最后两项通常已有结果；batch runner 会因为 `skip_existing_success=true` 自动跳过已有 `metrics.json`，没有则补跑。

## 4. 通过标准

这条线可以进入写作整理，前提是：

- 新 seed 的 split manifest 与 teacher split md5 一致。
- 没有跨 train/val/test 的重复 `sample_id` 或 `record_id`。
- R2Plus1D CT+Text full combo 的多 seed 均值仍然明显优于 `resnet3d18` CT+Text full combo。
- 相对 0417 `ct_text_sc`，BAcc/F1 的提升在 paired bootstrap 中仍然稳定。
- logits-only 消融不能完全解释 full combo 的收益，或者即使 logits-only 相近，也能明确把结论收窄为 backbone/input 贡献，而不是宣称蒸馏组合贡献。

如果 seed 复跑波动很大，或 logits-only 与 full combo 基本相同，则不要把结论写成“多蒸馏方法有效”。应收窄为：

- R2Plus1D backbone 和 CT+Text student 组合值得保留；
- full-combo 蒸馏最多作为候选 recipe，而不是独立贡献。

## 5. 推荐执行

在服务器上的项目根目录执行：

```bash
python run_student_kd_r2plus1d_validation.py make-config
```

如果服务器上的 text feature 路径以之前日志为准，可以覆盖：

```bash
python run_student_kd_r2plus1d_validation.py make-config \
  --text-feature-tsv /home/apulis-dev/userdata/Data/Text/text_features_v1.tsv
```

然后跑 batch：

```bash
python run_student_kd_r2plus1d_validation.py run-batch
```

如果只想先跑最关键的 3 个 seed：

```bash
python run_student_kd_r2plus1d_validation.py run-batch \
  --only ct_text_student_kd_mvn_r2plus1d_18_full_combo_seed43,ct_text_student_kd_mvn_r2plus1d_18_full_combo_seed44,ct_text_student_kd_mvn_r2plus1d_18_full_combo_seed45
```

训练完成后汇总：

```bash
python run_student_kd_r2plus1d_validation.py summarize \
  --outputs-root outputs \
  --legacy-root /path/to/outputs0417
```

汇总会生成：

- `outputs/r2plus1d_validation_analysis/validation_metrics.csv`
- `outputs/r2plus1d_validation_analysis/validation_metrics.md`
- `outputs/r2plus1d_validation_analysis/seed_stability.csv`
- `outputs/r2plus1d_validation_analysis/split_integrity.csv`
- `outputs/r2plus1d_validation_analysis/paired_bootstrap.csv`
- `outputs/r2plus1d_validation_analysis/paired_error_summary.csv`
- `outputs/r2plus1d_validation_analysis/paired_errors_*.csv`

## 6. 后续路线

验证通过后，下一步进入表格和写作整理。主表优先保留：

- 0417 teacher
- 0417 `ct_text_sc`
- 0514 `ct_text_student_kd_mvn_resnet3d18_full_combo`
- 0514 `ct_text_student_kd_mvn_r2plus1d_18_full_combo`
- R2Plus1D logits-only 消融
- R2Plus1D CT-only 消融
- R2Plus1D CT+Text full combo 多 seed 均值

验证不通过时，不继续扩大同类 sweep。优先回退到更稳的 `ct_text_sc` 或重新设计 student/backbone 方案。
