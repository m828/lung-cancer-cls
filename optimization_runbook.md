# Extension Optimization Runbook

This runbook covers extension validation only. It does not modify the locked binary R3 main result under `outputs0535`.

## Triclass First Profiles

Start with:

- `bacc_select`
- `macro_f1_select`
- `clinical_composite`

Smoke:

```bash
DRY_RUN=1 SMOKE=1 RUN_MODE=mini bash scripts/run_triclass_kd_optimization_suite.sh --root ./
SMOKE=1 RUN_MODE=mini bash scripts/run_triclass_kd_optimization_suite.sh --root ./
```

Switch model selection:

```bash
SMOKE=1 TRI_KD_SELECTION_METRIC=accuracy bash scripts/run_triclass_kd_optimization_suite.sh --root ./
SMOKE=1 TRI_KD_SELECTION_METRIC=macro_auroc bash scripts/run_triclass_kd_optimization_suite.sh --root ./
SMOKE=1 TRI_KD_SELECTION_METRIC=balanced_accuracy bash scripts/run_triclass_kd_optimization_suite.sh --root ./
```

Full seeds:

```bash
SMOKE=0 RUN_MODE=full STAGE=train bash scripts/run_triclass_kd_optimization_suite.sh --root ./
python3 experiments/analysis/analyze_triclass_kd_optimization.py --root ../outputs0538_triclass_kd_optimization
```

Enter the paper only if a profile improves TRI-S0 on BAcc, macro-F1, malignant recall, and macro AUROC without hiding negative deltas. If results remain negative, report it as three-class failure-mode analysis: gene teacher helps, but deployable CT+Text KD did not reliably mitigate malignant-to-benign collapse.

## LIDC First Profiles

Start with:

- `baseline_default`
- `kdinit_full_ft_bacc`
- `kdinit_diff_lr_01`
- `kdinit_freeze5`

Smoke:

```bash
DRY_RUN=1 USE_EXISTING_LIDC=1 SMOKE=1 RUN_MODE=mini bash scripts/run_lidc_transfer_optimization_suite.sh --root ./
USE_EXISTING_LIDC=1 SMOKE=1 RUN_MODE=mini bash scripts/run_lidc_transfer_optimization_suite.sh --root ./
```

Switch model selection:

```bash
USE_EXISTING_LIDC=1 SMOKE=1 LIDC_SELECTION_METRIC=accuracy bash scripts/run_lidc_transfer_optimization_suite.sh --root ./
USE_EXISTING_LIDC=1 SMOKE=1 LIDC_SELECTION_METRIC=auroc bash scripts/run_lidc_transfer_optimization_suite.sh --root ./
USE_EXISTING_LIDC=1 SMOKE=1 LIDC_SELECTION_METRIC=balanced_accuracy bash scripts/run_lidc_transfer_optimization_suite.sh --root ./
```

Full folds:

```bash
USE_EXISTING_LIDC=1 SMOKE=0 RUN_MODE=full STAGE=train bash scripts/run_lidc_transfer_optimization_suite.sh --root ./
python3 experiments/analysis/analyze_lidc_transfer_optimization.py --root ../outputs0539_lidc_transfer_optimization
```

Enter the paper only if KDInit improves paired folds against `baseline_default` on clinically relevant metrics without selecting by LIDC test. If still negative, report LIDC as public transfer validation showing no clear gain, with any ECE improvement described cautiously.
