#!/usr/bin/env python3
"""Generate additional analysis plans (non-training).

Outputs to --root (default outputs0531_analysis_plan/):
  1. calibration_analysis_plan.md
  2. error_analysis_plan.md
  3. external_lidc_gap_plan.md
  4. triclass_gap_plan.md
  5. subgroup_robustness_plan.md
  6. supplement_failed_backbones_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"  Wrote {path}")


def calibration_plan(out: Path) -> None:
    content = """# Calibration Analysis Plan

## Goal
Evaluate and improve model calibration for the DenseNet3D121 strict main candidate.

## Data Available
- `test_predictions.csv` in each run directory contains per-sample predicted probabilities
- `metrics.json` already reports ECE and Brier score

## Analyses (No Retrain Required)

### 1. Reliability Diagram
- Bin predictions into 10 bins by predicted probability
- Plot predicted probability vs observed frequency
- Compare: Group A (supervised), Group C (KD no gene), Group D (KD with gene)
- Tool: `sklearn.calibration.calibration_curve`

### 2. Temperature Scaling
- Fit temperature parameter T on validation set predictions
- Apply to test predictions: `softmax(logits / T)`
- Report ECE before/after temperature scaling
- Implementation: simple optimization over T using NLL on val set

### 3. Expected Calibration Error (ECE)
- Already in metrics.json
- Recompute with 15 bins (standard) for consistency
- Report per-group ECE ± std across seeds

### 4. Brier Score
- Already in metrics.json
- Decompose into reliability + resolution + uncertainty
- Report per-group Brier ± std

### 5. Negative Log-Likelihood (NLL)
- Compute from test_predictions.csv
- Report per-group NLL ± std

## Implementation
```python
# From test_predictions.csv
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

df = pd.read_csv("test_predictions.csv")
y_true = df["true_label"]
y_prob = df["predicted_probability"]
brier = brier_score_loss(y_true, y_prob)
nll = log_loss(y_true, y_prob)
```

## Output
- `calibration_results.csv`
- `calibration_reliability_diagram.png`
- `calibration_temperature_scaling.csv`
"""
    write(out / "calibration_analysis_plan.md", content)


def error_analysis_plan(out: Path) -> None:
    content = """# Error Analysis Plan

## Goal
Identify systematic error patterns in the DenseNet3D121 strict main candidate.

## Data Available
- `test_predictions.csv`: per-sample predictions with true labels
- `split_manifest.csv`: sample IDs and split assignments
- Metadata CSV (remote): clinical information

## Analyses (No Retrain Required)

### 1. Top False Positives
- Samples predicted as malignant but actually normal
- Rank by predicted probability (highest first)
- Report top 10 with sample ID, predicted prob, true label

### 2. Top False Negatives
- Samples predicted as normal but actually malignant
- Rank by predicted probability (lowest first)
- Report top 10 with sample ID, predicted prob, true label

### 3. Confusion Matrix Analysis
- Per-group confusion matrix (TP/FP/FN/TN)
- Per-class precision, recall, F1
- Identify which class has higher error rate

### 4. Error Consistency Across Seeds
- For each test sample, count how many seeds predict correctly
- Identify "hard" samples (wrong in ≥3/4 seeds)
- Identify "easy" samples (correct in all seeds)

### 5. Probability Distribution
- Plot probability distribution for true positives vs false positives
- Identify overlap region where model is uncertain

## Implementation
```python
import pandas as pd

df = pd.read_csv("test_predictions.csv")
fp = df[(df["predicted_label"] == 1) & (df["true_label"] == 0)]
fn = df[(df["predicted_label"] == 0) & (df["true_label"] == 1)]
top_fp = fp.nlargest(10, "predicted_probability")
top_fn = fn.nsmallest(10, "predicted_probability")
```

## Output
- `error_analysis_top_fp.csv`
- `error_analysis_top_fn.csv`
- `error_analysis_hard_samples.csv`
- `error_analysis_confusion_matrix.csv`
"""
    write(out / "error_analysis_plan.md", content)


def lidc_gap_plan(out: Path) -> None:
    content = """# External LIDC Validation Gap Plan

## Current Status
- **No LIDC outputs found** in any scanned directory
- LIDC pipeline requires separate data preparation and training

## Required Experiments

### 1. LIDC Baseline
- Train CT-only model on LIDC data
- Use same architecture as main experiment (DenseNet3D121 if compatible)
- 5-fold cross-validation

### 2. LIDC Student-Init
- Initialize student from internal model
- Fine-tune on LIDC data
- Compare with baseline

### 3. LIDC Internal Supervised Init
- Use internal supervised model as initialization
- Fine-tune on LIDC data
- Compare with student-init

### 4. LIDC Fold-wise Metrics
- Per-fold AUROC, BAcc, F1
- Mean ± std across folds

### 5. LIDC Predictions
- Per-sample predictions for each fold
- Enable downstream analysis

## Data Requirements
- LIDC-IDRI dataset with consistent preprocessing
- Compatible CT volume format (150 slices, 256×256)
- Binary label: malignant vs benign (or configurable)

## Implementation Notes
- Separate training pipeline from internal experiments
- May require code changes for LIDC-specific data loading
- Do NOT mix with internal split experiments

## Output
- `outputs_lidc/` directory
- `lidc_fold_metrics.csv`
- `lidc_predictions.csv`
"""
    write(out / "external_lidc_gap_plan.md", content)


def triclass_gap_plan(out: Path) -> None:
    content = """# Triclass Imbalance Ablation Gap Plan

## Current Status
- **No triclass outputs found**
- Current paper focuses on binary classification (malignant vs normal)
- Triclass is an independent research line

## If Triclass Is Needed

### Required Experiments
1. **CE baseline**: Standard cross-entropy
2. **Focal loss**: gamma=2.0
3. **LDAM-DRW**: Label-Distribution-Aware Margin with Deferred Re-Weighting
4. **Balanced Softmax**: Adjust logits by class frequency
5. **Confusion matrix**: Per-class TP/FP/FN/TN
6. **Macro-F1**: Unweighted average F1
7. **Per-class recall**: Sensitivity for each class

### Code Requirements
- Check if `train_multimodal.py` supports `--num-classes 3`
- Check if `--loss focal` is already supported (YES: `--loss ce` or `--loss focal`)
- LDAM and Balanced Softmax may require code changes

### Label Definition
- normal: healthy controls
- benign: benign nodules
- malignant: malignant nodules

## Recommendation
- Defer until binary main results are finalized
- Triclass does not affect binary main conclusions
- Can be added as supplementary analysis

## Output
- `outputs_triclass/` directory (separate from binary)
- `triclass_metrics.csv`
- `triclass_confusion_matrix.csv`
"""
    write(out / "triclass_gap_plan.md", content)


def subgroup_plan(out: Path) -> None:
    content = """# Subgroup Robustness Analysis Plan

## Current Status
- **DATA_NOT_AVAILABLE**: Metadata CSV is on remote training machine
- Cannot verify available subgroup columns locally

## Potential Subgroups (if metadata available)

### 1. Age
- Young (<60) vs Old (≥60)
- Per-subgroup AUROC, BAcc, F1
- Test for significant difference

### 2. Sex
- Male vs Female
- Per-subgroup AUROC, BAcc, F1
- Test for significant difference

### 3. Scanner / Manufacturer
- If available in metadata
- Per-scanner performance

### 4. Slice Thickness
- Thin (≤2.5mm) vs Thick (>2.5mm)
- Per-subgroup performance

### 5. Center / Site
- If multi-center data
- Per-center performance

## Implementation (After Data Access)
```python
import pandas as pd

meta = pd.read_csv("metadata.csv")
predictions = pd.read_csv("test_predictions.csv")
merged = predictions.merge(meta, on="sample_id")

for group_name, group_df in merged.groupby("age_group"):
    auc = compute_auroc(group_df["true_label"], group_df["predicted_probability"])
    print(f"{group_name}: AUROC={auc:.4f}")
```

## Next Steps
1. Access metadata CSV on training machine
2. Identify available subgroup columns
3. Run subgroup analysis script
4. Report per-subgroup metrics and significance tests

## Output
- `subgroup_robustness_results.csv`
- `subgroup_robustness_summary.md`
"""
    write(out / "subgroup_robustness_plan.md", content)


def failed_backbones_plan(out: Path) -> None:
    content = """# Supplement: Failed Backbone Summary

## Overview
This document summarizes backbone experiments that failed or showed instability
under strict-no-leakage conditions.

## R2Plus1D (r2plus1d_18)

### Full-Combo Strict (0529)
- seed42: AUROC=0.7556, BAcc=0.7092 — **TRAINING COLLAPSE**
- Cause: R2Plus1D + strict text + full-combo KD is unstable

### Logits-Only Strict (0530 rescue)
- seed43: AUROC=0.9681, BAcc=0.8824
- seed44: AUROC=0.9293, BAcc=0.8462
- seed45: AUROC=0.9569, BAcc=0.7210
- **BAcc std = 0.071** — too unstable for main candidate

### Lite-Combo Strict (0530 rescue)
- alpha=0.1: AUROC=0.9341 — below 0.95 threshold
- alpha=0.2: AUROC=0.9062 — collapsed

### Lite-Combo bs=2
- OOM on all attempts

### Conclusion
R2Plus1D is fundamentally unstable under strict conditions. Not recommended.

## ResNet3D18 (resnet3d18)

### Full-Combo Strict bs=1 (0530 backbone_swap)
- seed43: AUROC=0.8717, BAcc=0.5924 — **COLLAPSE**
- seed44: AUROC=0.9101, BAcc=0.5504 — **COLLAPSE**
- seed45: AUROC=0.9443, BAcc=0.8950

### Full-Combo Strict bs=2 (0530 backbone_swap_bs_2)
- seed43: AUROC=0.9624, BAcc=0.8681
- seed44: AUROC=0.9672, BAcc=0.8630
- seed45: AUROC=0.9651, BAcc=0.8966
- **BAcc std = 0.016** — borderline stable

### Full-Combo Strict bs=4 (0530 backbone_swap_bs_4)
- seed43: AUROC=0.9717, BAcc=0.8597
- seed44: AUROC=0.9672, BAcc=0.7101 — **COLLAPSE**
- seed45: AUROC=0.9772, BAcc=0.9412

### Conclusion
ResNet3D18 has frequent seed collapses. Not suitable as main candidate.

## Swin3D-Tiny (swin3d_tiny)

### bs=1 (0530 backbone_swap)
- seed42: AUROC=0.8185, BAcc=0.7050 — poor performance

### bs=2, bs=4
- **OOM** on all attempts (CUDA out of memory)

### Conclusion
Swin3D-Tiny requires reduced resolution, making results incomparable with other backbones.

## MC3-18 (mc3_18)

### bs=1 (0530 backbone_swap)
- seed42: AUROC=0.9460, BAcc=0.7471

### bs=2 (0530 backbone_swap_bs_2)
- seed42: AUROC=0.9634, BAcc=0.9311

### bs=4 (0530 backbone_swap_bs_4)
- **OOM** (CUDA out of memory)

### Conclusion
MC3-18 has limited data (single seed). Not sufficient for main candidate.

## DenseNet3D121 (densenet3d_121) — SUCCESS

### bs=4 (0530 backbone_swap_bs_4)
- seed42: AUROC=0.9810, BAcc=0.9345
- seed43: AUROC=0.9784, BAcc=0.8857
- seed44: AUROC=0.9745, BAcc=0.9244
- seed45: AUROC=0.9819, BAcc=0.9529
- **AUROC mean = 0.9790 ± 0.0033** — stable and high

### Conclusion
DenseNet3D121 is the recommended backbone for strict experiments.

## Summary Table

| Backbone | bs | Seeds | AUROC mean±std | BAcc mean±std | Status |
|---|---|---|---|---|---|
| R2Plus1D | 1 | 3 (logits) | 0.951±0.017 | 0.817±0.071 | UNSTABLE |
| ResNet3D18 | 2 | 3 | 0.965±0.002 | 0.876±0.016 | BORDERLINE |
| ResNet3D18 | 4 | 3 | 0.972±0.004 | 0.837±0.100 | UNSTABLE |
| MC3-18 | 2 | 1 | 0.963 | 0.931 | INSUFFICIENT |
| Swin3D | any | 0 | - | - | OOM |
| **DenseNet3D121** | **4** | **4** | **0.979±0.003** | **0.924±0.028** | **SELECTED** |
"""
    write(out / "supplement_failed_backbones_summary.md", content)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0531_analysis_plan"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"Generating analysis plans in {root}")
    calibration_plan(root)
    error_analysis_plan(root)
    lidc_gap_plan(root)
    triclass_gap_plan(root)
    subgroup_plan(root)
    failed_backbones_plan(root)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
