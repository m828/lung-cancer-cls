# Expected Output Schema

## Every new training run

- `command.txt`, `environment.txt`, `git_commit.txt`, `source_status.txt`
- `source_diff.patch`, `source_manifest.json`, `run_mode.txt`
- `resolved_run_config.json`, `run_provenance_start.json`
- `split_manifest.csv`, `teacher_logits_audit.json`, `kd_weight_mapping.csv`
- `kd_sample_weights_summary.json`, `training_history.json`, `metrics.json`
- `train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv`
- `best_model.pt`, `run_complete.json`
- `checkpoints/best_val_loss.pt`
- `checkpoints/best_val_auroc.pt`
- `checkpoints/best_val_f1.pt`
- `checkpoints/best_val_bacc.pt`
- `checkpoints/best_val_composite.pt`
- `checkpoints/last.pt`
- `checkpoint_evaluations.json`, `checkpoint_predictions/*.csv`

`run_complete.json` records mode, seed, duration, metric hash, student checkpoint path,
and student checkpoint SHA-256. Teacher checkpoint/cache paths and hashes are recorded in
the provenance and metrics files.

Each checkpoint evaluation records the validation-selected operating threshold when its
validation/test prediction pair is available. Resume state preserves training history,
early-stopping state, the primary best epoch, and all criterion-specific best records.

## Stage summaries

- `01_binary_factorial`: per-seed metrics, mean/SD, six paired effects, interaction,
  bootstrap intervals, threshold and identity audits.
- `02_shuffled_confidence`: three-arm summary, paired effects, exact weight-multiset audit,
  and SampleID donor mappings.
- `03_cnv_permutation`: teacher/student summaries and contrasts plus whole-row, split-local
  CNV donor mappings and integrity hashes.
- `04_checkpoint_sensitivity`: checkpoint inventory, test metrics by validation criterion,
  summary, stability report, and missing-file report.
- `05_teacher_correction_analysis`: four correction groups by seed, case list, student
  metrics, paired bootstrap, exact McNemar inputs, and ensemble sensitivity.
- `06_triclass_confusion_analysis`: class-map audit, raw/normalized matrices per seed,
  mean/SD matrices, class and macro metrics, vector PDFs, and PNG previews.

Rows marked `MISSING` are retained in inventories but excluded from means. Smoke fixtures
and full results occupy separate paths. A stage returns nonzero when any required arm/seed is
missing, so a partial table cannot be mistaken for a complete factorial result.
