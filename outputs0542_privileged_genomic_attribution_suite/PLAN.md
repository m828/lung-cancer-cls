# 0542 Privileged Genomic Attribution Suite Plan

1. Audit existing binary and triclass checkpoints, manifests, cached logits, predictions, and training settings.
2. Add reusable identity-alignment, KD-weight, CNV-permutation, checkpoint, metric, and provenance utilities.
3. Extend the cached-logits student entrypoint without changing legacy defaults.
4. Add guarded, resumable launchers for factorial, shuffled-confidence, and CNV-permutation experiments.
5. Add paired identity-level analyses, checkpoint sensitivity, teacher-correction analysis, and triclass confusion analysis.
6. Add focused unit tests and run bounded smoke tests only.
7. Write the runbook, expected schemas, limitations, and exact full-run commands.

Full four-seed training is prohibited unless both `SMOKE=0` and `RUN_MODE=full` are set.
