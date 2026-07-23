# Existing Pipeline Audit

- CT-text and CT-text-CNV teachers have byte-identical split manifests for seeds 42-45: **True**.
- The two teacher families match on the audited training protocol fields for seeds 42-45: **True**. The CT-text-CNV teacher additionally contains the CNV branch and associated fusion parameters.
- Both teacher families have metrics and predictions for seeds 42-45: **True**.
- Local teacher checkpoint binaries cover seeds 42-45: **False**. The local synchronized result copy contains metrics/predictions but no `.pt` files; checkpoint-dependent full runs must execute where the original checkpoints exist.
- Cached T0/T1 teacher logits cover seeds 42-45: **True**.
- The locked CT-text reference is not training-matched to R3: **True**. Verified differences: batch_size: locked=4, R3=12; lr: locked=0.0003, R3=0.0001; checkpoint criterion: locked='auroc', R3='composite'; mixed precision: locked generic trainer=disabled, R3 cached-KD trainer=enabled.
- Existing cached KD correctly retains hard-label CE and normalizes weighted KL by the within-batch weight sum.
- Existing R3 runs save only `best_model.pt`; the 0542 extension adds validation-loss/AUROC/F1/BAcc/composite and last checkpoints without changing legacy defaults.
- Both original and repeat seed-44 teacher directories exist. R3 cache metadata points to the original seed-44 teachers, so teacher-cache factorial arms follow the original artifact and do not merge it with the repeat. The read-only teacher-correction analysis separately uses the paper-locked `seed44_repeat` run as its hard-label student reference.
- Triclass TRI-T cache, TRI-S0 predictions, and TRI-SKD predictions are present for seeds 42-45: **True**.
- TRI-S0 contains 36 test identities per seed outside the teacher/KD aligned cohort; the confusion comparison explicitly uses the 227 common identities and reports this restriction.

## Reuse

Existing manifests, T0/T1 cached logits, teacher predictions, locked CT-text predictions, R3 predictions, and triclass predictions are reused read-only. New training is required for `S0_MATCHED`, uniform/confidence matched arms not already run under R3 settings, shuffled confidence, and the permuted-CNV teacher/student.

## Audit exceptions

- Teacher protocol mismatches: none.
- Teacher manifest mismatches: none.
