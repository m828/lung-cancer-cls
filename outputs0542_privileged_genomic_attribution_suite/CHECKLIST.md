# 0542 Implementation Checklist

- [x] Preserve old outputs and manuscript as read-only inputs.
- [x] Audit current baseline, R3 configuration, cached logits, and local checkpoint availability.
- [x] Create common audit and statistical utilities.
- [x] Extend cached-logits training with explicit attribution modes and multi-checkpoint saving.
- [x] Implement split-local whole-row CNV permutation preparation.
- [x] Add six experiment configurations.
- [x] Add guarded launch scripts with resume, force, and dry-run behavior.
- [x] Add all requested analysis scripts.
- [x] Add focused unit tests.
- [x] Run static checks and unit tests (15 passed; one PyTorch-only check environment-skipped).
- [x] Attempt bounded smoke without full training; analysis paths passed and training was explicitly environment-blocked.
- [x] Write audit inventories, implementation report, runbook, and full-run commands.
