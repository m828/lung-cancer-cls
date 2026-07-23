# Smoke Test Report

## Material Passport

- Artifact: `0542-smoke-validation`
- Type: implementation and analysis validation
- Status: `PARTIAL_PASS_ENVIRONMENT_BLOCKED`
- Scientific result: no
- Full training executed: no

## Results

| Module | Status | Verification |
|---|---|---|
| Python syntax | PASS | `compileall` on all new/modified suite files |
| Shell syntax | PASS | `bash -n` on all launchers |
| Focused unit tests | PASS | 15 passed |
| Atomic PyTorch serialization test | BLOCKED | 1 skipped because PyTorch is absent |
| Smoke command guard | PASS | full mode requires both `SMOKE=0` and `RUN_MODE=full` |
| Full command dry run | PASS | 40 command records (32 training, 8 preparation/cache); zero jobs started |
| Factorial analysis smoke | PASS | five synthetic fixture arms, 100 paired bootstrap replicates |
| Shuffled-confidence analysis smoke | PASS | weight multiset preserved, mapping changed |
| CNV-permutation analysis smoke | PASS | whole-row, split-local mapping audit passed |
| Checkpoint summary smoke | PASS | five validation criteria plus `last` parsed |
| Teacher-correction analysis | PASS | real existing predictions, seeds 42-45, 10,000 bootstrap replicates |
| Triclass confusion analysis | PASS | real existing predictions, seeds 42-45, PDF and PNG generated |
| S0_MATCHED 1-2 epoch training | BLOCKED | no local PyTorch/torchvision environment |
| Four KD factorial training modes | BLOCKED | same environment blocker |
| Shuffled-confidence training | BLOCKED | same environment blocker |
| Permuted-CNV teacher/student training | BLOCKED | same environment blocker and local teacher `.pt` files absent |

## Environment blocker

The smoke entrypoint stopped before data creation or training with exit code 3 and reported
missing `torch, torchvision, numpy, pandas, sklearn, matplotlib` in system Python. Two attempts
to install PyTorch into an isolated `/tmp` virtual environment failed because both PyPI and the
PyTorch CPU wheel endpoint timed out. No repository environment was modified.

## Commands exercised

```bash
PYTHONPATH=src:. /home/mmy/.hermes/hermes-agent/venv/bin/python -m pytest -q \
  tests/test_kd_weight_modes.py tests/test_cnv_permutation.py \
  tests/test_checkpoint_saving.py tests/test_prediction_alignment.py \
  tests/test_teacher_correction_groups.py tests/test_triclass_confusion_analysis.py

SMOKE=1 RUN_MODE=smoke SEEDS=42 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_smoke.sh
```

Analysis-only smoke fixtures were isolated outside scientific stage outputs and explicitly
marked non-scientific. They were removed after validation. The formal server smoke command is
provided in `full_run_commands.md`.

The final training-smoke preflight exited with code 3 before data preparation because the local
system Python lacks the required scientific stack. This is an environment blocker, not a model
or data failure; the bounded server command remains the required execution check.
