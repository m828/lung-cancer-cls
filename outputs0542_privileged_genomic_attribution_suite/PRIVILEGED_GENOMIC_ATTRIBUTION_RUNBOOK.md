# Privileged Genomic Attribution Suite Runbook

## Scope

This suite isolates teacher modality, KD weighting, confidence-to-case correspondence,
patient-level CNV correspondence, checkpoint selection, case-level teacher corrections,
and triclass class redistribution. It does not modify any manuscript or pre-0542 output.

The full binary training matrix contains 32 training jobs:

- 20 student jobs: five factorial arms x four seeds.
- 4 student jobs: shuffled-confidence control x four seeds.
- 4 permuted-CNV teacher jobs.
- 4 permuted-CNV student jobs.

Cache generation and CNV permutation preparation add eight non-training jobs. Existing
T0/T1 caches are reused read-only in the factorial and shuffle stages.

## Safety Gate

Full training is rejected unless both variables are explicit:

```bash
SMOKE=0 RUN_MODE=full
```

Any other combination is either rejected or forced to smoke mode. New runs are written
under a `smoke/` or `full/` component, so their summaries cannot be mixed silently.

## Server Prerequisites

1. Activate the existing project environment with PyTorch, torchvision, NumPy, pandas,
   scikit-learn, and matplotlib.
2. Run from `/home/apulis-dev/userdata/mmy/lung-cancer-cls-main0518`.
3. Confirm that the synchronized `outputs0531`, `outputs0534`, and `outputs0535` trees
   resolve through `RESULTS_ROOT` and that T0/T1 teacher checkpoints exist there.
4. Confirm the inferred data paths with a dry run before training.

The local workstation copy contains teacher metrics, predictions, and cached logits but
does not contain teacher `.pt` files. Checkpoint-dependent smoke/full training therefore
belongs on the server or another environment with the original checkpoint binaries.

## Dry Run

```bash
SMOKE=0 \
RUN_MODE=full \
SEEDS=42,43,44,45 \
STAGES=factorial,shuffle,cnv_perm,checkpoint,teacher_correction,triclass \
USE_EXISTING_TEACHERS=1 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --dry-run
```

Dry run creates command/provenance stubs but starts no model or analysis process.

## Bounded Smoke

Use the real server environment:

```bash
SMOKE=1 \
RUN_MODE=smoke \
SEEDS=42 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_smoke.sh
```

The smoke fixture uses 24/12/12 class-interleaved train/val/test cases, one teacher epoch,
and at most two student epochs. It exercises all five factorial arms, shuffled confidence,
the permuted-CNV teacher/student chain, checkpoint evaluation, teacher correction, and
triclass confusion analysis. Smoke performance is never interpreted scientifically.

## Full Run

```bash
SMOKE=0 \
RUN_MODE=full \
SEEDS=42,43,44,45 \
STAGES=factorial,shuffle,cnv_perm,checkpoint,teacher_correction,triclass \
USE_EXISTING_TEACHERS=1 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --resume
```

The default is `RESUME=1`, `FORCE=0`. A completed run is skipped. An incomplete student
run resumes from `checkpoints/last.pt` when present, including its historical best-checkpoint
records, training history, and early-stopping state. `--force` archives a prior 0542 run
to a timestamped sibling before starting; it never overwrites pre-0542 outputs.

## Stage Runs

```bash
STAGES=factorial bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --resume
STAGES=shuffle bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --resume
STAGES=cnv_perm bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --resume
STAGES=checkpoint,teacher_correction,triclass bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --resume
```

Set the same `SMOKE`, `RUN_MODE`, `SEEDS`, and `OUTPUT_ROOT` variables for every command.

## Required Invariants

- Uniform and confidence arms for a teacher must have the same cached-target SHA-256.
- Confidence is computed from untempered teacher softmax; soft targets use temperature 8.
- Shuffled confidence permutes training weights only and preserves their exact multiset.
- CNV permutation uses whole rows independently within train, val, and test.
- The primary student checkpoint remains `best_val_composite.pt`/`best_model.pt`.
- Test metrics are evaluated only after validation-based checkpoint selection.
- Checkpoint sensitivity uses each checkpoint's validation predictions to select its operating
  threshold before applying that threshold to the corresponding test predictions.
- Paired bootstrap resamples common test identities within each seed, then averages seed effects.
- Smoke and full directories must never be merged for analysis.

## Failure Handling

Each run has an independent log under `logs/`. A nonzero exit writes `failed.exit_code` and
does not delete successful seeds. Identity, label, split, cache-confidence, shuffled-weight,
or CNV-permutation audit failures are fatal for that run and report the offending IDs.
