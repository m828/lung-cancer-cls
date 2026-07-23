# Full Run Commands

## Preflight only

```bash
cd /home/apulis-dev/userdata/mmy/lung-cancer-cls-main0518
SMOKE=0 RUN_MODE=full SEEDS=42,43,44,45 \
STAGES=factorial,shuffle,cnv_perm,checkpoint,teacher_correction,triclass \
USE_EXISTING_TEACHERS=1 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --dry-run
```

## Bounded smoke

```bash
cd /home/apulis-dev/userdata/mmy/lung-cancer-cls-main0518
SMOKE=1 RUN_MODE=smoke SEEDS=42 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_smoke.sh
```

## Formal run, not executed in this implementation turn

```bash
cd /home/apulis-dev/userdata/mmy/lung-cancer-cls-main0518
SMOKE=0 RUN_MODE=full SEEDS=42,43,44,45 \
STAGES=factorial,shuffle,cnv_perm,checkpoint,teacher_correction,triclass \
USE_EXISTING_TEACHERS=1 \
OUTPUT_ROOT=outputs0542_privileged_genomic_attribution_suite \
bash scripts/run_privileged_genomic_attribution_suite.sh --root ./ --resume
```

Expected training jobs: 32. Expected preparation/cache commands: 8. The full dry run
generated all 40 command records without launching training.

