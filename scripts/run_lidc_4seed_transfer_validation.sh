#!/usr/bin/env bash
# LIDC transfer validation (4-seed fixed protocol) for baseline vs KDInit.
# Intentionally minimal: lock to two profiles and optionally replay on fold/seed grid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0540_lidc_4seed_transfer_validation}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-full}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

USE_EXISTING_LIDC="${USE_EXISTING_LIDC:-1}"
LIDC_MANIFEST_DIR="${LIDC_MANIFEST_DIR:-/home/apulis-dev/userdata/mmy/Data/lidc_bvm_manifest_consensus}"
LIDC_DATA_BASE="${LIDC_DATA_BASE:-/home/apulis-dev/userdata/mmy/Data/lidc_idri_consensus_3d_fold}"

MODEL="${MODEL:-densenet3d_121}"
DEPTH_SIZE="${DEPTH_SIZE:-64}"
VOLUME_HW="${VOLUME_HW:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"

R3_ROOT="${R3_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement/refined_candidates}"
R3_EXPORT_ROOT="${R3_EXPORT_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement/R3_checkpoint_export}"
R3_CHECKPOINT="${R3_CHECKPOINT:-}"

INIT_PREFIX="${INIT_PREFIX-__AUTO__}"
SEEDS_ENV="${SEEDS:-}"
FOLDS_ENV="${FOLDS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            shift 2
            ;;
        --out-root)
            if [[ $# -lt 2 ]]; then echo "[ERROR] --out-root requires a path" >&2; exit 2; fi
            OUT_ROOT="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

normalize_lidc_data_base() {
    local base="${1%/}"
    if [[ "${base}" =~ ^(.+_fold)[0-9]+$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "${base}"
    fi
}

LIDC_DATA_BASE="$(normalize_lidc_data_base "${LIDC_DATA_BASE}")"

PROFILE_DEFAULTS=(baseline_default kdinit_diff_lr_01)

if [[ -z "${FOLDS_ENV}" ]]; then
    if [[ "${SMOKE}" == "1" ]]; then
        FOLDS=(0)
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        FOLDS=(0 1)
    else
        FOLDS=(0)
    fi
else
    IFS=',' read -r -a FOLDS <<<"${FOLDS_ENV}"
fi

if [[ -z "${SEEDS_ENV}" ]]; then
    if [[ "${SMOKE}" == "1" ]]; then
        SEEDS=(42)
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        SEEDS=(42 43)
    else
        SEEDS=(42 43 44 45)
    fi
else
    IFS=',' read -r -a SEEDS <<<"${SEEDS_ENV}"
fi

mkdir -p \
  "${OUT_ROOT}/profiles" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used" \
  "${OUT_ROOT}/runbook"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_lidc_4seed_transfer_validation.sh.snapshot"

validate_metric() {
    local metric="$1"
    case "${metric}" in
        accuracy|balanced_accuracy|auroc|macro_auroc|f1|macro_f1|recall|specificity|transfer_composite) return 0 ;;
        *)
            echo "[ERROR] unsupported selection metric: ${metric}" >&2
            return 2
            ;;
    esac
}

find_r3_checkpoint() {
    if [[ -n "${R3_CHECKPOINT}" ]]; then
        echo "${R3_CHECKPOINT}"
        return 0
    fi
    local cand
    for cand in \
        "${R3_EXPORT_ROOT}/best_ct_encoder.pt" \
        "${R3_EXPORT_ROOT}/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed42/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed43/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed44/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed45/best_model.pt"; do
        if [[ -f "${cand}" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    echo ""
    return 1
}

R3_CKPT_RESOLVED="$(find_r3_checkpoint || true)"

resolve_init_prefix() {
    if [[ "${INIT_PREFIX}" != "__AUTO__" ]]; then
        echo "${INIT_PREFIX}"
    elif [[ -n "${R3_CKPT_RESOLVED}" && "$(basename "${R3_CKPT_RESOLVED}")" == "best_ct_encoder.pt" ]]; then
        echo ""
    else
        echo "ct_encoder."
    fi
}
INIT_PREFIX_RESOLVED="$(resolve_init_prefix)"

fold_data_root() {
    echo "${LIDC_DATA_BASE}${1}"
}

fold_manifest() {
    echo "${LIDC_MANIFEST_DIR}/split_manifest.csv"
}

profile_dir() {
    local profile="$1"; local fold="$2"; local seed="$3"
    echo "${OUT_ROOT}/profiles/${profile}/fold${fold}/seed${seed}"
}

audit_json_for() {
    local profile="$1"; local fold="$2"; local seed="$3"
    echo "$(profile_dir "${profile}" "${fold}" "${seed}")/lidc_kdinit_loading_audit.json"
}

audit_md_for() {
    local profile="$1"; local fold="$2"; local seed="$3"
    echo "$(profile_dir "${profile}" "${fold}" "${seed}")/lidc_kdinit_loading_audit.md"
}

run_cmd() {
    local name="$1"
    local out_path="$2"
    shift 2
    local logfile="${OUT_ROOT}/logs/${name}.log"
    if [[ -e "${out_path}" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] ${name}: exists ${out_path}"
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name}"
        echo "          out: ${out_path}"
        echo "          cmd: $*"
        return 0
    fi
    mkdir -p "$(dirname "${logfile}")" "$(dirname "${out_path}")"
    echo "[RUN ] ${name}"
    echo "       out: ${out_path}"
    echo "       log: ${logfile}"
    set +e
    "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        echo "[FAIL] ${name}: exit ${rc}; continuing"
    fi
}

check_existing_lidc() {
    local missing=0
    for path in \
        "${LIDC_MANIFEST_DIR}/nodule_manifest.csv" \
        "${LIDC_MANIFEST_DIR}/split_manifest.csv"; do
        if [[ -f "${path}" ]]; then
            echo "[OK] existing LIDC file: ${path}"
        else
            echo "[MISSING] existing LIDC file: ${path}"
            missing=1
        fi
    done
    for fold in "${FOLDS[@]}"; do
        local path="$(fold_data_root "${fold}")"
        if [[ -d "${path}" ]]; then
            echo "[OK] existing LIDC fold: ${path}"
        else
            echo "[MISSING] existing LIDC fold: ${path}"
            missing=1
        fi
    done
    if (( missing != 0 && DRY_RUN != 1 )); then
        return 1
    fi
    return 0
}

profile_defaults() {
    # Locked baseline and KD profile definitions.
    local profile="$1"
    case "${profile}" in
        baseline_default)
            LIDC_PROFILE_LR="${LR}"
            LIDC_SELECTION_METRIC="balanced_accuracy"
            LIDC_INIT_MODE="none"
            LIDC_TRANSFER_MODE="full_finetune"
            LIDC_ENCODER_LR_MULT="1.0"
            LIDC_HEAD_LR_MULT="1.0"
            ;;
        kdinit_diff_lr_01)
            LIDC_PROFILE_LR="${LR}"
            LIDC_SELECTION_METRIC="balanced_accuracy"
            LIDC_INIT_MODE="r3_ct_encoder"
            LIDC_TRANSFER_MODE="differential_lr"
            LIDC_ENCODER_LR_MULT="0.1"
            LIDC_HEAD_LR_MULT="1.0"
            ;;
        *)
            echo "[ERROR] Unknown LIDC profile: ${profile}" >&2
            return 2
            ;;
    esac
}

run_audit() {
    local profile="$1" fold="$2" seed="$3" outdir="$4"
    if [[ "${profile}" != "kdinit_diff_lr_01" ]]; then
        return 0
    fi
    if [[ -z "${R3_CKPT_RESOLVED}" ]]; then
        echo "[MISSING] KD audit skip: R3 checkpoint unresolved"
        return 0
    fi
    run_cmd "audit_${profile}_fold${fold}_seed${seed}" "$(audit_json_for "${profile}" "${fold}" "${seed}")" \
        python3 experiments/analysis/audit_lidc_kdinit_loading.py \
        --checkpoint "${R3_CKPT_RESOLVED}" \
        --lidc-model "${MODEL}" \
        --lidc-num-classes 2 \
        --init-prefix "${INIT_PREFIX_RESOLVED}" \
        --source-backbone densenet3d_121 \
        --output-json "$(audit_json_for "${profile}" "${fold}" "${seed}")" \
        --output-md "$(audit_md_for "${profile}" "${fold}" "${seed}")"
}

run_train_case() {
    local profile="$1" fold="$2" seed="$3"
    local outdir
    outdir="$(profile_dir "${profile}" "${fold}" "${seed}")"
    profile_defaults "${profile}"
    validate_metric "${LIDC_SELECTION_METRIC}" || return 2

    local init_args=()
    if [[ "${LIDC_INIT_MODE}" == "r3_ct_encoder" ]]; then
        if [[ -z "${R3_CKPT_RESOLVED}" ]]; then
            echo "[MISSING] ${profile} fold${fold} seed${seed}: no R3 checkpoint"
            return 0
        fi
        init_args=(--init-checkpoint "${R3_CKPT_RESOLVED}" --init-checkpoint-prefix "${INIT_PREFIX_RESOLVED}")
    elif [[ "${LIDC_INIT_MODE}" != "none" ]]; then
        echo "[ERROR] Unsupported LIDC_INIT_MODE=${LIDC_INIT_MODE}" >&2
        return 2
    fi

    echo "[PLAN] ${profile} fold${fold} seed${seed}: init=${LIDC_INIT_MODE} transfer=${LIDC_TRANSFER_MODE} selection=${LIDC_SELECTION_METRIC} lr=${LIDC_PROFILE_LR} enc_lr_mult=${LIDC_ENCODER_LR_MULT} head_lr_mult=${LIDC_HEAD_LR_MULT}"
    run_cmd "${profile}_fold${fold}_seed${seed}" "${outdir}/metrics.json" \
        python3 train.py \
        --output-dir "${outdir}" \
        --dataset-type lidc_idri \
        --data-root "$(fold_data_root "${fold}")" \
        --model "${MODEL}" \
        --use-3d-input \
        --depth-size "${DEPTH_SIZE}" \
        --volume-hw "${VOLUME_HW}" \
        --class-mode binary \
        --binary-task benign_vs_malignant \
        --use-predefined-split \
        --split-manifest-csv "$(fold_manifest "${fold}")" \
        --split-fold "${fold}" \
        --split-mode train_val_test \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --seed "${seed}" \
        --lr "${LIDC_PROFILE_LR}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --optimizer adamw --scheduler cosine \
        --sampling-strategy weighted \
        --class-weight-strategy effective_num \
        --selection-metric "${LIDC_SELECTION_METRIC}" \
        --transfer-mode "${LIDC_TRANSFER_MODE}" \
        --encoder-lr-mult "${LIDC_ENCODER_LR_MULT}" \
        --head-lr-mult "${LIDC_HEAD_LR_MULT}" \
        "${init_args[@]}"
}

stage_audit() {
    local profile fold seed
    for profile in "${PROFILE_DEFAULTS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if [[ "${profile}" == "kdinit_diff_lr_01" ]]; then
                    run_audit "${profile}" "${fold}" "${seed}" "$(profile_dir "${profile}" "${fold}" "${seed}")"
                else
                    echo "[SKIP] audit baseline_default fold${fold} seed${seed}: no init"
                fi
            done
        done
    done
}

stage_train() {
    if [[ "${USE_EXISTING_LIDC}" == "1" ]]; then
        check_existing_lidc || return 0
    fi
    local profile fold seed
    for profile in "${PROFILE_DEFAULTS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            if [[ ! -f "$(fold_manifest "${fold}")" && "${DRY_RUN}" != "1" ]]; then
                echo "[MISSING] split manifest for fold${fold}: $(fold_manifest "${fold}")"
                continue
            fi
            if [[ ! -d "$(fold_data_root "${fold}")" && "${DRY_RUN}" != "1" ]]; then
                echo "[MISSING] fold data root for fold${fold}: $(fold_data_root "${fold}")"
                continue
            fi
            for seed in "${SEEDS[@]}"; do
                if [[ "${profile}" == "kdinit_diff_lr_01" ]]; then
                    run_audit "${profile}" "${fold}" "${seed}" "$(profile_dir "${profile}" "${fold}" "${seed}")"
                fi
                run_train_case "${profile}" "${fold}" "${seed}"
            done
        done
    done
}

stage_analyze() {
    run_cmd "analyze_lidc_4seed_transfer_validation" "${OUT_ROOT}/lidc_4seed_summary.md" \
        python3 experiments/analysis/analyze_lidc_4seed_transfer_validation.py --root "${OUT_ROOT}"
}

cat <<EOF
============================================================
LIDC 4-seed transfer validation (fixed profile mode)
PROJECT_ROOT         : ${PROJECT_ROOT}
OUT_ROOT             : ${OUT_ROOT}
STAGE                : ${STAGE}
RUN_MODE             : ${RUN_MODE}
SMOKE                : ${SMOKE}
DRY_RUN              : ${DRY_RUN}
USE_EXISTING_LIDC    : ${USE_EXISTING_LIDC}
FOLDS                : ${FOLDS[*]}
SEEDS                : ${SEEDS[*]}
LIDC_MANIFEST_DIR    : ${LIDC_MANIFEST_DIR}
LIDC_DATA_BASE       : ${LIDC_DATA_BASE}
LIDC_PROFILE         : baseline_default, kdinit_diff_lr_01
MODEL                : ${MODEL}
R3_CHECKPOINT        : ${R3_CKPT_RESOLVED:-<missing>}
INIT_PREFIX          : ${INIT_PREFIX_RESOLVED}
============================================================
EOF

if [[ "${USE_EXISTING_LIDC}" != "1" ]]; then
    echo "[ERROR] This suite is fixed-input mode and requires USE_EXISTING_LIDC=1." >&2
    echo "        Set USE_EXISTING_LIDC=1 to use preprocessed LIDC data directly."
    exit 2
fi

if [[ "${STAGE}" == "audit" || "${STAGE}" == "all" ]]; then stage_audit; fi
if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then stage_train; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

cat > "${OUT_ROOT}/lidc_4seed_runbook.md" <<'EOF'
# LIDC 4-seed transfer validation runbook

## Purpose

Lock and reproduce fixed-method LIDC transfer validation on preprocessed folds.

- Profiles:
  - `baseline_default`
  - `kdinit_diff_lr_01`
- Data source:
  - Manifest: `${LIDC_MANIFEST_DIR}/split_manifest.csv`
  - Folds: `${LIDC_DATA_BASE}{0..4}`

## Smoke run (quick check)

```bash
SMOKE=1 USE_EXISTING_LIDC=1 FOLDS=0 SEEDS=42 \
DRY_RUN=1 bash scripts/run_lidc_4seed_transfer_validation.sh --root ./
```

## Full 4-seed run (fold0 default)

```bash
USE_EXISTING_LIDC=1 FOLDS=0 SEEDS=42,43,44,45 \
DRY_RUN=1 bash scripts/run_lidc_4seed_transfer_validation.sh --root ./
```

to actually run: set `DRY_RUN=0` and add `STAGE=train` / `STAGE=all`.

## Full protocol run (all folds if needed)

```bash
USE_EXISTING_LIDC=1 FOLDS=0,1,2,3,4 SEEDS=42,43,44,45 \
STAGE=train bash scripts/run_lidc_4seed_transfer_validation.sh --root ./
```

## Analysis

```bash
python experiments/analysis/analyze_lidc_4seed_transfer_validation.py --root outputs0540_lidc_4seed_transfer_validation
```

## How to judge paper-ready interpretation

- If one fold only (FOLDS=0), treat as **single-fold 4-seed repeated transfer validation**, not full 5-fold external validation.\n+- Compare `lidc_4seed_comparison.md`, `lidc_4seed_seedwise.csv`, and `lidc_4seed_summary.md`.
- If one fold only (FOLDS=0), treat as **single-fold 4-seed repeated transfer validation**, not full 5-fold external validation.
- Compare `lidc_4seed_comparison.md`, `lidc_4seed_seedwise.csv`, and `lidc_4seed_summary.md`.

## Distinguishing from full 5-fold

- Full 5-fold means `FOLDS=0,1,2,3,4`.
- Single-fold mode means only `FOLDS=0` and should be interpreted as a fixed-split repeated experiment.

EOF

echo "[DONE] outputs: ${OUT_ROOT}"
