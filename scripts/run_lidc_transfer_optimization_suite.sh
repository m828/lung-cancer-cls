#!/usr/bin/env bash
# Configurable LIDC transfer optimization suite.
# Uses existing processed LIDC folds by default and never regenerates splits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0539_lidc_transfer_optimization}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-mini}"
SMOKE="${SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_EXISTING_LIDC="${USE_EXISTING_LIDC:-1}"
LIDC_MANIFEST_DIR="${LIDC_MANIFEST_DIR:-/home/apulis-dev/userdata/mmy/Data/lidc_bvm_manifest_consensus}"
LIDC_DATA_BASE="${LIDC_DATA_BASE:-/home/apulis-dev/userdata/mmy/Data/lidc_idri_consensus_3d_fold}"
MODEL="${MODEL:-densenet3d_121}"
DEPTH_SIZE="${DEPTH_SIZE:-64}"
VOLUME_HW="${VOLUME_HW:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LIDC_PROFILE_ENV="${LIDC_PROFILE:-}"
LIDC_SELECTION_METRIC_USER="${LIDC_SELECTION_METRIC:-}"
LIDC_INIT_MODE_USER="${LIDC_INIT_MODE:-}"
LIDC_TRANSFER_MODE_USER="${LIDC_TRANSFER_MODE:-}"
LIDC_FREEZE_EPOCHS_USER="${LIDC_FREEZE_EPOCHS:-}"
LIDC_ENCODER_LR_MULT_USER="${LIDC_ENCODER_LR_MULT:-}"
LIDC_HEAD_LR_MULT_USER="${LIDC_HEAD_LR_MULT:-}"
LIDC_LR_USER="${LIDC_LR:-}"
LIDC_WEIGHT_DECAY_USER="${LIDC_WEIGHT_DECAY:-}"
LIDC_EPOCHS_USER="${LIDC_EPOCHS:-}"
LIDC_BATCH_SIZE_USER="${LIDC_BATCH_SIZE:-}"
LIDC_AUG_PROFILE_USER="${LIDC_AUG_PROFILE:-}"
LIDC_LABEL_SMOOTHING_USER="${LIDC_LABEL_SMOOTHING:-}"
LIDC_LOSS_USER="${LIDC_LOSS:-}"
LIDC_FOCAL_GAMMA_USER="${LIDC_FOCAL_GAMMA:-}"
R3_ROOT="${R3_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement/refined_candidates}"
R3_EXPORT_ROOT="${R3_EXPORT_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement/R3_checkpoint_export}"
R3_CHECKPOINT="${R3_CHECKPOINT:-}"
INIT_PREFIX="${INIT_PREFIX-__AUTO__}"
LIDC_ALLOWED_SELECTION_METRICS=(accuracy balanced_accuracy auroc f1 recall specificity transfer_composite)

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

mkdir -p "${OUT_ROOT}/profiles" "${OUT_ROOT}/logs" "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_lidc_transfer_optimization_suite.sh.snapshot"

if [[ "${SMOKE}" == "1" ]]; then
    FOLDS=(0)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    FOLDS=(0 1)
else
    FOLDS=(0 1 2 3 4)
fi

if [[ -n "${LIDC_PROFILE_ENV}" ]]; then
    IFS=',' read -r -a PROFILES <<<"${LIDC_PROFILE_ENV}"
elif [[ "${RUN_MODE}" == "sweep" || "${RUN_MODE}" == "full" ]]; then
    PROFILES=(baseline_default kdinit_full_ft_bacc kdinit_full_ft_auc kdinit_low_lr kdinit_diff_lr_01 kdinit_diff_lr_005 kdinit_freeze5 kdinit_linear_probe kdinit_strong_aug kdinit_focal)
else
    PROFILES=(baseline_default kdinit_full_ft_bacc kdinit_diff_lr_01 kdinit_freeze5)
fi

find_r3_checkpoint() {
    if [[ -n "${R3_CHECKPOINT}" ]]; then echo "${R3_CHECKPOINT}"; return 0; fi
    local cand
    for cand in \
        "${R3_EXPORT_ROOT}/best_ct_encoder.pt" \
        "${R3_EXPORT_ROOT}/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed42/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed45/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed44/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed43/best_model.pt"; do
        if [[ -f "${cand}" ]]; then echo "${cand}"; return 0; fi
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

profile_defaults() {
    local profile="$1"
    LIDC_INIT_MODE="none"
    LIDC_TRANSFER_MODE="full_finetune"
    LIDC_SELECTION_METRIC="balanced_accuracy"
    LIDC_FREEZE_EPOCHS="0"
    LIDC_ENCODER_LR_MULT="1.0"
    LIDC_HEAD_LR_MULT="1.0"
    LIDC_LR="1e-4"
    LIDC_WEIGHT_DECAY="1e-4"
    LIDC_EPOCHS="30"
    LIDC_BATCH_SIZE="8"
    LIDC_AUG_PROFILE="basic"
    LIDC_LABEL_SMOOTHING="0.0"
    LIDC_LOSS="ce"
    LIDC_FOCAL_GAMMA="2.0"
    case "${profile}" in
        baseline_default) ;;
        kdinit_full_ft_bacc) LIDC_INIT_MODE="r3_ct_encoder" ;;
        kdinit_full_ft_auc) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_SELECTION_METRIC="auroc" ;;
        kdinit_low_lr) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_LR="5e-5" ;;
        kdinit_diff_lr_01) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_TRANSFER_MODE="differential_lr"; LIDC_ENCODER_LR_MULT="0.1" ;;
        kdinit_diff_lr_005) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_TRANSFER_MODE="differential_lr"; LIDC_ENCODER_LR_MULT="0.05" ;;
        kdinit_freeze5) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_TRANSFER_MODE="freeze_then_finetune"; LIDC_FREEZE_EPOCHS="5" ;;
        kdinit_linear_probe) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_TRANSFER_MODE="linear_probe_then_finetune"; LIDC_FREEZE_EPOCHS="5" ;;
        kdinit_strong_aug) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_TRANSFER_MODE="differential_lr"; LIDC_ENCODER_LR_MULT="0.1"; LIDC_AUG_PROFILE="strong" ;;
        kdinit_focal) LIDC_INIT_MODE="r3_ct_encoder"; LIDC_TRANSFER_MODE="differential_lr"; LIDC_ENCODER_LR_MULT="0.1"; LIDC_LOSS="focal"; LIDC_FOCAL_GAMMA="2.0" ;;
        *) echo "[ERROR] Unknown LIDC_PROFILE=${profile}" >&2; exit 2 ;;
    esac
    [[ -n "${LIDC_SELECTION_METRIC_USER}" ]] && LIDC_SELECTION_METRIC="${LIDC_SELECTION_METRIC_USER}"
    [[ -n "${LIDC_INIT_MODE_USER}" ]] && LIDC_INIT_MODE="${LIDC_INIT_MODE_USER}"
    [[ -n "${LIDC_TRANSFER_MODE_USER}" ]] && LIDC_TRANSFER_MODE="${LIDC_TRANSFER_MODE_USER}"
    [[ -n "${LIDC_FREEZE_EPOCHS_USER}" ]] && LIDC_FREEZE_EPOCHS="${LIDC_FREEZE_EPOCHS_USER}"
    [[ -n "${LIDC_ENCODER_LR_MULT_USER}" ]] && LIDC_ENCODER_LR_MULT="${LIDC_ENCODER_LR_MULT_USER}"
    [[ -n "${LIDC_HEAD_LR_MULT_USER}" ]] && LIDC_HEAD_LR_MULT="${LIDC_HEAD_LR_MULT_USER}"
    [[ -n "${LIDC_LR_USER}" ]] && LIDC_LR="${LIDC_LR_USER}"
    [[ -n "${LIDC_WEIGHT_DECAY_USER}" ]] && LIDC_WEIGHT_DECAY="${LIDC_WEIGHT_DECAY_USER}"
    [[ -n "${LIDC_EPOCHS_USER}" ]] && LIDC_EPOCHS="${LIDC_EPOCHS_USER}"
    [[ -n "${LIDC_BATCH_SIZE_USER}" ]] && LIDC_BATCH_SIZE="${LIDC_BATCH_SIZE_USER}"
    [[ -n "${LIDC_AUG_PROFILE_USER}" ]] && LIDC_AUG_PROFILE="${LIDC_AUG_PROFILE_USER}"
    [[ -n "${LIDC_LABEL_SMOOTHING_USER}" ]] && LIDC_LABEL_SMOOTHING="${LIDC_LABEL_SMOOTHING_USER}"
    [[ -n "${LIDC_LOSS_USER}" ]] && LIDC_LOSS="${LIDC_LOSS_USER}"
    [[ -n "${LIDC_FOCAL_GAMMA_USER}" ]] && LIDC_FOCAL_GAMMA="${LIDC_FOCAL_GAMMA_USER}"
    return 0
}

validate_lidc_selection_metric() {
    local metric="$1"
    local allowed
    for allowed in "${LIDC_ALLOWED_SELECTION_METRICS[@]}"; do
        if [[ "${metric}" == "${allowed}" ]]; then
            return 0
        fi
    done
    echo "[ERROR] Unsupported LIDC_SELECTION_METRIC=${metric}. Allowed: ${LIDC_ALLOWED_SELECTION_METRICS[*]}" >&2
    return 2
}

training_loss_name() {
    case "$1" in
        class_balanced_ce) echo "ce" ;;
        ce|focal) echo "$1" ;;
        *) echo "[ERROR] Unknown LIDC_LOSS=$1" >&2; exit 2 ;;
    esac
}

class_weight_strategy_for() {
    case "$1" in
        class_balanced_ce) echo "effective_num" ;;
        *) echo "effective_num" ;;
    esac
}

fold_data_root() { echo "${LIDC_DATA_BASE}${1}"; }
fold_manifest() { echo "${LIDC_MANIFEST_DIR}/split_manifest.csv"; }
profile_fold_out() { echo "${OUT_ROOT}/profiles/${1}/fold${2}"; }
audit_json_for() { echo "$(profile_fold_out "$1" "$2")/lidc_kdinit_loading_audit.json"; }
audit_md_for() { echo "$(profile_fold_out "$1" "$2")/lidc_kdinit_loading_audit.md"; }

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
    if (( rc != 0 )); then echo "[FAIL] ${name}: exit ${rc}; continuing"; fi
}

check_existing_lidc() {
    local missing=0 path fold
    for path in "${LIDC_MANIFEST_DIR}/nodule_manifest.csv" "${LIDC_MANIFEST_DIR}/split_manifest.csv"; do
        [[ -f "${path}" ]] && echo "[OK] existing LIDC file: ${path}" || { echo "[MISSING] existing LIDC file: ${path}"; missing=1; }
    done
    for fold in 0 1 2 3 4; do
        path="$(fold_data_root "${fold}")"
        [[ -d "${path}" ]] && echo "[OK] existing LIDC fold${fold}: ${path}" || { echo "[MISSING] existing LIDC fold${fold}: ${path}"; missing=1; }
    done
    if (( missing != 0 && DRY_RUN != 1 )); then return 1; fi
    return 0
}

run_audit_for() {
    local profile="$1"
    local fold="$2"
    if [[ "${LIDC_INIT_MODE}" != "r3_ct_encoder" ]]; then
        return 0
    fi
    if [[ -z "${R3_CKPT_RESOLVED}" ]]; then
        echo "[MISSING] ${profile} fold${fold}: R3 checkpoint not found; set R3_CHECKPOINT=/path/to/best_model.pt"
        return 0
    fi
    run_cmd "audit_${profile}_fold${fold}" "$(audit_json_for "${profile}" "${fold}")" \
        python3 experiments/analysis/audit_lidc_kdinit_loading.py \
        --checkpoint "${R3_CKPT_RESOLVED}" \
        --lidc-model "${MODEL}" \
        --lidc-num-classes 2 \
        --init-prefix "${INIT_PREFIX_RESOLVED}" \
        --source-backbone densenet3d_121 \
        --output-json "$(audit_json_for "${profile}" "${fold}")" \
        --output-md "$(audit_md_for "${profile}" "${fold}")"
}

write_audit() {
    cat >"${OUT_ROOT}/lidc_optimization_code_audit.md" <<EOF
# LIDC Transfer Optimization Code Audit

This suite is a public-data transfer validation optimizer, not the locked binary main result.

- Output root: \`${OUT_ROOT}\`
- USE_EXISTING_LIDC: \`${USE_EXISTING_LIDC}\`
- Manifest dir: \`${LIDC_MANIFEST_DIR}\`
- Fold data prefix: \`${LIDC_DATA_BASE}\`
- LIDC backbone: \`${MODEL}\`
- Source R3 checkpoint: \`${R3_CKPT_RESOLVED:-<missing>}\`
- Source R3 backbone: \`densenet3d_121\`
- Init prefix: \`${INIT_PREFIX_RESOLVED}\`
- Supported selection metrics: accuracy, balanced_accuracy, auroc, f1, recall, specificity, transfer_composite
- Supported transfer modes: full_finetune, freeze_encoder, freeze_then_finetune, differential_lr, linear_probe_then_finetune
- Supported profiles: baseline_default, kdinit_full_ft_bacc, kdinit_full_ft_auc, kdinit_low_lr, kdinit_diff_lr_01, kdinit_diff_lr_005, kdinit_freeze5, kdinit_linear_probe, kdinit_strong_aug, kdinit_focal
- KDInit profile/fold runs write \`lidc_kdinit_loading_audit.json/md\` before training.
- Existing folds are consumed as-is; this script does not regenerate LIDC splits.
EOF
}

stage_train() {
    if [[ "${USE_EXISTING_LIDC}" != "1" ]]; then
        echo "[ERROR] This optimization suite expects USE_EXISTING_LIDC=1 and will not regenerate folds." >&2
        return 2
    fi
    check_existing_lidc || return 0
    local profile fold outdir loss_name class_weight init_args name
    for profile in "${PROFILES[@]}"; do
        profile_defaults "${profile}"
        loss_name="$(training_loss_name "${LIDC_LOSS}")"
        class_weight="$(class_weight_strategy_for "${LIDC_LOSS}")"
        for fold in "${FOLDS[@]}"; do
            outdir="$(profile_fold_out "${profile}" "${fold}")"
            name="lidc_${profile}_fold${fold}"
            validate_lidc_selection_metric "${LIDC_SELECTION_METRIC}"
            echo "[PROFILE] ${profile} fold${fold}: init=${LIDC_INIT_MODE} transfer=${LIDC_TRANSFER_MODE} selection=${LIDC_SELECTION_METRIC} loss=${LIDC_LOSS} lr=${LIDC_LR} out=${outdir}"
            echo "[AUDIT] path: $(audit_json_for "${profile}" "${fold}")"
            run_audit_for "${profile}" "${fold}"
            init_args=()
            if [[ "${LIDC_INIT_MODE}" == "r3_ct_encoder" ]]; then
                if [[ -z "${R3_CKPT_RESOLVED}" ]]; then
                    echo "[MISSING] ${profile} fold${fold}: R3 checkpoint not found"
                    continue
                fi
                init_args=(--init-checkpoint "${R3_CKPT_RESOLVED}" --init-checkpoint-prefix "${INIT_PREFIX_RESOLVED}")
            elif [[ "${LIDC_INIT_MODE}" != "none" ]]; then
                echo "[ERROR] Unsupported LIDC_INIT_MODE=${LIDC_INIT_MODE}" >&2
                return 2
            fi
            run_cmd "${name}" "${outdir}/metrics.json" \
                python3 train.py \
                --output-dir "${outdir}" \
                --dataset-type lidc_idri \
                --data-root "$(fold_data_root "${fold}")" \
                --model "${MODEL}" \
                --use-3d-input --depth-size "${DEPTH_SIZE}" --volume-hw "${VOLUME_HW}" \
                --class-mode binary --binary-task benign_vs_malignant \
                --use-predefined-split \
                --split-manifest-csv "$(fold_manifest "${fold}")" \
                --split-fold "${fold}" \
                --split-mode train_val_test \
                --epochs "${LIDC_EPOCHS}" --batch-size "${LIDC_BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
                --lr "${LIDC_LR}" --weight-decay "${LIDC_WEIGHT_DECAY}" \
                --optimizer adamw --scheduler cosine \
                --aug-profile "${LIDC_AUG_PROFILE}" \
                --loss "${loss_name}" --label-smoothing "${LIDC_LABEL_SMOOTHING}" --focal-gamma "${LIDC_FOCAL_GAMMA}" \
                --sampling-strategy weighted --class-weight-strategy "${class_weight}" \
                --selection-metric "${LIDC_SELECTION_METRIC}" \
                --transfer-mode "${LIDC_TRANSFER_MODE}" \
                --freeze-epochs "${LIDC_FREEZE_EPOCHS}" \
                --encoder-lr-mult "${LIDC_ENCODER_LR_MULT}" \
                --head-lr-mult "${LIDC_HEAD_LR_MULT}" \
                "${init_args[@]}"
        done
    done
}

stage_analyze() {
    run_cmd "analyze_lidc_transfer_optimization" "${OUT_ROOT}/lidc_transfer_optimization_summary.md" \
        python3 experiments/analysis/analyze_lidc_transfer_optimization.py --root "${OUT_ROOT}"
}

cat <<EOF
============================================================
LIDC transfer optimization suite
PROJECT_ROOT     : ${PROJECT_ROOT}
OUT_ROOT         : ${OUT_ROOT}
STAGE            : ${STAGE}
RUN_MODE         : ${RUN_MODE}
SMOKE            : ${SMOKE}
DRY_RUN          : ${DRY_RUN}
USE_EXISTING_LIDC: ${USE_EXISTING_LIDC}
FOLDS            : ${FOLDS[*]}
PROFILES         : ${PROFILES[*]}
LIDC_MANIFEST_DIR: ${LIDC_MANIFEST_DIR}
LIDC_DATA_BASE   : ${LIDC_DATA_BASE}
R3_CHECKPOINT    : ${R3_CKPT_RESOLVED:-<missing>}
INIT_PREFIX      : ${INIT_PREFIX_RESOLVED}
============================================================
EOF

if [[ "${STAGE}" == "audit" || "${STAGE}" == "all" ]]; then write_audit; fi
if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then stage_train; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
