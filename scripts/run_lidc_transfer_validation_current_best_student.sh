#!/usr/bin/env bash
# LIDC-IDRI transfer validation for the current best gene-distilled CT+Text student.
# DRY_RUN=1 prints concrete commands or MISSING reasons only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0537_lidc_transfer_validation}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-mini}"
SMOKE="${SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DEPTH_SIZE="${DEPTH_SIZE:-64}"
VOLUME_HW="${VOLUME_HW:-64}"
MODEL="${MODEL:-densenet3d_121}"

LIDC_RAW_ROOT="${LIDC_RAW_ROOT:-/workspace/data-lung/LIDC-IDRI}"
LIDC_MANIFEST_ROOT="${LIDC_MANIFEST_ROOT:-${OUT_ROOT}/lidc_bvm_manifest_consensus}"
LIDC_DATA_ROOT_BASE="${LIDC_DATA_ROOT_BASE:-/workspace/data-lung/lidc_idri_consensus_3d_fold}"
R3_ROOT="${R3_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement/refined_candidates}"
R3_CHECKPOINT="${R3_CHECKPOINT:-}"
INIT_PREFIX="${INIT_PREFIX:-ct_encoder.}"

mkdir -p \
  "${OUT_ROOT}/baseline" \
  "${OUT_ROOT}/kd_init" \
  "${OUT_ROOT}/folds" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_lidc_transfer_validation_current_best_student.sh.snapshot"

if [[ "${SMOKE}" == "1" ]]; then
    FOLDS=(0)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    FOLDS=(0 1)
else
    FOLDS=(0 1 2 3 4)
fi

find_r3_checkpoint() {
    if [[ -n "${R3_CHECKPOINT}" ]]; then
        echo "${R3_CHECKPOINT}"
        return 0
    fi
    local cand
    for cand in \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed42/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed45/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed44/best_model.pt" \
        "${R3_ROOT}/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed43/best_model.pt"; do
        if [[ -f "${cand}" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    echo ""
    return 1
}

R3_CKPT_RESOLVED="$(find_r3_checkpoint || true)"

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
    return 0
}

fold_data_root() {
    local fold="$1"
    echo "${LIDC_DATA_ROOT_BASE}${fold}"
}

fold_manifest() {
    local fold="$1"
    echo "$(fold_data_root "${fold}")/processed_split_manifest.csv"
}

baseline_out() {
    local fold="$1"
    echo "${OUT_ROOT}/baseline/LIDC-B_${MODEL}_fold${fold}"
}

kdinit_out() {
    local fold="$1"
    echo "${OUT_ROOT}/kd_init/LIDC-KDInit_R3ct_${MODEL}_fold${fold}"
}

stage_manifest() {
    if [[ -f "${LIDC_MANIFEST_ROOT}/split_manifest.csv" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] lidc_manifest: exists ${LIDC_MANIFEST_ROOT}/split_manifest.csv"
        return 0
    fi
    if [[ ! -d "${LIDC_RAW_ROOT}" && "${DRY_RUN}" != "1" ]]; then
        echo "[MISSING] LIDC raw root: ${LIDC_RAW_ROOT}"
        return 0
    fi
    run_cmd "build_lidc_manifest" "${LIDC_MANIFEST_ROOT}/split_manifest.csv" \
        python3 build_lidc_idri_split_manifest.py \
        --input-root "${LIDC_RAW_ROOT}" \
        --output-dir "${LIDC_MANIFEST_ROOT}" \
        --metadata-source auto \
        --annotation-policy consensus \
        --consensus-min-readers 2 \
        --xy-tolerance-px 15 \
        --z-tolerance-mm 3 \
        --label-policy score12_vs_score45 \
        --split-scheme patient_kfold \
        --n-splits 5 \
        --val-ratio 0.1 \
        --seed 42
}

stage_prepare() {
    local fold
    for fold in "${FOLDS[@]}"; do
        if [[ -f "$(fold_manifest "${fold}")" && "${DRY_RUN}" != "1" ]]; then
            echo "[SKIP] prepare_lidc_fold${fold}: exists $(fold_manifest "${fold}")"
            continue
        fi
        if [[ ! -f "${LIDC_MANIFEST_ROOT}/nodule_manifest.csv" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] nodule manifest for fold${fold}: ${LIDC_MANIFEST_ROOT}/nodule_manifest.csv"
            continue
        fi
        run_cmd "prepare_lidc_fold${fold}" "$(fold_manifest "${fold}")" \
            python3 prepare_lidc_idri_consensus_3d.py \
            --input-root "${LIDC_RAW_ROOT}" \
            --manifest-csv "${LIDC_MANIFEST_ROOT}/nodule_manifest.csv" \
            --split-manifest-csv "${LIDC_MANIFEST_ROOT}/split_manifest.csv" \
            --split-fold "${fold}" \
            --output-root "$(fold_data_root "${fold}")" \
            --depth-size "${DEPTH_SIZE}" \
            --volume-hw "${VOLUME_HW}" \
            --context-scale 1.5 \
            --min-size-xy 32 \
            --min-size-z 8
    done
}

train_args_for_fold() {
    local fold="$1"
    echo \
        --dataset-type lidc_idri \
        --data-root "$(fold_data_root "${fold}")" \
        --model "${MODEL}" \
        --use-3d-input --depth-size "${DEPTH_SIZE}" --volume-hw "${VOLUME_HW}" \
        --class-mode binary --binary-task benign_vs_malignant \
        --use-predefined-split \
        --split-manifest-csv "$(fold_manifest "${fold}")" \
        --split-fold "${fold}" \
        --split-mode train_val_test \
        --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
        --optimizer adamw --scheduler cosine \
        --sampling-strategy weighted --class-weight-strategy effective_num \
        --selection-metric auroc
}

stage_train() {
    local fold
    for fold in "${FOLDS[@]}"; do
        if [[ ! -d "$(fold_data_root "${fold}")" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] LIDC fold${fold} data root: $(fold_data_root "${fold}")"
            continue
        fi
        if [[ ! -f "$(fold_manifest "${fold}")" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] LIDC fold${fold} split manifest: $(fold_manifest "${fold}")"
            continue
        fi
        run_cmd "LIDC-B_fold${fold}" "$(baseline_out "${fold}")/metrics.json" \
            python3 train.py \
            --output-dir "$(baseline_out "${fold}")" \
            $(train_args_for_fold "${fold}")

        if [[ -z "${R3_CKPT_RESOLVED}" ]]; then
            echo "[MISSING] LIDC-KDInit_fold${fold}: R3 best_model.pt not found under ${R3_ROOT}; set R3_CHECKPOINT=/path/to/best_model.pt"
        elif [[ ! -f "${R3_CKPT_RESOLVED}" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] LIDC-KDInit_fold${fold}: ${R3_CKPT_RESOLVED}"
        else
            run_cmd "LIDC-KDInit_fold${fold}" "$(kdinit_out "${fold}")/metrics.json" \
                python3 train.py \
                --output-dir "$(kdinit_out "${fold}")" \
                $(train_args_for_fold "${fold}") \
                --init-checkpoint "${R3_CKPT_RESOLVED}" \
                --init-checkpoint-prefix "${INIT_PREFIX}"
        fi
    done
}

stage_analyze() {
    run_cmd "analyze_lidc_transfer_validation" "${OUT_ROOT}/lidc_summary.md" \
        python3 experiments/analysis/analyze_lidc_transfer_validation.py --root "${OUT_ROOT}"
}

cat <<EOF
============================================================
LIDC transfer validation: current best R3 CT encoder
PROJECT_ROOT     : ${PROJECT_ROOT}
OUT_ROOT         : ${OUT_ROOT}
STAGE            : ${STAGE}
RUN_MODE         : ${RUN_MODE}
SMOKE            : ${SMOKE}
DRY_RUN          : ${DRY_RUN}
FOLDS            : ${FOLDS[*]}
LIDC_RAW_ROOT    : ${LIDC_RAW_ROOT}
LIDC_MANIFEST    : ${LIDC_MANIFEST_ROOT}
LIDC_DATA_BASE   : ${LIDC_DATA_ROOT_BASE}
R3_ROOT          : ${R3_ROOT}
R3_CHECKPOINT    : ${R3_CKPT_RESOLVED:-<missing>}
GROUPS           : LIDC-B CT baseline; LIDC-KDInit CT encoder initialized from R3
============================================================
EOF

if [[ "${STAGE}" == "manifest" || "${STAGE}" == "all" ]]; then stage_manifest; fi
if [[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]]; then stage_prepare; fi
if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then stage_train; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
