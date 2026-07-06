#!/usr/bin/env bash
# Prepare the R3 CT encoder checkpoint required by LIDC-KDInit.
#
# This script first searches for an existing R3 checkpoint.  If none exists, it
# can rerun the locked R3 seed checkpoint-saving configuration.  It does not
# start LIDC training; it only prints a LIDC dry-run command after export.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
REFINE_ROOT="${REFINE_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement}"
SEARCH_ROOTS="${SEARCH_ROOTS:-${REFINE_ROOT} ${PARENT_ROOT}/outputs0534_best_student_kd_search ${PARENT_ROOT}}"
OUT_DIR="${OUT_DIR:-${REFINE_ROOT}/R3_checkpoint_export}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
R3_SEED="${R3_SEED:-42}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-2}"

R3_PREFIX="${R3_PREFIX:-R3_confidence_a0.1_T8_bs12_lr1e-4_composite}"
R3_RUN_DIR="${R3_RUN_DIR:-${REFINE_ROOT}/refined_candidates/${R3_PREFIX}_seed${R3_SEED}}"
R3_METRICS="${R3_METRICS:-${R3_RUN_DIR}/metrics.json}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs" "${OUT_DIR}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_DIR}/scripts_used/export_r3_checkpoint_for_lidc.sh.snapshot"

infer_r3_config_py='
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
keys = [
    "data_root", "metadata_csv", "ct_root", "text_feature_tsv", "reference_manifest",
    "cached_teacher_targets",
]
out = {k: "" for k in keys}
if p.is_file():
    try:
        cfg = json.loads(p.read_text(encoding="utf-8")).get("config", {})
    except Exception:
        cfg = {}
    for k in keys:
        if cfg.get(k):
            out[k] = str(cfg[k])
for k in keys:
    print(f"{k}\t{out[k]}")
'

declare -A INFERRED
while IFS=$'\t' read -r key value; do
    INFERRED[$key]="$value"
done < <(python3 -c "${infer_r3_config_py}" "${R3_METRICS}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"
REFERENCE_MANIFEST="${REFERENCE_MANIFEST:-${PARENT_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42/split_manifest.csv}"
if [[ -n "${INFERRED[reference_manifest]:-}" && -f "${INFERRED[reference_manifest]}" ]]; then
    REFERENCE_MANIFEST="${REFERENCE_MANIFEST:-${INFERRED[reference_manifest]}}"
fi

default_cache="${PARENT_ROOT}/outputs0534_best_student_kd_search/cached_teacher_targets/T1_seed${R3_SEED}.csv"
if [[ ! -f "${default_cache}" ]]; then
    default_cache="${REFINE_ROOT}/cached_teacher_targets/T1_seed${R3_SEED}.csv"
fi
T1_CACHE="${T1_CACHE:-${default_cache}}"
if [[ -n "${INFERRED[cached_teacher_targets]:-}" && -f "${INFERRED[cached_teacher_targets]}" ]]; then
    T1_CACHE="${T1_CACHE:-${INFERRED[cached_teacher_targets]}}"
fi

find_existing_r3_checkpoint() {
    local root path
    for root in ${SEARCH_ROOTS}; do
        [[ -d "${root}" ]] || continue
        while IFS= read -r path; do
            case "${path}" in
                *"${R3_PREFIX}"* | *"R3_repeat1_confidence_a0.1_T8_bs12_lr1e-4_composite"*)
                    echo "${path}"
                    return 0
                    ;;
            esac
        done < <(find "${root}" -type f \( -name 'best_model.pt' -o -name 'best.pt' -o -name 'checkpoint.pt' -o -name 'model.pt' -o -name '*.ckpt' \) -print)
    done
    return 1
}

run_cmd() {
    local name="$1"
    shift
    local logfile="${OUT_DIR}/logs/${name}.log"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name}"
        echo "          cmd: $*"
        return 0
    fi
    echo "[RUN ] ${name}"
    echo "       log: ${logfile}"
    set +e
    "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        echo "[FAIL] ${name}: exit ${rc}"
        return "${rc}"
    fi
}

copy_existing_checkpoint() {
    local ckpt="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] copy_existing_checkpoint"
        echo "          cmd: cp -f ${ckpt} ${OUT_DIR}/best_model.pt"
        return 0
    fi
    cp -f "${ckpt}" "${OUT_DIR}/best_model.pt"
}

check_r3_training_inputs() {
    local missing=0
    local label path
    for item in \
        "DATA_ROOT:${DATA_ROOT}" \
        "METADATA_CSV:${METADATA_CSV}" \
        "CT_ROOT:${CT_ROOT}" \
        "TEXT_FEATURE_TSV:${TEXT_FEATURE_TSV}" \
        "REFERENCE_MANIFEST:${REFERENCE_MANIFEST}" \
        "T1_CACHE:${T1_CACHE}"; do
        label="${item%%:*}"
        path="${item#*:}"
        if [[ -z "${path}" ]]; then
            echo "[MISSING] ${label}: <empty>"
            missing=1
        elif [[ ! -e "${path}" ]]; then
            echo "[MISSING] ${label}: ${path}"
            missing=1
        else
            echo "[OK] ${label}: ${path}"
        fi
    done
    return "${missing}"
}

train_r3_checkpoint() {
    check_r3_training_inputs || {
        echo "[BLOCKED] Cannot regenerate R3 checkpoint until missing inputs are available or overridden."
        echo "          Override DATA_ROOT/METADATA_CSV/CT_ROOT/TEXT_FEATURE_TSV/REFERENCE_MANIFEST/T1_CACHE as needed."
        return 2
    }
    run_cmd "train_R3_checkpoint_seed${R3_SEED}" \
        python3 scripts/train_student_kd_cached_logits.py \
        --output-dir "${OUT_DIR}" \
        --cached-teacher-targets "${T1_CACHE}" \
        --seed "${R3_SEED}" \
        --distillation-alpha 0.1 --distillation-temperature 8 \
        --batch-size 12 --grad-accum-steps 1 --lr 1e-4 --weight-decay 1e-4 \
        --optimizer adamw --scheduler cosine --epochs "${EPOCHS}" --amp \
        --early-stopping-patience 10 --num-workers "${NUM_WORKERS}" \
        --kd-weighting confidence --kd-weight-floor 0.05 --kd-weight-max 1.0 \
        --class-mode binary --binary-task malignant_vs_normal \
        --strict-no-leakage --disable-text-numeric-features \
        --reference-manifest "${REFERENCE_MANIFEST}" \
        --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" \
        --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}" \
        --ct-model densenet3d_121 --modalities ct,text \
        --depth-size 128 --volume-hw 256 \
        --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256 \
        --dropout 0.3 --loss ce --label-smoothing 0.05 \
        --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999 \
        --selection-metric auroc --composite-selection-metric
}

export_ct_encoder() {
    if [[ ! -f "${OUT_DIR}/best_model.pt" && "${DRY_RUN}" != "1" ]]; then
        echo "[MISSING] ${OUT_DIR}/best_model.pt"
        return 2
    fi
    run_cmd "export_best_ct_encoder" \
        python3 scripts/export_ct_encoder_checkpoint.py \
        --checkpoint "${OUT_DIR}/best_model.pt" \
        --output "${OUT_DIR}/best_ct_encoder.pt" \
        --metadata-output "${OUT_DIR}/checkpoint_metadata.json" \
        --source-run-dir "${OUT_DIR}" \
        --lidc-model densenet3d_121 \
        --lidc-num-classes 2 \
        --verify-lidc-load
}

lidc_dry_run() {
    local ckpt="${OUT_DIR}/best_ct_encoder.pt"
    if [[ ! -f "${ckpt}" && "${DRY_RUN}" != "1" ]]; then
        echo "[MISSING] LIDC-KDInit init checkpoint: ${ckpt}"
        return 2
    fi
    echo "[INFO] LIDC-KDInit dry-run command:"
    echo "       DRY_RUN=1 SMOKE=1 STAGE=train R3_CHECKPOINT=${ckpt} INIT_PREFIX=ct_encoder. bash scripts/run_lidc_transfer_validation_current_best_student.sh"
    DRY_RUN=1 SMOKE=1 STAGE=train R3_CHECKPOINT="${ckpt}" INIT_PREFIX=ct_encoder. \
        bash scripts/run_lidc_transfer_validation_current_best_student.sh
}

cat <<EOF
============================================================
R3 checkpoint export for LIDC-KDInit
PROJECT_ROOT       : ${PROJECT_ROOT}
REFINE_ROOT        : ${REFINE_ROOT}
OUT_DIR            : ${OUT_DIR}
R3_SEED            : ${R3_SEED}
R3_RUN_DIR         : ${R3_RUN_DIR}
R3_METRICS         : ${R3_METRICS}
SEARCH_ROOTS       : ${SEARCH_ROOTS}
DRY_RUN            : ${DRY_RUN}
FORCE_RETRAIN      : ${FORCE_RETRAIN}
============================================================
EOF

existing_ckpt=""
if [[ "${FORCE_RETRAIN}" != "1" ]]; then
    existing_ckpt="$(find_existing_r3_checkpoint || true)"
fi

if [[ -n "${existing_ckpt}" ]]; then
    echo "[FOUND] existing R3 checkpoint: ${existing_ckpt}"
    copy_existing_checkpoint "${existing_ckpt}"
else
    echo "[INFO] existing R3 checkpoint not found under: ${SEARCH_ROOTS}"
    train_r3_checkpoint || true
fi

export_ct_encoder || true
lidc_dry_run || true

echo "[DONE] export target: ${OUT_DIR}"
