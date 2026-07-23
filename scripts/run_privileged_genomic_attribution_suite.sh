#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/attribution_suite_common.sh"
ATTR_REMAINING_ARGS=()
attribution_parse_common_args "$@"
attribution_guard_mode
attribution_seed_array
cd "${ATTR_PROJECT_ROOT}"
export PYTHONPATH="${ATTR_PROJECT_ROOT}/src:${ATTR_PROJECT_ROOT}:${PYTHONPATH:-}"
ACTIVE_SEEDS="$(IFS=,; echo "${ATTR_SEED_ARRAY[*]}")"

STAGES="${STAGES:-factorial,shuffle,cnv_perm,checkpoint,teacher_correction,triclass}"
RESULTS_ROOT_RESOLVED="$(attribution_results_root)"
mkdir -p "${OUTPUT_ROOT}/reports" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/manifests"

common_flags=(--root "${ATTR_PROJECT_ROOT}")
if [[ "${RESUME}" == "1" ]]; then common_flags+=(--resume); else common_flags+=(--no-resume); fi
if [[ "${FORCE}" == "1" ]]; then common_flags+=(--force); fi
if [[ "${DRY_RUN}" == "1" ]]; then common_flags+=(--dry-run); fi

contains_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
failures=0

if contains_stage factorial; then
    bash scripts/run_binary_privileged_factorial_4seed.sh "${common_flags[@]}" || failures=$((failures + 1))
fi
if contains_stage shuffle; then
    bash scripts/run_binary_shuffled_confidence_4seed.sh "${common_flags[@]}" || failures=$((failures + 1))
fi
if contains_stage cnv_perm; then
    bash scripts/run_binary_cnv_permutation_4seed.sh "${common_flags[@]}" || failures=$((failures + 1))
fi
if contains_stage checkpoint && [[ "${DRY_RUN}" == "0" ]]; then
    python3 experiments/analysis/analyze_checkpoint_selection_sensitivity.py \
        --root "${OUTPUT_ROOT}/04_checkpoint_sensitivity" \
        --run-mode "${RUN_MODE}" \
        --search-root "${OUTPUT_ROOT}/01_binary_factorial" \
        --search-root "${OUTPUT_ROOT}/02_shuffled_confidence" \
        --search-root "${OUTPUT_ROOT}/03_cnv_permutation" \
        --search-root "${RESULTS_ROOT_RESOLVED}/outputs0535_student_kd_refinement/refined_candidates" \
        || failures=$((failures + 1))
fi
if contains_stage teacher_correction && [[ "${DRY_RUN}" == "0" ]]; then
    correction_args=(
        --output-dir "${OUTPUT_ROOT}/05_teacher_correction_analysis"
        --seeds "${ACTIVE_SEEDS}"
        --bootstrap-iters "$([[ "${RUN_MODE}" == "smoke" ]] && echo 100 || echo 10000)"
        --ct-text-teacher-pattern "${RESULTS_ROOT_RESOLVED}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed{seed}"
        --full-teacher-pattern "${RESULTS_ROOT_RESOLVED}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed{seed}"
        --supervised-pattern "${RESULTS_ROOT_RESOLVED}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed{seed}"
        --kd-pattern "${RESULTS_ROOT_RESOLVED}/outputs0535_student_kd_refinement/refined_candidates/R3_confidence_a0.1_T8_bs12_lr1e-4_composite_seed{seed}"
        --supervised-run-override "44=${RESULTS_ROOT_RESOLVED}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed44_repeat"
    )
    if [[ "${RUN_MODE}" == "smoke" ]]; then correction_args+=(--smoke); fi
    python3 experiments/analysis/analyze_teacher_correction_transfer.py "${correction_args[@]}" || failures=$((failures + 1))
fi
if contains_stage triclass && [[ "${DRY_RUN}" == "0" ]]; then
    python3 experiments/analysis/analyze_triclass_confusion_profiles.py \
        --source-root "${RESULTS_ROOT_RESOLVED}/outputs0541_triclass_teacher_student_selection_4seed" \
        --output-dir "${OUTPUT_ROOT}/06_triclass_confusion_analysis" \
        --seeds "${ACTIVE_SEEDS}" || failures=$((failures + 1))
fi

cat > "${OUTPUT_ROOT}/manifests/suite_invocation.txt" <<EOF
RUN_MODE=${RUN_MODE}
SMOKE=${SMOKE}
SEEDS=${SEEDS}
STAGES=${STAGES}
RESULTS_ROOT=${RESULTS_ROOT_RESOLVED}
OUTPUT_ROOT=${OUTPUT_ROOT}
RESUME=${RESUME}
FORCE=${FORCE}
DRY_RUN=${DRY_RUN}
FAILURE_COUNT=${failures}
EOF

echo "[DONE] privileged genomic attribution suite failures=${failures}"
exit "${failures}"
