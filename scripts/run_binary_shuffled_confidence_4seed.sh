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

RESULTS_ROOT_RESOLVED="$(attribution_results_root)"
attribution_infer_paths "${RESULTS_ROOT_RESOLVED}"
attribution_validate_paths
attribution_student_common_args
STAGE_ROOT="${OUTPUT_ROOT}/02_shuffled_confidence"
FACTORIAL_ROOT="${OUTPUT_ROOT}/01_binary_factorial"
mkdir -p "${STAGE_ROOT}/weight_permutation_manifests" "${STAGE_ROOT}/audits"

link_reused_arm() {
    local name="$1" source_arm="$2"
    local target="${STAGE_ROOT}/${name}" source="${FACTORIAL_ROOT}/${source_arm}"
    if [[ -L "${target}" || -e "${target}" ]]; then return 0; fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] link ${target} -> ${source}"
    elif [[ -e "${source}" ]]; then
        ln -s "${source}" "${target}"
    else
        echo "[MISSING] factorial source for ${name}: ${source}" >&2
        return 2
    fi
}
link_reused_arm uniform KD_CT_TEXT_CNV_UNIFORM || exit $?
link_reused_arm true_confidence KD_CT_TEXT_CNV_CONFIDENCE || exit $?

failures=0
for seed in "${ATTR_SEED_ARRAY[@]}"; do
    cache="$(attribution_cache_for "${RESULTS_ROOT_RESOLVED}" T1 "${seed}")"
    teacher_dir="$(attribution_teacher_dir "${RESULTS_ROOT_RESOLVED}" T1 "${seed}")"
    if [[ ! -f "${cache}" && "${DRY_RUN}" != "1" ]]; then
        echo "[MISSING] T1 cached logits for seed${seed}: ${cache}" >&2
        failures=$((failures + 1)); continue
    fi
    outdir="${STAGE_ROOT}/shuffled_confidence/${RUN_MODE}/seed${seed}"
    resume_args=()
    if [[ "${RESUME}" == "1" && -f "${outdir}/checkpoints/last.pt" ]]; then
        resume_args=(--resume-checkpoint "${outdir}/checkpoints/last.pt")
    fi
    attribution_run "shuffled_confidence_${RUN_MODE}_seed${seed}" "${outdir}" \
        python3 scripts/train_student_kd_cached_logits.py \
        "${ATTR_STUDENT_COMMON[@]}" --seed "${seed}" --output-dir "${outdir}" \
        --teacher-type ct_text_cnv --teacher-checkpoint "${teacher_dir}/best_model.pt" \
        --cached-logits-path "${cache}" --kd-weight-mode shuffled_confidence \
        --confidence-shuffle-seed "$((seed + 10000))" "${resume_args[@]}" || failures=$((failures + 1))
    if [[ -f "${outdir}/kd_weight_mapping.csv" ]]; then
        cp -f "${outdir}/kd_weight_mapping.csv" "${STAGE_ROOT}/weight_permutation_manifests/${RUN_MODE}_seed${seed}.csv"
    fi
done

bootstrap_iters="$([[ "${RUN_MODE}" == "smoke" ]] && echo 100 || echo 10000)"
ACTIVE_SEEDS="$(IFS=,; echo "${ATTR_SEED_ARRAY[*]}")"
analysis_args=(--root "${STAGE_ROOT}" --bootstrap-iters "${bootstrap_iters}" --expected-seeds "${ACTIVE_SEEDS}")
if [[ "${RUN_MODE}" == "smoke" ]]; then analysis_args+=(--smoke); fi
if [[ "${DRY_RUN}" == "0" ]]; then
    python3 experiments/analysis/analyze_shuffled_confidence_control.py "${analysis_args[@]}" || failures=$((failures + 1))
fi
echo "[DONE] shuffled-confidence stage failures=${failures}"
exit "${failures}"
