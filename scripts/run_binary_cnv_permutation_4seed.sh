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
if [[ "${DRY_RUN}" != "1" && ( -z "${GENE_TSV:-}" || ! -f "${GENE_TSV}" ) ]]; then
    echo "[FATAL] GENE_TSV is required for CNV permutation: ${GENE_TSV:-<unset>}" >&2
    exit 2
fi
attribution_student_common_args
STAGE_ROOT="${OUTPUT_ROOT}/03_cnv_permutation"
FACTORIAL_ROOT="${OUTPUT_ROOT}/01_binary_factorial"
mkdir -p "${STAGE_ROOT}"/{teachers,students,permutation_manifests,audits,permuted_features,cached_teacher_targets}
GENE_COLUMN_ARGS=()
[[ -n "${GENE_ID_COL:-}" ]] && GENE_COLUMN_ARGS+=(--gene-id-col "${GENE_ID_COL}")
[[ -n "${GENE_LABEL_COL:-}" ]] && GENE_COLUMN_ARGS+=(--gene-label-col "${GENE_LABEL_COL}")

safe_link() {
    local source="$1" target="$2"
    if [[ -L "${target}" || -e "${target}" ]]; then return 0; fi
    mkdir -p "$(dirname "${target}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] link ${target} -> ${source}"
    elif [[ -e "${source}" ]]; then
        ln -s "${source}" "${target}"
    else
        echo "[MISSING] reuse source: ${source}" >&2
        return 2
    fi
}

failures=0
for seed in "${ATTR_SEED_ARRAY[@]}"; do
    ct_teacher="$(attribution_teacher_dir "${RESULTS_ROOT_RESOLVED}" T0 "${seed}")"
    real_teacher="$(attribution_teacher_dir "${RESULTS_ROOT_RESOLVED}" T1 "${seed}")"
    safe_link "${ct_teacher}" "${STAGE_ROOT}/teachers/ct_text/seed${seed}" || failures=$((failures + 1))
    safe_link "${real_teacher}" "${STAGE_ROOT}/teachers/real_cnv/seed${seed}" || failures=$((failures + 1))
    safe_link "${FACTORIAL_ROOT}/KD_CT_TEXT_CONFIDENCE/${RUN_MODE}/seed${seed}" "${STAGE_ROOT}/students/ct_text_confidence/${RUN_MODE}/seed${seed}" || failures=$((failures + 1))
    safe_link "${FACTORIAL_ROOT}/KD_CT_TEXT_CNV_CONFIDENCE/${RUN_MODE}/seed${seed}" "${STAGE_ROOT}/students/real_cnv_confidence/${RUN_MODE}/seed${seed}" || failures=$((failures + 1))

    permutation_seed=$((seed + 20000))
    permutation_dir="${STAGE_ROOT}/permuted_features/${RUN_MODE}/seed${seed}"
    permuted_tsv="${permutation_dir}/permuted_cnv.tsv"
    mapping_csv="${STAGE_ROOT}/permutation_manifests/${RUN_MODE}_seed${seed}.csv"
    permutation_force=()
    if [[ "${FORCE}" == "1" ]]; then permutation_force=(--force); fi
    attribution_run "prepare_cnv_permutation_${RUN_MODE}_seed${seed}" "${permutation_dir}" \
        python3 scripts/prepare_cnv_permutation.py \
        --gene-tsv "${GENE_TSV}" --split-manifest "${REF_MANIFEST}" \
        --output-tsv "${permuted_tsv}" --mapping-csv "${mapping_csv}" \
        --cnv-permutation-seed "${permutation_seed}" "${GENE_COLUMN_ARGS[@]}" "${permutation_force[@]}" || { failures=$((failures + 1)); continue; }

    teacher_out="${STAGE_ROOT}/teachers/permuted_cnv/${RUN_MODE}/seed${seed}"
    teacher_smoke=()
    teacher_epochs=50
    if [[ "${RUN_MODE}" == "smoke" ]]; then teacher_smoke=(--smoke --num-workers 0); teacher_epochs=2; fi
    attribution_run "permuted_cnv_teacher_${RUN_MODE}_seed${seed}" "${teacher_out}" \
        python3 train_multimodal.py \
        --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" --output-dir "${teacher_out}" \
        --reference-manifest "${REF_MANIFEST}" --ct-root "${CT_ROOT}" --gene-tsv "${permuted_tsv}" \
        --text-feature-tsv "${TEXT_FEATURE_TSV}" "${GENE_COLUMN_ARGS[@]}" --modalities ct,text,cnv \
        --class-mode binary --binary-task malignant_vs_normal --strict-no-leakage --disable-text-numeric-features \
        --ct-model "${CT_MODEL:-densenet3d_121}" --depth-size "${DEPTH_SIZE:-128}" --volume-hw "${VOLUME_HW:-256}" \
        --ct-feature-dim "${CT_FEATURE_DIM:-128}" --text-feature-dim "${TEXT_FEATURE_DIM:-256}" --gene-hidden-dim "${GENE_HIDDEN_DIM:-256}" --fusion-hidden-dim "${FUSION_HIDDEN_DIM:-256}" \
        --dropout 0.3 --epochs "${teacher_epochs}" --batch-size 4 --seed "${seed}" \
        --lr 3e-4 --weight-decay 1e-4 --optimizer adamw --scheduler cosine \
        --loss ce --label-smoothing 0.05 --sampling-strategy weighted \
        --class-weight-strategy effective_num --effective-num-beta 0.999 --selection-metric auroc \
        "${teacher_smoke[@]}" || { failures=$((failures + 1)); continue; }

    cache_dir="${STAGE_ROOT}/cached_teacher_targets/${RUN_MODE}/seed${seed}"
    cache_name="permuted_cnv_seed${seed}"
    attribution_run "cache_permuted_cnv_${RUN_MODE}_seed${seed}" "${cache_dir}" \
        python3 scripts/cache_teacher_soft_targets.py \
        --teacher-run-dir "${teacher_out}" --output-dir "${cache_dir}" --cache-name "${cache_name}" \
        --seed "${seed}" --batch-size 8 --num-workers "$([[ "${RUN_MODE}" == "smoke" ]] && echo 0 || echo "${NUM_WORKERS:-2}")" \
        --reference-manifest "${teacher_out}/split_manifest.csv" \
        --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" --ct-root "${CT_ROOT}" \
        --gene-tsv "${permuted_tsv}" --text-feature-tsv "${TEXT_FEATURE_TSV}" "${GENE_COLUMN_ARGS[@]}" \
        --class-mode binary --binary-task malignant_vs_normal --strict-no-leakage --disable-text-numeric-features || { failures=$((failures + 1)); continue; }

    student_out="${STAGE_ROOT}/students/permuted_cnv_confidence/${RUN_MODE}/seed${seed}"
    resume_args=()
    if [[ "${RESUME}" == "1" && -f "${student_out}/checkpoints/last.pt" ]]; then resume_args=(--resume-checkpoint "${student_out}/checkpoints/last.pt"); fi
    attribution_run "permuted_cnv_student_${RUN_MODE}_seed${seed}" "${student_out}" \
        python3 scripts/train_student_kd_cached_logits.py \
        "${ATTR_STUDENT_COMMON[@]}" --seed "${seed}" --output-dir "${student_out}" \
        --teacher-type ct_text_permuted_cnv --teacher-checkpoint "${teacher_out}/best_model.pt" \
        --cached-logits-path "${cache_dir}/${cache_name}.csv" --kd-weight-mode confidence \
        --cnv-permutation-seed "${permutation_seed}" "${resume_args[@]}" || failures=$((failures + 1))
done

bootstrap_iters="$([[ "${RUN_MODE}" == "smoke" ]] && echo 100 || echo 10000)"
ACTIVE_SEEDS="$(IFS=,; echo "${ATTR_SEED_ARRAY[*]}")"
analysis_args=(--root "${STAGE_ROOT}" --bootstrap-iters "${bootstrap_iters}" --expected-seeds "${ACTIVE_SEEDS}")
if [[ "${RUN_MODE}" == "smoke" ]]; then analysis_args+=(--smoke); fi
if [[ "${DRY_RUN}" == "0" ]]; then
    python3 experiments/analysis/analyze_cnv_permutation_control.py "${analysis_args[@]}" || failures=$((failures + 1))
fi
echo "[DONE] CNV-permutation stage failures=${failures}"
exit "${failures}"
