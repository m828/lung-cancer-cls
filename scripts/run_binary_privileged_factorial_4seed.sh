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
STAGE_ROOT="${OUTPUT_ROOT}/01_binary_factorial"
mkdir -p "${STAGE_ROOT}" "${OUTPUT_ROOT}/00_audit"

ensure_cache() {
    local teacher="$1" seed="$2"
    local cache
    cache="$(attribution_cache_for "${RESULTS_ROOT_RESOLVED}" "${teacher}" "${seed}")"
    if [[ -f "${cache}" || "${DRY_RUN}" == "1" ]]; then printf '%s\n' "${cache}"; return 0; fi
    local teacher_dir cache_dir cache_name
    teacher_dir="$(attribution_teacher_dir "${RESULTS_ROOT_RESOLVED}" "${teacher}" "${seed}")"
    cache_dir="${OUTPUT_ROOT}/00_audit/generated_teacher_caches/${teacher}_seed${seed}"
    cache_name="${teacher}_seed${seed}"
    local extra=()
    if [[ "${teacher}" == "T1" ]]; then
        extra=(--gene-tsv "${GENE_TSV}")
        [[ -n "${GENE_ID_COL:-}" ]] && extra+=(--gene-id-col "${GENE_ID_COL}")
        [[ -n "${GENE_LABEL_COL:-}" ]] && extra+=(--gene-label-col "${GENE_LABEL_COL}")
    fi
    if [[ ! -f "${teacher_dir}/best_model.pt" ]]; then
        echo "[MISSING] teacher checkpoint required to regenerate cache: ${teacher_dir}/best_model.pt" >&2
        return 2
    fi
    attribution_run "cache_${teacher}_seed${seed}" "${cache_dir}" \
        python3 scripts/cache_teacher_soft_targets.py \
        --teacher-run-dir "${teacher_dir}" --output-dir "${cache_dir}" --cache-name "${cache_name}" \
        --seed "${seed}" --batch-size 8 --num-workers "${NUM_WORKERS:-2}" \
        --class-mode binary --binary-task malignant_vs_normal \
        --strict-no-leakage --disable-text-numeric-features \
        --reference-manifest "${REF_MANIFEST}" --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" \
        --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}" "${extra[@]}" >&2 || return $?
    printf '%s\n' "${cache_dir}/${cache_name}.csv"
}

run_student_arm() {
    local arm="$1" seed="$2" mode="$3" teacher="$4" cache="$5"
    local outdir="${STAGE_ROOT}/${arm}/${RUN_MODE}/seed${seed}"
    local resume_args=()
    if [[ "${RESUME}" == "1" && -f "${outdir}/checkpoints/last.pt" ]]; then
        resume_args=(--resume-checkpoint "${outdir}/checkpoints/last.pt")
    fi
    local teacher_args=()
    if [[ -n "${teacher}" ]]; then
        local teacher_dir
        teacher_dir="$(attribution_teacher_dir "${RESULTS_ROOT_RESOLVED}" "${teacher}" "${seed}")"
        teacher_args=(--teacher-type "$([[ "${teacher}" == "T0" ]] && echo ct_text || echo ct_text_cnv)" --teacher-checkpoint "${teacher_dir}/best_model.pt" --cached-logits-path "${cache}")
    fi
    attribution_run "${arm}_${RUN_MODE}_seed${seed}" "${outdir}" \
        python3 scripts/train_student_kd_cached_logits.py \
        "${ATTR_STUDENT_COMMON[@]}" --seed "${seed}" --output-dir "${outdir}" \
        --kd-weight-mode "${mode}" "${teacher_args[@]}" "${resume_args[@]}"
}

failures=0
for seed in "${ATTR_SEED_ARRAY[@]}"; do
    t0_cache="$(ensure_cache T0 "${seed}")" || { failures=$((failures + 1)); continue; }
    t1_cache="$(ensure_cache T1 "${seed}")" || { failures=$((failures + 1)); continue; }
    run_student_arm S0_MATCHED "${seed}" none "" "" || failures=$((failures + 1))
    run_student_arm KD_CT_TEXT_UNIFORM "${seed}" uniform T0 "${t0_cache}" || failures=$((failures + 1))
    run_student_arm KD_CT_TEXT_CONFIDENCE "${seed}" confidence T0 "${t0_cache}" || failures=$((failures + 1))
    run_student_arm KD_CT_TEXT_CNV_UNIFORM "${seed}" uniform T1 "${t1_cache}" || failures=$((failures + 1))
    run_student_arm KD_CT_TEXT_CNV_CONFIDENCE "${seed}" confidence T1 "${t1_cache}" || failures=$((failures + 1))
done

ACTIVE_SEEDS="$(IFS=,; echo "${ATTR_SEED_ARRAY[*]}")"
python3 - "${STAGE_ROOT}" "${RUN_MODE}" "${ACTIVE_SEEDS}" <<'PY'
import csv,sys
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]; seeds=[int(x) for x in sys.argv[3].replace(',', ' ').split()]
arms=[
 ('S0_MATCHED','none','none'),
 ('KD_CT_TEXT_UNIFORM','ct_text','uniform'),
 ('KD_CT_TEXT_CONFIDENCE','ct_text','confidence'),
 ('KD_CT_TEXT_CNV_UNIFORM','ct_text_cnv','uniform'),
 ('KD_CT_TEXT_CNV_CONFIDENCE','ct_text_cnv','confidence'),
]
rows=[]
for arm,teacher,weight in arms:
 for seed in seeds:
  out=root/arm/mode/f'seed{seed}'
  rows.append({'arm':arm,'seed':seed,'teacher_type':teacher,'kd_weight_mode':weight,'run_mode':mode,'output_dir':str(out),'status':'complete' if (out/'run_complete.json').is_file() else 'MISSING'})
for name in ('binary_factorial_config_matrix.csv','binary_factorial_run_manifest.csv'):
 with (root/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
PY

bootstrap_iters="$([[ "${RUN_MODE}" == "smoke" ]] && echo 100 || echo 10000)"
analysis_args=(--root "${STAGE_ROOT}" --bootstrap-iters "${bootstrap_iters}" --expected-seeds "${ACTIVE_SEEDS}")
if [[ "${RUN_MODE}" == "smoke" ]]; then analysis_args+=(--smoke); fi
if [[ "${DRY_RUN}" == "0" ]]; then
    python3 experiments/analysis/analyze_binary_privileged_factorial.py "${analysis_args[@]}" || failures=$((failures + 1))
fi
echo "[DONE] factorial stage failures=${failures}"
exit "${failures}"
