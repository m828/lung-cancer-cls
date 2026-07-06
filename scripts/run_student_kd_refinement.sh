#!/usr/bin/env bash
# Local refinement of the best deployable CT+Text cached-logits KD student (outputs0535).
#
# This script is intentionally small: 4 candidate configs x selected seeds.
# It reuses 0534 cached teacher targets when present and writes new training
# outputs only under outputs0535_student_kd_refinement.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-}"
SOURCE_ROOT="${SOURCE_ROOT:-${PARENT_ROOT}/outputs0534_best_student_kd_search}"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0535_student_kd_refinement}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-full}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CONF_KD_WEIGHT_FLOOR="${CONF_KD_WEIGHT_FLOOR:-0.05}"
CANDIDATES="${CANDIDATES:-R1,R2,R3,R4}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

mkdir -p \
  "${OUT_ROOT}/cached_teacher_targets" \
  "${OUT_ROOT}/refined_candidates" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_student_kd_refinement.sh.snapshot"

PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"

resolve_results_root() {
    if [[ -n "${RESULTS_ROOT}" ]]; then
        return 0
    fi
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        if [[ -f "${cand}/${PREFERRED_REL}" || -d "${cand}/outputs0531_teacher_homogeneous_gene_test" ]]; then
            RESULTS_ROOT="${cand}"
            return 0
        fi
    done
    return 1
}

if ! resolve_results_root; then
    echo "[FATAL] Set RESULTS_ROOT manually." >&2
    exit 1
fi

resolve_ref_manifest() {
    local candidates=(
        "${RESULTS_ROOT}/${PREFERRED_REL}"
        "${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed42/split_manifest.csv"
    )
    local cand
    for cand in "${candidates[@]}"; do
        if [[ -f "${cand}" ]]; then
            REF_MANIFEST="${cand}"
            return 0
        fi
    done
    return 1
}

if ! resolve_ref_manifest; then
    echo "[FATAL] No reference manifest found." >&2
    exit 1
fi

infer_paths_py='
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
candidates = [
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_cnv_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0531_gene_privileged_ablation" / "ct_text_sc_densenet3d121_strict_bs4_seed42" / "metrics.json",
]
keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv", "gene_tsv"]
out = {k: "" for k in keys}
for p in candidates:
    if not p.is_file():
        continue
    try:
        cfg = json.loads(p.read_text(encoding="utf-8")).get("config", {})
    except Exception:
        continue
    for k in keys:
        if not out[k] and cfg.get(k):
            out[k] = str(cfg[k])
for k in keys:
    print(f"{k}\t{out[k]}")
'
declare -A INFERRED
while IFS=$'\t' read -r k v; do
    INFERRED[$k]="$v"
done < <(python3 -c "${infer_paths_py}" "${RESULTS_ROOT}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"
GENE_TSV="${GENE_TSV:-${INFERRED[gene_tsv]:-}}"

for required in DATA_ROOT METADATA_CSV CT_ROOT TEXT_FEATURE_TSV; do
    if [[ -z "${!required}" ]]; then
        echo "[FATAL] Missing ${required}. Set it explicitly." >&2
        exit 1
    fi
done

TEACHER_T0_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed"
TEACHER_T1_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed"
S0_BASE="${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed"

if [[ "${SMOKE}" == "1" ]]; then
    SEEDS=(42)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    SEEDS=(42 43)
else
    SEEDS=(42 43 44 45)
fi

COMMON_ARGS=(
    --class-mode binary --binary-task malignant_vs_normal
    --strict-no-leakage --disable-text-numeric-features
    --reference-manifest "${REF_MANIFEST}"
    --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}"
    --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}"
)

if [[ -n "${GENE_TSV}" ]]; then
    GENE_ARGS=(--gene-tsv "${GENE_TSV}")
else
    GENE_ARGS=()
fi

STUDENT_ARGS=(
    "${COMMON_ARGS[@]}"
    --ct-model densenet3d_121 --modalities ct,text
    --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256
    --dropout 0.3 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
    --selection-metric auroc --composite-selection-metric
)

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
        echo "[DRY_RUN] ${name} -> $*"
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

cache_path() {
    local teacher="$1"
    local seed="$2"
    local source="${SOURCE_ROOT}/cached_teacher_targets/${teacher}_seed${seed}.csv"
    local local_cache="${OUT_ROOT}/cached_teacher_targets/${teacher}_seed${seed}.csv"
    if [[ -f "${source}" ]]; then
        echo "${source}"
    else
        echo "${local_cache}"
    fi
}

cache_output_path() {
    local teacher="$1"
    local seed="$2"
    echo "${OUT_ROOT}/cached_teacher_targets/${teacher}_seed${seed}.csv"
}

teacher_dir_for() {
    local teacher="$1"
    local seed="$2"
    if [[ "${teacher}" == "T0" ]]; then
        echo "${TEACHER_T0_BASE}${seed}"
    else
        echo "${TEACHER_T1_BASE}${seed}"
    fi
}

ensure_cache() {
    local teacher="$1"
    local seed="$2"
    local existing
    existing="$(cache_path "${teacher}" "${seed}")"
    if [[ -f "${existing}" ]]; then
        echo "[CACHE] ${teacher} seed${seed}: ${existing}"
        return 0
    fi

    local teacher_dir
    teacher_dir="$(teacher_dir_for "${teacher}" "${seed}")"
    if [[ ! -f "${teacher_dir}/metrics.json" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] cache_${teacher}_seed${seed}: missing teacher ${teacher_dir}"
        return 1
    fi
    local -a extra
    if [[ "${teacher}" == "T1" ]]; then
        extra=("${GENE_ARGS[@]}")
    else
        extra=()
    fi
    run_cmd "cache_${teacher}_seed${seed}" "$(cache_output_path "${teacher}" "${seed}")" \
        python3 scripts/cache_teacher_soft_targets.py \
        --teacher-run-dir "${teacher_dir}" \
        --output-dir "${OUT_ROOT}/cached_teacher_targets" \
        --cache-name "${teacher}_seed${seed}" \
        --seed "${seed}" --batch-size "${CACHE_BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        "${COMMON_ARGS[@]}" "${extra[@]}"
}

s0_train_predictions() {
    local seed="$1"
    echo "${S0_BASE}${seed}/train_predictions.csv"
}

run_refined_candidate() {
    local candidate_id="$1"
    local kd_weighting="$2"
    local alpha="$3"
    local temp="$4"
    local lr="$5"
    local seed="$6"
    local t1_cache
    t1_cache="$(cache_path T1 "${seed}")"
    if [[ ! -f "${t1_cache}" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] ${candidate_id}_seed${seed}: missing T1 cache ${t1_cache}"
        return 0
    fi

    local candidate_label="${candidate_id}"
    if [[ -n "${RUN_SUFFIX}" ]]; then
        candidate_label="${candidate_id}_${RUN_SUFFIX}"
    fi
    local name="${candidate_label}_${kd_weighting}_a${alpha}_T${temp}_bs12_lr${lr}_composite_seed${seed}"
    local outdir="${OUT_ROOT}/refined_candidates/${name}"
    local kd_args=(--kd-weighting "${kd_weighting}" --kd-weight-max 1.0)

    if [[ "${kd_weighting}" == "confidence" ]]; then
        kd_args+=(--kd-weight-floor "${CONF_KD_WEIGHT_FLOOR}")
    else
        local t0_cache
        t0_cache="$(cache_path T0 "${seed}")"
        if [[ ! -f "${t0_cache}" && "${DRY_RUN}" != "1" ]]; then
            echo "[SKIP] ${name}: missing T0 cache ${t0_cache}"
            return 0
        fi
        kd_args+=(--kd-weight-floor 0.0 --reference-teacher-targets "${t0_cache}")
        local s0_pred
        s0_pred="$(s0_train_predictions "${seed}")"
        if [[ -f "${s0_pred}" || "${DRY_RUN}" == "1" ]]; then
            kd_args+=(--s0-predictions "${s0_pred}")
        else
            echo "[WARN] ${name}: missing S0 train predictions; advantage_learnable will fall back to T0 support"
        fi
    fi

    run_cmd "${name}" "${outdir}/metrics.json" \
        python3 scripts/train_student_kd_cached_logits.py \
        --output-dir "${outdir}" \
        --cached-teacher-targets "${t1_cache}" \
        --seed "${seed}" \
        --distillation-alpha "${alpha}" --distillation-temperature "${temp}" \
        --batch-size 12 --grad-accum-steps 1 --lr "${lr}" --weight-decay 1e-4 \
        --optimizer adamw --scheduler cosine --epochs "${EPOCHS}" --amp \
        --early-stopping-patience 10 --num-workers "${NUM_WORKERS}" \
        "${kd_args[@]}" \
        "${STUDENT_ARGS[@]}"
}

candidate_enabled() {
    local candidate_id="$1"
    local selected=",${CANDIDATES},"
    [[ "${selected}" == *",${candidate_id},"* || "${selected}" == *",all,"* ]]
}

stage_cache() {
    local seed
    for seed in "${SEEDS[@]}"; do
        ensure_cache T1 "${seed}" || true
        if candidate_enabled R4; then
            ensure_cache T0 "${seed}" || true
        fi
    done
}

stage_refine() {
    local seed
    for seed in "${SEEDS[@]}"; do
        ensure_cache T1 "${seed}" || { echo "[SKIP] refine seed${seed}: T1 cache unavailable"; continue; }
        if candidate_enabled R4; then
            ensure_cache T0 "${seed}" || true
        fi

        if candidate_enabled R1; then
            run_refined_candidate "R1" "confidence" "0.075" "6" "2e-4" "${seed}"
        fi
        if candidate_enabled R2; then
            run_refined_candidate "R2" "confidence" "0.05" "6" "2e-4" "${seed}"
        fi
        if candidate_enabled R3; then
            run_refined_candidate "R3" "confidence" "0.1" "8" "1e-4" "${seed}"
        fi
        if candidate_enabled R4; then
            run_refined_candidate "R4" "advantage_learnable" "0.075" "6" "2e-4" "${seed}"
        fi
    done
}

stage_analyze() {
    local name="analyze_student_kd_refinement"
    local logfile="${OUT_ROOT}/logs/${name}.log"
    local cmd=(
        python3 experiments/analysis/analyze_student_kd_refinement.py \
        --root "${OUT_ROOT}" \
        --baseline-root "${RESULTS_ROOT}/outputs0531_gene_privileged_ablation"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name} -> ${cmd[*]}"
        return 0
    fi
    mkdir -p "$(dirname "${logfile}")"
    echo "[RUN ] ${name}"
    echo "       out: ${OUT_ROOT}/best_refined_student_candidate.md"
    echo "       log: ${logfile}"
    set +e
    "${cmd[@]}" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        echo "[FAIL] ${name}: exit ${rc}; continuing"
    fi
    return 0
}

cat <<EOF
============================================================
Student KD local refinement
PROJECT_ROOT : ${PROJECT_ROOT}
RESULTS_ROOT : ${RESULTS_ROOT}
SOURCE_ROOT  : ${SOURCE_ROOT}
OUT_ROOT     : ${OUT_ROOT}
REF_MANIFEST : ${REF_MANIFEST}
STAGE        : ${STAGE}
RUN_MODE     : ${RUN_MODE}
DRY_RUN      : ${DRY_RUN}
EPOCHS       : ${EPOCHS}
SEEDS        : ${SEEDS[*]}
CANDIDATES   : ${CANDIDATES}
RUN_SUFFIX   : ${RUN_SUFFIX:-<none>}
CONFIGS      : R1 confidence a0.075 T6 lr2e-4; R2 confidence a0.05 T6 lr2e-4; R3 confidence a0.1 T8 lr1e-4; R4 advantage_learnable a0.075 T6 lr2e-4
============================================================
EOF

if [[ "${STAGE}" == "cache" || "${STAGE}" == "all" ]]; then stage_cache; fi
if [[ "${STAGE}" == "refine" || "${STAGE}" == "all" ]]; then stage_refine; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
