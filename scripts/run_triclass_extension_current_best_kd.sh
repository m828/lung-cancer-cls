#!/usr/bin/env bash
# Internal three-class extension for current best gene-enhanced KD idea.
# No training is launched when DRY_RUN=1. Outputs are isolated under outputs0536.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PARENT_ROOT}}"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0536_triclass_extension}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-mini}"
SMOKE="${SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-12}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
KD_ALPHA="${KD_ALPHA:-0.1}"
KD_TEMPERATURE="${KD_TEMPERATURE:-8}"

mkdir -p \
  "${OUT_ROOT}/teacher_ct_cnv_text" \
  "${OUT_ROOT}/supervised_ct_text" \
  "${OUT_ROOT}/student_kd_ct_text_from_gene_teacher" \
  "${OUT_ROOT}/cached_teacher_targets" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_triclass_extension_current_best_kd.sh.snapshot"

infer_paths_py='
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
candidates = [
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_cnv_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0417" / "ct_cnv_text_teacher_mc_tvt" / "metrics.json",
    root / "outputs0417" / "ct_text_mc_tvt" / "metrics.json",
]
keys = ["data_root", "metadata_csv", "ct_root", "gene_tsv", "text_feature_tsv"]
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
GENE_TSV="${GENE_TSV:-${INFERRED[gene_tsv]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"

if [[ "${SMOKE}" == "1" ]]; then
    SEEDS=(42)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    SEEDS=(42 43)
else
    SEEDS=(42 43 44 45)
fi

COMMON_ARGS=(
    --class-mode multiclass
    --strict-no-leakage --disable-text-numeric-features
    --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}"
    --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}"
    --ct-model densenet3d_121 --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256 --gene-hidden-dim 256 --fusion-hidden-dim 256
    --dropout 0.3 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
    --optimizer adamw --scheduler cosine --selection-metric balanced_accuracy
    --split-mode train_val_test --use-predefined-split
    --epochs "${EPOCHS}" --num-workers "${NUM_WORKERS}"
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

preflight_common() {
    local missing=0
    for var in DATA_ROOT METADATA_CSV CT_ROOT TEXT_FEATURE_TSV GENE_TSV; do
        if [[ -z "${!var}" ]]; then
            echo "[MISSING] ${var}: set it explicitly before non-dry run"
            missing=1
        elif [[ ! -e "${!var}" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] ${var}: ${!var}"
            missing=1
        fi
    done
    if (( missing != 0 && DRY_RUN != 1 )); then
        return 1
    fi
    return 0
}

teacher_dir() {
    echo "${OUT_ROOT}/teacher_ct_cnv_text/TRI-T_ct_cnv_text_densenet3d121_seed${1}"
}

supervised_dir() {
    echo "${OUT_ROOT}/supervised_ct_text/TRI-S0_ct_text_densenet3d121_seed${1}"
}

teacher_manifest() {
    echo "$(teacher_dir "$1")/split_manifest.csv"
}

cache_csv() {
    echo "${OUT_ROOT}/cached_teacher_targets/TRI-T_seed${1}.csv"
}

student_dir() {
    echo "${OUT_ROOT}/student_kd_ct_text_from_gene_teacher/TRI-SKD_confidence_a0.1_T8_seed${1}"
}

stage_teacher() {
    preflight_common || return 0
    local seed
    for seed in "${SEEDS[@]}"; do
        run_cmd "TRI-T_seed${seed}" "$(teacher_dir "${seed}")/metrics.json" \
            python3 train_multimodal.py \
            --output-dir "$(teacher_dir "${seed}")" \
            --modalities ct,cnv,text --gene-tsv "${GENE_TSV}" \
            --seed "${seed}" --batch-size 4 --lr 1e-4 --weight-decay 1e-4 \
            "${COMMON_ARGS[@]}"
    done
}

stage_supervised() {
    preflight_common || return 0
    local seed manifest_arg=()
    for seed in "${SEEDS[@]}"; do
        manifest_arg=()
        if [[ -f "$(teacher_manifest "${seed}")" || "${DRY_RUN}" == "1" ]]; then
            manifest_arg=(--reference-manifest "$(teacher_manifest "${seed}")")
        else
            echo "[WARN] TRI-S0_seed${seed}: teacher split manifest missing; falling back to metadata predefined split"
        fi
        run_cmd "TRI-S0_seed${seed}" "$(supervised_dir "${seed}")/metrics.json" \
            python3 train_multimodal.py \
            --output-dir "$(supervised_dir "${seed}")" \
            --modalities ct,text \
            --seed "${seed}" --batch-size 4 --lr 1e-4 --weight-decay 1e-4 \
            "${COMMON_ARGS[@]}" "${manifest_arg[@]}"
    done
}

ensure_cache() {
    local seed="$1"
    if [[ -f "$(cache_csv "${seed}")" ]]; then
        echo "[CACHE] TRI-T seed${seed}: $(cache_csv "${seed}")"
        return 0
    fi
    if [[ ! -f "$(teacher_dir "${seed}")/metrics.json" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] cache_TRI-T_seed${seed}: missing teacher $(teacher_dir "${seed}")"
        return 1
    fi
    run_cmd "cache_TRI-T_seed${seed}" "$(cache_csv "${seed}")" \
        python3 scripts/cache_teacher_soft_targets.py \
        --teacher-run-dir "$(teacher_dir "${seed}")" \
        --output-dir "${OUT_ROOT}/cached_teacher_targets" \
        --cache-name "TRI-T_seed${seed}" \
        --reference-manifest "$(teacher_manifest "${seed}")" \
        --gene-tsv "${GENE_TSV}" \
        --seed "${seed}" --batch-size "${CACHE_BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        --class-mode multiclass \
        --strict-no-leakage --disable-text-numeric-features \
        --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" \
        --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}"
}

stage_student() {
    preflight_common || return 0
    local seed
    for seed in "${SEEDS[@]}"; do
        ensure_cache "${seed}" || { echo "[SKIP] TRI-SKD_seed${seed}: teacher cache unavailable"; continue; }
        if [[ ! -f "$(cache_csv "${seed}")" && "${DRY_RUN}" != "1" ]]; then
            echo "[SKIP] TRI-SKD_seed${seed}: missing cache $(cache_csv "${seed}")"
            continue
        fi
        run_cmd "TRI-SKD_seed${seed}" "$(student_dir "${seed}")/metrics.json" \
            python3 scripts/train_student_kd_cached_logits.py \
            --output-dir "$(student_dir "${seed}")" \
            --cached-teacher-targets "$(cache_csv "${seed}")" \
            --reference-manifest "$(teacher_manifest "${seed}")" \
            --modalities ct,text \
            --seed "${seed}" \
            --class-mode multiclass \
            --distillation-alpha "${KD_ALPHA}" --distillation-temperature "${KD_TEMPERATURE}" \
            --kd-weighting confidence --kd-weight-floor 0.05 --kd-weight-max 1.0 \
            --batch-size "${STUDENT_BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
            --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
            --optimizer adamw --scheduler cosine --epochs "${EPOCHS}" --amp \
            --early-stopping-patience 10 --num-workers "${NUM_WORKERS}" \
            --composite-selection-metric \
            --strict-no-leakage --disable-text-numeric-features \
            --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model densenet3d_121 --depth-size 128 --volume-hw 256 \
            --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256 \
            --dropout 0.3 --loss ce --label-smoothing 0.05 \
            --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999 \
            --selection-metric balanced_accuracy
    done
}

stage_analyze() {
    run_cmd "analyze_triclass_extension" "${OUT_ROOT}/triclass_summary.md" \
        python3 experiments/analysis/analyze_triclass_extension.py --root "${OUT_ROOT}"
}

cat <<EOF
============================================================
Internal triclass extension: current best KD idea
PROJECT_ROOT : ${PROJECT_ROOT}
RESULTS_ROOT : ${RESULTS_ROOT}
OUT_ROOT     : ${OUT_ROOT}
STAGE        : ${STAGE}
RUN_MODE     : ${RUN_MODE}
SMOKE        : ${SMOKE}
DRY_RUN      : ${DRY_RUN}
SEEDS        : ${SEEDS[*]}
DATA_ROOT    : ${DATA_ROOT:-<missing>}
METADATA_CSV : ${METADATA_CSV:-<missing>}
CT_ROOT      : ${CT_ROOT:-<missing>}
GENE_TSV     : ${GENE_TSV:-<missing>}
TEXT_TSV     : ${TEXT_FEATURE_TSV:-<missing>}
GROUPS       : TRI-T ct,cnv,text; TRI-S0 ct,text; TRI-SKD ct,text from TRI-T cached logits confidence KD
============================================================
EOF

if [[ "${STAGE}" == "teacher" || "${STAGE}" == "all" ]]; then stage_teacher; fi
if [[ "${STAGE}" == "supervised" || "${STAGE}" == "all" ]]; then stage_supervised; fi
if [[ "${STAGE}" == "student" || "${STAGE}" == "all" ]]; then stage_student; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
