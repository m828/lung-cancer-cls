#!/usr/bin/env bash
# Teacher/student checkpoint-selection sensitivity for the internal triclass KD line.
# This wrapper writes only under outputs0541 by default.  DRY_RUN=1 prints commands.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0541_triclass_teacher_student_selection_4seed}"
SOURCE_TRI_ROOT="${SOURCE_TRI_ROOT:-${PARENT_ROOT}/outputs0536_triclass_extension}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-smoke}"
SMOKE="${SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-${BATCH_SIZE}}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
KD_ALPHA="${KD_ALPHA:-0.1}"
KD_TEMPERATURE="${KD_TEMPERATURE:-8}"
TRI_TEACHER_SELECTION_METRICS="${TRI_TEACHER_SELECTION_METRICS:-accuracy,balanced_accuracy,macro_f1,macro_auroc,malignant_recall,triclass_clinical_composite}"
TRI_SENSITIVITY_TEACHER_METRICS="${TRI_SENSITIVITY_TEACHER_METRICS:-accuracy,balanced_accuracy,macro_f1,triclass_clinical_composite}"
TRI_TEACHER_SELECTION_METRIC="${TRI_TEACHER_SELECTION_METRIC:-macro_f1}"
TRI_STUDENT_SELECTION_METRIC="${TRI_STUDENT_SELECTION_METRIC:-macro_f1}"
TEACHER_PRIMARY_SELECTION_METRIC="${TEACHER_PRIMARY_SELECTION_METRIC:-macro_f1}"
SEEDS_ENV="${SEEDS:-}"
RESULTS_ROOT_HINT="${RESULTS_ROOT:-}"
ALLOWED_TRI_METRICS=(accuracy balanced_accuracy auroc macro_auroc macro_f1 malignant_recall triclass_clinical_composite triclass_calibrated_composite)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            if [[ $# -lt 2 ]]; then echo "[ERROR] --root requires a path" >&2; exit 2; fi
            RESULTS_ROOT_HINT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$2")"
            shift 2
            ;;
        --out-root)
            if [[ $# -lt 2 ]]; then echo "[ERROR] --out-root requires a path" >&2; exit 2; fi
            OUT_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$2")"
            shift 2
            ;;
        --source-tri-root)
            if [[ $# -lt 2 ]]; then echo "[ERROR] --source-tri-root requires a path" >&2; exit 2; fi
            SOURCE_TRI_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$2")"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p \
  "${OUT_ROOT}/teacher_ct_cnv_text" \
  "${OUT_ROOT}/teacher_selected" \
  "${OUT_ROOT}/cached_teacher_targets" \
  "${OUT_ROOT}/seed42_teacher_selection_sensitivity" \
  "${OUT_ROOT}/student_4seed" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_triclass_teacher_student_selection_4seed.sh.snapshot"

split_csv() {
    local raw="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<<"${raw}"
}

validate_metric() {
    local metric="$1"
    local allowed
    for allowed in "${ALLOWED_TRI_METRICS[@]}"; do
        if [[ "${metric}" == "${allowed}" ]]; then
            return 0
        fi
    done
    echo "[ERROR] Unsupported triclass selection metric: ${metric}. Allowed: ${ALLOWED_TRI_METRICS[*]}" >&2
    return 2
}

split_csv "${TRI_TEACHER_SELECTION_METRICS}" TEACHER_SELECTION_METRICS_ARR
split_csv "${TRI_SENSITIVITY_TEACHER_METRICS}" SENSITIVITY_TEACHER_METRICS_ARR
validate_metric "${TRI_TEACHER_SELECTION_METRIC}"
validate_metric "${TRI_STUDENT_SELECTION_METRIC}"
validate_metric "${TEACHER_PRIMARY_SELECTION_METRIC}"
for metric in "${TEACHER_SELECTION_METRICS_ARR[@]}"; do validate_metric "${metric}"; done
for metric in "${SENSITIVITY_TEACHER_METRICS_ARR[@]}"; do validate_metric "${metric}"; done

if [[ -n "${SEEDS_ENV}" ]]; then
    IFS=',' read -r -a SEEDS <<<"${SEEDS_ENV}"
elif [[ "${SMOKE}" == "1" || "${RUN_MODE}" == "smoke" || "${RUN_MODE}" == "sensitivity" ]]; then
    SEEDS=(42)
else
    SEEDS=(42 43 44 45)
fi

ACTIVE_TEACHER_METRICS=()
if [[ "${RUN_MODE}" == "sensitivity" || "${SMOKE}" == "1" ]]; then
    ACTIVE_TEACHER_METRICS=("${SENSITIVITY_TEACHER_METRICS_ARR[@]}")
else
    ACTIVE_TEACHER_METRICS=("${TRI_TEACHER_SELECTION_METRIC}")
fi

resolve_json="${SOURCE_TRI_ROOT}/resolved_current_main_data_config.json"
if [[ ! -f "${resolve_json}" ]]; then
    resolve_json="${OUT_ROOT}/resolved_current_main_data_config.json"
    resolver=(python3 scripts/resolve_current_main_data_config.py --project-root "${PROJECT_ROOT}" --out "${resolve_json}")
    [[ -n "${RESULTS_ROOT_HINT}" ]] && resolver+=(--results-root "${RESULTS_ROOT_HINT}")
    "${resolver[@]}" >/dev/null
fi

declare -A RESOLVED
while IFS=$'\t' read -r key value; do
    RESOLVED[$key]="$value"
done < <(python3 - "${resolve_json}" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
keys = [
    "data_root", "metadata_csv", "ct_root", "gene_tsv", "text_feature_tsv",
    "text_source", "text_source_type", "text_health_csv", "text_disease_csv",
    "bert_model_path", "text_embedding_backend", "text_hash_dim",
    "text_batch_size", "text_max_length", "text_cache_tsv",
    "metadata_sample_id_col", "patient_id_col", "metadata_text_id_col",
    "text_record_id_col", "ct_path_col", "label_col", "split_col",
    "allowed_text_cols", "allowed_numeric_cols", "forbidden_feature_keywords",
    "gene_id_col", "gene_label_col", "strict_no_leakage",
    "disable_text_numeric_features",
]
for key in keys:
    value = cfg.get(key)
    if value is None:
        value = ""
    print(f"{key}\t{value}")
PY
)

DATA_ROOT="${RESOLVED[data_root]:-}"
METADATA_CSV="${RESOLVED[metadata_csv]:-}"
CT_ROOT="${RESOLVED[ct_root]:-}"
GENE_TSV="${RESOLVED[gene_tsv]:-}"
TEXT_FEATURE_TSV="${RESOLVED[text_feature_tsv]:-}"
TEXT_SOURCE="${RESOLVED[text_source]:-}"
TEXT_SOURCE_TYPE="${RESOLVED[text_source_type]:-unknown}"
TEXT_HEALTH_CSV="${RESOLVED[text_health_csv]:-}"
TEXT_DISEASE_CSV="${RESOLVED[text_disease_csv]:-}"
BERT_MODEL_PATH="${RESOLVED[bert_model_path]:-}"
TEXT_EMBEDDING_BACKEND="${RESOLVED[text_embedding_backend]:-bert}"
TEXT_HASH_DIM="${RESOLVED[text_hash_dim]:-128}"
TEXT_BATCH_SIZE="${RESOLVED[text_batch_size]:-8}"
TEXT_MAX_LENGTH="${RESOLVED[text_max_length]:-128}"
TEXT_CACHE_TSV="${RESOLVED[text_cache_tsv]:-}"
METADATA_SAMPLE_ID_COL="${RESOLVED[metadata_sample_id_col]:-SampleID}"
PATIENT_ID_COL="${RESOLVED[patient_id_col]:-}"
METADATA_TEXT_ID_COL="${RESOLVED[metadata_text_id_col]:-record_id}"
TEXT_RECORD_ID_COL="${RESOLVED[text_record_id_col]:-record_id}"
CT_PATH_COL="${RESOLVED[ct_path_col]:-CT_numpy_cloud路径}"
LABEL_COL="${RESOLVED[label_col]:-样本类型}"
SPLIT_COL="${RESOLVED[split_col]:-CT_train_val_split}"
ALLOWED_TEXT_COLS="${RESOLVED[allowed_text_cols]:-}"
ALLOWED_NUMERIC_COLS="${RESOLVED[allowed_numeric_cols]:-}"
FORBIDDEN_FEATURE_KEYWORDS="${RESOLVED[forbidden_feature_keywords]:-}"
GENE_ID_COL="${RESOLVED[gene_id_col]:-}"
GENE_LABEL_COL="${RESOLVED[gene_label_col]:-}"
STRICT_NO_LEAKAGE="${RESOLVED[strict_no_leakage]:-True}"
DISABLE_TEXT_NUMERIC_FEATURES="${RESOLVED[disable_text_numeric_features]:-True}"

if [[ -n "${TEXT_SOURCE}" && -n "${GENE_TSV}" && "${TEXT_SOURCE}" == "${GENE_TSV}" ]]; then
    echo "[ERROR] text feature source equals gene feature source. This is almost certainly wrong." >&2
    exit 2
fi

TEXT_ARGS=()
if [[ "${TEXT_SOURCE_TYPE}" == "text_feature_tsv" && -n "${TEXT_FEATURE_TSV}" ]]; then
    TEXT_ARGS=(--text-feature-tsv "${TEXT_FEATURE_TSV}")
elif [[ "${TEXT_SOURCE_TYPE}" == "raw_text_csv" ]]; then
    [[ -n "${TEXT_HEALTH_CSV}" ]] && TEXT_ARGS+=(--text-health-csv "${TEXT_HEALTH_CSV}")
    [[ -n "${TEXT_DISEASE_CSV}" ]] && TEXT_ARGS+=(--text-disease-csv "${TEXT_DISEASE_CSV}")
    [[ -n "${BERT_MODEL_PATH}" ]] && TEXT_ARGS+=(--bert-model-path "${BERT_MODEL_PATH}")
    TEXT_ARGS+=(--text-embedding-backend "${TEXT_EMBEDDING_BACKEND}")
    TEXT_ARGS+=(--text-hash-dim "${TEXT_HASH_DIM}" --text-batch-size "${TEXT_BATCH_SIZE}" --text-max-length "${TEXT_MAX_LENGTH}")
    [[ -n "${TEXT_CACHE_TSV}" ]] && TEXT_ARGS+=(--text-cache-tsv "${TEXT_CACHE_TSV}")
elif [[ "${TEXT_SOURCE_TYPE}" == "text_cache_tsv" && -n "${TEXT_CACHE_TSV}" ]]; then
    TEXT_ARGS=(--text-cache-tsv "${TEXT_CACHE_TSV}")
else
    echo "[ERROR] Could not resolve a usable text source. text_source_type=${TEXT_SOURCE_TYPE}" >&2
    exit 2
fi

COLUMN_ARGS=(
    --label-col "${LABEL_COL}"
    --metadata-sample-id-col "${METADATA_SAMPLE_ID_COL}"
    --metadata-text-id-col "${METADATA_TEXT_ID_COL}"
    --split-col "${SPLIT_COL}"
    --ct-path-col "${CT_PATH_COL}"
    --text-record-id-col "${TEXT_RECORD_ID_COL}"
)
[[ -n "${PATIENT_ID_COL}" ]] && COLUMN_ARGS+=(--patient-id-col "${PATIENT_ID_COL}")
[[ -n "${ALLOWED_TEXT_COLS}" ]] && COLUMN_ARGS+=(--allowed-text-cols "${ALLOWED_TEXT_COLS}")
[[ -n "${ALLOWED_NUMERIC_COLS}" ]] && COLUMN_ARGS+=(--allowed-numeric-cols "${ALLOWED_NUMERIC_COLS}")
[[ -n "${FORBIDDEN_FEATURE_KEYWORDS}" ]] && COLUMN_ARGS+=(--forbidden-feature-keywords "${FORBIDDEN_FEATURE_KEYWORDS}")

GENE_ARGS=(--gene-tsv "${GENE_TSV}")
[[ -n "${GENE_ID_COL}" ]] && GENE_ARGS+=(--gene-id-col "${GENE_ID_COL}")
[[ -n "${GENE_LABEL_COL}" ]] && GENE_ARGS+=(--gene-label-col "${GENE_LABEL_COL}")

STRICT_ARGS=()
[[ "${STRICT_NO_LEAKAGE}" == "True" || "${STRICT_NO_LEAKAGE}" == "true" || "${STRICT_NO_LEAKAGE}" == "1" ]] && STRICT_ARGS+=(--strict-no-leakage)
[[ "${DISABLE_TEXT_NUMERIC_FEATURES}" == "True" || "${DISABLE_TEXT_NUMERIC_FEATURES}" == "true" || "${DISABLE_TEXT_NUMERIC_FEATURES}" == "1" ]] && STRICT_ARGS+=(--disable-text-numeric-features)

DATA_ARGS=(
    --class-mode multiclass
    "${STRICT_ARGS[@]}"
    --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}"
    --ct-root "${CT_ROOT}"
    "${TEXT_ARGS[@]}"
    "${COLUMN_ARGS[@]}"
)

COMMON_MODEL_ARGS=(
    --ct-model densenet3d_121 --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256 --gene-hidden-dim 256 --fusion-hidden-dim 256
    --dropout 0.3 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
    --optimizer adamw --scheduler cosine
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
}

teacher_dir() {
    echo "${OUT_ROOT}/teacher_ct_cnv_text/TRI-T_multi_select_seed${1}"
}

teacher_manifest() {
    echo "$(teacher_dir "$1")/split_manifest.csv"
}

teacher_checkpoint() {
    local metric="$1"
    local seed="$2"
    echo "$(teacher_dir "${seed}")/best_${metric}.pt"
}

selected_teacher_dir() {
    local metric="$1"
    local seed="$2"
    echo "${OUT_ROOT}/teacher_selected/teacher_select_${metric}_seed${seed}"
}

cache_csv() {
    local metric="$1"
    local seed="$2"
    echo "${OUT_ROOT}/cached_teacher_targets/teacher_select_${metric}_seed${seed}.csv"
}

student_dir() {
    local metric="$1"
    local seed="$2"
    if [[ "${RUN_MODE}" == "sensitivity" || "${SMOKE}" == "1" ]]; then
        echo "${OUT_ROOT}/seed42_teacher_selection_sensitivity/teacher_select_${metric}/TRI-SKD_teacher_${metric}_student_${TRI_STUDENT_SELECTION_METRIC}_seed${seed}"
    else
        echo "${OUT_ROOT}/student_4seed/teacher_select_${metric}/TRI-SKD_teacher_${metric}_student_${TRI_STUDENT_SELECTION_METRIC}_seed${seed}"
    fi
}

prepare_selected_teacher_dir() {
    local metric="$1"
    local seed="$2"
    local source_dir selected_dir ckpt
    source_dir="$(teacher_dir "${seed}")"
    selected_dir="$(selected_teacher_dir "${metric}" "${seed}")"
    ckpt="$(teacher_checkpoint "${metric}" "${seed}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "${selected_dir}"
        return 0
    fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "[SKIP] teacher_select_${metric}_seed${seed}: missing checkpoint ${ckpt}" >&2
        return 1
    fi
    mkdir -p "${selected_dir}"
    cp -f "${source_dir}/metrics.json" "${selected_dir}/metrics.json"
    cp -f "${source_dir}/split_manifest.csv" "${selected_dir}/split_manifest.csv"
    cp -f "${ckpt}" "${selected_dir}/best_model.pt"
    python3 - "${selected_dir}" "${source_dir}" "${ckpt}" "${metric}" "${seed}" <<'PY'
import json, sys
from pathlib import Path
selected_dir = Path(sys.argv[1])
source_dir = Path(sys.argv[2])
ckpt = Path(sys.argv[3])
metric = sys.argv[4]
seed = sys.argv[5]
payload = {
    "seed": seed,
    "teacher_selection_metric": metric,
    "source_teacher_dir": str(source_dir),
    "source_checkpoint": str(ckpt),
    "selected_teacher_dir": str(selected_dir),
    "best_model_semantics": f"best_{metric}.pt copied to best_model.pt for cache_teacher_soft_targets.py",
}
(selected_dir / "selected_teacher_checkpoint.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
    echo "${selected_dir}"
}

preflight_common() {
    local missing=0
    for item in \
        "DATA_ROOT:${DATA_ROOT}" \
        "METADATA_CSV:${METADATA_CSV}" \
        "CT_ROOT:${CT_ROOT}" \
        "GENE_TSV:${GENE_TSV}"; do
        local label="${item%%:*}"
        local path="${item#*:}"
        if [[ -z "${path}" ]]; then
            echo "[MISSING] ${label}: <empty>"
            missing=1
        elif [[ ! -e "${path}" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] ${label}: ${path}"
            missing=1
        fi
    done
    if [[ "${TEXT_SOURCE_TYPE}" == "text_feature_tsv" && -n "${TEXT_FEATURE_TSV}" && ! -e "${TEXT_FEATURE_TSV}" && "${DRY_RUN}" != "1" ]]; then
        echo "[MISSING] TEXT_FEATURE_TSV: ${TEXT_FEATURE_TSV}"
        missing=1
    fi
    if (( missing != 0 && DRY_RUN != 1 )); then
        return 1
    fi
    return 0
}

write_runbook() {
    cat >"${OUT_ROOT}/triclass_teacher_student_selection_runbook.md" <<EOF
# Triclass Teacher/Student Selection Runbook

## Purpose

This suite separates two checkpoint decisions:

- Teacher checkpoint selection: selects the CT+CNV+Text teacher epoch used to cache logits / soft labels.
- Student checkpoint selection: selects the CT+Text KD student epoch after teacher soft labels are fixed.

The default student selection is \`${TRI_STUDENT_SELECTION_METRIC}\`.  The default teacher selection for 4-seed runs is \`${TRI_TEACHER_SELECTION_METRIC}\`; \`accuracy\` is supported as sensitivity only because it can hide malignant-class collapse under class imbalance.

## Teacher Metrics

Multi-metric teacher checkpoints are saved in one teacher training run:

- \`best_accuracy.pt\`
- \`best_balanced_accuracy.pt\`
- \`best_macro_f1.pt\`
- \`best_macro_auroc.pt\`
- \`best_malignant_recall.pt\`
- \`best_triclass_clinical_composite.pt\`

\`triclass_clinical_composite = 0.25 * balanced_accuracy + 0.25 * macro_f1 + 0.20 * malignant_recall + 0.20 * macro_auroc + 0.10 * benign_recall\`.

Metric resolution is strict.  Missing metrics raise an error; no AUROC fallback is allowed.

## Recommended Order

1. Dry-run: \`DRY_RUN=1 SMOKE=1 STAGE=all bash scripts/run_triclass_teacher_student_selection_4seed.sh --root ./\`
2. Teacher smoke: \`SMOKE=1 STAGE=teacher bash scripts/run_triclass_teacher_student_selection_4seed.sh --root ./\`
3. Seed42 sensitivity: \`SMOKE=1 STAGE=all RUN_MODE=sensitivity bash scripts/run_triclass_teacher_student_selection_4seed.sh --root ./\`
4. 4-seed macro-F1 main: \`SMOKE=0 RUN_MODE=full TRI_TEACHER_SELECTION_METRIC=macro_f1 TRI_STUDENT_SELECTION_METRIC=macro_f1 bash scripts/run_triclass_teacher_student_selection_4seed.sh --root ./\`
5. 4-seed clinical-composite sensitivity: \`SMOKE=0 RUN_MODE=full TRI_TEACHER_SELECTION_METRIC=triclass_clinical_composite TRI_STUDENT_SELECTION_METRIC=macro_f1 bash scripts/run_triclass_teacher_student_selection_4seed.sh --root ./\`

## Reporting Boundary

Seed42 teacher-selection sensitivity is preliminary.  Only fixed-split 4-seed repeated runs should be treated as paper-facing experiment evidence.  Accuracy-selected teacher results should be reported as sensitivity unless they also preserve malignant recall and macro-F1.
EOF
}

stage_teacher() {
    preflight_common || return 0
    local seed
    for seed in "${SEEDS[@]}"; do
        run_cmd "TRI-T_multi_select_seed${seed}" "$(teacher_dir "${seed}")/metrics.json" \
            python3 train_multimodal.py \
            --output-dir "$(teacher_dir "${seed}")" \
            --modalities ct,cnv,text "${GENE_ARGS[@]}" \
            --seed "${seed}" --batch-size "${BATCH_SIZE}" --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
            --selection-metric "${TEACHER_PRIMARY_SELECTION_METRIC}" \
            --extra-checkpoint-metrics "${TRI_TEACHER_SELECTION_METRICS}" \
            --split-mode train_val_test --use-predefined-split \
            --epochs "${EPOCHS}" --num-workers "${NUM_WORKERS}" \
            "${DATA_ARGS[@]}" "${COMMON_MODEL_ARGS[@]}"
    done
}

ensure_cache() {
    local metric="$1"
    local seed="$2"
    if [[ -f "$(cache_csv "${metric}" "${seed}")" ]]; then
        echo "[CACHE] teacher_select_${metric} seed${seed}: $(cache_csv "${metric}" "${seed}")"
        return 0
    fi
    local selected_dir
    selected_dir="$(prepare_selected_teacher_dir "${metric}" "${seed}")" || return 1
    run_cmd "cache_teacher_select_${metric}_seed${seed}" "$(cache_csv "${metric}" "${seed}")" \
        python3 scripts/cache_teacher_soft_targets.py \
        --teacher-run-dir "${selected_dir}" \
        --output-dir "${OUT_ROOT}/cached_teacher_targets" \
        --cache-name "teacher_select_${metric}_seed${seed}" \
        --reference-manifest "$(teacher_manifest "${seed}")" \
        "${GENE_ARGS[@]}" \
        --seed "${seed}" --batch-size "${CACHE_BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        --split-mode train_val_test \
        "${DATA_ARGS[@]}"
}

stage_cache() {
    preflight_common || return 0
    local seed metric
    for seed in "${SEEDS[@]}"; do
        for metric in "${ACTIVE_TEACHER_METRICS[@]}"; do
            ensure_cache "${metric}" "${seed}" || true
        done
    done
}

stage_student() {
    preflight_common || return 0
    local seed metric
    for seed in "${SEEDS[@]}"; do
        for metric in "${ACTIVE_TEACHER_METRICS[@]}"; do
            ensure_cache "${metric}" "${seed}" || { echo "[SKIP] student teacher_select_${metric} seed${seed}: cache unavailable"; continue; }
            if [[ ! -f "$(cache_csv "${metric}" "${seed}")" && "${DRY_RUN}" != "1" ]]; then
                echo "[SKIP] student teacher_select_${metric} seed${seed}: missing cache $(cache_csv "${metric}" "${seed}")"
                continue
            fi
            run_cmd "student_teacher_select_${metric}_seed${seed}" "$(student_dir "${metric}" "${seed}")/metrics.json" \
                python3 scripts/train_student_kd_cached_logits.py \
                --output-dir "$(student_dir "${metric}" "${seed}")" \
                --cached-teacher-targets "$(cache_csv "${metric}" "${seed}")" \
                --reference-manifest "$(teacher_manifest "${seed}")" \
                --modalities ct,text \
                --seed "${seed}" \
                --distillation-alpha "${KD_ALPHA}" --distillation-temperature "${KD_TEMPERATURE}" \
                --kd-weighting confidence --kd-weight-floor 0.05 --kd-weight-max 1.0 \
                --batch-size "${STUDENT_BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
                --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
                --optimizer adamw --scheduler cosine --epochs "${EPOCHS}" --amp \
                --early-stopping-patience 10 --num-workers "${NUM_WORKERS}" \
                --ct-model densenet3d_121 --depth-size 128 --volume-hw 256 \
                --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256 \
                --dropout 0.3 --loss ce --label-smoothing 0.05 \
                --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999 \
                --selection-metric "${TRI_STUDENT_SELECTION_METRIC}" \
                "${DATA_ARGS[@]}"
        done
    done
}

stage_analyze() {
    run_cmd "analyze_triclass_teacher_student_selection_4seed" "${OUT_ROOT}/triclass_teacher_student_comparison.md" \
        python3 experiments/analysis/analyze_triclass_teacher_student_selection_4seed.py \
        --root "${OUT_ROOT}" \
        --source-root "${SOURCE_TRI_ROOT}"
}

write_runbook

cat <<EOF
============================================================
Triclass teacher/student checkpoint selection suite
PROJECT_ROOT          : ${PROJECT_ROOT}
SOURCE_TRI_ROOT       : ${SOURCE_TRI_ROOT}
OUT_ROOT              : ${OUT_ROOT}
RESOLVED_CONFIG       : ${resolve_json}
STAGE                 : ${STAGE}
RUN_MODE              : ${RUN_MODE}
SMOKE                 : ${SMOKE}
DRY_RUN               : ${DRY_RUN}
SEEDS                 : ${SEEDS[*]}
teacher selection metrics saved : ${TEACHER_SELECTION_METRICS_ARR[*]}
teacher metrics for student/cache: ${ACTIVE_TEACHER_METRICS[*]}
student selection metric         : ${TRI_STUDENT_SELECTION_METRIC}
metadata_csv          : ${METADATA_CSV}
ct_root               : ${CT_ROOT}
gene_tsv              : ${GENE_TSV}
text source           : ${TEXT_SOURCE_TYPE} ${TEXT_SOURCE}
class_mode            : multiclass
no hint               : yes
no feature alignment  : yes
fallback behavior     : strict metric resolution; no AUROC fallback
============================================================
EOF

for seed in "${SEEDS[@]}"; do
    echo "[PLAN] seed${seed} teacher_dir=$(teacher_dir "${seed}")"
    for metric in "${ACTIVE_TEACHER_METRICS[@]}"; do
        echo "[PLAN] seed${seed} teacher_select_${metric}_checkpoint=$(teacher_checkpoint "${metric}" "${seed}")"
        echo "[PLAN] seed${seed} teacher_select_${metric}_cache=$(cache_csv "${metric}" "${seed}")"
        echo "[PLAN] seed${seed} teacher_select_${metric}_student_dir=$(student_dir "${metric}" "${seed}")"
    done
done

if [[ "${STAGE}" == "teacher" || "${STAGE}" == "all" ]]; then stage_teacher; fi
if [[ "${STAGE}" == "cache" || "${STAGE}" == "all" ]]; then stage_cache; fi
if [[ "${STAGE}" == "student" || "${STAGE}" == "all" ]]; then stage_student; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
