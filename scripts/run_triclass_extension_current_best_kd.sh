#!/usr/bin/env bash
# Internal three-class extension for current best gene-enhanced KD idea.
# DRY_RUN=1 prints concrete commands only.  Data paths are resolved from the
# current successful malignant-vs-normal main line, not from user-exported TSVs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0536_triclass_extension}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-mini}"
SMOKE="${SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
SOURCE_FROM_CURRENT_MAIN="${SOURCE_FROM_CURRENT_MAIN:-1}"
SEEDS_ENV="${SEEDS:-}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_CLASSES="${NUM_CLASSES:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
KD_ALPHA="${KD_ALPHA:-0.1}"
KD_TEMPERATURE="${KD_TEMPERATURE:-8}"
RESULTS_ROOT_HINT="${RESULTS_ROOT:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --root requires a path" >&2
                exit 2
            fi
            RESULTS_ROOT_HINT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$2")"
            shift 2
            ;;
        --out-root)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --out-root requires a path" >&2
                exit 2
            fi
            OUT_ROOT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$2")"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "${NUM_CLASSES}" != "3" ]]; then
    echo "[ERROR] Current internal extension supports NUM_CLASSES=3 only via class_mode=multiclass." >&2
    exit 2
fi
if [[ "${SOURCE_FROM_CURRENT_MAIN}" != "1" ]]; then
    echo "[ERROR] SOURCE_FROM_CURRENT_MAIN=0 is disabled for this wrapper; use the current main data config resolver." >&2
    exit 2
fi

mkdir -p \
  "${OUT_ROOT}/teacher_ct_cnv_text" \
  "${OUT_ROOT}/supervised_ct_text" \
  "${OUT_ROOT}/student_kd_ct_text_from_gene_teacher" \
  "${OUT_ROOT}/cached_teacher_targets" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_triclass_extension_current_best_kd.sh.snapshot"

RESOLVED_CONFIG_JSON="${OUT_ROOT}/resolved_current_main_data_config.json"
resolver_cmd=(
    python3 scripts/resolve_current_main_data_config.py
    --project-root "${PROJECT_ROOT}"
    --out "${RESOLVED_CONFIG_JSON}"
)
if [[ -n "${RESULTS_ROOT_HINT}" ]]; then
    resolver_cmd+=(--results-root "${RESULTS_ROOT_HINT}")
fi
"${resolver_cmd[@]}" >/dev/null

declare -A RESOLVED
while IFS=$'\t' read -r key value; do
    RESOLVED[$key]="$value"
done < <(python3 - "${RESOLVED_CONFIG_JSON}" <<'PY'
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
    "disable_text_numeric_features", "main_binary_task", "source_metrics",
]
for key in keys:
    value = cfg.get(key)
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
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
if [[ -n "${TEXT_FEATURE_TSV}" && -n "${GENE_TSV}" && "${TEXT_FEATURE_TSV}" == "${GENE_TSV}" ]]; then
    echo "[ERROR] text feature source equals gene feature source. This is almost certainly wrong." >&2
    exit 2
fi

if [[ -n "${SEEDS_ENV}" ]]; then
    IFS=',' read -r -a SEEDS <<<"${SEEDS_ENV}"
elif [[ "${SMOKE}" == "1" ]]; then
    SEEDS=(42)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    SEEDS=(42 43)
else
    SEEDS=(42 43 44 45)
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
    echo "[ERROR] Could not resolve a usable text source from current main config." >&2
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

COMMON_TRAIN_ARGS=(
    "${DATA_ARGS[@]}"
    --ct-model densenet3d_121 --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256 --gene-hidden-dim 256 --fusion-hidden-dim 256
    --dropout 0.3 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
    --optimizer adamw --scheduler cosine --selection-metric balanced_accuracy
    --split-mode train_val_test --use-predefined-split
    --epochs "${EPOCHS}" --num-workers "${NUM_WORKERS}"
)

# train_student_kd_cached_logits.py does not expose --use-predefined-split.
# TRI-SKD keeps the same split contract through --reference-manifest from TRI-T
# plus the resolved metadata split column in DATA_ARGS.
STUDENT_TRAIN_ARGS=(
    "${DATA_ARGS[@]}"
    --ct-model densenet3d_121 --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256
    --dropout 0.3 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
    --selection-metric balanced_accuracy
    --split-mode train_val_test
)

write_audit() {
    python3 - "${OUT_ROOT}" "${RESOLVED_CONFIG_JSON}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
cfg = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
same = cfg.get("same_text_and_gene_source")
text_source = cfg.get("text_source") or "<unresolved>"
gene_source = cfg.get("gene_source") or cfg.get("gene_tsv") or "<unresolved>"
metadata = cfg.get("metadata_csv") or "<unresolved>"
ct_root = cfg.get("ct_root") or "<unresolved>"
source_metrics = cfg.get("source_metrics") or []
lines = [
    "# Triclass Code Audit",
    "",
    "## Current Support",
    "- Current training entry: `train_multimodal.py` delegates to `lung_cancer_cls.multimodal_teacher_student.main_train`.",
    "- Existing CLI supports `--class-mode multiclass`, which maps internal labels to normal / benign / malignant via the existing label mapper.",
    "- Existing model creation uses `num_classes=len(class_names)`, so multiclass uses three output logits without modifying model core code.",
    "- Existing metrics include multiclass-compatible accuracy, macro AUROC when computable, macro-F1, balanced accuracy, per-class outputs in prediction CSVs, and confusion matrix data from the shared evaluator.",
    "",
    "## Resolved Data Contract",
    "- Reuses current malignant-vs-normal main experiment data config: yes.",
    f"- Metadata CSV: `{metadata}`",
    f"- CT root: `{ct_root}`",
    f"- Text source type: `{cfg.get('text_source_type')}`",
    f"- Text source: `{text_source}`",
    f"- CNV/Gene source: `{gene_source}`",
    f"- Text source equals gene source: `{same}`",
    "- Safety check: wrapper exits with `[ERROR] text feature source equals gene feature source. This is almost certainly wrong.` if the two sources are identical.",
    "",
    "## Column Contract",
    f"- sample id column: `{cfg.get('metadata_sample_id_col')}`",
    f"- record id column in metadata: `{cfg.get('metadata_text_id_col')}`",
    f"- record id column in text features: `{cfg.get('text_record_id_col')}`",
    f"- CT path column: `{cfg.get('ct_path_col')}`",
    f"- label column: `{cfg.get('label_col')}`",
    f"- split column: `{cfg.get('split_col')}`",
    "",
    "## Label Mapping",
    "- Binary main line uses `malignant_vs_normal` and drops benign samples.",
    "- Triclass extension uses `--class-mode multiclass` on the same metadata and keeps all mapped classes.",
    "- Mapping: `健康对照 -> normal`, `良性结节 -> benign`, `肺癌 -> malignant`.",
    "",
    "## Source Metrics Used For Resolution",
]
lines.extend(f"- `{p}`" for p in source_metrics)
lines.extend([
    "",
    "## Risk Points",
    "- Non-dry-run still requires the resolved server paths to exist on the execution host.",
    "- This script intentionally does not use old 1120 triclass outputs as data sources.",
    "- Teacher / supervised / student runs write only under `outputs0536_triclass_extension/`.",
    "",
    "## Recommended Smoke Command",
    "`DRY_RUN=1 SMOKE=1 STAGE=teacher bash scripts/run_triclass_extension_current_best_kd.sh --root ./`",
])
out.mkdir(parents=True, exist_ok=True)
(out / "triclass_code_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}
write_audit

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
    if [[ "${TEXT_SOURCE_TYPE}" == "text_feature_tsv" ]]; then
        if [[ -z "${TEXT_FEATURE_TSV}" ]]; then
            echo "[MISSING] TEXT_FEATURE_TSV: <empty>"
            missing=1
        elif [[ ! -e "${TEXT_FEATURE_TSV}" && "${DRY_RUN}" != "1" ]]; then
            echo "[MISSING] TEXT_FEATURE_TSV: ${TEXT_FEATURE_TSV}"
            missing=1
        fi
    fi
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
            --modalities ct,cnv,text "${GENE_ARGS[@]}" \
            --seed "${seed}" --batch-size "${BATCH_SIZE}" --lr 1e-4 --weight-decay 1e-4 \
            "${COMMON_TRAIN_ARGS[@]}"
    done
}

stage_supervised() {
    preflight_common || return 0
    local seed
    local manifest_arg=()
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
            --seed "${seed}" --batch-size "${BATCH_SIZE}" --lr 1e-4 --weight-decay 1e-4 \
            "${COMMON_TRAIN_ARGS[@]}" "${manifest_arg[@]}"
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
        "${GENE_ARGS[@]}" \
        --seed "${seed}" --batch-size "${CACHE_BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
        --split-mode train_val_test \
        "${DATA_ARGS[@]}"
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
            --distillation-alpha "${KD_ALPHA}" --distillation-temperature "${KD_TEMPERATURE}" \
            --kd-weighting confidence --kd-weight-floor 0.05 --kd-weight-max 1.0 \
            --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
            --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
            --optimizer adamw --scheduler cosine --epochs "${EPOCHS}" --amp \
            --early-stopping-patience 10 --num-workers "${NUM_WORKERS}" \
            --composite-selection-metric \
            "${STUDENT_TRAIN_ARGS[@]}"
    done
}

stage_analyze() {
    run_cmd "analyze_triclass_extension" "${OUT_ROOT}/triclass_summary.md" \
        python3 experiments/analysis/analyze_triclass_extension.py --root "${OUT_ROOT}"
}

cat <<EOF
============================================================
Internal triclass extension: current best KD idea
PROJECT_ROOT       : ${PROJECT_ROOT}
RESULTS_ROOT_HINT  : ${RESULTS_ROOT_HINT:-<auto>}
OUT_ROOT           : ${OUT_ROOT}
RESOLVED_CONFIG    : ${RESOLVED_CONFIG_JSON}
STAGE              : ${STAGE}
RUN_MODE           : ${RUN_MODE}
SMOKE              : ${SMOKE}
DRY_RUN            : ${DRY_RUN}
SEEDS              : ${SEEDS[*]}
SOURCE_FROM_CURRENT_MAIN: ${SOURCE_FROM_CURRENT_MAIN}
metadata_csv       : ${METADATA_CSV:-<missing>}
ct_root            : ${CT_ROOT:-<missing>}
gene_tsv           : ${GENE_TSV:-<missing>}
text source type   : ${TEXT_SOURCE_TYPE}
text source        : ${TEXT_SOURCE:-<missing>}
class_mode / label_mode: multiclass / triclass
num_classes        : ${NUM_CLASSES}
label_col          : ${LABEL_COL}
record_id_col      : ${TEXT_RECORD_ID_COL}
groups             : TRI-T / TRI-S0 / TRI-SKD
GROUPS             : TRI-T ct,cnv,text; TRI-S0 ct,text; TRI-SKD ct,text from TRI-T cached logits confidence KD
============================================================
EOF

if [[ "${STAGE}" == "teacher" || "${STAGE}" == "all" ]]; then stage_teacher; fi
if [[ "${STAGE}" == "supervised" || "${STAGE}" == "all" ]]; then stage_supervised; fi
if [[ "${STAGE}" == "student" || "${STAGE}" == "all" ]]; then stage_student; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
