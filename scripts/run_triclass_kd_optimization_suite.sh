#!/usr/bin/env bash
# Configurable three-class KD optimization suite.
# Read-only with respect to outputs0536; writes only under outputs0538 by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0538_triclass_kd_optimization}"
SOURCE_TRI_ROOT="${SOURCE_TRI_ROOT:-${PARENT_ROOT}/outputs0536_triclass_extension}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-mini}"
SMOKE="${SMOKE:-1}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRI_OPT_PROFILE_ENV="${TRI_OPT_PROFILE:-}"
TRI_KD_SELECTION_METRIC_USER="${TRI_KD_SELECTION_METRIC:-}"
TRI_LOSS_USER="${TRI_LOSS:-}"
TRI_FOCAL_GAMMA_USER="${TRI_FOCAL_GAMMA:-}"
TRI_CLASS_WEIGHT_STRATEGY_USER="${TRI_CLASS_WEIGHT_STRATEGY:-}"
TRI_EFFECTIVE_NUM_BETA_USER="${TRI_EFFECTIVE_NUM_BETA:-}"
TRI_SAMPLING_STRATEGY_USER="${TRI_SAMPLING_STRATEGY:-}"
TRI_KD_ALPHA_USER="${TRI_KD_ALPHA:-}"
TRI_KD_TEMPERATURE_USER="${TRI_KD_TEMPERATURE:-}"
TRI_KD_WEIGHTING_USER="${TRI_KD_WEIGHTING:-}"
TRI_MALIGNANT_KD_BOOST_USER="${TRI_MALIGNANT_KD_BOOST:-}"
TRI_TEACHER_PROB_SMOOTHING_USER="${TRI_TEACHER_PROB_SMOOTHING:-}"
TRI_STUDENT_LABEL_SMOOTHING_USER="${TRI_STUDENT_LABEL_SMOOTHING:-}"
SEEDS_ENV="${SEEDS:-}"
RESULTS_ROOT_HINT="${RESULTS_ROOT:-}"
TRI_ALLOWED_SELECTION_METRICS=(balanced_accuracy accuracy auroc macro_auroc macro_f1 malignant_recall triclass_clinical_composite triclass_calibrated_composite)

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
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${OUT_ROOT}/profiles" "${OUT_ROOT}/logs" "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_triclass_kd_optimization_suite.sh.snapshot"

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
for key in [
    "data_root", "metadata_csv", "ct_root", "gene_tsv", "text_feature_tsv",
    "text_source", "text_source_type", "text_cache_tsv",
    "metadata_sample_id_col", "patient_id_col", "metadata_text_id_col",
    "text_record_id_col", "ct_path_col", "label_col", "split_col",
    "allowed_text_cols", "allowed_numeric_cols", "forbidden_feature_keywords",
    "strict_no_leakage", "disable_text_numeric_features",
]:
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
TEXT_SOURCE_TYPE="${RESOLVED[text_source_type]:-}"
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
STRICT_NO_LEAKAGE="${RESOLVED[strict_no_leakage]:-True}"
DISABLE_TEXT_NUMERIC_FEATURES="${RESOLVED[disable_text_numeric_features]:-True}"

if [[ -n "${TEXT_SOURCE}" && -n "${GENE_TSV}" && "${TEXT_SOURCE}" == "${GENE_TSV}" ]]; then
    echo "[ERROR] text feature source equals gene feature source. This is almost certainly wrong." >&2
    exit 2
fi

TEXT_ARGS=()
if [[ "${TEXT_SOURCE_TYPE}" == "text_feature_tsv" && -n "${TEXT_FEATURE_TSV}" ]]; then
    TEXT_ARGS=(--text-feature-tsv "${TEXT_FEATURE_TSV}")
elif [[ "${TEXT_SOURCE_TYPE}" == "text_cache_tsv" && -n "${TEXT_CACHE_TSV}" ]]; then
    TEXT_ARGS=(--text-cache-tsv "${TEXT_CACHE_TSV}")
else
    echo "[ERROR] TRI-KD optimization currently requires resolved text_feature_tsv/text_cache_tsv for cached KD. Resolved text_source_type=${TEXT_SOURCE_TYPE}" >&2
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

STRICT_ARGS=()
[[ "${STRICT_NO_LEAKAGE}" == "True" || "${STRICT_NO_LEAKAGE}" == "true" || "${STRICT_NO_LEAKAGE}" == "1" ]] && STRICT_ARGS+=(--strict-no-leakage)
[[ "${DISABLE_TEXT_NUMERIC_FEATURES}" == "True" || "${DISABLE_TEXT_NUMERIC_FEATURES}" == "true" || "${DISABLE_TEXT_NUMERIC_FEATURES}" == "1" ]] && STRICT_ARGS+=(--disable-text-numeric-features)

if [[ -n "${SEEDS_ENV}" ]]; then
    IFS=',' read -r -a SEEDS <<<"${SEEDS_ENV}"
elif [[ "${SMOKE}" == "1" ]]; then
    SEEDS=(42)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    SEEDS=(42 43)
else
    SEEDS=(42 43 44 45)
fi

if [[ -n "${TRI_OPT_PROFILE_ENV}" ]]; then
    IFS=',' read -r -a PROFILES <<<"${TRI_OPT_PROFILE_ENV}"
elif [[ "${RUN_MODE}" == "sweep" || "${RUN_MODE}" == "full" ]]; then
    PROFILES=(bacc_select acc_select auc_select macro_f1_select clinical_composite class_balanced focal malignant_boost low_alpha high_temp)
else
    PROFILES=(bacc_select macro_f1_select clinical_composite)
fi

profile_defaults() {
    local profile="$1"
    TRI_KD_SELECTION_METRIC="balanced_accuracy"
    TRI_LOSS="ce"
    TRI_FOCAL_GAMMA="2.0"
    TRI_CLASS_WEIGHT_STRATEGY="effective_num"
    TRI_EFFECTIVE_NUM_BETA="0.999"
    TRI_SAMPLING_STRATEGY="weighted"
    TRI_KD_ALPHA="0.1"
    TRI_KD_TEMPERATURE="8"
    TRI_KD_WEIGHTING="confidence"
    TRI_MALIGNANT_KD_BOOST="1.0"
    TRI_TEACHER_PROB_SMOOTHING="0.0"
    TRI_STUDENT_LABEL_SMOOTHING="0.05"
    case "${profile}" in
        bacc_select) ;;
        acc_select) TRI_KD_SELECTION_METRIC="accuracy" ;;
        auc_select) TRI_KD_SELECTION_METRIC="macro_auroc" ;;
        macro_f1_select) TRI_KD_SELECTION_METRIC="macro_f1" ;;
        clinical_composite) TRI_KD_SELECTION_METRIC="triclass_clinical_composite" ;;
        class_balanced) TRI_LOSS="class_balanced_ce"; TRI_CLASS_WEIGHT_STRATEGY="effective_num"; TRI_SAMPLING_STRATEGY="class_balanced" ;;
        focal) TRI_LOSS="focal"; TRI_FOCAL_GAMMA="2.0" ;;
        malignant_boost) TRI_KD_SELECTION_METRIC="triclass_clinical_composite"; TRI_LOSS="class_balanced_ce"; TRI_KD_WEIGHTING="class_confidence"; TRI_MALIGNANT_KD_BOOST="1.5"; TRI_SAMPLING_STRATEGY="malignant_oversample" ;;
        low_alpha) TRI_KD_ALPHA="0.05" ;;
        high_temp) TRI_KD_TEMPERATURE="12" ;;
        *) echo "[ERROR] Unknown TRI_OPT_PROFILE=${profile}" >&2; exit 2 ;;
    esac
    [[ -n "${TRI_KD_SELECTION_METRIC_USER}" ]] && TRI_KD_SELECTION_METRIC="${TRI_KD_SELECTION_METRIC_USER}"
    [[ -n "${TRI_LOSS_USER}" ]] && TRI_LOSS="${TRI_LOSS_USER}"
    [[ -n "${TRI_FOCAL_GAMMA_USER}" ]] && TRI_FOCAL_GAMMA="${TRI_FOCAL_GAMMA_USER}"
    [[ -n "${TRI_CLASS_WEIGHT_STRATEGY_USER}" ]] && TRI_CLASS_WEIGHT_STRATEGY="${TRI_CLASS_WEIGHT_STRATEGY_USER}"
    [[ -n "${TRI_EFFECTIVE_NUM_BETA_USER}" ]] && TRI_EFFECTIVE_NUM_BETA="${TRI_EFFECTIVE_NUM_BETA_USER}"
    [[ -n "${TRI_SAMPLING_STRATEGY_USER}" ]] && TRI_SAMPLING_STRATEGY="${TRI_SAMPLING_STRATEGY_USER}"
    [[ -n "${TRI_KD_ALPHA_USER}" ]] && TRI_KD_ALPHA="${TRI_KD_ALPHA_USER}"
    [[ -n "${TRI_KD_TEMPERATURE_USER}" ]] && TRI_KD_TEMPERATURE="${TRI_KD_TEMPERATURE_USER}"
    [[ -n "${TRI_KD_WEIGHTING_USER}" ]] && TRI_KD_WEIGHTING="${TRI_KD_WEIGHTING_USER}"
    [[ -n "${TRI_MALIGNANT_KD_BOOST_USER}" ]] && TRI_MALIGNANT_KD_BOOST="${TRI_MALIGNANT_KD_BOOST_USER}"
    [[ -n "${TRI_TEACHER_PROB_SMOOTHING_USER}" ]] && TRI_TEACHER_PROB_SMOOTHING="${TRI_TEACHER_PROB_SMOOTHING_USER}"
    [[ -n "${TRI_STUDENT_LABEL_SMOOTHING_USER}" ]] && TRI_STUDENT_LABEL_SMOOTHING="${TRI_STUDENT_LABEL_SMOOTHING_USER}"
    return 0
}

validate_tri_selection_metric() {
    local metric="$1"
    local allowed
    for allowed in "${TRI_ALLOWED_SELECTION_METRICS[@]}"; do
        if [[ "${metric}" == "${allowed}" ]]; then
            return 0
        fi
    done
    echo "[ERROR] Unsupported TRI_KD_SELECTION_METRIC=${metric}. Allowed: ${TRI_ALLOWED_SELECTION_METRICS[*]}" >&2
    return 2
}

training_loss_name() {
    case "$1" in
        class_balanced_ce) echo "ce" ;;
        class_balanced_focal) echo "focal" ;;
        ce|focal) echo "$1" ;;
        *) echo "[ERROR] Unknown TRI_LOSS=$1" >&2; exit 2 ;;
    esac
}

reference_manifest_for() {
    local seed="$1"
    echo "${SOURCE_TRI_ROOT}/teacher_ct_cnv_text/TRI-T_ct_cnv_text_densenet3d121_seed${seed}/split_manifest.csv"
}

teacher_cache_for() {
    local seed="$1"
    echo "${SOURCE_TRI_ROOT}/cached_teacher_targets/TRI-T_seed${seed}.csv"
}

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

write_audit() {
    cat >"${OUT_ROOT}/triclass_optimization_code_audit.md" <<EOF
# Triclass KD Optimization Code Audit

This suite is an extension validation optimizer, not the locked binary main result.

- Source triclass root: \`${SOURCE_TRI_ROOT}\`
- Output root: \`${OUT_ROOT}\`
- Metadata: \`${METADATA_CSV}\`
- CT root: \`${CT_ROOT}\`
- Gene/CNV TSV: \`${GENE_TSV}\`
- Text source type: \`${TEXT_SOURCE_TYPE}\`
- Text source: \`${TEXT_SOURCE}\`
- Reuses teacher caches from: \`${SOURCE_TRI_ROOT}/cached_teacher_targets/TRI-T_seed*.csv\`
- Reuses TRI-S0 supervised results from: \`${SOURCE_TRI_ROOT}/supervised_ct_text\`
- Class mode: multiclass, labels normal/benign/malignant, num_classes=3
- Supported selection metrics: balanced_accuracy, accuracy, auroc, macro_auroc, macro_f1, malignant_recall, triclass_clinical_composite, triclass_calibrated_composite
- Missing selection metrics raise an error through \`resolve_selection_score\`; there is no silent AUROC fallback.
- Supported profiles: bacc_select, acc_select, auc_select, macro_f1_select, clinical_composite, class_balanced, focal, malignant_boost, low_alpha, high_temp

Safety checks:

- TEXT source equals Gene source: \`no\`
- outputs0536 is read-only input for this suite.
- outputs0535 binary main result is not touched.
EOF
}

stage_train() {
    local profile seed cache manifest outdir loss_name name
    for profile in "${PROFILES[@]}"; do
        profile_defaults "${profile}"
        loss_name="$(training_loss_name "${TRI_LOSS}")"
        for seed in "${SEEDS[@]}"; do
            cache="$(teacher_cache_for "${seed}")"
            manifest="$(reference_manifest_for "${seed}")"
            if [[ ! -f "${cache}" ]]; then
                echo "[MISSING] ${profile} seed${seed}: teacher cache ${cache}"
                continue
            fi
            if [[ ! -f "${manifest}" ]]; then
                echo "[MISSING] ${profile} seed${seed}: reference manifest ${manifest}"
                continue
            fi
            outdir="${OUT_ROOT}/profiles/${profile}/TRI-SKD_${profile}_seed${seed}"
            name="triclass_${profile}_seed${seed}"
            validate_tri_selection_metric "${TRI_KD_SELECTION_METRIC}"
            echo "[PROFILE] ${profile} seed${seed}: selection=${TRI_KD_SELECTION_METRIC} loss=${TRI_LOSS} train_loss=${loss_name} class_weight=${TRI_CLASS_WEIGHT_STRATEGY} gamma=${TRI_FOCAL_GAMMA} sampling=${TRI_SAMPLING_STRATEGY} kd=${TRI_KD_WEIGHTING} alpha=${TRI_KD_ALPHA} T=${TRI_KD_TEMPERATURE} out=${outdir}"
            echo "[CACHE] reuse ${cache}"
            run_cmd "${name}" "${outdir}/metrics.json" \
                python3 scripts/train_student_kd_cached_logits.py \
                --output-dir "${outdir}" \
                --cached-teacher-targets "${cache}" \
                --reference-manifest "${manifest}" \
                --seed "${seed}" \
                --class-mode multiclass \
                "${STRICT_ARGS[@]}" \
                --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}" \
                --ct-root "${CT_ROOT}" \
                "${TEXT_ARGS[@]}" \
                "${COLUMN_ARGS[@]}" \
                --ct-model densenet3d_121 --modalities ct,text \
                --depth-size 128 --volume-hw 256 \
                --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256 \
                --dropout 0.3 --loss "${loss_name}" --label-smoothing "${TRI_STUDENT_LABEL_SMOOTHING}" \
                --focal-gamma "${TRI_FOCAL_GAMMA}" \
                --sampling-strategy "${TRI_SAMPLING_STRATEGY}" \
                --class-weight-strategy "${TRI_CLASS_WEIGHT_STRATEGY}" \
                --effective-num-beta "${TRI_EFFECTIVE_NUM_BETA}" \
                --selection-metric "${TRI_KD_SELECTION_METRIC}" \
                --distillation-alpha "${TRI_KD_ALPHA}" --distillation-temperature "${TRI_KD_TEMPERATURE}" \
                --kd-weighting "${TRI_KD_WEIGHTING}" --kd-weight-floor 0.0 --kd-weight-max 1.0 \
                --malignant-kd-boost "${TRI_MALIGNANT_KD_BOOST}" \
                --teacher-prob-smoothing "${TRI_TEACHER_PROB_SMOOTHING}" \
                --batch-size "${BATCH_SIZE}" --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
                --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
                --optimizer adamw --scheduler cosine --epochs "${EPOCHS}" --amp \
                --early-stopping-patience 10 --num-workers "${NUM_WORKERS}"
        done
    done
}

stage_analyze() {
    run_cmd "analyze_triclass_kd_optimization" "${OUT_ROOT}/triclass_kd_optimization_summary.md" \
        python3 experiments/analysis/analyze_triclass_kd_optimization.py \
        --root "${OUT_ROOT}" \
        --source-root "${SOURCE_TRI_ROOT}"
}

cat <<EOF
============================================================
Triclass KD optimization suite
PROJECT_ROOT : ${PROJECT_ROOT}
SOURCE_TRI   : ${SOURCE_TRI_ROOT}
OUT_ROOT     : ${OUT_ROOT}
STAGE        : ${STAGE}
RUN_MODE     : ${RUN_MODE}
SMOKE        : ${SMOKE}
DRY_RUN      : ${DRY_RUN}
SEEDS        : ${SEEDS[*]}
PROFILES     : ${PROFILES[*]}
metadata_csv : ${METADATA_CSV}
ct_root      : ${CT_ROOT}
gene_tsv     : ${GENE_TSV}
text source  : ${TEXT_SOURCE_TYPE} ${TEXT_SOURCE}
groups       : TRI-SKD optimization profiles vs TRI-S0 from outputs0536
============================================================
EOF

if [[ "${STAGE}" == "audit" || "${STAGE}" == "all" ]]; then write_audit; fi
if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then stage_train; fi
if [[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]]; then stage_analyze; fi

echo "[DONE] outputs: ${OUT_ROOT}"
