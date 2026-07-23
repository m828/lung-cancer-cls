#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${SMOKE:-1}" != "1" ]]; then
    echo "[FATAL] this entrypoint only permits SMOKE=1" >&2
    exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs0542_privileged_genomic_attribution_suite}"
FIXTURE_ROOT="${OUTPUT_ROOT}/00_audit/smoke_fixture"
RESULTS_ROOT="${FIXTURE_ROOT}/results"
DATA_ROOT="${FIXTURE_ROOT}/data"
mkdir -p "${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test" "${RESULTS_ROOT}/outputs0534_best_student_kd_search/cached_teacher_targets" "${OUTPUT_ROOT}/logs"

missing_modules="$(python3 - <<'PY'
import importlib.util
required=('torch','torchvision','numpy','pandas','sklearn','matplotlib')
print(','.join(name for name in required if importlib.util.find_spec(name) is None))
PY
)"
if [[ -n "${missing_modules}" ]]; then
    mkdir -p "${OUTPUT_ROOT}/reports"
    printf '%s\n' \
        '# Smoke Test Environment Blocker' \
        '' \
        "Training smoke was not started because required Python modules are unavailable: \`${missing_modules}\`." \
        'No full training was attempted.' \
        > "${OUTPUT_ROOT}/reports/smoke_environment_blocker.md"
    echo "[BLOCKED] missing Python modules: ${missing_modules}" >&2
    exit 3
fi

python3 scripts/create_attribution_smoke_fixture.py --root "${DATA_ROOT}"

train_teacher() {
    local teacher="$1" modalities="$2" output="$3"
    if [[ -f "${output}/metrics.json" && -f "${output}/best_model.pt" ]]; then
        echo "[SKIP] smoke teacher ${teacher}"
        return 0
    fi
    local gene_args=()
    if [[ "${modalities}" == *cnv* ]]; then gene_args=(--gene-tsv "${DATA_ROOT}/gene.tsv"); fi
    python3 train_multimodal.py \
        --data-root "${DATA_ROOT}" --metadata-csv "${DATA_ROOT}/metadata.csv" --output-dir "${output}" \
        --reference-manifest "${DATA_ROOT}/split_manifest.csv" --ct-root "${DATA_ROOT}/ct" \
        --text-feature-tsv "${DATA_ROOT}/text.tsv" "${gene_args[@]}" --modalities "${modalities}" \
        --class-mode binary --binary-task malignant_vs_normal --disable-text-numeric-features --strict-no-leakage \
        --ct-model attention3d_cnn --depth-size 8 --volume-hw 32 --ct-feature-dim 16 \
        --text-feature-dim 8 --gene-hidden-dim 8 --fusion-hidden-dim 16 --dropout 0.1 \
        --epochs 1 --batch-size 4 --num-workers 0 --lr 1e-3 --weight-decay 1e-4 \
        --optimizer adamw --scheduler cosine --loss ce --label-smoothing 0.0 \
        --sampling-strategy weighted --class-weight-strategy none --selection-metric auroc --seed 42 --cpu --smoke
}

T0_DIR="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42"
T1_DIR="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed42"
train_teacher T0 ct,text "${T0_DIR}"
train_teacher T1 ct,text,cnv "${T1_DIR}"

cache_teacher() {
    local teacher="$1" teacher_dir="$2" gene_args=()
    local output="${RESULTS_ROOT}/outputs0534_best_student_kd_search/cached_teacher_targets/${teacher}_seed42.csv"
    if [[ -f "${output}" ]]; then echo "[SKIP] smoke cache ${teacher}"; return 0; fi
    if [[ "${teacher}" == "T1" ]]; then gene_args=(--gene-tsv "${DATA_ROOT}/gene.tsv"); fi
    python3 scripts/cache_teacher_soft_targets.py \
        --teacher-run-dir "${teacher_dir}" \
        --output-dir "${RESULTS_ROOT}/outputs0534_best_student_kd_search/cached_teacher_targets" \
        --cache-name "${teacher}_seed42" --seed 42 --batch-size 4 --num-workers 0 \
        --reference-manifest "${DATA_ROOT}/split_manifest.csv" --data-root "${DATA_ROOT}" \
        --metadata-csv "${DATA_ROOT}/metadata.csv" --ct-root "${DATA_ROOT}/ct" \
        --text-feature-tsv "${DATA_ROOT}/text.tsv" "${gene_args[@]}" \
        --class-mode binary --binary-task malignant_vs_normal --disable-text-numeric-features --strict-no-leakage --cpu
}
cache_teacher T0 "${T0_DIR}"
cache_teacher T1 "${T1_DIR}"

export SMOKE=1 RUN_MODE=smoke SEEDS=42 OUTPUT_ROOT RESULTS_ROOT
export DATA_ROOT METADATA_CSV="${DATA_ROOT}/metadata.csv" CT_ROOT="${DATA_ROOT}/ct"
export TEXT_FEATURE_TSV="${DATA_ROOT}/text.tsv" GENE_TSV="${DATA_ROOT}/gene.tsv" REF_MANIFEST="${DATA_ROOT}/split_manifest.csv"
export CT_MODEL=attention3d_cnn DEPTH_SIZE=8 VOLUME_HW=32 CT_FEATURE_DIM=16 TEXT_FEATURE_DIM=8 GENE_HIDDEN_DIM=8 FUSION_HIDDEN_DIM=16
export NUM_WORKERS=0

bash scripts/run_binary_privileged_factorial_4seed.sh --root "${PROJECT_ROOT}" --resume
bash scripts/run_binary_shuffled_confidence_4seed.sh --root "${PROJECT_ROOT}" --resume
bash scripts/run_binary_cnv_permutation_4seed.sh --root "${PROJECT_ROOT}" --resume
python3 experiments/analysis/analyze_checkpoint_selection_sensitivity.py --root "${OUTPUT_ROOT}/04_checkpoint_sensitivity" --run-mode smoke
python3 experiments/analysis/analyze_teacher_correction_transfer.py \
    --output-dir "${OUTPUT_ROOT}/05_teacher_correction_analysis/smoke_fixture" --seeds 42 --bootstrap-iters 100 --smoke \
    --ct-text-teacher-pattern "${T0_DIR}" --full-teacher-pattern "${T1_DIR}" \
    --supervised-pattern "${OUTPUT_ROOT}/01_binary_factorial/S0_MATCHED/smoke/seed42" \
    --kd-pattern "${OUTPUT_ROOT}/01_binary_factorial/KD_CT_TEXT_CNV_CONFIDENCE/smoke/seed42"

TRICLASS_SOURCE_ROOT="${TRICLASS_SOURCE_ROOT:-${PROJECT_ROOT}/../outputs0541_triclass_teacher_student_selection_4seed}"
if [[ -d "${TRICLASS_SOURCE_ROOT}" ]]; then
    python3 experiments/analysis/analyze_triclass_confusion_profiles.py \
        --source-root "${TRICLASS_SOURCE_ROOT}" \
        --output-dir "${OUTPUT_ROOT}/06_triclass_confusion_analysis" \
        --seeds 42,43,44,45
else
    echo "[BLOCKED] triclass source unavailable: ${TRICLASS_SOURCE_ROOT}" >&2
    exit 3
fi

echo "[OK] bounded attribution smoke suite complete"
