#!/bin/bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(${SEEDS:-42 2024 2025 3407 8888})

CT_CSV="${CT_CSV:-/home/apulis-dev/userdata/Data/Multi/多模态统一检索表_CT本地路径_文本划分0205_修复.csv}"
CT_ROOT="${CT_ROOT:-/home/apulis-dev/userdata/Data/CT1500}"
GENE_TSV="${GENE_TSV:-/home/apulis-dev/userdata/Data/Gene/FDEM_CNV_merge_pcc.tsv}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${EXP_DIR}/text_features/huawei_text_features.tsv}"

"${PYTHON_BIN}" "${EXP_DIR}/generate_split_manifests.py" \
  --ct_csv "${CT_CSV}" \
  --ct_root "${CT_ROOT}" \
  --gene_tsv "${GENE_TSV}" \
  --text_feature_tsv "${TEXT_FEATURE_TSV}"

"${PYTHON_BIN}" "${EXP_DIR}/check_split_integrity.py"

TEXT_ARGS=()
if [[ -f "${TEXT_FEATURE_TSV}" ]]; then
  TEXT_ARGS+=(--text_feature_tsv "${TEXT_FEATURE_TSV}")
fi

for SEED in "${SEEDS[@]}"; do
  echo "Running split_7_1_2 seed=${SEED}"
  "${PYTHON_BIN}" "${EXP_DIR}/run_lungcls_experiment.py" \
    --split_name split_7_1_2 \
    --ct_csv "${CT_CSV}" \
    --ct_root "${CT_ROOT}" \
    --gene_tsv "${GENE_TSV}" \
    --seed "${SEED}" \
    "${TEXT_ARGS[@]}" \
    "$@"
done
