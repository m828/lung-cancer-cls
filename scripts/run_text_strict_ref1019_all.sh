#!/usr/bin/env bash
# =============================================================================
# scripts/run_text_strict_ref1019_all.sh
#
# 一键重跑涉及 text 的 strict-no-leakage 关键实验，全部对齐到同一个 1019 split。
#
# 用法：
#   bash scripts/run_text_strict_ref1019_all.sh                # 默认只跑 seed42
#   RUN_MULTI_SEED=1 bash scripts/run_text_strict_ref1019_all.sh  # 包括 seed43/44/45
#
# 仅做：
#   1) text-only 严格无泄露（验证 text 单模态在剥离 num__ 后仍有判别力）
#   2) CT+Text supervised baseline（不含 KD）
#   3) ct+cnv+text strict teacher 重训
#   4) CT-only student（从 strict teacher 蒸馏）
#   5) CT+Text student（full-combo，最关键的 strict 主结果复跑）
#   6) CT+Text student（logits-only 对照）
#   7) CT+Text student（ResNet3D18 backbone 对照）
#
# 设计原则：
#   - 不覆盖 outputs/ 下已有历史 run；所有新结果写到 outputs/text_strict_ref1019_rerun/
#   - 所有 run 必须使用同一个 1019 reference manifest；如不存在则自动搜索
#   - 每个 run 完成后用 check_text_strict_rerun.py 生成汇总
# =============================================================================

set -u  # 不开 -e，因为单个 run 失败也要继续做其他 run + 汇总

# ----------------------------- 路径解析 --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_MM="${PROJECT_ROOT}/train_multimodal.py"
TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"

if [[ ! -f "${TRAIN_MM}" || ! -f "${TRAIN_KD}" ]]; then
    echo "[FATAL] train_multimodal.py / train_student_kd.py not found under ${PROJECT_ROOT}" >&2
    exit 1
fi

# ----------------------------- 输出根 ----------------------------------------
OUTPUT_ROOT="outputs/text_strict_ref1019_rerun"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# ----------------------------- 自动推断 reference manifest -------------------
# 优先使用内网 outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv
# 否则在 outputs/ 下搜索 total=1019 train=652 val=163 test=204 的 manifest
REF_MANIFEST_PREFERRED="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"
REF_MANIFEST=""

count_split() {
    # 统计 split_manifest.csv 的 train/val/test 行数（不含 header）
    local f="$1"
    [[ -f "$f" ]] || return 1
    python3 - "$f" <<'PY'
import sys, csv
f = sys.argv[1]
counts = {"train": 0, "val": 0, "test": 0}
total = 0
with open(f, encoding='utf-8-sig') as fh:
    reader = csv.DictReader(fh)
    if 'assigned_split' not in reader.fieldnames:
        print("NO_ASSIGNED_SPLIT_COL")
        sys.exit(2)
    for row in reader:
        total += 1
        s = (row.get('assigned_split') or '').strip()
        if s in counts:
            counts[s] += 1
print(f"{total},{counts['train']},{counts['val']},{counts['test']}")
PY
}

resolve_ref_manifest() {
    if [[ -f "${REF_MANIFEST_PREFERRED}" ]]; then
        local c
        c="$(count_split "${REF_MANIFEST_PREFERRED}")" || true
        if [[ "${c}" == "1019,652,163,204" ]]; then
            REF_MANIFEST="${REF_MANIFEST_PREFERRED}"
            return 0
        fi
        echo "[WARN] Preferred reference manifest ${REF_MANIFEST_PREFERRED} exists but split counts are ${c}, falling back to search." >&2
    fi
    # 搜索 outputs/ 下所有 split_manifest.csv
    local found=""
    while IFS= read -r mf; do
        local c
        c="$(count_split "${mf}" 2>/dev/null)" || continue
        if [[ "${c}" == "1019,652,163,204" ]]; then
            found="${mf}"
            break
        fi
    done < <(find outputs -name "split_manifest.csv" 2>/dev/null)
    if [[ -n "${found}" ]]; then
        REF_MANIFEST="${found}"
        return 0
    fi
    return 1
}

if ! resolve_ref_manifest; then
    echo "[FATAL] Could not find a split_manifest.csv with total=1019 train=652 val=163 test=204 under outputs/" >&2
    echo "        Please make sure the reference teacher run (with the 1019 split) exists locally." >&2
    exit 1
fi
REF_MANIFEST_COUNTS="$(count_split "${REF_MANIFEST}")"

# ----------------------------- 自动推断数据路径 ------------------------------
# 从历史 run 的 metrics.json["config"] 中读 data_root / metadata_csv / ct_root / gene_tsv / text_feature_tsv
# 候选 run 顺序：strict 优先 → 含 text 的 student → teacher → text-only
infer_paths_py='
import json, sys
from pathlib import Path
candidates = [
    "outputs/ct_text_student_kd_mvn_r2plus1d_18_full_combo",
    "outputs/ct_cnv_text_teacher_mvn_tvt",
    "outputs/ct_text_mvn_sc_tvt",
    "outputs/text_only_strict_ref1019",
    "outputs/ct_text_student_kd_mvn_r2plus1d_18_logits_only",
    "outputs/ct_text_student_kd_mvn_resnet3d18_full_combo",
]
keys = ["data_root", "metadata_csv", "ct_root", "gene_tsv", "text_feature_tsv"]
out = {k: "" for k in keys}
source = {k: "" for k in keys}
for run in candidates:
    p = Path(run) / "metrics.json"
    if not p.is_file():
        continue
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    cfg = d.get("config") or {}
    for k in keys:
        if not out[k]:
            v = cfg.get(k)
            if v:
                out[k] = str(v)
                source[k] = str(p)
for k in keys:
    print(f"{k}\t{out[k]}\t{source[k]}")
'

declare -A INFERRED INFERRED_FROM
while IFS=$'\t' read -r k v src; do
    INFERRED[$k]="$v"
    INFERRED_FROM[$k]="$src"
done < <(python3 -c "${infer_paths_py}")

# 允许通过环境变量手动覆盖
DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
GENE_TSV="${GENE_TSV:-${INFERRED[gene_tsv]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"

# PATIENT_ID_COL：metadata CSV 中的患者 ID 列名。无历史值，必须手动设置或保持为空（脚本会跳过 patient-level 检查）。
PATIENT_ID_COL="${PATIENT_ID_COL:-}"

# ----------------------------- seed 控制 -------------------------------------
RUN_MULTI_SEED="${RUN_MULTI_SEED:-0}"
SEEDS=(42)
if [[ "${RUN_MULTI_SEED}" == "1" ]]; then
    SEEDS=(42 43 44 45)
fi

# ----------------------------- 打印 banner -----------------------------------
echo "============================================================"
echo "Strict-no-leakage rerun — text-related experiments"
echo "============================================================"
echo "PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "REF_MANIFEST       : ${REF_MANIFEST}"
echo "REF_MANIFEST_COUNTS: total,train,val,test = ${REF_MANIFEST_COUNTS}"
echo ""
echo "Data paths (auto-inferred — override via env vars if wrong):"
printf "  %-18s = %s\n" "DATA_ROOT"        "${DATA_ROOT}"
printf "  %-18s = %s\n" "METADATA_CSV"     "${METADATA_CSV}"
printf "  %-18s = %s\n" "CT_ROOT"          "${CT_ROOT}"
printf "  %-18s = %s\n" "GENE_TSV"         "${GENE_TSV}"
printf "  %-18s = %s\n" "TEXT_FEATURE_TSV" "${TEXT_FEATURE_TSV}"
printf "  %-18s = %s\n" "PATIENT_ID_COL"   "${PATIENT_ID_COL:-<unset — patient-level split check skipped>}"
echo ""
echo "Inference sources:"
for k in data_root metadata_csv ct_root gene_tsv text_feature_tsv; do
    printf "  %-18s ← %s\n" "$k" "${INFERRED_FROM[$k]:-<none>}"
done
echo ""
echo "OUTPUT_ROOT        : ${OUTPUT_ROOT}"
echo "LOG_DIR            : ${LOG_DIR}"
echo "RUN_MULTI_SEED     : ${RUN_MULTI_SEED}  (seeds = ${SEEDS[*]})"
echo "============================================================"

# 必填路径检查
missing=()
[[ -z "${DATA_ROOT}"        ]] && missing+=("DATA_ROOT")
[[ -z "${METADATA_CSV}"     ]] && missing+=("METADATA_CSV")
[[ -z "${CT_ROOT}"          ]] && missing+=("CT_ROOT")
[[ -z "${TEXT_FEATURE_TSV}" ]] && missing+=("TEXT_FEATURE_TSV")
if (( ${#missing[@]} > 0 )); then
    echo "[FATAL] Required path(s) could not be inferred: ${missing[*]}"
    echo "        Set them via env vars before re-running, e.g.:"
    echo "          DATA_ROOT=... METADATA_CSV=... CT_ROOT=... TEXT_FEATURE_TSV=... bash $0"
    exit 1
fi

# ----------------------------- 通用工具函数 ----------------------------------
# patient-id-col flag（仅在已设置时附加）
patient_args() {
    if [[ -n "${PATIENT_ID_COL}" ]]; then
        echo "--patient-id-col ${PATIENT_ID_COL}"
    fi
}

# 共享 strict-text flag
strict_text_args() {
    echo "--strict-no-leakage --disable-text-numeric-features"
}

# 共享训练超参（与 0521 R2Plus1D student 一致；teacher 自己单独设置 batch_size=2）
common_train_args() {
    cat <<EOF
--class-mode binary
--binary-task malignant_vs_normal
--selection-metric auroc
--split-mode train_val_test
--epochs 50
--optimizer adamw
--lr 0.0003
--weight-decay 0.0001
--scheduler cosine
--loss ce
--label-smoothing 0.05
--sampling-strategy weighted
--class-weight-strategy effective_num
--effective-num-beta 0.999
--depth-size 128
--volume-hw 256
--ct-feature-dim 128
--text-feature-dim 256
--fusion-hidden-dim 256
--dropout 0.3
EOF
}

# 已存在则跳过（基于 metrics.json 存在性）
already_done() {
    local out="$1"
    [[ -f "${out}/metrics.json" ]]
}

# 运行命令并把 echo 出来；保存日志
run_cmd() {
    local name="$1"; shift
    local outdir="$1"; shift
    local logf="${LOG_DIR}/${name}.log"

    if already_done "${outdir}"; then
        echo "[SKIP] ${name} — ${outdir}/metrics.json already exists"
        return 0
    fi

    mkdir -p "${outdir}"
    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN]  ${name}"
    echo "       output : ${outdir}"
    echo "       log    : ${logf}"
    echo "       cmd    : $*"
    echo "------------------------------------------------------------"
    # tee 到日志，同时把 stdout/stderr 一起带走
    "$@" 2>&1 | tee "${logf}"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "[FAIL] ${name} exited with code ${rc} (see ${logf})"
    else
        echo "[OK]   ${name}"
    fi
    return ${rc}
}

# ----------------------------- 公共训练参数数组 ------------------------------
# 把 common_train_args 切成数组（避免 word splitting 出问题）
read -r -a COMMON_ARGS <<<"$(common_train_args | tr '\n' ' ')"
read -r -a STRICT_ARGS <<<"$(strict_text_args | tr '\n' ' ')"
read -r -a PATIENT_ARGS <<<"$(patient_args | tr '\n' ' ')"

# ============================================================================
# 1) text-only strict
# ============================================================================
run_text_only_strict() {
    local outdir="${OUTPUT_ROOT}/text_only_strict_ref1019"
    run_cmd "text_only_strict_ref1019" "${outdir}" \
        python3 "${TRAIN_MM}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --output-dir "${outdir}" \
            --modalities text \
            --reference-manifest "${REF_MANIFEST}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --seed 42 \
            --batch-size 8 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 2) CT+Text supervised (no KD) — R2Plus1D
# ============================================================================
run_ct_text_sc_r2plus1d_strict() {
    local outdir="${OUTPUT_ROOT}/ct_text_sc_r2plus1d_strict_ref1019"
    run_cmd "ct_text_sc_r2plus1d_strict_ref1019" "${outdir}" \
        python3 "${TRAIN_MM}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --seed 42 \
            --batch-size 1 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 3) ct+cnv+text strict teacher  (用于第 4/5/6/7 中 student 的 teacher)
#    必须先于 student 跑完。
# ============================================================================
TEACHER_OUTDIR="${OUTPUT_ROOT}/ct_cnv_text_teacher_strict_ref1019"

run_teacher_strict() {
    run_cmd "ct_cnv_text_teacher_strict_ref1019" "${TEACHER_OUTDIR}" \
        python3 "${TRAIN_MM}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --output-dir "${TEACHER_OUTDIR}" \
            --modalities ct,cnv,text \
            --reference-manifest "${REF_MANIFEST}" \
            --ct-root "${CT_ROOT}" \
            --gene-tsv "${GENE_TSV}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model resnet3d18 \
            --seed 42 \
            --batch-size 2 \
            --epochs 100 \
            --class-mode binary \
            --binary-task malignant_vs_normal \
            --selection-metric auroc \
            --split-mode train_val_test \
            --optimizer adamw \
            --lr 0.0003 \
            --weight-decay 0.0001 \
            --scheduler cosine \
            --loss ce \
            --label-smoothing 0.05 \
            --sampling-strategy weighted \
            --class-weight-strategy effective_num \
            --effective-num-beta 0.999 \
            --depth-size 128 \
            --volume-hw 256 \
            --ct-feature-dim 128 \
            --text-feature-dim 256 \
            --fusion-hidden-dim 256 \
            --dropout 0.3 \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

teacher_ready() {
    [[ -f "${TEACHER_OUTDIR}/metrics.json" && -f "${TEACHER_OUTDIR}/best_model.pt" ]] \
        || [[ -f "${TEACHER_OUTDIR}/metrics.json" ]]
}

# ============================================================================
# 4) CT-only student（从 strict teacher 蒸馏） — R2Plus1D full combo
# ============================================================================
run_ct_student_kd_r2plus1d_full_combo_strict() {
    local outdir="${OUTPUT_ROOT}/ct_student_kd_r2plus1d_full_combo_strict_ref1019"
    if ! teacher_ready; then
        echo "[SKIP] CT student — teacher run ${TEACHER_OUTDIR} not ready"; return 1
    fi
    run_cmd "ct_student_kd_r2plus1d_full_combo_strict_ref1019" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_OUTDIR}" \
            --distill-methods "logits,fused,hint,relation,attention,ct" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,relation=0.25,attention=0.25,ct=0.5" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --seed 42 \
            --batch-size 1 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 5) CT+Text student full-combo — 最关键的 strict 主结果（支持多 seed）
# ============================================================================
run_ct_text_student_kd_r2plus1d_full_combo_strict() {
    local seed="$1"
    if ! teacher_ready; then
        echo "[SKIP] CT+Text student full-combo seed${seed} — teacher run ${TEACHER_OUTDIR} not ready"; return 1
    fi
    local suffix=""
    if [[ "${seed}" != "42" ]]; then
        suffix="_seed${seed}"
    fi
    local outdir="${OUTPUT_ROOT}/ct_text_student_kd_r2plus1d_full_combo_strict_ref1019${suffix}"
    run_cmd "ct_text_student_kd_r2plus1d_full_combo_strict_ref1019${suffix}" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_OUTDIR}" \
            --distill-methods "logits,fused,hint,relation,attention,ct,text" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --seed "${seed}" \
            --batch-size 1 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 6) CT+Text student logits-only (对照 full-combo)
# ============================================================================
run_ct_text_student_kd_r2plus1d_logits_only_strict() {
    local outdir="${OUTPUT_ROOT}/ct_text_student_kd_r2plus1d_logits_only_strict_ref1019"
    if ! teacher_ready; then
        echo "[SKIP] CT+Text logits-only — teacher run ${TEACHER_OUTDIR} not ready"; return 1
    fi
    run_cmd "ct_text_student_kd_r2plus1d_logits_only_strict_ref1019" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_OUTDIR}" \
            --distill-methods "logits" \
            --distill-feature-loss cosine \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --seed 42 \
            --batch-size 1 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 7) CT+Text student full-combo on ResNet3D18（backbone 对照）
# ============================================================================
run_ct_text_student_kd_resnet3d18_full_combo_strict() {
    local outdir="${OUTPUT_ROOT}/ct_text_student_kd_resnet3d18_full_combo_strict_ref1019"
    if ! teacher_ready; then
        echo "[SKIP] CT+Text resnet3d18 full-combo — teacher run ${TEACHER_OUTDIR} not ready"; return 1
    fi
    run_cmd "ct_text_student_kd_resnet3d18_full_combo_strict_ref1019" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model resnet3d18 \
            --teacher-run-dir "${TEACHER_OUTDIR}" \
            --distill-methods "logits,fused,hint,relation,attention,ct,text" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --seed 42 \
            --batch-size 2 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ----------------------------- 执行顺序 --------------------------------------
# 1) text-only 和 2) CT+Text supervised：和 teacher 互不依赖，可以先跑也可以后跑
# 3) teacher：必须先于 4/5/6/7
# 4-7) 各类 student
#
# 失败一个 run 不会中断整个脚本（set -u 但没有 -e）

run_text_only_strict || true
run_ct_text_sc_r2plus1d_strict || true

run_teacher_strict || true

run_ct_student_kd_r2plus1d_full_combo_strict || true

for s in "${SEEDS[@]}"; do
    run_ct_text_student_kd_r2plus1d_full_combo_strict "${s}" || true
done

run_ct_text_student_kd_r2plus1d_logits_only_strict || true
run_ct_text_student_kd_resnet3d18_full_combo_strict || true

# ----------------------------- 汇总 ------------------------------------------
echo ""
echo "============================================================"
echo "All runs done (or skipped). Generating summary..."
echo "============================================================"

CHECK_PY="${PROJECT_ROOT}/experiments/analysis/check_text_strict_rerun.py"
if [[ -f "${CHECK_PY}" ]]; then
    python3 "${CHECK_PY}" \
        --root "${OUTPUT_ROOT}" \
        --reference-manifest "${REF_MANIFEST}"
else
    echo "[WARN] ${CHECK_PY} not found — skip summary generation."
fi

echo ""
echo "Summary outputs:"
echo "  ${OUTPUT_ROOT}/strict_text_rerun_summary.csv"
echo "  ${OUTPUT_ROOT}/strict_text_rerun_summary.md"
echo "Logs:"
echo "  ${LOG_DIR}/"
echo "Done."
