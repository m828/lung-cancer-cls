#!/usr/bin/env bash
# scripts/run_strict_rescue_0530.sh
#
# 基于 outputs0529 诊断结果的 strict rescue 小实验。
#
# 背景：outputs0529 中 R2Plus1D + CT+Text + full-combo strict (seed42) 出现训练塌陷
# (test AUROC 0.7556)，诊断指出这是 R2Plus1D × full-combo × strict-text × bs=1 的优化
# 病态，而非数据/checkpoint/CT 分支问题。
#
# 三个目标：
#   1) lite-combo 测试：丢掉 0 梯度的 relation/attention，lower alpha → 看是否能恢复 ≥ 0.95
#   2) 补 logits-only strict 多 seed (43/44/45)
#   3) (条件 4) 若 seed42 lite-combo AUROC ≥ 0.95，再补 lite-combo seeds 43/44/45
#
# 重要约束：
#   - 不覆盖 outputs0529
#   - 不重新随机划分，复用 1019 reference split
#   - 已存在 metrics.json 的 run 直接 skip
#   - 内网只有 outputs/ 目录；data path 自动从历史 run 推断；可用 env var 覆盖
#   - 数据路径 / metadata / patient_id_col 不要写死
#
# 运行方式：
#   bash scripts/run_strict_rescue_0530.sh                       # 默认：只跑步骤 1+2
#   RUN_CONDITIONAL_SEEDS=1 bash scripts/run_strict_rescue_0530.sh
#       # 步骤 1+2 完成后，如果 lite-combo seed42 AUROC ≥ 0.95，自动补 lite-combo seeds 43/44/45
#   LITE_BATCH_SIZE=2 bash scripts/run_strict_rescue_0530.sh
#       # 如果显存允许把 lite-combo 的 batch_size 改成 2
#
# 完成后会调用 experiments/analysis/rescue_metrics_summary.py 生成 csv + md 汇总。

set -uo pipefail

# ----------------------------- 路径定位 --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"
if [[ ! -f "${TRAIN_KD}" ]]; then
    echo "[FATAL] train_student_kd.py not found at ${TRAIN_KD}"; exit 1
fi

OUTPUT_ROOT="outputs0530_strict_rescue"
LOG_DIR="${OUTPUT_ROOT}/logs"
SCRIPTS_USED_DIR="${OUTPUT_ROOT}/scripts_used"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SCRIPTS_USED_DIR}"

# 保存当前脚本副本以便事后审计
cp -f "${BASH_SOURCE[0]}" "${SCRIPTS_USED_DIR}/run_strict_rescue_0530.sh.snapshot"

# ----------------------------- 参考 split -----------------------------------
REF_MANIFEST_PREFERRED="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"
REF_MANIFEST=""

count_split() {
    python3 - <<PY "$1"
import csv, sys
p = sys.argv[1]
total = 0
cnt = {'train':0,'val':0,'test':0}
with open(p,'r',encoding='utf-8-sig',newline='') as f:
    reader = csv.DictReader(f)
    col = None
    for c in ('assigned_split','split','Split'):
        if reader.fieldnames and c in reader.fieldnames:
            col = c; break
    if col is None:
        print('0,0,0,0'); sys.exit(0)
    for row in reader:
        total += 1
        v = (row.get(col) or '').strip().lower()
        if v in cnt: cnt[v] += 1
print(f"{total},{cnt['train']},{cnt['val']},{cnt['test']}")
PY
}

resolve_ref_manifest() {
    if [[ -f "${REF_MANIFEST_PREFERRED}" ]]; then
        c="$(count_split "${REF_MANIFEST_PREFERRED}")" || true
        if [[ "${c}" == "1019,652,163,204" ]]; then
            REF_MANIFEST="${REF_MANIFEST_PREFERRED}"
            return 0
        fi
        echo "[WARN] Preferred ref manifest exists but counts are ${c}, falling back to search." >&2
    fi
    while IFS= read -r f; do
        c="$(count_split "$f")" || continue
        if [[ "${c}" == "1019,652,163,204" ]]; then
            REF_MANIFEST="$f"; return 0
        fi
    done < <(find outputs -type f -name split_manifest.csv 2>/dev/null)
    return 1
}

if ! resolve_ref_manifest; then
    echo "[FATAL] no 1019/652/163/204 split_manifest.csv found under outputs/." >&2
    exit 1
fi
REF_MANIFEST_COUNTS="$(count_split "${REF_MANIFEST}")"

# ----------------------------- teacher run dir -------------------------------
# 优先用 outputs0529 (本次任务里上一个批次的 strict teacher);
# 如不可见，再回退到 outputs/ 下同名 strict teacher (内网 outputs/ 映射)
TEACHER_CANDIDATES=(
    "outputs0529/ct_cnv_text_teacher_strict_ref1019"
    "outputs/text_strict_ref1019_rerun/ct_cnv_text_teacher_strict_ref1019"
    "outputs/ct_cnv_text_teacher_strict_ref1019"
)
TEACHER_RUN_DIR=""
for cand in "${TEACHER_CANDIDATES[@]}"; do
    if [[ -f "${cand}/metrics.json" ]]; then
        TEACHER_RUN_DIR="${cand}"; break
    fi
done
if [[ -z "${TEACHER_RUN_DIR}" ]]; then
    echo "[FATAL] strict teacher run dir not found. Tried:" >&2
    printf '  %s\n' "${TEACHER_CANDIDATES[@]}" >&2
    exit 1
fi

# ----------------------------- 数据路径推断 ---------------------------------
# 从已存在 run 的 metrics.json["config"] 中推断 data_root / metadata_csv / ct_root /
# text_feature_tsv (rescue 不需要 gene_tsv，但若存在也带上 — student KD 不用)
infer_paths_py='
import json, sys
from pathlib import Path
candidates = [
    "outputs0529/ct_text_student_kd_r2plus1d_full_combo_strict_ref1019",
    "outputs0529/ct_text_student_kd_r2plus1d_logits_only_strict_ref1019",
    "outputs0529/ct_cnv_text_teacher_strict_ref1019",
    "outputs/text_strict_ref1019_rerun/ct_text_student_kd_r2plus1d_full_combo_strict_ref1019",
    "outputs/ct_text_student_kd_mvn_r2plus1d_18_full_combo",
    "outputs/ct_cnv_text_teacher_mvn_tvt",
]
keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv"]
out = {k: "" for k in keys}
source = {k: "" for k in keys}
for run in candidates:
    p = Path(run) / "metrics.json"
    if not p.is_file(): continue
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    cfg = d.get("config") or {}
    for k in keys:
        if not out[k]:
            v = cfg.get(k)
            if v:
                out[k] = str(v); source[k] = str(p)
for k in keys:
    print(f"{k}\t{out[k]}\t{source[k]}")
'

declare -A INFERRED INFERRED_FROM
while IFS=$'\t' read -r k v src; do
    INFERRED[$k]="$v"; INFERRED_FROM[$k]="$src"
done < <(python3 -c "${infer_paths_py}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"
PATIENT_ID_COL="${PATIENT_ID_COL:-}"

# ----------------------------- 控制开关 -------------------------------------
LITE_BATCH_SIZE="${LITE_BATCH_SIZE:-1}"               # full-combo lite 默认 bs=1，可改 2
RUN_CONDITIONAL_SEEDS="${RUN_CONDITIONAL_SEEDS:-0}"   # 是否在 seed42 lite ≥0.95 时补 43/44/45
COND_AUROC_THRESH="${COND_AUROC_THRESH:-0.95}"

# ----------------------------- banner ---------------------------------------
echo "============================================================"
echo "Strict-no-leakage rescue — outputs0530_strict_rescue"
echo "============================================================"
echo "PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "REF_MANIFEST       : ${REF_MANIFEST}"
echo "REF_MANIFEST_COUNTS: total,train,val,test = ${REF_MANIFEST_COUNTS}"
echo "TEACHER_RUN_DIR    : ${TEACHER_RUN_DIR}"
echo ""
echo "Data paths (auto-inferred — override via env vars if wrong):"
printf "  %-18s = %s\n" "DATA_ROOT"        "${DATA_ROOT}"
printf "  %-18s = %s\n" "METADATA_CSV"     "${METADATA_CSV}"
printf "  %-18s = %s\n" "CT_ROOT"          "${CT_ROOT}"
printf "  %-18s = %s\n" "TEXT_FEATURE_TSV" "${TEXT_FEATURE_TSV}"
printf "  %-18s = %s\n" "PATIENT_ID_COL"   "${PATIENT_ID_COL:-<unset>}"
echo ""
echo "Inference sources:"
for k in data_root metadata_csv ct_root text_feature_tsv; do
    printf "  %-18s ← %s\n" "$k" "${INFERRED_FROM[$k]:-<none>}"
done
echo ""
echo "OUTPUT_ROOT        : ${OUTPUT_ROOT}"
echo "LOG_DIR            : ${LOG_DIR}"
echo "LITE_BATCH_SIZE    : ${LITE_BATCH_SIZE}"
echo "RUN_CONDITIONAL_SEEDS : ${RUN_CONDITIONAL_SEEDS}"
echo "COND_AUROC_THRESH  : ${COND_AUROC_THRESH}"
echo "============================================================"

missing=()
[[ -z "${DATA_ROOT}"        ]] && missing+=("DATA_ROOT")
[[ -z "${METADATA_CSV}"     ]] && missing+=("METADATA_CSV")
[[ -z "${CT_ROOT}"          ]] && missing+=("CT_ROOT")
[[ -z "${TEXT_FEATURE_TSV}" ]] && missing+=("TEXT_FEATURE_TSV")
if (( ${#missing[@]} > 0 )); then
    echo "[FATAL] Required path(s) could not be inferred: ${missing[*]}"
    echo "        Set them via env vars before re-running."
    exit 1
fi

# ----------------------------- 工具函数 -------------------------------------
patient_args() {
    if [[ -n "${PATIENT_ID_COL}" ]]; then
        echo "--patient-id-col ${PATIENT_ID_COL}"
    fi
}
read -r -a PATIENT_ARGS <<<"$(patient_args)"

# 与 outputs0529 主模型一致的训练超参（除了被 rescue 改的 alpha / distill_methods）
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
read -r -a COMMON_ARGS <<<"$(common_train_args | tr '\n' ' ')"
STRICT_ARGS=(--strict-no-leakage --disable-text-numeric-features)

already_done() {
    [[ -f "$1/metrics.json" ]]
}

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
    "$@" 2>&1 | tee "${logf}"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "[FAIL] ${name} exited with code ${rc} (see ${logf})"
    else
        echo "[OK]   ${name}"
    fi
    return ${rc}
}

# 从 metrics.json 里读出 test AUROC，给条件 4 的门控用
read_test_auroc() {
    python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "metrics.json"
if not p.is_file():
    print(""); raise SystemExit
try:
    d = json.loads(p.read_text(encoding="utf-8"))
    v = (d.get("test_metrics") or {}).get("auroc")
    print(f"{float(v):.6f}" if v is not None else "")
except Exception:
    print("")
PY
}

# ============================================================================
# 1) lite-combo (drop relation+attention) — alpha=0.2 seed42
# ============================================================================
LITE_A02_SEED42_OUT="${OUTPUT_ROOT}/r2plus1d_lite_combo_alpha02_strict_seed42"

run_lite_combo_alpha02_seed42() {
    run_cmd "r2plus1d_lite_combo_alpha02_strict_seed42" "${LITE_A02_SEED42_OUT}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${LITE_A02_SEED42_OUT}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_RUN_DIR}" \
            --distill-methods "logits,fused,hint,ct,text" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,ct=0.5,text=0.25" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.2 \
            --distillation-temperature 4.0 \
            --seed 42 \
            --batch-size "${LITE_BATCH_SIZE}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 2) lite-combo alpha=0.1 seed42
# ============================================================================
LITE_A01_SEED42_OUT="${OUTPUT_ROOT}/r2plus1d_lite_combo_alpha01_strict_seed42"

run_lite_combo_alpha01_seed42() {
    run_cmd "r2plus1d_lite_combo_alpha01_strict_seed42" "${LITE_A01_SEED42_OUT}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${LITE_A01_SEED42_OUT}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_RUN_DIR}" \
            --distill-methods "logits,fused,hint,ct,text" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,ct=0.5,text=0.25" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.1 \
            --distillation-temperature 4.0 \
            --seed 42 \
            --batch-size "${LITE_BATCH_SIZE}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 3) logits-only strict 多 seed (43/44/45) — 补 outputs0529 的 seed42
# ============================================================================
run_logits_only_seed() {
    local seed="$1"
    local outdir="${OUTPUT_ROOT}/r2plus1d_logits_only_strict_seed${seed}"
    run_cmd "r2plus1d_logits_only_strict_seed${seed}" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_RUN_DIR}" \
            --distill-methods "logits" \
            --distill-feature-loss cosine \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --seed "${seed}" \
            --batch-size 1 \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ============================================================================
# 4) (条件) lite-combo alpha=0.2 seeds 43/44/45 — 仅在 seed42 ≥ COND_AUROC_THRESH 时跑
# ============================================================================
run_lite_combo_alpha02_seed() {
    local seed="$1"
    local outdir="${OUTPUT_ROOT}/r2plus1d_lite_combo_alpha02_strict_seed${seed}"
    run_cmd "r2plus1d_lite_combo_alpha02_strict_seed${seed}" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model r2plus1d_18 \
            --teacher-run-dir "${TEACHER_RUN_DIR}" \
            --distill-methods "logits,fused,hint,ct,text" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,ct=0.5,text=0.25" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.2 \
            --distillation-temperature 4.0 \
            --seed "${seed}" \
            --batch-size "${LITE_BATCH_SIZE}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ----------------------------- 执行顺序 -------------------------------------
# 步骤 1+2 与步骤 3 之间无依赖，但为了便于看 log，按顺序串跑。
# 步骤 4 是条件性的，依赖步骤 1 的产物。

run_lite_combo_alpha02_seed42 || true
run_lite_combo_alpha01_seed42 || true

for s in 43 44 45; do
    run_logits_only_seed "${s}" || true
done

# 条件分支 — 默认关闭，需 RUN_CONDITIONAL_SEEDS=1 才启动
if [[ "${RUN_CONDITIONAL_SEEDS}" == "1" ]]; then
    auc="$(read_test_auroc "${LITE_A02_SEED42_OUT}")"
    if [[ -z "${auc}" ]]; then
        echo "[COND] cannot read AUROC from ${LITE_A02_SEED42_OUT}; skip conditional seeds."
    else
        awk_keep=$(awk -v a="${auc}" -v t="${COND_AUROC_THRESH}" 'BEGIN { print (a+0 >= t+0) ? 1 : 0 }')
        if [[ "${awk_keep}" == "1" ]]; then
            echo "[COND] lite-combo alpha=0.2 seed42 AUROC=${auc} ≥ ${COND_AUROC_THRESH} — launching seeds 43/44/45."
            for s in 43 44 45; do
                run_lite_combo_alpha02_seed "${s}" || true
            done
        else
            echo "[COND] lite-combo alpha=0.2 seed42 AUROC=${auc} < ${COND_AUROC_THRESH} — skip seeds 43/44/45."
        fi
    fi
else
    echo "[COND] RUN_CONDITIONAL_SEEDS=0 — conditional lite seeds 43/44/45 not attempted."
fi

# ----------------------------- 汇总 -----------------------------------------
echo ""
echo "============================================================"
echo "All rescue runs done (or skipped). Generating summary..."
echo "============================================================"

SUMMARY_PY="${PROJECT_ROOT}/experiments/analysis/rescue_metrics_summary.py"
if [[ -f "${SUMMARY_PY}" ]]; then
    python3 "${SUMMARY_PY}" \
        --root "${OUTPUT_ROOT}" \
        --reference-manifest "${REF_MANIFEST}" \
        --baseline-dir "outputs0529"
else
    echo "[WARN] ${SUMMARY_PY} not found — skipping summary generation."
fi

echo ""
echo "Summary outputs (if SUMMARY_PY ran):"
echo "  ${OUTPUT_ROOT}/rescue_metrics.csv"
echo "  ${OUTPUT_ROOT}/rescue_metrics.md"
echo "  ${OUTPUT_ROOT}/rescue_summary.md"
echo "Logs:"
echo "  ${LOG_DIR}/"
echo "Done."
