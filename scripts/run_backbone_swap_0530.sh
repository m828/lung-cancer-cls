#!/usr/bin/env bash
# scripts/run_backbone_swap_0530.sh
#
# Strict rescue — backbone swap.
#
# 背景：outputs0530_strict_rescue 显示 R2Plus1D × strict × bs=1 在 full-combo / lite-combo /
# logits-only 三种 KD 配方下都没法稳定 ≥0.95（最好 0.9569 / std 0.018），断言瓶颈
# 不在 KD 配方而在 R2Plus1D + strict text 的优化稳定性本身。
#
# 本脚本不改任何代码，只在 build_model() 已支持的 3D backbone 注册表中切换 --ct-model，
# 把同样的 full-combo + strict + 1019 split + bs=1 配方跑在其它 backbone 上，
# 看是否能替代 R2Plus1D 拿到「AUROC ≥ 0.95 且 4-seed std < 0.015」的主候选位置。
#
# 实验范围（默认）：
#   tier-1 — resnet3d18  : seeds 43/44/45 (补 outputs0529 seed42=0.9710，凑 4 seed)
#   tier-2 — densenet3d_121 : seeds 42/43/44/45 (公平第二选择，参数少、bs=1 友好)
#
# 可选（默认关）：
#   tier-3 — mc3_18 seed 42       (R2Plus1D 的兄弟，探一下整个家族的 strict 行为)
#   tier-4 — swin3d_tiny seed 42  (高显存，建议不要跑，bs=1+depth=128 很可能 OOM)
#
# 重要约束 (与 run_strict_rescue_0530.sh 一致)：
#   - 不覆盖 outputs0529 / outputs0530_strict_rescue
#   - 不重新随机划分，复用 1019 reference split
#   - 已存在 metrics.json 的 run 直接 skip
#   - 数据路径 / metadata / patient_id_col 不写死，自动从历史 run 推断
#   - 训练超参与 outputs0529 strict main 一致，除了 --ct-model
#   - KD 配方使用 full-combo（与 outputs0529 ResNet3D strict 0.9710 那个 run 同构）
#
# 运行方式：
#   bash scripts/run_backbone_swap_0530.sh
#       # 默认：resnet3d18 seeds 43/44/45 + densenet3d_121 seeds 42/43/44/45
#
#   RUN_MC3=1 bash scripts/run_backbone_swap_0530.sh
#       # 加跑 mc3_18 seed 42
#
#   RUN_SWIN3D=1 bash scripts/run_backbone_swap_0530.sh
#       # 加跑 swin3d_tiny seed 42（OOM 风险，自己看显存）
#
#   BATCH_SIZE=2 bash scripts/run_backbone_swap_0530.sh
#       # 显存够把 bs 改 2（注意：与 R2Plus1D strict 的 bs=1 不严格 apple-to-apple）
#
# 完成后会调用 experiments/analysis/backbone_swap_metrics_summary.py 生成 csv + md 汇总。

set -uo pipefail

# ----------------------------- 路径定位 --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"
if [[ ! -f "${TRAIN_KD}" ]]; then
    echo "[FATAL] train_student_kd.py not found at ${TRAIN_KD}"; exit 1
fi

OUTPUT_ROOT="outputs0530_backbone_swap"
LOG_DIR="${OUTPUT_ROOT}/logs"
SCRIPTS_USED_DIR="${OUTPUT_ROOT}/scripts_used"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SCRIPTS_USED_DIR}"

cp -f "${BASH_SOURCE[0]}" "${SCRIPTS_USED_DIR}/run_backbone_swap_0530.sh.snapshot"

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
            REF_MANIFEST="${REF_MANIFEST_PREFERRED}"; return 0
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
infer_paths_py='
import json, sys
from pathlib import Path
candidates = [
    "outputs0530_strict_rescue/r2plus1d_logits_only_strict_seed43",
    "outputs0529/ct_text_student_kd_resnet3d18_full_combo_strict_ref1019",
    "outputs0529/ct_text_student_kd_r2plus1d_full_combo_strict_ref1019",
    "outputs0529/ct_text_student_kd_r2plus1d_logits_only_strict_ref1019",
    "outputs0529/ct_cnv_text_teacher_strict_ref1019",
    "outputs/text_strict_ref1019_rerun/ct_text_student_kd_resnet3d18_full_combo_strict_ref1019",
    "outputs/ct_text_student_kd_mvn_resnet3d18_full_combo",
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
BATCH_SIZE="${BATCH_SIZE:-1}"             # 与 outputs0529 strict main 对齐；显存够可改 2
RUN_MC3="${RUN_MC3:-0}"                   # 加跑 mc3_18 seed42（探兄弟 backbone）
RUN_SWIN3D="${RUN_SWIN3D:-0}"             # 加跑 swin3d_tiny seed42（OOM 风险）

# ----------------------------- banner ---------------------------------------
echo "============================================================"
echo "Strict rescue — BACKBONE SWAP — outputs0530_backbone_swap"
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
echo "BATCH_SIZE         : ${BATCH_SIZE}"
echo "RUN_MC3            : ${RUN_MC3}"
echo "RUN_SWIN3D         : ${RUN_SWIN3D}"
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

# 与 outputs0529 ResNet3D18 strict full-combo run 一致的训练超参
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

# full-combo KD 配方（与 outputs0529 ResNet3D 那个 0.9710 strict run 同构）
FULL_COMBO_DISTILL_METHODS="logits,fused,hint,relation,attention,ct,text"

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

# 通用 backbone-swap run（full-combo strict）
run_backbone_full_combo() {
    local backbone="$1"
    local seed="$2"
    local tag="${backbone}_full_combo_strict_seed${seed}"
    local outdir="${OUTPUT_ROOT}/${tag}"
    run_cmd "${tag}" "${outdir}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model "${backbone}" \
            --teacher-run-dir "${TEACHER_RUN_DIR}" \
            --distill-methods "${FULL_COMBO_DISTILL_METHODS}" \
            --distill-method-weights "logits=1,fused=0.5,hint=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --seed "${seed}" \
            --batch-size "${BATCH_SIZE}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ----------------------------- 执行计划 -------------------------------------
# tier-1 resnet3d18：补 43/44/45（outputs0529 已有 seed42=0.9710，凑 4 seed）
for s in 43 44 45; do
    run_backbone_full_combo "resnet3d18" "${s}" || true
done

# tier-2 densenet3d_121：跑 42/43/44/45
for s in 42 43 44 45; do
    run_backbone_full_combo "densenet3d_121" "${s}" || true
done

# tier-3 (可选) mc3_18：仅 seed42 探路
if [[ "${RUN_MC3}" == "1" ]]; then
    run_backbone_full_combo "mc3_18" 42 || true
fi

# tier-4 (可选) swin3d_tiny：仅 seed42 探路，OOM 自负
if [[ "${RUN_SWIN3D}" == "1" ]]; then
    run_backbone_full_combo "swin3d_tiny" 42 || true
fi

# ----------------------------- 汇总 -----------------------------------------
echo ""
echo "============================================================"
echo "All backbone-swap runs done (or skipped). Generating summary..."
echo "============================================================"

SUMMARY_PY="${PROJECT_ROOT}/experiments/analysis/backbone_swap_metrics_summary.py"
if [[ -f "${SUMMARY_PY}" ]]; then
    python3 "${SUMMARY_PY}" \
        --root "${OUTPUT_ROOT}" \
        --reference-manifest "${REF_MANIFEST}" \
        --baseline-dir "outputs0529" \
        --rescue-dir "outputs0530_strict_rescue"
else
    echo "[WARN] ${SUMMARY_PY} not found — skipping summary generation."
fi

echo ""
echo "Summary outputs (if SUMMARY_PY ran):"
echo "  ${OUTPUT_ROOT}/backbone_swap_metrics.csv"
echo "  ${OUTPUT_ROOT}/backbone_swap_metrics.md"
echo "  ${OUTPUT_ROOT}/backbone_swap_summary.md"
echo "Logs:"
echo "  ${LOG_DIR}/"
echo "Done."
