#!/usr/bin/env bash
# scripts/run_gene_privileged_ablation_densenet_strict.sh
#
# Gene privileged information ablation — DenseNet3D121 strict.
#
# 背景：当前 strict 主候选为 DenseNet3D121 + CT+Text + full-combo KD + bs=4。
# 本实验回答：基因特权信息 (CNV) 在 teacher 训练中是否带来额外增益。
#
# 四组实验：
#   Group A: DenseNet3D121 CT+Text supervised strict (无 KD, 无基因) — 4 seeds
#   Group B: CT+Text teacher strict (无 CNV) — seed42
#   Group C: DenseNet3D121 CT+Text student KD from CT+Text teacher — 4 seeds
#   Group D: DenseNet3D121 CT+Text student KD from CT+CNV+Text teacher — 4 seeds (复用 0530)
#
# 运行方式：
#   bash scripts/run_gene_privileged_ablation_densenet_strict.sh
#
# 完成后调用 experiments/analysis/analyze_gene_privileged_ablation.py 生成汇总。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_MM="${PROJECT_ROOT}/train_multimodal.py"
TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"

EXPECTED="1019,652,163,204"

# ========================= RESULTS_ROOT 双环境自适应检测 =========================
# 两套目录结构：
#   (1) 本地归档环境：仓库在 ${PROJECT_ROOT}，结果目录在上一级 ${PROJECT_ROOT}/..
#   (2) 内网训练环境：结果目录就在仓库根 ${PROJECT_ROOT} 下
# 不写死，按 marker 文件存在性自动选择；也可通过环境变量 RESULTS_ROOT 手动覆盖。
RESULTS_ROOT="${RESULTS_ROOT:-}"
PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"

count_split() {
    python3 - <<PY "$1"
import csv, sys
p = sys.argv[1]
cnt = {'train':0,'val':0,'test':0}
total = 0
try:
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
except Exception:
    print('0,0,0,0'); sys.exit(0)
print(f"{total},{cnt['train']},{cnt['val']},{cnt['test']}")
PY
}

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
detect_results_root() {
    # 优先级 1：marker 文件存在性
    if [[ -f "${PROJECT_ROOT}/${PREFERRED_REL}" ]]; then
        RESULTS_ROOT="${PROJECT_ROOT}"; return 0
    fi
    if [[ -f "${PARENT_ROOT}/${PREFERRED_REL}" ]]; then
        RESULTS_ROOT="${PARENT_ROOT}"; return 0
    fi
    # 优先级 2：在两个候选根下递归搜 1019/652/163/204 的 split_manifest.csv
    local cand
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        local f c
        while IFS= read -r f; do
            c="$(count_split "$f")" || continue
            if [[ "${c}" == "${EXPECTED}" ]]; then
                RESULTS_ROOT="${cand}"; return 0
            fi
        done < <(find "${cand}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    done
    return 1
}

if [[ -z "${RESULTS_ROOT}" ]]; then
    if ! detect_results_root; then
        echo "[FATAL] RESULTS_ROOT auto-detect failed." >&2
        echo "        Looked under: ${PROJECT_ROOT} and ${PARENT_ROOT}" >&2
        echo "        Set RESULTS_ROOT=/path/to/results manually and re-run." >&2
        exit 1
    fi
fi

OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/outputs0531_gene_privileged_ablation}"
OUTPUT_ROOT="${OUT_ROOT}"
LOG_DIR="${OUTPUT_ROOT}/logs"
SCRIPTS_USED_DIR="${OUTPUT_ROOT}/scripts_used"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SCRIPTS_USED_DIR}"
cp -f "${BASH_SOURCE[0]}" "${SCRIPTS_USED_DIR}/run_gene_privileged_ablation_densenet_strict.sh.snapshot"

# ========================= 参考 split =========================
REF_MANIFEST_PREFERRED="${RESULTS_ROOT}/${PREFERRED_REL}"
REF_MANIFEST=""

# count_split() 已在 RESULTS_ROOT 检测段定义并复用。

resolve_ref_manifest() {
    # 候选优先级：preferred -> 0529 teacher -> 0530 densenet seed42 -> 递归搜索
    local cands=(
        "${REF_MANIFEST_PREFERRED}"
        "${RESULTS_ROOT}/outputs0529/ct_cnv_text_teacher_strict_ref1019/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42/split_manifest.csv"
    )
    local cand c
    for cand in "${cands[@]}"; do
        if [[ -f "${cand}" ]]; then
            c="$(count_split "${cand}")" || true
            if [[ "${c}" == "${EXPECTED}" ]]; then
                REF_MANIFEST="${cand}"; return 0
            fi
        fi
    done
    local f
    while IFS= read -r f; do
        c="$(count_split "$f")" || continue
        if [[ "${c}" == "${EXPECTED}" ]]; then
            REF_MANIFEST="$f"; return 0
        fi
    done < <(find "${RESULTS_ROOT}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    return 1
}

if ! resolve_ref_manifest; then
    echo "[FATAL] no ${EXPECTED} split_manifest.csv found under ${RESULTS_ROOT}." >&2; exit 1
fi

# ========================= 数据路径推断 =========================
infer_paths_py='
import json, sys
from pathlib import Path
root = sys.argv[1]
candidates = [
    "outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42",
    "outputs0530_backbone_swap/densenet3d_121_full_combo_strict_seed42",
    "outputs0529/ct_cnv_text_teacher_strict_ref1019",
    "outputs/ct_cnv_text_teacher_mvn_tvt",
]
keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv", "gene_tsv"]
out = {k: "" for k in keys}
source = {k: "" for k in keys}
for run in candidates:
    p = Path(root) / run / "metrics.json"
    if not p.is_file(): continue
    try: d = json.loads(p.read_text(encoding="utf-8"))
    except: continue
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
done < <(python3 -c "${infer_paths_py}" "${RESULTS_ROOT}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"
GENE_TSV="${GENE_TSV:-${INFERRED[gene_tsv]:-}}"
PATIENT_ID_COL="${PATIENT_ID_COL:-}"

# ========================= teacher 路径 =========================
# CT+CNV+Text teacher (已有)
TEACHER_CNV_CANDIDATES=(
    "${RESULTS_ROOT}/outputs0529/ct_cnv_text_teacher_strict_ref1019"
    "${RESULTS_ROOT}/outputs/text_strict_ref1019_rerun/ct_cnv_text_teacher_strict_ref1019"
    "${RESULTS_ROOT}/outputs/ct_cnv_text_teacher_strict_ref1019"
)
TEACHER_CNV_DIR=""
for cand in "${TEACHER_CNV_CANDIDATES[@]}"; do
    if [[ -f "${cand}/metrics.json" ]]; then
        TEACHER_CNV_DIR="${cand}"; break
    fi
done

# CT+Text teacher (待训练, Group B)
TEACHER_CT_TEXT_DIR="${OUTPUT_ROOT}/ct_text_teacher_strict_ref1019_seed42"

# Group D 复用源目录 (CT+CNV+Text teacher KD, 已有结果)
GROUP_D_SRC_BASE="${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4"
group_d_found=0
for seed in 42 43 44 45; do
    [[ -f "${GROUP_D_SRC_BASE}/densenet3d_121_full_combo_strict_seed${seed}/metrics.json" ]] && group_d_found=$((group_d_found+1))
done

yesno() { [[ -e "$1" ]] && echo "yes" || echo "no"; }

# ========================= banner =========================
echo "============================================================"
echo "Gene privileged ablation — DenseNet3D121 strict"
echo "============================================================"
echo "PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "RESULTS_ROOT       : ${RESULTS_ROOT}"
echo "OUT_ROOT           : ${OUT_ROOT}"
echo "REF_MANIFEST       : ${REF_MANIFEST}"
echo "TEACHER_CNV_DIR    : ${TEACHER_CNV_DIR}"
echo "TEACHER_CT_TEXT_DIR: ${TEACHER_CT_TEXT_DIR}"
echo ""
echo "Environment detection:"
printf "  %-40s : %s\n" "outputs/ exists"                  "$(yesno "${RESULTS_ROOT}/outputs")"
printf "  %-40s : %s\n" "outputs0529/ exists"              "$(yesno "${RESULTS_ROOT}/outputs0529")"
printf "  %-40s : %s\n" "outputs0530_backbone_swap_bs_4/ exists" "$(yesno "${GROUP_D_SRC_BASE}")"
printf "  %-40s : %s\n" "Group D reuse dirs found (of 4)"  "${group_d_found}"
printf "  %-40s : %s\n" "strict CT+CNV+Text teacher found" "$([[ -n "${TEACHER_CNV_DIR}" ]] && echo "yes" || echo "no")"
echo ""
echo "Data paths:"
printf "  %-18s = %s\n" "DATA_ROOT"        "${DATA_ROOT}"
printf "  %-18s = %s\n" "METADATA_CSV"     "${METADATA_CSV}"
printf "  %-18s = %s\n" "CT_ROOT"          "${CT_ROOT}"
printf "  %-18s = %s\n" "TEXT_FEATURE_TSV" "${TEXT_FEATURE_TSV}"
printf "  %-18s = %s\n" "GENE_TSV"         "${GENE_TSV}"
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
if [[ -z "${TEACHER_CNV_DIR}" ]]; then
    echo "[FATAL] CT+CNV+Text teacher not found." >&2; exit 1
fi

# ========================= 工具函数 =========================
patient_args() {
    if [[ -n "${PATIENT_ID_COL}" ]]; then echo "--patient-id-col ${PATIENT_ID_COL}"; fi
}
read -r -a PATIENT_ARGS <<<"$(patient_args)"

already_done() { [[ -f "$1/metrics.json" ]]; }

# 校验已有 run 的 config 是否满足目标要求。
# 用法: validate_run <outdir> <ct_model> <batch_size> <teacher_kind:none|ct_text|ct_cnv_text>
# 返回 0 = 满足(可安全 skip); 1 = CONFIG_MISMATCH; 2 = 不可读
validate_run() {
    python3 - "$@" <<'PY'
import json, sys, csv
from pathlib import Path
outdir, want_ct, want_bs, teacher_kind = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = Path(outdir)
mp = d / "metrics.json"
problems = []
try:
    m = json.loads(mp.read_text(encoding="utf-8"))
except Exception as e:
    print(f"UNREADABLE metrics.json: {e}"); sys.exit(2)
cfg = m.get("config") or {}

def check(cond, msg):
    if not cond: problems.append(msg)

# ct_model / batch_size / modalities
check(str(cfg.get("ct_model")) == want_ct, f"ct_model={cfg.get('ct_model')} != {want_ct}")
check(str(cfg.get("batch_size")) == str(want_bs), f"batch_size={cfg.get('batch_size')} != {want_bs}")
mods = cfg.get("modalities") or m.get("modalities")
mset = set(mods) if isinstance(mods, (list, tuple)) else set(str(mods).split(","))
check(mset == {"ct", "text"}, f"modalities={sorted(mset)} != ['ct','text']")
# strict flags
check(bool(cfg.get("strict_no_leakage")) is True, "strict_no_leakage != true")
check(bool(cfg.get("disable_text_numeric_features")) is True, "disable_text_numeric_features != true")
# num__ features used == 0 (via modality_feature_dims.text_num)
dims = m.get("modality_feature_dims") or {}
if "text_num" in dims:
    check(int(dims.get("text_num") or 0) == 0, f"text_num={dims.get('text_num')} != 0")
# teacher
tmods = m.get("teacher_modalities")
tset = set(tmods) if isinstance(tmods, (list, tuple)) else (set(str(tmods).split(",")) if tmods else set())
tdir = m.get("teacher_run_dir") or cfg.get("teacher_run_dir")
if teacher_kind == "none":
    check(not tdir, f"unexpected teacher_run_dir={tdir} for supervised run")
elif teacher_kind == "ct_text":
    check(bool(tdir), "missing teacher_run_dir")
    check("cnv" not in tset, f"teacher_modalities={sorted(tset)} should NOT contain cnv")
elif teacher_kind == "ct_cnv_text":
    check(bool(tdir), "missing teacher_run_dir")
    check({"ct","cnv","text"}.issubset(tset), f"teacher_modalities={sorted(tset)} must contain ct,cnv,text")
# split 1019/652/163/204 + test_predictions.csv == 204
tp = d / "test_predictions.csv"
if tp.is_file():
    with tp.open(encoding="utf-8-sig", newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1
    check(n == 204, f"test_predictions.csv rows={n} != 204")
else:
    problems.append("test_predictions.csv missing")
# audit / leakage files
check((d / "text_feature_audit.json").is_file(), "text_feature_audit.json missing")
check((d / "leakage_warnings.json").is_file(), "leakage_warnings.json missing")

if problems:
    print("CONFIG_MISMATCH: " + "; ".join(problems)); sys.exit(1)
print("OK"); sys.exit(0)
PY
}

run_cmd() {
    local name="$1"; shift
    local outdir="$1"; shift
    local want_ct="$1"; shift
    local want_bs="$1"; shift
    local teacher_kind="$1"; shift
    local logf="${LOG_DIR}/${name}.log"
    if already_done "${outdir}"; then
        local vmsg vrc
        vmsg="$(validate_run "${outdir}" "${want_ct}" "${want_bs}" "${teacher_kind}")"; vrc=$?
        if (( vrc == 0 )); then
            echo "[SKIP] ${name} — metrics.json exists and config validated OK"
            return 0
        else
            echo "[CONFIG_MISMATCH] ${name} — existing run does NOT meet requirements; NOT overwriting."
            echo "    ${outdir}"
            echo "    ${vmsg}"
            echo "    -> Inspect/move it manually, or set a different OUT_ROOT, then re-run."
            return 0
        fi
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

# ========================= 通用训练参数 =========================
# 与 outputs0530_backbone_swap_bs_4/densenet3d_121 完全对齐
COMMON_SUPERVISED_ARGS=(
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
)
STRICT_ARGS=(--strict-no-leakage --disable-text-numeric-features)

# KD 配方 — 与 outputs0530_backbone_swap_bs_4/densenet3d_121 完全一致
DISTILL_METHODS="logits,fused,hint,relation,attention,ct,text"
DISTILL_METHOD_WEIGHTS="logits=1,fused=0.5,hint=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25"
DISTILL_ALPHA=0.5
DISTILL_TEMP=4.0

# ========================= Group A: DenseNet3D121 CT+Text supervised =========================
run_group_a() {
    local seed="$1"
    local tag="ct_text_sc_densenet3d121_strict_bs4_seed${seed}"
    local outdir="${OUTPUT_ROOT}/${tag}"
    run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "none" \
        python3 "${TRAIN_MM}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model densenet3d_121 \
            --batch-size 4 \
            --seed "${seed}" \
            "${COMMON_SUPERVISED_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ========================= Group B: CT+Text teacher strict =========================
run_group_b() {
    local outdir="${TEACHER_CT_TEXT_DIR}"
    run_cmd "ct_text_teacher_strict_ref1019_seed42" "${outdir}" "resnet3d18" "2" "none" \
        python3 "${TRAIN_MM}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model resnet3d18 \
            --batch-size 2 \
            --seed 42 \
            "${COMMON_SUPERVISED_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ========================= Group C: KD from CT+Text teacher =========================
run_group_c() {
    local seed="$1"
    local tag="densenet3d121_kd_from_ct_text_teacher_bs4_seed${seed}"
    local outdir="${OUTPUT_ROOT}/${tag}"
    # 确认 Group B teacher 存在
    if [[ ! -f "${TEACHER_CT_TEXT_DIR}/metrics.json" ]]; then
        echo "[SKIP] ${tag} — Group B teacher not ready at ${TEACHER_CT_TEXT_DIR}"
        return 0
    fi
    run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct_text" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --ct-model densenet3d_121 \
            --teacher-run-dir "${TEACHER_CT_TEXT_DIR}" \
            --distill-methods "${DISTILL_METHODS}" \
            --distill-method-weights "${DISTILL_METHOD_WEIGHTS}" \
            --distill-feature-loss cosine \
            --distill-normalize-features \
            --distillation-alpha "${DISTILL_ALPHA}" \
            --distillation-temperature "${DISTILL_TEMP}" \
            --batch-size 4 \
            --seed "${seed}" \
            "${COMMON_SUPERVISED_ARGS[@]}" \
            "${STRICT_ARGS[@]}" \
            "${PATIENT_ARGS[@]}"
}

# ========================= Group D: KD from CT+CNV+Text teacher (复用已有) =========================
# 已有结果: outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed{42..45}
# 链接到 OUTPUT_ROOT 下方便分析脚本统一读取
link_group_d() {
    for seed in 42 43 44 45; do
        local src="${GROUP_D_SRC_BASE}/densenet3d_121_full_combo_strict_seed${seed}"
        local dst="${OUTPUT_ROOT}/densenet3d121_kd_from_ct_cnv_text_teacher_bs4_seed${seed}"
        if [[ -f "${src}/metrics.json" ]]; then
            if [[ ! -e "${dst}" ]]; then
                ln -s "${src}" "${dst}"
                echo "[LINK] Group D seed${seed} -> ${src}"
            else
                echo "[SKIP] Group D seed${seed} — link already exists"
            fi
        else
            echo "[WARN] Group D seed${seed} — ${src}/metrics.json not found"
        fi
    done
}

# ========================= 执行 =========================
echo ""
echo "==================== Group B: CT+Text teacher strict ===================="
run_group_b || true

echo ""
echo "==================== Group A: DenseNet3D121 CT+Text supervised ===================="
for s in 42 43 44 45; do
    run_group_a "${s}" || true
done

echo ""
echo "==================== Group C: KD from CT+Text teacher ===================="
for s in 42 43 44 45; do
    run_group_c "${s}" || true
done

echo ""
echo "==================== Group D: KD from CT+CNV+Text teacher (reuse) ===================="
link_group_d

# ========================= 汇总 =========================
echo ""
echo "============================================================"
echo "All runs done. Generating summary..."
echo "============================================================"

SUMMARY_PY="${PROJECT_ROOT}/experiments/analysis/analyze_gene_privileged_ablation.py"
if [[ -f "${SUMMARY_PY}" ]]; then
    python3 "${SUMMARY_PY}" --root "${OUTPUT_ROOT}"
else
    echo "[WARN] ${SUMMARY_PY} not found — skipping summary generation."
fi

echo ""
echo "Outputs:"
echo "  ${OUTPUT_ROOT}/gene_privileged_ablation_metrics.csv"
echo "  ${OUTPUT_ROOT}/gene_privileged_ablation_metrics.md"
echo "  ${OUTPUT_ROOT}/gene_privileged_ablation_summary.md"
echo "Done."
