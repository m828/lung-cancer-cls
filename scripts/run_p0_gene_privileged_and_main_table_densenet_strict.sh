#!/usr/bin/env bash
# scripts/run_p0_gene_privileged_and_main_table_densenet_strict.sh
#
# P0 一键补跑：主表同构化 + 基因特权信息消融 + 5 seed + teacher 多 seed
#
# 实验清单：
#   P0-1: DenseNet3D121 CT+Text supervised strict bs=4 seeds 42-45  (Group A)
#   P0-2: CT+Text teacher strict seed42                             (Group B)
#   P0-3: DenseNet3D121 CT+Text student KD from CT+Text teacher bs=4 seeds 42-45  (Group C)
#   P0-4: DenseNet3D121 CT+Text student KD from CT+CNV+Text teacher bs=4 seed46   (5 seed)
#   P0-5: CT+CNV+Text teacher strict seeds 43-45                    (teacher multi-seed)
#
# 运行方式：
#   bash scripts/run_p0_gene_privileged_and_main_table_densenet_strict.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_MM="${PROJECT_ROOT}/train_multimodal.py"
TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"

EXPECTED="1019,652,163,204"

# ========================= RESULTS_ROOT 双环境自适应检测 =========================
# (1) 本地归档环境：结果目录在仓库上一级 ${PROJECT_ROOT}/..
# (2) 内网训练环境：结果目录就在仓库根 ${PROJECT_ROOT} 下
# 不写死；按 marker 文件自动选择；可用环境变量 RESULTS_ROOT 手动覆盖。
RESULTS_ROOT="${RESULTS_ROOT:-}"
PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"

count_split() {
    python3 - <<PY "$1"
import csv, sys
p = sys.argv[1]
cnt = {'train':0,'val':0,'test':0}; total = 0
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
    if [[ -f "${PROJECT_ROOT}/${PREFERRED_REL}" ]]; then RESULTS_ROOT="${PROJECT_ROOT}"; return 0; fi
    if [[ -f "${PARENT_ROOT}/${PREFERRED_REL}" ]];  then RESULTS_ROOT="${PARENT_ROOT}";  return 0; fi
    local cand f c
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        while IFS= read -r f; do
            c="$(count_split "$f")" || continue
            if [[ "${c}" == "${EXPECTED}" ]]; then RESULTS_ROOT="${cand}"; return 0; fi
        done < <(find "${cand}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    done
    return 1
}

if [[ -z "${RESULTS_ROOT}" ]]; then
    if ! detect_results_root; then
        echo "[FATAL] RESULTS_ROOT auto-detect failed (looked under ${PROJECT_ROOT} and ${PARENT_ROOT})." >&2
        echo "        Set RESULTS_ROOT=/path/to/results manually and re-run." >&2
        exit 1
    fi
fi

OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/outputs0531_gene_privileged_ablation}"
OUTPUT_ROOT="${OUT_ROOT}"
LOG_DIR="${OUTPUT_ROOT}/logs"
SCRIPTS_USED_DIR="${OUTPUT_ROOT}/scripts_used"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SCRIPTS_USED_DIR}"
cp -f "${BASH_SOURCE[0]}" "${SCRIPTS_USED_DIR}/run_p0_gene_privileged_and_main_table_densenet_strict.sh.snapshot"

# ========================= 参考 split =========================
REF_MANIFEST=""
resolve_ref_manifest() {
    local cands=(
        "${RESULTS_ROOT}/${PREFERRED_REL}"
        "${RESULTS_ROOT}/outputs0529/ct_cnv_text_teacher_strict_ref1019/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42/split_manifest.csv"
    )
    local cand c f
    for cand in "${cands[@]}"; do
        if [[ -f "${cand}" ]]; then
            c="$(count_split "${cand}")" || true
            [[ "${c}" == "${EXPECTED}" ]] && { REF_MANIFEST="${cand}"; return 0; }
        fi
    done
    while IFS= read -r f; do
        c="$(count_split "$f")" || continue
        [[ "${c}" == "${EXPECTED}" ]] && { REF_MANIFEST="$f"; return 0; }
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
    "outputs0529/ct_cnv_text_teacher_strict_ref1019",
]
keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv", "gene_tsv"]
out = {k: "" for k in keys}
for run in candidates:
    p = Path(root) / run / "metrics.json"
    if not p.is_file(): continue
    try: d = json.loads(p.read_text(encoding="utf-8"))
    except: continue
    cfg = d.get("config") or {}
    for k in keys:
        if not out[k]:
            v = cfg.get(k)
            if v: out[k] = str(v)
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
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"
GENE_TSV="${GENE_TSV:-${INFERRED[gene_tsv]:-}}"

# ========================= Teacher 路径 =========================
TEACHER_CNV_DIR=""
for cand in \
    "${RESULTS_ROOT}/outputs0529/ct_cnv_text_teacher_strict_ref1019" \
    "${RESULTS_ROOT}/outputs/text_strict_ref1019_rerun/ct_cnv_text_teacher_strict_ref1019"; do
    if [[ -f "${cand}/metrics.json" ]]; then TEACHER_CNV_DIR="${cand}"; break; fi
done
TEACHER_CT_TEXT_DIR="${OUTPUT_ROOT}/ct_text_teacher_strict_ref1019_seed42"

GROUP_D_SRC_BASE="${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4"
group_d_found=0
for seed in 42 43 44 45; do
    [[ -f "${GROUP_D_SRC_BASE}/densenet3d_121_full_combo_strict_seed${seed}/metrics.json" ]] && group_d_found=$((group_d_found+1))
done
yesno() { [[ -e "$1" ]] && echo "yes" || echo "no"; }

# ========================= Banner =========================
echo "============================================================"
echo "P0: Gene privileged ablation + main table + 5 seed + teacher"
echo "============================================================"
echo "PROJECT_ROOT    : ${PROJECT_ROOT}"
echo "RESULTS_ROOT    : ${RESULTS_ROOT}"
echo "OUT_ROOT        : ${OUT_ROOT}"
echo "REF_MANIFEST    : ${REF_MANIFEST}"
echo "TEACHER_CNV_DIR : ${TEACHER_CNV_DIR}"
echo "TEACHER_CT_TEXT : ${TEACHER_CT_TEXT_DIR}"
echo ""
echo "Environment detection:"
printf "  %-40s : %s\n" "outputs/ exists"                  "$(yesno "${RESULTS_ROOT}/outputs")"
printf "  %-40s : %s\n" "outputs0529/ exists"              "$(yesno "${RESULTS_ROOT}/outputs0529")"
printf "  %-40s : %s\n" "outputs0530_backbone_swap_bs_4/ exists" "$(yesno "${GROUP_D_SRC_BASE}")"
printf "  %-40s : %s\n" "Group D reuse dirs found (of 4)"  "${group_d_found}"
printf "  %-40s : %s\n" "strict CT+CNV+Text teacher found" "$([[ -n "${TEACHER_CNV_DIR}" ]] && echo "yes" || echo "no")"
echo ""
echo "DATA_ROOT       : ${DATA_ROOT}"
echo "METADATA_CSV    : ${METADATA_CSV}"
echo "CT_ROOT         : ${CT_ROOT}"
echo "TEXT_FEATURE_TSV: ${TEXT_FEATURE_TSV}"
echo "GENE_TSV        : ${GENE_TSV}"
echo "============================================================"

missing=()
[[ -z "${DATA_ROOT}" ]] && missing+=("DATA_ROOT")
[[ -z "${METADATA_CSV}" ]] && missing+=("METADATA_CSV")
[[ -z "${CT_ROOT}" ]] && missing+=("CT_ROOT")
[[ -z "${TEXT_FEATURE_TSV}" ]] && missing+=("TEXT_FEATURE_TSV")
if (( ${#missing[@]} > 0 )); then
    echo "[FATAL] Missing: ${missing[*]}"; exit 1
fi
if [[ -z "${TEACHER_CNV_DIR}" ]]; then
    echo "[FATAL] CT+CNV+Text teacher not found"; exit 1
fi

# ========================= 工具函数 =========================
already_done() { [[ -f "$1/metrics.json" ]]; }

# validate_run <outdir> <want_ct_model> <want_bs> <want_modalities> <teacher_kind>
#   want_modalities: "ct,text" | "ct,cnv,text"
#   teacher_kind:    none | ct_text | ct_cnv_text
# 返回 0 = 满足(可安全 skip); 1 = CONFIG_MISMATCH; 2 = 不可读
validate_run() {
    python3 - "$@" <<'PY'
import json, sys, csv
from pathlib import Path
outdir, want_ct, want_bs, want_mods_s, teacher_kind = sys.argv[1:6]
want_mods = set(x for x in want_mods_s.split(",") if x)
d = Path(outdir)
try:
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
except Exception as e:
    print(f"UNREADABLE metrics.json: {e}"); sys.exit(2)
cfg = m.get("config") or {}
problems = []
def check(c, msg):
    if not c: problems.append(msg)

check(str(cfg.get("ct_model")) == want_ct, f"ct_model={cfg.get('ct_model')} != {want_ct}")
check(str(cfg.get("batch_size")) == str(want_bs), f"batch_size={cfg.get('batch_size')} != {want_bs}")
mods = cfg.get("modalities") or m.get("modalities")
mset = set(mods) if isinstance(mods, (list, tuple)) else set(str(mods).split(","))
check(mset == want_mods, f"modalities={sorted(mset)} != {sorted(want_mods)}")
check(bool(cfg.get("strict_no_leakage")) is True, "strict_no_leakage != true")
check(bool(cfg.get("disable_text_numeric_features")) is True, "disable_text_numeric_features != true")
# num__ text features used == 0
dims = m.get("modality_feature_dims") or {}
if "text_num" in dims:
    check(int(dims.get("text_num") or 0) == 0, f"text_num={dims.get('text_num')} != 0")
# teacher
tmods = m.get("teacher_modalities")
tset = set(tmods) if isinstance(tmods, (list, tuple)) else (set(str(tmods).split(",")) if tmods else set())
tdir = m.get("teacher_run_dir") or cfg.get("teacher_run_dir")
if teacher_kind == "none":
    check(not tdir, f"unexpected teacher_run_dir={tdir}")
elif teacher_kind == "ct_text":
    check(bool(tdir), "missing teacher_run_dir")
    check("cnv" not in tset, f"teacher_modalities={sorted(tset)} should NOT contain cnv")
elif teacher_kind == "ct_cnv_text":
    check(bool(tdir), "missing teacher_run_dir")
    check({"ct","cnv","text"}.issubset(tset), f"teacher_modalities={sorted(tset)} must contain ct,cnv,text")
# split + predictions
tp = d / "test_predictions.csv"
if tp.is_file():
    with tp.open(encoding="utf-8-sig", newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1
    check(n == 204, f"test_predictions.csv rows={n} != 204")
else:
    problems.append("test_predictions.csv missing")
check((d / "text_feature_audit.json").is_file(), "text_feature_audit.json missing")
check((d / "leakage_warnings.json").is_file(), "leakage_warnings.json missing")

if problems:
    print("CONFIG_MISMATCH: " + "; ".join(problems)); sys.exit(1)
print("OK"); sys.exit(0)
PY
}

run_cmd() {
    local name="$1"; shift; local outdir="$1"; shift
    local want_ct="$1"; shift; local want_bs="$1"; shift
    local want_mods="$1"; shift; local teacher_kind="$1"; shift
    local logf="${LOG_DIR}/${name}.log"
    if already_done "${outdir}"; then
        local vmsg vrc
        vmsg="$(validate_run "${outdir}" "${want_ct}" "${want_bs}" "${want_mods}" "${teacher_kind}")"; vrc=$?
        if (( vrc == 0 )); then
            echo "[SKIP] ${name} — metrics.json exists and config validated OK"
            return 0
        else
            echo "[CONFIG_MISMATCH] ${name} — existing run does NOT meet requirements; NOT overwriting."
            echo "    ${outdir}"
            echo "    ${vmsg}"
            echo "    -> Inspect/move it, or set a different OUT_ROOT, then re-run."
            return 0
        fi
    fi
    mkdir -p "${outdir}"
    echo ""; echo "------------------------------------------------------------"
    echo "[RUN]  ${name}"; echo "       output: ${outdir}"; echo "       log: ${logf}"
    echo "------------------------------------------------------------"
    "$@" 2>&1 | tee "${logf}"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then echo "[FAIL] ${name} exit ${rc}"
    else echo "[OK]   ${name}"; fi
    return ${rc}
}

verify_run() {
    local outdir="$1"
    local name="$2"
    local ok=true
    [[ ! -f "${outdir}/metrics.json" ]] && echo "  [WARN] ${name}: missing metrics.json" && ok=false
    [[ ! -f "${outdir}/split_manifest.csv" ]] && echo "  [WARN] ${name}: missing split_manifest.csv" && ok=false
    [[ ! -f "${outdir}/test_predictions.csv" ]] && echo "  [WARN] ${name}: missing test_predictions.csv" && ok=false
    [[ ! -f "${outdir}/text_feature_audit.json" ]] && echo "  [WARN] ${name}: missing text_feature_audit.json" && ok=false
    [[ ! -f "${outdir}/leakage_warnings.json" ]] && echo "  [WARN] ${name}: missing leakage_warnings.json" && ok=false
    $ok && echo "  [OK] ${name} verified"
}

# ========================= 通用训练参数 =========================
COMMON_ARGS=(
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
DISTILL_METHODS="logits,fused,hint,relation,attention,ct,text"
DISTILL_WEIGHTS="logits=1,fused=0.5,hint=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25"

# ========================= P0-5: CT+CNV+Text teacher seeds 43-45 =========================
echo ""
echo "==================== P0-5: CT+CNV+Text teacher seeds 43-45 ===================="
for seed in 43 44 45; do
    tag="ct_cnv_text_teacher_strict_ref1019_seed${seed}"
    outdir="${OUTPUT_ROOT}/${tag}"
    run_cmd "${tag}" "${outdir}" "resnet3d18" "2" "ct,cnv,text" "none" \
        python3 "${TRAIN_MM}" \
            --output-dir "${outdir}" \
            --modalities ct,cnv,text \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            --gene-tsv "${GENE_TSV}" \
            --ct-model resnet3d18 \
            --batch-size 2 \
            --seed "${seed}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" || true
    verify_run "${outdir}" "${tag}"
done

# ========================= P0-2: CT+Text teacher strict =========================
echo ""
echo "==================== P0-2: CT+Text teacher strict seed42 ===================="
run_cmd "ct_text_teacher_strict_ref1019_seed42" "${TEACHER_CT_TEXT_DIR}" "resnet3d18" "2" "ct,text" "none" \
    python3 "${TRAIN_MM}" \
        --output-dir "${TEACHER_CT_TEXT_DIR}" \
        --modalities ct,text \
        --reference-manifest "${REF_MANIFEST}" \
        --data-root "${DATA_ROOT}" \
        --metadata-csv "${METADATA_CSV}" \
        --ct-root "${CT_ROOT}" \
        --text-feature-tsv "${TEXT_FEATURE_TSV}" \
        --ct-model resnet3d18 \
        --batch-size 2 \
        --seed 42 \
        "${COMMON_ARGS[@]}" \
        "${STRICT_ARGS[@]}" || true
verify_run "${TEACHER_CT_TEXT_DIR}" "ct_text_teacher_strict_ref1019_seed42"

# ========================= P0-1: DenseNet3D121 CT+Text supervised =========================
echo ""
echo "==================== P0-1: DenseNet3D121 CT+Text supervised bs=4 ===================="
for seed in 42 43 44 45; do
    tag="ct_text_sc_densenet3d121_strict_bs4_seed${seed}"
    outdir="${OUTPUT_ROOT}/${tag}"
    run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "none" \
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
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" || true
    verify_run "${outdir}" "${tag}"
done

# ========================= P0-3: KD from CT+Text teacher =========================
echo ""
echo "==================== P0-3: DenseNet3D121 KD from CT+Text teacher bs=4 ===================="
if [[ -f "${TEACHER_CT_TEXT_DIR}/metrics.json" ]]; then
    for seed in 42 43 44 45; do
        tag="densenet3d121_kd_from_ct_text_teacher_bs4_seed${seed}"
        outdir="${OUTPUT_ROOT}/${tag}"
        run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "ct_text" \
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
                --distill-method-weights "${DISTILL_WEIGHTS}" \
                --distill-feature-loss cosine \
                --distill-normalize-features \
                --distillation-alpha 0.5 \
                --distillation-temperature 4.0 \
                --batch-size 4 \
                --seed "${seed}" \
                "${COMMON_ARGS[@]}" \
                "${STRICT_ARGS[@]}" || true
        verify_run "${outdir}" "${tag}"
    done
else
    echo "[SKIP] P0-3 — CT+Text teacher not ready"
fi

# ========================= P0-4: seed46 for 5-seed main result =========================
echo ""
echo "==================== P0-4: DenseNet3D121 KD from CT+CNV+Text teacher seed46 ===================="
tag="densenet3d121_kd_from_ct_cnv_text_teacher_bs4_seed46"
outdir="${OUTPUT_ROOT}/${tag}"
run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "ct_cnv_text" \
    python3 "${TRAIN_KD}" \
        --output-dir "${outdir}" \
        --modalities ct,text \
        --reference-manifest "${REF_MANIFEST}" \
        --data-root "${DATA_ROOT}" \
        --metadata-csv "${METADATA_CSV}" \
        --ct-root "${CT_ROOT}" \
        --text-feature-tsv "${TEXT_FEATURE_TSV}" \
        --ct-model densenet3d_121 \
        --teacher-run-dir "${TEACHER_CNV_DIR}" \
        --distill-methods "${DISTILL_METHODS}" \
        --distill-method-weights "${DISTILL_WEIGHTS}" \
        --distill-feature-loss cosine \
        --distill-normalize-features \
        --distillation-alpha 0.5 \
        --distillation-temperature 4.0 \
        --batch-size 4 \
        --seed 46 \
        "${COMMON_ARGS[@]}" \
        "${STRICT_ARGS[@]}" || true
verify_run "${outdir}" "${tag}"

# ========================= Link existing Group D =========================
echo ""
echo "==================== Link existing Group D ===================="
for seed in 42 43 44 45; do
    src="${GROUP_D_SRC_BASE}/densenet3d_121_full_combo_strict_seed${seed}"
    dst="${OUTPUT_ROOT}/densenet3d121_kd_from_ct_cnv_text_teacher_bs4_seed${seed}"
    if [[ -f "${src}/metrics.json" ]] && [[ ! -e "${dst}" ]]; then
        ln -s "${src}" "${dst}"
        echo "[LINK] seed${seed} -> ${src}"
    elif [[ -e "${dst}" ]]; then
        echo "[SKIP] seed${seed} — link exists"
    fi
done

# ========================= Summary =========================
echo ""
echo "============================================================"
echo "P0 runs complete. Verifying..."
echo "============================================================"
for d in "${OUTPUT_ROOT}"/*/; do
    name=$(basename "$d")
    [[ "$name" == "logs" || "$name" == "scripts_used" ]] && continue
    verify_run "$d" "$name"
done

echo ""
echo "Run analysis: python3 experiments/analysis/analyze_p0_gene_privileged_ablation.py --root ${OUTPUT_ROOT}"
echo "Done."
