#!/usr/bin/env bash
# scripts/run_teacher_homogeneous_gene_test.sh
#
# 同构 teacher / student 基因信息验证
#
# 背景：前一轮消融 (0531) 使用 ResNet3D18 teacher + DenseNet3D121 student，
# backbone 不一致导致 KD 增益归因不清。本脚本将 teacher 和 student 统一为 DenseNet3D121，
# 严格比较 CT+Text teacher vs CT+CNV+Text teacher 对 student 的影响。
#
# 实验设计：
#   Stage 1 (teacher):
#     T0: DenseNet3D121 CT+Text teacher strict         seeds 42-45
#     T1: DenseNet3D121 CT+CNV+Text teacher strict     seeds 42-45
#   Stage 2 (student):
#     S0: 复用已有 DenseNet3D121 CT+Text supervised    seeds 42-45 (Group A)
#     S1-logits:  KD from T0, logits-only              seeds 42-45
#     S2-logits:  KD from T1, logits-only              seeds 42-45
#     S1-light:   KD from T0, light-combo (no hint)    seeds 42-45
#     S2-light:   KD from T1, light-combo (no hint)    seeds 42-45
#
# KD recipe 注意：
#   - logits-only: --distill-methods logits
#   - light-combo: --distill-methods logits,fused,relation,attention,ct,text
#   - 不使用 hint，因为 teacher 有 CNV 而 student 无 CNV 时 fusion_input_dim 不匹配
#
# 运行方式：
#   bash scripts/run_teacher_homogeneous_gene_test.sh               # smoke test (seed42 only)
#   FULL_SEEDS=1 bash scripts/run_teacher_homogeneous_gene_test.sh  # seeds 42-45
#   SKIP_TEACHER=1 bash scripts/run_teacher_homogeneous_gene_test.sh  # 跳过 teacher 阶段
#   RUN_LIGHT_COMBO=1 bash scripts/run_teacher_homogeneous_gene_test.sh  # 加跑 light-combo

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_MM="${PROJECT_ROOT}/train_multimodal.py"
TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"

# ========================= RESULTS_ROOT 双环境自适应 =========================
RESULTS_ROOT="${RESULTS_ROOT:-}"
PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"
PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

resolve_results_root() {
    if [[ -n "${RESULTS_ROOT}" ]]; then return 0; fi
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        if [[ -f "${cand}/${PREFERRED_REL}" ]]; then
            RESULTS_ROOT="${cand}"; return 0
        fi
    done
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        while IFS= read -r f; do
            c="$(python3 -c "
import csv,sys
p=sys.argv[1]
cnt={'train':0,'val':0,'test':0};total=0
with open(p,'r',encoding='utf-8-sig',newline='') as fh:
    r=csv.DictReader(fh);col=None
    for c in ('assigned_split','split','Split'):
        if r.fieldnames and c in r.fieldnames: col=c;break
    if not col: print('0,0,0,0');sys.exit(0)
    for row in r:
        total+=1;v=(row.get(col) or '').strip().lower()
        if v in cnt: cnt[v]+=1
print(f'{total},{cnt[\"train\"]},{cnt[\"val\"]},{cnt[\"test\"]}')
" "$f" 2>/dev/null)" || continue
            if [[ "${c}" == "1019,652,163,204" ]]; then
                RESULTS_ROOT="${cand}"; return 0
            fi
        done < <(find "${cand}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    done
    return 1
}

if ! resolve_results_root; then
    echo "[FATAL] RESULTS_ROOT auto-detect failed." >&2
    echo "        Set RESULTS_ROOT=/path/to/results manually." >&2; exit 1
fi

# ========================= OUT_ROOT =========================
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs" "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_teacher_homogeneous_gene_test.sh.snapshot"

# ========================= REF_MANIFEST =========================
resolve_ref_manifest() {
    for cand in \
        "${RESULTS_ROOT}/${PREFERRED_REL}" \
        "${RESULTS_ROOT}/outputs0529/ct_cnv_text_teacher_strict_ref1019/split_manifest.csv" \
        "${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42/split_manifest.csv"; do
        if [[ -f "${cand}" ]]; then REF_MANIFEST="${cand}"; return 0; fi
    done
    while IFS= read -r f; do
        c="$(python3 -c "
import csv,sys
p=sys.argv[1];cnt={'train':0,'val':0,'test':0};total=0
with open(p,'r',encoding='utf-8-sig',newline='') as fh:
    r=csv.DictReader(fh);col=None
    for c in ('assigned_split','split','Split'):
        if r.fieldnames and c in r.fieldnames: col=c;break
    if not col: print('0,0,0,0');sys.exit(0)
    for row in r:
        total+=1;v=(row.get(col) or '').strip().lower()
        if v in cnt: cnt[v]+=1
print(f'{total},{cnt[\"train\"]},{cnt[\"val\"]},{cnt[\"test\"]}')
" "$f" 2>/dev/null)" || continue
        if [[ "${c}" == "1019,652,163,204" ]]; then REF_MANIFEST="${f}"; return 0; fi
    done < <(find "${RESULTS_ROOT}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    return 1
}

if ! resolve_ref_manifest; then
    echo "[FATAL] No 1019/652/163/204 split_manifest.csv found." >&2; exit 1
fi

# ========================= 数据路径推断 =========================
infer_paths_py='
import json, sys
from pathlib import Path
candidates = [
    "outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42",
    "outputs0529/ct_cnv_text_teacher_strict_ref1019",
    "outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed42",
]
keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv", "gene_tsv"]
root = Path(sys.argv[1])
out = {k: "" for k in keys}
for run in candidates:
    p = root / run / "metrics.json"
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

# ========================= 控制开关 =========================
SMOKE_SEED="${SMOKE_SEED:-42}"
FULL_SEEDS="${FULL_SEEDS:-0}"         # 1 = seeds 42-45
RUN_STAGE="${RUN_STAGE:-all}"         # all / teacher_only / student_only
SKIP_TEACHER="${SKIP_TEACHER:-0}"     # 1 = 跳过 teacher 阶段 (兼容旧接口)
SKIP_STUDENT="${SKIP_STUDENT:-0}"     # 1 = 跳过 student 阶段
RUN_LIGHT_COMBO="${RUN_LIGHT_COMBO:-0}"  # 1 = 加跑 light-combo
DRY_RUN="${DRY_RUN:-0}"              # 1 = 只打印命令，不启动训练/不改动结果

# RUN_STAGE 映射
if [[ "${RUN_STAGE}" == "student_only" ]]; then SKIP_TEACHER=1; fi
if [[ "${RUN_STAGE}" == "teacher_only" ]]; then SKIP_STUDENT=1; fi

if [[ "${FULL_SEEDS}" == "1" ]]; then
    SEEDS=(42 43 44 45)
else
    SEEDS=("${SMOKE_SEED}")
fi

# ========================= Banner =========================
yesno() { [[ -e "$1" ]] && echo "yes" || echo "no"; }

echo "============================================================"
echo "Teacher Homogeneous Gene Test — DenseNet3D121 strict"
echo "============================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "RESULTS_ROOT : ${RESULTS_ROOT}"
echo "OUT_ROOT     : ${OUT_ROOT}"
echo "REF_MANIFEST : ${REF_MANIFEST}"
echo "FULL_SEEDS   : ${FULL_SEEDS} (seeds: ${SEEDS[*]})"
echo "RUN_STAGE    : ${RUN_STAGE}"
echo "SKIP_TEACHER : ${SKIP_TEACHER}"
echo "SKIP_STUDENT : ${SKIP_STUDENT}"
echo "RUN_LIGHT_COMBO: ${RUN_LIGHT_COMBO}"
echo "DRY_RUN     : ${DRY_RUN}"
echo ""
echo "Environment:"
printf "  %-40s : %s\n" "outputs/ exists" "$(yesno "${RESULTS_ROOT}/outputs")"
printf "  %-40s : %s\n" "outputs0529/ exists" "$(yesno "${RESULTS_ROOT}/outputs0529")"
printf "  %-40s : %s\n" "outputs0531_gene_privileged_ablation/ exists" "$(yesno "${RESULTS_ROOT}/outputs0531_gene_privileged_ablation")"
echo ""
echo "Data paths:"
printf "  %-18s = %s\n" "DATA_ROOT" "${DATA_ROOT}"
printf "  %-18s = %s\n" "METADATA_CSV" "${METADATA_CSV}"
printf "  %-18s = %s\n" "CT_ROOT" "${CT_ROOT}"
printf "  %-18s = %s\n" "TEXT_FEATURE_TSV" "${TEXT_FEATURE_TSV}"
printf "  %-18s = %s\n" "GENE_TSV" "${GENE_TSV}"
echo "============================================================"

missing=()
[[ -z "${DATA_ROOT}" ]] && missing+=("DATA_ROOT")
[[ -z "${METADATA_CSV}" ]] && missing+=("METADATA_CSV")
[[ -z "${CT_ROOT}" ]] && missing+=("CT_ROOT")
[[ -z "${TEXT_FEATURE_TSV}" ]] && missing+=("TEXT_FEATURE_TSV")
if [[ "${SKIP_TEACHER}" != "1" && -z "${GENE_TSV}" ]]; then missing+=("GENE_TSV"); fi
if (( ${#missing[@]} > 0 )); then
    echo "[FATAL] Missing paths: ${missing[*]}"; exit 1
fi

# ========================= 工具函数 =========================
already_done() { [[ -f "$1/metrics.json" ]]; }

validate_config() {
    local outdir="$1" want_ct="$2" want_bs="$3" want_mods="$4" want_teacher="${5:-}" want_methods="${6:-}"
    local mj="${outdir}/metrics.json"
    python3 - "$mj" "$want_ct" "$want_bs" "$want_mods" "$want_teacher" "$want_methods" <<'PY'
import csv, json, sys
from pathlib import Path
mj, want_ct, want_bs, want_mods, want_teacher, want_methods = sys.argv[1:7]
run_dir = Path(mj).parent
m = json.loads(Path(mj).read_text(encoding="utf-8"))
cfg = m.get("config") or {}
mfd = m.get("modality_feature_dims") or {}
issues = []

def norm_csv(v):
    if isinstance(v, str):
        return ",".join(x.strip() for x in v.split(",") if x.strip())
    return ",".join(v or [])

def split_counts(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        col = next((c for c in ("assigned_split", "split", "Split") if reader.fieldnames and c in reader.fieldnames), None)
        if not col:
            return None
        cnt = {"train": 0, "val": 0, "test": 0}; total = 0
        for row in reader:
            total += 1; v = (row.get(col) or "").strip().lower()
            if v in cnt: cnt[v] += 1
        return total, cnt["train"], cnt["val"], cnt["test"]

if cfg.get("ct_model") != want_ct: issues.append(f"ct_model={cfg.get('ct_model')}!={want_ct}")
if str(cfg.get("batch_size")) != want_bs: issues.append(f"bs={cfg.get('batch_size')}!={want_bs}")
actual_mods = norm_csv(cfg.get("modalities"))
if actual_mods != want_mods: issues.append(f"mods={actual_mods}!={want_mods}")
if cfg.get("class_mode") != "binary": issues.append(f"class_mode={cfg.get('class_mode')}!=binary")
if cfg.get("binary_task") != "malignant_vs_normal": issues.append(f"binary_task={cfg.get('binary_task')}!=malignant_vs_normal")
if cfg.get("selection_metric") != "auroc": issues.append(f"selection_metric={cfg.get('selection_metric')}!=auroc")
if not cfg.get("strict_no_leakage"): issues.append("STRICT_NOT_ENABLED")
if not cfg.get("disable_text_numeric_features"): issues.append("NUM_NOT_DISABLED")
if mfd.get("text_num") not in (0, None): issues.append(f"text_num={mfd.get('text_num')}")
if want_teacher:
    actual_teacher = cfg.get("teacher_run_dir") or ""
    try:
        if Path(actual_teacher).resolve() != Path(want_teacher).resolve():
            issues.append(f"teacher_run_dir={actual_teacher}!={want_teacher}")
    except OSError:
        issues.append(f"teacher_run_dir={actual_teacher}!={want_teacher}")
actual_methods = norm_csv(cfg.get("distill_methods"))
if want_methods and actual_methods != want_methods:
    issues.append(f"distill_methods={actual_methods}!={want_methods}")
if "hint" in actual_methods.split(","):
    issues.append("FORBIDDEN_HINT_METHOD")
split_path = run_dir / "split_manifest.csv"
if not split_path.is_file() and cfg.get("reference_manifest"):
    split_path = Path(str(cfg.get("reference_manifest")))
counts = split_counts(split_path)
if counts != (1019, 652, 163, 204):
    issues.append(f"SPLIT_MISMATCH={counts}")
tp = run_dir / "test_predictions.csv"
if not tp.is_file():
    issues.append("MISSING_TEST_PREDICTIONS")
else:
    with tp.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        id_col = next((c for c in ("sample_id", "record_id") if reader.fieldnames and c in reader.fieldnames), None)
        n_rows = sum(1 for _ in reader)
    if n_rows != 204:
        issues.append(f"TEST_PRED_ROWS={n_rows}!=204")
    if not id_col:
        issues.append("MISSING_SAMPLE_OR_RECORD_ID")
if not (run_dir / "text_feature_audit.json").is_file():
    issues.append("MISSING_AUDIT")
if not (run_dir / "leakage_warnings.json").is_file():
    issues.append("MISSING_WARNINGS")
if issues:
    print("CONFIG_MISMATCH: " + "; ".join(issues))
    sys.exit(1)
print("OK")
sys.exit(0)
PY
}
run_cmd() {
    local name="$1"; shift; local outdir="$1"; shift
    local want_ct="$1"; shift; local want_bs="$1"; shift; local want_mods="$1"; shift
    local want_teacher="$1"; shift; local want_methods="$1"; shift
    local logf="${OUT_ROOT}/logs/${name}.log"
    if already_done "${outdir}"; then
        local verdict
        verdict="$(validate_config "${outdir}" "${want_ct}" "${want_bs}" "${want_mods}" "${want_teacher}" "${want_methods}" 2>&1)" || true
        if [[ "${verdict}" == OK* ]]; then
            echo "[SKIP] ${name} — validated OK"; return 0
        else
            echo "[CONFIG_MISMATCH] ${name} — ${verdict}"
            echo "    -> Not overwriting. Fix manually or re-run."; return 1
        fi
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[DRY-RUN] %s:' "${name}"
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    mkdir -p "${outdir}"
    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN]  ${name}"
    echo "       output: ${outdir}"
    echo "       log: ${logf}"
    echo "------------------------------------------------------------"
    "$@" 2>&1 | tee "${logf}"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then echo "[FAIL] ${name} exit ${rc}"
    else echo "[OK]   ${name}"; fi
    return ${rc}
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

LOGITS_ONLY_METHODS="logits"
LIGHT_COMBO_METHODS="logits,fused,relation,attention,ct,text"
LIGHT_COMBO_WEIGHTS="logits=1,fused=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25"

# ========================= Stage 1: Teacher =========================
if [[ "${SKIP_TEACHER}" != "1" ]]; then
    echo ""
    echo "==================== Stage 1: Teacher Training ===================="

    # T0: DenseNet3D121 CT+Text teacher
    for seed in "${SEEDS[@]}"; do
        tag="T0_densenet3d121_ct_text_teacher_seed${seed}"
        outdir="${OUT_ROOT}/densenet3d121_ct_text_teacher_strict_seed${seed}"
        run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "" "" \
            python3 "${TRAIN_MM}" \
                --output-dir "${outdir}" \
                --modalities ct,text \
                --ct-model densenet3d_121 \
                --batch-size 4 \
                --seed "${seed}" \
                --reference-manifest "${REF_MANIFEST}" \
                --data-root "${DATA_ROOT}" \
                --metadata-csv "${METADATA_CSV}" \
                --ct-root "${CT_ROOT}" \
                --text-feature-tsv "${TEXT_FEATURE_TSV}" \
                "${COMMON_ARGS[@]}" \
                "${STRICT_ARGS[@]}" || true
    done

    # T1: DenseNet3D121 CT+CNV+Text teacher
    for seed in "${SEEDS[@]}"; do
        tag="T1_densenet3d121_ct_cnv_text_teacher_seed${seed}"
        outdir="${OUT_ROOT}/densenet3d121_ct_cnv_text_teacher_strict_seed${seed}"
        run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,cnv,text" "" "" \
            python3 "${TRAIN_MM}" \
                --output-dir "${outdir}" \
                --modalities ct,cnv,text \
                --ct-model densenet3d_121 \
                --batch-size 4 \
                --seed "${seed}" \
                --reference-manifest "${REF_MANIFEST}" \
                --data-root "${DATA_ROOT}" \
                --metadata-csv "${METADATA_CSV}" \
                --ct-root "${CT_ROOT}" \
                --text-feature-tsv "${TEXT_FEATURE_TSV}" \
                --gene-tsv "${GENE_TSV}" \
                "${COMMON_ARGS[@]}" \
                "${STRICT_ARGS[@]}" || true
    done
else
    echo ""
    echo "==================== Stage 1: Teacher (SKIPPED) ===================="
fi

# ========================= Stage 2: Student =========================
if [[ "${SKIP_STUDENT}" == "1" ]]; then
    echo ""
    echo "==================== Stage 2: Student (SKIPPED, RUN_STAGE=${RUN_STAGE}) ===================="
else
echo ""
echo "==================== Stage 2: Student Training ===================="

# S0: 复用已有 Group A
echo ""
echo "--- S0: Reusing existing DenseNet3D121 CT+Text supervised (Group A) ---"
for seed in "${SEEDS[@]}"; do
    src="${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed${seed}"
    dst="${OUT_ROOT}/ct_text_sc_densenet3d121_strict_bs4_seed${seed}"
    if [[ -e "${dst}" ]]; then
        verdict="$(validate_config "${dst}" "densenet3d_121" "4" "ct,text" "" "" 2>&1)" || true
        if [[ "${verdict}" == OK* ]]; then
            echo "[SKIP] S0 seed${seed} — validated OK"
        else
            echo "[CONFIG_MISMATCH] S0 seed${seed} — ${verdict}"
        fi
    elif [[ -f "${src}/metrics.json" ]]; then
        verdict="$(validate_config "${src}" "densenet3d_121" "4" "ct,text" "" "" 2>&1)" || true
        if [[ "${verdict}" != OK* ]]; then
            echo "[CONFIG_MISMATCH] S0 seed${seed} source — ${verdict}"
        elif [[ "${DRY_RUN}" == "1" ]]; then
            echo "[DRY-RUN] ln -s ${src} ${dst}"
        else
            ln -s "${src}" "${dst}"
            echo "[LINK] S0 seed${seed} -> ${src}"
        fi
    else
        echo "[WARN] S0 seed${seed} — source not found at ${src}"
    fi
done

# S1-logits: KD from T0, logits-only
echo ""
echo "--- S1-logits: KD from CT+Text teacher, logits-only ---"
for seed in "${SEEDS[@]}"; do
    teacher_dir="${OUT_ROOT}/densenet3d121_ct_text_teacher_strict_seed${seed}"
    tag="S1_logits_from_ct_text_teacher_seed${seed}"
    outdir="${OUT_ROOT}/densenet3d121_kd_from_ct_text_teacher_logits_only_seed${seed}"
    if [[ ! -f "${teacher_dir}/metrics.json" ]]; then
        echo "[SKIP] ${tag} — T0 teacher not ready"; continue
    fi
    run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "${teacher_dir}" "${LOGITS_ONLY_METHODS}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --ct-model densenet3d_121 \
            --batch-size 4 \
            --seed "${seed}" \
            --teacher-run-dir "${teacher_dir}" \
            --distill-methods "${LOGITS_ONLY_METHODS}" \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" || true
done

# S2-logits: KD from T1, logits-only
echo ""
echo "--- S2-logits: KD from CT+CNV+Text teacher, logits-only ---"
for seed in "${SEEDS[@]}"; do
    teacher_dir="${OUT_ROOT}/densenet3d121_ct_cnv_text_teacher_strict_seed${seed}"
    tag="S2_logits_from_ct_cnv_text_teacher_seed${seed}"
    outdir="${OUT_ROOT}/densenet3d121_kd_from_ct_cnv_text_teacher_logits_only_seed${seed}"
    if [[ ! -f "${teacher_dir}/metrics.json" ]]; then
        echo "[SKIP] ${tag} — T1 teacher not ready"; continue
    fi
    run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "${teacher_dir}" "${LOGITS_ONLY_METHODS}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --ct-model densenet3d_121 \
            --batch-size 4 \
            --seed "${seed}" \
            --teacher-run-dir "${teacher_dir}" \
            --distill-methods "${LOGITS_ONLY_METHODS}" \
            --distillation-alpha 0.5 \
            --distillation-temperature 4.0 \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" || true
done

# S1-light: KD from T0, light-combo (no hint)
if [[ "${RUN_LIGHT_COMBO}" == "1" ]]; then
    echo ""
    echo "--- S1-light: KD from CT+Text teacher, light-combo (no hint) ---"
    for seed in "${SEEDS[@]}"; do
        teacher_dir="${OUT_ROOT}/densenet3d121_ct_text_teacher_strict_seed${seed}"
        tag="S1_light_from_ct_text_teacher_seed${seed}"
        outdir="${OUT_ROOT}/densenet3d121_kd_from_ct_text_teacher_light_combo_seed${seed}"
        if [[ ! -f "${teacher_dir}/metrics.json" ]]; then
            echo "[SKIP] ${tag} — T0 teacher not ready"; continue
        fi
        run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "${teacher_dir}" "${LIGHT_COMBO_METHODS}" \
            python3 "${TRAIN_KD}" \
                --output-dir "${outdir}" \
                --modalities ct,text \
                --ct-model densenet3d_121 \
                --batch-size 4 \
                --seed "${seed}" \
                --teacher-run-dir "${teacher_dir}" \
                --distill-methods "${LIGHT_COMBO_METHODS}" \
                --distill-method-weights "${LIGHT_COMBO_WEIGHTS}" \
                --distill-feature-loss cosine \
                --distill-normalize-features \
                --distillation-alpha 0.5 \
                --distillation-temperature 4.0 \
                --reference-manifest "${REF_MANIFEST}" \
                --data-root "${DATA_ROOT}" \
                --metadata-csv "${METADATA_CSV}" \
                --ct-root "${CT_ROOT}" \
                --text-feature-tsv "${TEXT_FEATURE_TSV}" \
                "${COMMON_ARGS[@]}" \
                "${STRICT_ARGS[@]}" || true
    done

    echo ""
    echo "--- S2-light: KD from CT+CNV+Text teacher, light-combo (no hint) ---"
    for seed in "${SEEDS[@]}"; do
        teacher_dir="${OUT_ROOT}/densenet3d121_ct_cnv_text_teacher_strict_seed${seed}"
        tag="S2_light_from_ct_cnv_text_teacher_seed${seed}"
        outdir="${OUT_ROOT}/densenet3d121_kd_from_ct_cnv_text_teacher_light_combo_seed${seed}"
        if [[ ! -f "${teacher_dir}/metrics.json" ]]; then
            echo "[SKIP] ${tag} — T1 teacher not ready"; continue
        fi
        run_cmd "${tag}" "${outdir}" "densenet3d_121" "4" "ct,text" "${teacher_dir}" "${LIGHT_COMBO_METHODS}" \
            python3 "${TRAIN_KD}" \
                --output-dir "${outdir}" \
                --modalities ct,text \
                --ct-model densenet3d_121 \
                --batch-size 4 \
                --seed "${seed}" \
                --teacher-run-dir "${teacher_dir}" \
                --distill-methods "${LIGHT_COMBO_METHODS}" \
                --distill-method-weights "${LIGHT_COMBO_WEIGHTS}" \
                --distill-feature-loss cosine \
                --distill-normalize-features \
                --distillation-alpha 0.5 \
                --distillation-temperature 4.0 \
                --reference-manifest "${REF_MANIFEST}" \
                --data-root "${DATA_ROOT}" \
                --metadata-csv "${METADATA_CSV}" \
                --ct-root "${CT_ROOT}" \
                --text-feature-tsv "${TEXT_FEATURE_TSV}" \
                "${COMMON_ARGS[@]}" \
                "${STRICT_ARGS[@]}" || true
    done
else
    echo ""
    echo "--- Light-combo (SKIPPED, set RUN_LIGHT_COMBO=1 to enable) ---"
fi
fi  # end SKIP_STUDENT

# ========================= Summary =========================
echo ""
echo "============================================================"
echo "All runs done. Running analysis..."
echo "============================================================"

ANALYSIS_PY="${PROJECT_ROOT}/experiments/analysis/analyze_teacher_homogeneous_gene_test.py"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] python3 ${ANALYSIS_PY} --root ${OUT_ROOT} --results-root ${RESULTS_ROOT}"
elif [[ -f "${ANALYSIS_PY}" ]]; then
    python3 "${ANALYSIS_PY}" --root "${OUT_ROOT}" --results-root "${RESULTS_ROOT}"
else
    echo "[WARN] ${ANALYSIS_PY} not found — skip analysis"
fi

echo ""
echo "Outputs:"
echo "  ${OUT_ROOT}/teacher_homogeneous_metrics.csv"
echo "  ${OUT_ROOT}/teacher_homogeneous_metrics.md"
echo "  ${OUT_ROOT}/teacher_homogeneous_summary.md"
echo "  ${OUT_ROOT}/student_transfer_metrics.csv"
echo "  ${OUT_ROOT}/student_transfer_metrics.md"
echo "  ${OUT_ROOT}/student_transfer_summary.md"
echo "  ${OUT_ROOT}/paired_bootstrap_teacher.csv"
echo "  ${OUT_ROOT}/paired_bootstrap_student.csv"
echo "Done."
