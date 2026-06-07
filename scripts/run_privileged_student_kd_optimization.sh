#!/usr/bin/env bash
# scripts/run_privileged_student_kd_optimization.sh
#
# Student KD optimization for gene privileged teacher transfer.
#
# 前提：T1 (DenseNet3D121 CT+CNV+Text teacher) > T0 (CT+Text teacher) 已验证。
# 目标：优化 CT+Text student 从 gene teacher 获得的 KD 收益。
#
# 实验设计：
#   Exp 1: logits-only alpha/T sweep (from T1)
#   Exp 2: hint-free light-combo variants (from T1)
#   Exp 3: T0 teacher control (best configs from Exp 1/2)
#
# 运行方式：
#   bash scripts/run_privileged_student_kd_optimization.sh                    # smoke (seed42, alpha02 only)
#   FULL_SEEDS=1 bash scripts/run_privileged_student_kd_optimization.sh       # seeds 42-45
#   FULL_SWEEP=1 bash scripts/run_privileged_student_kd_optimization.sh       # all alpha/T combos

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"

# ========================= RESULTS_ROOT =========================
RESULTS_ROOT="${RESULTS_ROOT:-}"
PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"
PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

resolve_results_root() {
    if [[ -n "${RESULTS_ROOT}" ]]; then return 0; fi
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        if [[ -f "${cand}/${PREFERRED_REL}" ]]; then RESULTS_ROOT="${cand}"; return 0; fi
    done
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        while IFS= read -r f; do
            c="$(python3 -c "
import csv,sys;p=sys.argv[1];cnt={'train':0,'val':0,'test':0};total=0
with open(p,'r',encoding='utf-8-sig',newline='') as fh:
    r=csv.DictReader(fh);col=None
    for c in ('assigned_split','split','Split'):
        if r.fieldnames and c in r.fieldnames: col=c;break
    if not col: print('0,0,0,0');sys.exit(0)
    for row in r: total+=1;v=(row.get(col) or '').strip().lower()
    if v in cnt: cnt[v]+=1
print(f'{total},{cnt[\"train\"]},{cnt[\"val\"]},{cnt[\"test\"]}')
" "$f" 2>/dev/null)" || continue
            if [[ "${c}" == "1019,652,163,204" ]]; then RESULTS_ROOT="${cand}"; return 0; fi
        done < <(find "${cand}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    done
    return 1
}

if ! resolve_results_root; then
    echo "[FATAL] RESULTS_ROOT auto-detect failed. Set RESULTS_ROOT manually." >&2; exit 1
fi

# ========================= OUT_ROOT =========================
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/outputs0532_privileged_student_kd_optimization}"
mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/logs" "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_privileged_student_kd_optimization.sh.snapshot"

# ========================= REF_MANIFEST =========================
resolve_ref_manifest() {
    for cand in \
        "${RESULTS_ROOT}/${PREFERRED_REL}" \
        "${RESULTS_ROOT}/outputs0529/ct_cnv_text_teacher_strict_ref1019/split_manifest.csv" \
        "${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42/split_manifest.csv"; do
        if [[ -f "${cand}" ]]; then REF_MANIFEST="${cand}"; return 0; fi
    done
    return 1
}
if ! resolve_ref_manifest; then
    echo "[FATAL] No 1019/652/163/204 split_manifest.csv found." >&2; exit 1
fi

# ========================= Data paths =========================
infer_paths_py='
import json,sys;from pathlib import Path
keys=["data_root","metadata_csv","ct_root","text_feature_tsv"]
root=Path(sys.argv[1]);out={k:"" for k in keys}
for run in ["outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42","outputs0529/ct_cnv_text_teacher_strict_ref1019"]:
    p=root/run/"metrics.json"
    if not p.is_file(): continue
    try: d=json.loads(p.read_text(encoding="utf-8"))
    except: continue
    cfg=d.get("config") or {}
    for k in keys:
        if not out[k]:
            v=cfg.get(k)
            if v: out[k]=str(v)
for k in keys: print(f"{k}\t{out[k]}")
'
declare -A INFERRED
while IFS=$'\t' read -r k v; do INFERRED[$k]="$v"; done < <(python3 -c "${infer_paths_py}" "${RESULTS_ROOT}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"

# ========================= Teacher paths =========================
TEACHER_T0_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed"
TEACHER_T1_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed"

# Check T1 teacher exists
t1_ok=0
for s in 42 43 44 45; do
    [[ -f "${TEACHER_T1_BASE}${s}/metrics.json" ]] && t1_ok=$((t1_ok+1))
done
if [[ ${t1_ok} -eq 0 ]]; then
    echo "[FATAL] T1 teachers not found at ${TEACHER_T1_BASE}{42..45}"
    echo "        Run first: FULL_SEEDS=1 bash scripts/run_teacher_homogeneous_gene_test.sh"
    exit 1
fi

# ========================= Controls =========================
SMOKE_SEED="${SMOKE_SEED:-42}"
FULL_SEEDS="${FULL_SEEDS:-0}"
FULL_SWEEP="${FULL_SWEEP:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${FULL_SEEDS}" == "1" ]]; then SEEDS=(42 43 44 45); else SEEDS=("${SMOKE_SEED}"); fi

# ========================= Banner =========================
echo "============================================================"
echo "Privileged Student KD Optimization"
echo "============================================================"
echo "PROJECT_ROOT  : ${PROJECT_ROOT}"
echo "RESULTS_ROOT  : ${RESULTS_ROOT}"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "REF_MANIFEST  : ${REF_MANIFEST}"
echo "FULL_SEEDS    : ${FULL_SEEDS} (seeds: ${SEEDS[*]})"
echo "FULL_SWEEP    : ${FULL_SWEEP}"
echo "DRY_RUN       : ${DRY_RUN}"
echo "T1 teacher ok : ${t1_ok}/4"
echo "DATA_ROOT     : ${DATA_ROOT}"
echo "============================================================"

missing=()
[[ -z "${DATA_ROOT}" ]] && missing+=("DATA_ROOT")
[[ -z "${METADATA_CSV}" ]] && missing+=("METADATA_CSV")
[[ -z "${CT_ROOT}" ]] && missing+=("CT_ROOT")
[[ -z "${TEXT_FEATURE_TSV}" ]] && missing+=("TEXT_FEATURE_TSV")
if (( ${#missing[@]} > 0 )); then echo "[FATAL] Missing: ${missing[*]}"; exit 1; fi

# ========================= Tools =========================
already_done() { [[ -f "$1/metrics.json" ]]; }

validate_config() {
    local outdir="$1" want_bs="$2" want_mods="$3" want_teacher="$4" want_methods="$5"
    python3 - "$outdir/metrics.json" "$want_bs" "$want_mods" "$want_teacher" "$want_methods" <<'PY'
import json,csv,sys
from pathlib import Path
mj,want_bs,want_mods,want_teacher,want_methods=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
p=Path(mj)
if not p.is_file(): print("NO_METRICS"); sys.exit(1)
m=json.loads(p.read_text(encoding="utf-8"));cfg=m.get("config") or {};mfd=m.get("modality_feature_dims") or {}
issues=[]
if str(cfg.get("batch_size"))!=want_bs: issues.append(f"bs={cfg.get('batch_size')}!={want_bs}")
actual_mods=",".join(cfg.get("modalities") or [])
if actual_mods!=want_mods: issues.append(f"mods={actual_mods}!={want_mods}")
if not cfg.get("strict_no_leakage"): issues.append("STRICT_OFF")
if not cfg.get("disable_text_numeric_features"): issues.append("NUM_ON")
if mfd.get("text_num") not in (0,None): issues.append(f"text_num={mfd.get('text_num')}")
# teacher check
actual_teacher=cfg.get("teacher_run_dir","")
if want_teacher and want_teacher not in actual_teacher: issues.append(f"teacher={actual_teacher}!={want_teacher}")
# distill methods check
actual_methods=set(cfg.get("distill_methods") or [])
if "hint" in actual_methods: issues.append("FORBIDDEN_HINT")
# split
d=p.parent
sp=d/"split_manifest.csv"
if sp.is_file():
    with sp.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f);col=None
        for c in ("assigned_split","split","Split"):
            if reader.fieldnames and c in reader.fieldnames: col=c;break
        if col:
            cnt={"train":0,"val":0,"test":0};total=0
            for row in reader: total+=1;v=(row.get(col) or "").strip().lower()
            if v in cnt: cnt[v]+=1
            if (total,cnt["train"],cnt["val"],cnt["test"])!=(1019,652,163,204): issues.append("SPLIT_BAD")
tp=d/"test_predictions.csv"
if tp.is_file():
    with tp.open(encoding="utf-8-sig") as f:
        import csv as c2; r=c2.reader(f);next(r);nrows=sum(1 for _ in r)
    if nrows!=204: issues.append(f"ROWS={nrows}")
if not (d/"text_feature_audit.json").is_file(): issues.append("NO_AUDIT")
if not (d/"leakage_warnings.json").is_file(): issues.append("NO_WARNINGS")
if issues: print("CONFIG_MISMATCH: "+"; ".join(issues));sys.exit(1)
else: print("OK");sys.exit(0)
PY
}

run_cmd() {
    local name="$1"; shift; local outdir="$1"; shift
    local want_bs="$1"; shift; local want_mods="$1"; shift
    local want_teacher="$1"; shift; local want_methods="$1"; shift
    local logf="${OUT_ROOT}/logs/${name}.log"
    if already_done "${outdir}"; then
        verdict="$(validate_config "${outdir}" "${want_bs}" "${want_mods}" "${want_teacher}" "${want_methods}" 2>&1)" || true
        if [[ "${verdict}" == OK* ]]; then echo "[SKIP] ${name} — validated OK"; return 0
        else echo "[CONFIG_MISMATCH] ${name} — ${verdict}"; return 1; fi
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name} — would run: $*"
        return 0
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

# ========================= Common args =========================
COMMON_ARGS=(
    --class-mode binary --binary-task malignant_vs_normal
    --selection-metric auroc --split-mode train_val_test
    --epochs 50 --optimizer adamw --lr 0.0003 --weight-decay 0.0001
    --scheduler cosine --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num
    --effective-num-beta 0.999 --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256
    --fusion-hidden-dim 256 --dropout 0.3
)
STRICT_ARGS=(--strict-no-leakage --disable-text-numeric-features)

run_student() {
    local name="$1" teacher_dir="$2" methods="$3" alpha="$4" temp="$5"
    local weights="$6" feature_loss="$7" normalize="$8"
    local outdir="${OUT_ROOT}/${name}"
    local methods_tag="${methods//,/_}"
    run_cmd "${name}" "${outdir}" "4" "ct,text" "${teacher_dir}" "${methods}" \
        python3 "${TRAIN_KD}" \
            --output-dir "${outdir}" \
            --modalities ct,text \
            --ct-model densenet3d_121 \
            --batch-size 4 \
            --seed "${seed}" \
            --teacher-run-dir "${teacher_dir}" \
            --distill-methods "${methods}" \
            --distillation-alpha "${alpha}" \
            --distillation-temperature "${temp}" \
            ${weights:+--distill-method-weights "${weights}"} \
            ${feature_loss:+--distill-feature-loss "${feature_loss}"} \
            ${normalize:+--distill-normalize-features} \
            --reference-manifest "${REF_MANIFEST}" \
            --data-root "${DATA_ROOT}" \
            --metadata-csv "${METADATA_CSV}" \
            --ct-root "${CT_ROOT}" \
            --text-feature-tsv "${TEXT_FEATURE_TSV}" \
            "${COMMON_ARGS[@]}" \
            "${STRICT_ARGS[@]}" || true
}

# ========================= Exp 1: Alpha/T sweep =========================
echo ""
echo "==================== Exp 1: Logits-only alpha/T sweep (from T1) ===================="

# Alpha sweep at T=4
for alpha in 0.1 0.2 0.3 0.5; do
    # In smoke mode, only run alpha=0.2
    if [[ "${FULL_SWEEP}" != "1" && "${alpha}" != "0.2" ]]; then continue; fi
    for seed in "${SEEDS[@]}"; do
        run_student "S2_logits_alpha${alpha/./}_T4_seed${seed}" \
            "${TEACHER_T1_BASE}${seed}" "logits" "${alpha}" "4.0" "" "" ""
    done
done

# Temperature sweep at alpha=0.2
if [[ "${FULL_SWEEP}" == "1" ]]; then
    for temp in 2.0 6.0 8.0; do
        for seed in "${SEEDS[@]}"; do
            run_student "S2_logits_alpha02_T${temp/.*}_seed${seed}" \
                "${TEACHER_T1_BASE}${seed}" "logits" "0.2" "${temp}" "" "" ""
        done
    done
fi

# ========================= Exp 2: Light-combo variants =========================
echo ""
echo "==================== Exp 2: Hint-free light-combo variants (from T1) ===================="

# Variant 1: logits,fused
for seed in "${SEEDS[@]}"; do
    run_student "S2_light_logits_fused_alpha02_T4_seed${seed}" \
        "${TEACHER_T1_BASE}${seed}" "logits,fused" "0.2" "4.0" \
        "logits=1,fused=0.5" "cosine" "--distill-normalize-features"
done

# Variant 2: logits,fused,ct,text
for seed in "${SEEDS[@]}"; do
    run_student "S2_light_logits_fused_ct_text_alpha02_T4_seed${seed}" \
        "${TEACHER_T1_BASE}${seed}" "logits,fused,ct,text" "0.2" "4.0" \
        "logits=1,fused=0.5,ct=0.5,text=0.25" "cosine" "--distill-normalize-features"
done

# Variant 3: logits,fused,relation,attention,ct,text (full light-combo)
for seed in "${SEEDS[@]}"; do
    run_student "S2_light_full_no_hint_alpha02_T4_seed${seed}" \
        "${TEACHER_T1_BASE}${seed}" "logits,fused,relation,attention,ct,text" "0.2" "4.0" \
        "logits=1,fused=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25" "cosine" "--distill-normalize-features"
done

# ========================= Exp 3: T0 control =========================
echo ""
echo "==================== Exp 3: T0 teacher control ===================="

# T0 logits-only alpha=0.2
for seed in "${SEEDS[@]}"; do
    t0_dir="${TEACHER_T0_BASE}${seed}"
    if [[ ! -f "${t0_dir}/metrics.json" ]]; then
        echo "[SKIP] T0 control seed${seed} — T0 teacher not ready"; continue
    fi
    run_student "S1_logits_alpha02_T4_seed${seed}" \
        "${t0_dir}" "logits" "0.2" "4.0" "" "" ""
done

# T0 light-combo no-hint alpha=0.2
if [[ "${FULL_SWEEP}" == "1" ]]; then
    for seed in "${SEEDS[@]}"; do
        t0_dir="${TEACHER_T0_BASE}${seed}"
        if [[ ! -f "${t0_dir}/metrics.json" ]]; then continue; fi
        run_student "S1_light_full_no_hint_alpha02_T4_seed${seed}" \
            "${t0_dir}" "logits,fused,relation,attention,ct,text" "0.2" "4.0" \
            "logits=1,fused=0.5,relation=0.25,attention=0.25,ct=0.5,text=0.25" "cosine" "--distill-normalize-features"
    done
fi

# ========================= Link existing S0 =========================
echo ""
echo "==================== Link existing S0 (supervised baseline) ===================="
for seed in "${SEEDS[@]}"; do
    src="${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed${seed}"
    dst="${OUT_ROOT}/S0_supervised_seed${seed}"
    if [[ -f "${src}/metrics.json" ]] && [[ ! -e "${dst}" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[DRY_RUN] Would link S0 seed${seed} -> ${src}"
        else
            ln -s "${src}" "${dst}"
            echo "[LINK] S0 seed${seed} -> ${src}"
        fi
    elif [[ -e "${dst}" ]]; then
        echo "[SKIP] S0 seed${seed} — exists"
    fi
done

# ========================= Summary =========================
echo ""
echo "============================================================"
echo "All runs done."
echo "============================================================"
echo "Run analysis:"
echo "  python3 experiments/analysis/analyze_privileged_student_kd_optimization.py --root ${OUT_ROOT}"
echo "Done."
