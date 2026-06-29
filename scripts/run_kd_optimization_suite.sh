#!/usr/bin/env bash
# scripts/run_kd_optimization_suite.sh
#
# KD optimization suite for CT+Text student under strict-no-leakage.
# Focus: distillation strategy + training strategy + batch stability + calibration/weighting.
#
# Stage usage:
#   bash scripts/run_kd_optimization_suite.sh
#   STAGE=alpha bash scripts/run_kd_optimization_suite.sh
#   STAGE=optimizer bash scripts/run_kd_optimization_suite.sh
#   STAGE=batch bash scripts/run_kd_optimization_suite.sh
#   STAGE=kd_method bash scripts/run_kd_optimization_suite.sh
#   STAGE=all RUN_MODE=smoke bash scripts/run_kd_optimization_suite.sh
#   STAGE=all RUN_MODE=mini FULL_SEEDS=1 bash scripts/run_kd_optimization_suite.sh
#   DRY_RUN=1 bash scripts/run_kd_optimization_suite.sh
#
# RUN_MODE:
#   smoke      -> seed42 only, minimal grid
#   mini       -> seeds 42,43 + reduced grid
#   full       -> seeds 42-45 + full requested grid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_KD="${PROJECT_ROOT}/train_student_kd.py"
WRAPPER="${PROJECT_ROOT}/scripts/train_student_kd_suite_wrapper.py"

RESULTS_ROOT="${RESULTS_ROOT:-}"
PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"
PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0533_kd_optimization_suite}"

mkdir -p "${OUT_ROOT}"
mkdir -p \
  "${OUT_ROOT}/alpha_sweep" \
  "${OUT_ROOT}/temperature_sweep" \
  "${OUT_ROOT}/optimizer_sweep" \
  "${OUT_ROOT}/batch_size_sweep" \
  "${OUT_ROOT}/light_combo_variants" \
  "${OUT_ROOT}/calibration_kd" \
  "${OUT_ROOT}/confidence_weighted_kd" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_kd_optimization_suite.sh.snapshot"

echo "[CONFIG] Script : ${BASH_SOURCE[0]}"
echo "[CONFIG] Root   : ${PROJECT_ROOT}"
echo "[CONFIG] Output : ${OUT_ROOT}"

# =======================================================
# Locate RESULTS_ROOT
resolve_results_root() {
    if [[ -n "${RESULTS_ROOT}" ]]; then
        return 0
    fi

    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        if [[ -f "${cand}/${PREFERRED_REL}" ]]; then
            RESULTS_ROOT="${cand}"
            return 0
        fi
    done

    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        while IFS= read -r f; do
            c="$(python3 - <<'PY' "$f"
import csv
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
total = 0
cnt = {"train": 0, "val": 0, "test": 0}
with p.open("r", encoding="utf-8-sig", newline="") as fh:
    r = csv.DictReader(fh)
    if not r.fieldnames:
        print("0,0,0,0")
        raise SystemExit
    col = None
    for c in ("assigned_split", "split", "Split"):
        if c in r.fieldnames:
            col = c
            break
    if col is None:
        print("0,0,0,0")
        raise SystemExit
    for row in r:
        total += 1
        v = (row.get(col) or "").strip().lower()
        if v in cnt:
            cnt[v] += 1
print(f"{total},{cnt['train']},{cnt['val']},{cnt['test']}")
PY
)" || true
            if [[ "${c}" == "1019,652,163,204" ]]; then
                RESULTS_ROOT="${cand}"
                return 0
            fi
        done < <(find "${cand}" -maxdepth 4 -type f -name split_manifest.csv 2>/dev/null)
    done
    return 1
}

if ! resolve_results_root; then
    echo "[FATAL] auto-detect RESULTS_ROOT failed. Set RESULTS_ROOT manually." >&2
    exit 1
fi

echo "[CONFIG] RESULTS_ROOT: ${RESULTS_ROOT}"

# =======================================================
# Locate reference split manifest (split control)
resolve_ref_manifest() {
    local candidates=(
        "${RESULTS_ROOT}/${PREFERRED_REL}"
        "${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed42/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4/densenet3d121_121_full_combo_strict_seed42/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0530_backbone_swap_bs_4/densenet3d_121_full_combo_strict_seed42/split_manifest.csv"
    )
    for cand in "${candidates[@]}"; do
        if [[ -f "${cand}" ]]; then
            REF_MANIFEST="${cand}"
            return 0
        fi
    done
    return 1
}

if ! resolve_ref_manifest; then
    echo "[FATAL] No reference split_manifest with required split (1019/652/163/204) found." >&2
    exit 1
fi

echo "[CONFIG] REF_MANIFEST: ${REF_MANIFEST}"

# =======================================================
# Infer shared paths from existing runs to avoid hard-coding envs.
infer_paths_py='
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
candidates = [
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_cnv_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0530_backbone_swap_bs_4" / "densenet3d_121_full_combo_strict_seed42" / "metrics.json",
    root / "outputs0531_gene_privileged_ablation" / "ct_text_sc_densenet3d121_strict_bs4_seed42" / "metrics.json",
]

keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv"]
out = {k: "" for k in keys}
for p in candidates:
    if not p.is_file():
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg = data.get("config") or {}
    except Exception:
        continue
    for k in keys:
        if not out[k]:
            v = cfg.get(k)
            if v:
                out[k] = str(v)

for k in keys:
    print(f"{k}\t{out[k]}")
'

declare -A INFERRED
while IFS=$'\t' read -r k v; do
    INFERRED[$k]="$v"
done < <(python3 -c "$infer_paths_py" "${RESULTS_ROOT}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"

missing=()
[[ -z "${DATA_ROOT}" ]] && missing+=("DATA_ROOT")
[[ -z "${METADATA_CSV}" ]] && missing+=("METADATA_CSV")
[[ -z "${CT_ROOT}" ]] && missing+=("CT_ROOT")
[[ -z "${TEXT_FEATURE_TSV}" ]] && missing+=("TEXT_FEATURE_TSV")
if (( ${#missing[@]} > 0 )); then
    echo "[FATAL] Missing core paths: ${missing[*]}" >&2
    exit 1
fi

# =======================================================
# Teacher / baseline paths (must be existing and fixed for strict comparison)
TEACHER_T0_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed"
TEACHER_T1_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed"
S0_BASE="${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed"
S1_BASE="${RESULTS_ROOT}/outputs0532_privileged_student_kd_optimization/S1_logits_alpha02_T4_seed"

if [[ ! -f "${TEACHER_T1_BASE}42/metrics.json" ]]; then
    echo "[FATAL] Required T1 teacher run missing: ${TEACHER_T1_BASE}42" >&2
    exit 1
fi

for seed in 42 43 44 45; do
    if [[ ! -f "${S0_BASE}${seed}/metrics.json" ]]; then
        echo "[WARN] Missing supervised baseline seed${seed}: ${S0_BASE}${seed}"
    fi
done

# =======================================================
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-smoke}"  # smoke|mini|full
FULL_SEEDS="${FULL_SEEDS:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${RUN_MODE}" == "full" ]] || [[ "${FULL_SEEDS}" == "1" ]]; then
    SEEDS=(42 43 44 45)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    SEEDS=(42 43)
else
    SEEDS=(42)
fi

COMMON_ARGS=(
    --class-mode binary --binary-task malignant_vs_normal
    --selection-metric auroc
    --epochs 50 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num
    --effective-num-beta 0.999
    --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256
    --fusion-hidden-dim 256 --dropout 0.3
    --reference-manifest "${REF_MANIFEST}"
    --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}"
    --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}"
    --ct-model densenet3d_121 --modalities ct,text
    --split-mode train_val_test
    --strict-no-leakage --disable-text-numeric-features
)

run_cmd() {
    local name="$1"
    local outdir="$2"
    shift 2
    local logfile="${OUT_ROOT}/logs/${name}.log"
    if [[ -f "${outdir}/metrics.json" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] ${name}: result exists"
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name} -> $*"
        return 0
    fi
    mkdir -p "${outdir}" "$(dirname "${logfile}")"
    echo ""
    echo "[RUN ] ${name}"
    echo "       outdir: ${outdir}"
    echo "       log  : ${logfile}"
    echo "------------------------------------------------------------"
    "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo "[FAIL] ${name} exit ${rc}"
        return ${rc}
    fi
}

run_student() {
    local group="$1"; local name="$2"; local seed="$3"; local teacher_dir="$4"
    local methods="$5"; local alpha="$6"; local temp="$7"
    local bs="$8"; local optimizer="$9"; local scheduler="${10}"
    local lr="${11}"; local wd="${12}"; local acc="${13:-1}"
    local method_weights="${14:-}"; local feature_loss="${15:-cosine}"; local normalize="${16:-1}"
    local -a wrapper_args=("${@:17}")

    local outdir="${OUT_ROOT}/${group}/${name}"
    if [[ ! -d "${teacher_dir}" || ! -f "${teacher_dir}/metrics.json" ]]; then
        echo "[SKIP] ${name}: missing teacher ${teacher_dir}"
        return 0
    fi
    if [[ -f "${outdir}/metrics.json" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] ${name}: exists"
        return 0
    fi

    local -a cmd=(python3 "${WRAPPER}")
    if (( acc > 1 )); then
        cmd+=(--accumulation-steps "${acc}")
    fi
    if (( ${#wrapper_args[@]} > 0 )); then
        cmd+=("${wrapper_args[@]}")
    fi
    cmd+=(-- "$TRAIN_KD"
        --output-dir "${outdir}"
        --seed "${seed}"
        --teacher-run-dir "${teacher_dir}"
        --distill-methods "${methods}"
        --distillation-alpha "${alpha}"
        --distillation-temperature "${temp}"
        --distill-feature-loss "${feature_loss}"
        --optimizer "${optimizer}"
        --scheduler "${scheduler}"
        --lr "${lr}"
        --weight-decay "${wd}"
        --batch-size "${bs}"
        "${COMMON_ARGS[@]}")

    if [[ -n "${method_weights}" ]]; then
        cmd+=(--distill-method-weights "${method_weights}")
    fi
    if [[ "${normalize}" == "1" ]]; then
        cmd+=(--distill-normalize-features)
    fi

    run_cmd "${name}" "${outdir}" "${cmd[@]}"
}

link_or_skip() {
    local src="$1"
    local dst="$2"
    if [[ -e "${dst}" ]]; then return 0; fi
    if [[ ! -f "${src}/metrics.json" ]]; then
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] LINK ${dst} -> ${src}"
        return 0
    fi
    ln -s "${src}" "${dst}"
}

stage_alpha_temps() {
    local alpha_list=("$@")
    local temp=4
    for seed in "${SEEDS[@]}"; do
        for alpha in "${alpha_list[@]}"; do
            atag="${alpha/./}"
            run_student "alpha_sweep" "S2_logits_alpha${atag}_T${temp}_seed${seed}" \
                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" "${alpha}" "${temp}" \
                4 adamw cosine 0.0003 0.0001 1
        done
    done
}

# =======================================================
echo ""
echo "============================================================"
echo "KD optimization suite"
echo "============================================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "OUT_ROOT     : ${OUT_ROOT}"
echo "RESULTS_ROOT : ${RESULTS_ROOT}"
echo "STAGE        : ${STAGE}"
echo "RUN_MODE     : ${RUN_MODE} (seeds=${SEEDS[*]})"
echo "DRY_RUN      : ${DRY_RUN}"
echo "============================================================"

if [[ "${STAGE}" == "alpha" || "${STAGE}" == "all" ]]; then
    echo "[Stage A] KD alpha / temperature sweeps"
    declare -a alpha_list
    if [[ "${RUN_MODE}" == "full" ]]; then
        alpha_list=(0.05 0.1 0.2 0.3 0.5)
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        alpha_list=(0.1 0.2 0.3)
    else
        alpha_list=(0.2)
    fi
    if (( ${#alpha_list[@]} == 0 )); then
        echo "[WARN] alpha configs empty, skipping alpha sweep."
    else
        echo "[DEBUG] alpha configs: ${#alpha_list[@]}"
        echo "[DEBUG] seeds: ${SEEDS[*]}"
        stage_alpha_temps "${alpha_list[@]}"
    fi

    if [[ "${RUN_MODE}" == "full" ]]; then
        temp_values=(2 6 8)
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        temp_values=(6)
    else
        temp_values=()
    fi
    if (( ${#temp_values[@]} == 0 )); then
        echo "[WARN] temperature configs empty, skipping temperature branch."
    else
        echo "[DEBUG] temperature configs: ${#temp_values[@]} -> ${temp_values[*]}"
    fi
    for temp in "${temp_values[@]}"; do
        ttag="${temp/./}"
        for seed in "${SEEDS[@]}"; do
            run_student "temperature_sweep" "S2_logits_alpha02_T${ttag}_seed${seed}" \
                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 "${temp}" 4 adamw cosine 0.0003 0.0001 1
        done
    done
fi

if [[ "${STAGE}" == "optimizer" || "${STAGE}" == "all" ]]; then
    echo "[Stage B] Optimizer / scheduler / LR / WD sweeps"
    if [[ "${RUN_MODE}" == "full" ]]; then
        OPTS=(adamw sgd)
        LRS=(1e-4 3e-4 5e-4)
        WDS=(1e-5 1e-4)
        ALLOWED_SCHED=("none" "cosine")
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        OPTS=(adamw sgd)
        LRS=(3e-4)
        WDS=(1e-4)
        ALLOWED_SCHED=("none" "cosine")
    else
        OPTS=(adamw)
        LRS=(3e-4)
        WDS=(1e-4)
        ALLOWED_SCHED=("cosine")
    fi

    for seed in "${SEEDS[@]}"; do
        for opt in "${OPTS[@]}"; do
            for lr in "${LRS[@]}"; do
                for wd in "${WDS[@]}"; do
                    for sched in "${ALLOWED_SCHED[@]}"; do
                        if [[ "${opt}" == "adamw" ]]; then
                            case "${sched}" in
                                none)
                                    run_student "optimizer_sweep" "S2_alpha02_T4_${opt}_none_lr${lr}_wd${wd}_seed${seed}" \
                                        "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 "${opt}" none "${lr}" "${wd}" 1
                                    ;;
                                cosine)
                                    run_student "optimizer_sweep" "S2_alpha02_T4_${opt}_cosine_lr${lr}_wd${wd}_seed${seed}" \
                                        "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 "${opt}" cosine "${lr}" "${wd}" 1
                                    if [[ "${RUN_MODE}" == "full" ]]; then
                                        run_student "optimizer_sweep" "S2_alpha02_T4_${opt}_warmup_cosine_lr${lr}_wd${wd}_seed${seed}" \
                                            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 "${opt}" cosine "${lr}" "${wd}" 1 --wrapper-scheduler warmup_cosine
                                    fi
                                    ;;
                                esac
                        elif [[ "${opt}" == "sgd" && "${sched}" != "none" ]]; then
                            run_student "optimizer_sweep" "S2_alpha02_T4_${opt}_cosine_lr${lr}_wd${wd}_seed${seed}" \
                                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 "${opt}" cosine "${lr}" "${wd}" 1
                        fi
                    done
                done
            done
        done
    done
fi

if [[ "${STAGE}" == "batch" || "${STAGE}" == "all" ]]; then
    echo "[Stage C] Batch size and accumulation (stability tests)"
    BS_VALS=(1 2 4)
    if [[ "${RUN_MODE}" == "full" ]] || [[ "${RUN_MODE}" == "mini" ]]; then
        BS_VALS=(1 2 4)
    else
        BS_VALS=(4)
    fi

    for seed in "${SEEDS[@]}"; do
        for bs in "${BS_VALS[@]}"; do
            run_student "batch_size_sweep" "S2_alpha02_T4_bs${bs}_seed${seed}" \
                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 "${bs}" adamw cosine 0.0003 0.0001 1
        done
        if [[ "${RUN_MODE}" != "smoke" ]]; then
            run_student "batch_size_sweep" "S2_alpha02_T4_bs1_acc4_seed${seed}" \
                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 1 adamw cosine 0.0003 0.0001 4
            run_student "batch_size_sweep" "S2_alpha02_T4_bs2_acc2_seed${seed}" \
                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 2 adamw cosine 0.0003 0.0001 2
        fi
    done
fi

if [[ "${STAGE}" == "kd_method" || "${STAGE}" == "all" ]]; then
    echo "[Stage D] KD method structures + calibration-aware + confidence-weighted"
    for seed in "${SEEDS[@]}"; do
        run_student "light_combo_variants" "S2_logits_alpha02_T4_seed${seed}" \
            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
            "logits=1" cosine 1
        run_student "light_combo_variants" "S2_light_logits_fused_alpha02_T4_seed${seed}" \
            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits,fused" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
            "logits=1,fused=0.5" cosine 1
        run_student "light_combo_variants" "S2_light_logits_fused_ct_text_alpha02_T4_seed${seed}" \
            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits,fused,ct,text" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
            "logits=1,fused=0.5,ct=0.5,text=0.25" cosine 1

        run_student "calibration_kd" "S2_logits_alpha02_T4_calibT2_seed${seed}" \
            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
            "logits=1" cosine 1 --wrapper-calibration-aware --wrapper-teacher-temperature-scale 2.0
        if [[ "${RUN_MODE}" == "full" ]]; then
            run_student "calibration_kd" "S2_logits_alpha02_T4_calibT3_seed${seed}" \
                "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
                "logits=1" cosine 1 --wrapper-calibration-aware --wrapper-teacher-temperature-scale 3.0
        fi

        run_student "confidence_weighted_kd" "S2_logits_alpha02_T4_confsoft_g0.5_seed${seed}" \
            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
            "logits=1" cosine 1 --confidence-mode soft --confidence-gamma 0.5 --confidence-floor 0.10
        run_student "confidence_weighted_kd" "S2_logits_alpha02_T4_confhard_seed${seed}" \
            "${seed}" "${TEACHER_T1_BASE}${seed}" "logits" 0.2 4 4 adamw cosine 0.0003 0.0001 1 \
            "logits=1" cosine 1 --confidence-mode hard --confidence-threshold 0.90 --confidence-floor 0.20
    done
fi

echo "[INFO] Link baseline comparisons into suite for downstream analysis."
for seed in "${SEEDS[@]}"; do
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/alpha_sweep/S0_supervised_seed${seed}"
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/temperature_sweep/S0_supervised_seed${seed}"
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/optimizer_sweep/S0_supervised_seed${seed}"
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/batch_size_sweep/S0_supervised_seed${seed}"
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/light_combo_variants/S0_supervised_seed${seed}"
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/calibration_kd/S0_supervised_seed${seed}"
    link_or_skip "${S0_BASE}${seed}" "${OUT_ROOT}/confidence_weighted_kd/S0_supervised_seed${seed}"

    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/alpha_sweep/S1_logits_alpha02_T4_seed${seed}"
    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/temperature_sweep/S1_logits_alpha02_T4_seed${seed}"
    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/optimizer_sweep/S1_logits_alpha02_T4_seed${seed}"
    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/batch_size_sweep/S1_logits_alpha02_T4_seed${seed}"
    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/light_combo_variants/S1_logits_alpha02_T4_seed${seed}"
    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/calibration_kd/S1_logits_alpha02_T4_seed${seed}"
    link_or_skip "${S1_BASE}${seed}" "${OUT_ROOT}/confidence_weighted_kd/S1_logits_alpha02_T4_seed${seed}"
done

cat <<EOF
============================================================
Done. Output root:
  ${OUT_ROOT}
Run analysis:
  python3 experiments/analysis/analyze_kd_optimization_suite.py --root ${OUT_ROOT}
============================================================
EOF
