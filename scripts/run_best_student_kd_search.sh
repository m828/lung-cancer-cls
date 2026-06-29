#!/usr/bin/env bash
# Final deployable CT+Text student KD search (outputs0534).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PARENT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-${PARENT_ROOT}/outputs0534_best_student_kd_search}"
STAGE="${STAGE:-all}"
RUN_MODE="${RUN_MODE:-smoke}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FULL_SEEDS="${FULL_SEEDS:-0}"

mkdir -p \
  "${OUT_ROOT}/cached_teacher_targets" \
  "${OUT_ROOT}/s1_ct_text_teacher_kd" \
  "${OUT_ROOT}/s2_gene_teacher_kd" \
  "${OUT_ROOT}/selective_kd" \
  "${OUT_ROOT}/threshold_calibration" \
  "${OUT_ROOT}/logs" \
  "${OUT_ROOT}/scripts_used"
cp -f "${BASH_SOURCE[0]}" "${OUT_ROOT}/scripts_used/run_best_student_kd_search.sh.snapshot"

PREFERRED_REL="outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"
resolve_results_root() {
    if [[ -n "${RESULTS_ROOT}" ]]; then return 0; fi
    for cand in "${PROJECT_ROOT}" "${PARENT_ROOT}"; do
        if [[ -f "${cand}/${PREFERRED_REL}" || -d "${cand}/outputs0531_teacher_homogeneous_gene_test" ]]; then
            RESULTS_ROOT="${cand}"
            return 0
        fi
    done
    return 1
}
if ! resolve_results_root; then
    echo "[FATAL] Set RESULTS_ROOT manually." >&2
    exit 1
fi

resolve_ref_manifest() {
    local candidates=(
        "${RESULTS_ROOT}/${PREFERRED_REL}"
        "${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42/split_manifest.csv"
        "${RESULTS_ROOT}/outputs0531_gene_privileged_ablation/ct_text_sc_densenet3d121_strict_bs4_seed42/split_manifest.csv"
    )
    for cand in "${candidates[@]}"; do
        if [[ -f "${cand}" ]]; then REF_MANIFEST="${cand}"; return 0; fi
    done
    return 1
}
if ! resolve_ref_manifest; then
    echo "[FATAL] No reference manifest found." >&2
    exit 1
fi

infer_paths_py='
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
candidates = [
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0531_teacher_homogeneous_gene_test" / "densenet3d121_ct_cnv_text_teacher_strict_seed42" / "metrics.json",
    root / "outputs0531_gene_privileged_ablation" / "ct_text_sc_densenet3d121_strict_bs4_seed42" / "metrics.json",
]
keys = ["data_root", "metadata_csv", "ct_root", "text_feature_tsv", "gene_tsv"]
out = {k: "" for k in keys}
for p in candidates:
    if not p.is_file():
        continue
    try:
        cfg = json.loads(p.read_text(encoding="utf-8")).get("config", {})
    except Exception:
        continue
    for k in keys:
        if not out[k] and cfg.get(k):
            out[k] = str(cfg[k])
for k in keys:
    print(f"{k}\t{out[k]}")
'
declare -A INFERRED
while IFS=$'\t' read -r k v; do INFERRED[$k]="$v"; done < <(python3 -c "$infer_paths_py" "${RESULTS_ROOT}")

DATA_ROOT="${DATA_ROOT:-${INFERRED[data_root]:-}}"
METADATA_CSV="${METADATA_CSV:-${INFERRED[metadata_csv]:-}}"
CT_ROOT="${CT_ROOT:-${INFERRED[ct_root]:-}}"
TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${INFERRED[text_feature_tsv]:-}}"
GENE_TSV="${GENE_TSV:-${INFERRED[gene_tsv]:-}}"

for required in DATA_ROOT METADATA_CSV CT_ROOT TEXT_FEATURE_TSV; do
    if [[ -z "${!required}" ]]; then
        echo "[FATAL] Missing ${required}. Set it explicitly." >&2
        exit 1
    fi
done

TEACHER_T0_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed"
TEACHER_T1_BASE="${RESULTS_ROOT}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed"

if [[ "${RUN_MODE}" == "full" || "${FULL_SEEDS}" == "1" ]]; then
    SEEDS=(42 43 44 45)
elif [[ "${RUN_MODE}" == "mini" ]]; then
    SEEDS=(42 43)
else
    SEEDS=(42)
fi
if [[ "${SMOKE}" == "1" ]]; then
    SEEDS=(42)
fi

COMMON_ARGS=(
    --class-mode binary --binary-task malignant_vs_normal
    --strict-no-leakage --disable-text-numeric-features
    --reference-manifest "${REF_MANIFEST}"
    --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}"
    --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}"
)
if [[ -n "${GENE_TSV}" ]]; then
    GENE_ARGS=(--gene-tsv "${GENE_TSV}")
else
    GENE_ARGS=()
fi

STUDENT_ARGS=(
    "${COMMON_ARGS[@]}"
    --ct-model densenet3d_121 --modalities ct,text
    --depth-size 128 --volume-hw 256
    --ct-feature-dim 128 --text-feature-dim 256 --fusion-hidden-dim 256
    --dropout 0.3 --loss ce --label-smoothing 0.05
    --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
    --selection-metric auroc --composite-selection-metric
)

run_cmd() {
    local name="$1"
    local out_path="$2"
    shift 2
    local logfile="${OUT_ROOT}/logs/${name}.log"
    if [[ -e "${out_path}" && "${DRY_RUN}" != "1" ]]; then
        echo "[SKIP] ${name}: exists ${out_path}"
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name} -> $*"
        return 0
    fi
    mkdir -p "$(dirname "${logfile}")" "$(dirname "${out_path}")"
    echo "[RUN ] ${name}"
    echo "       out: ${out_path}"
    echo "       log: ${logfile}"
    set +e
    "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        echo "[FAIL] ${name}: exit ${rc}; continuing"
    fi
    return 0
}

cache_file() {
    local teacher="$1"; local seed="$2"
    echo "${OUT_ROOT}/cached_teacher_targets/${teacher}_seed${seed}.csv"
}

run_cache() {
    for seed in "${SEEDS[@]}"; do
        for teacher in T0 T1; do
            if [[ "${teacher}" == "T0" ]]; then
                teacher_dir="${TEACHER_T0_BASE}${seed}"
                extra=()
            else
                teacher_dir="${TEACHER_T1_BASE}${seed}"
                extra=("${GENE_ARGS[@]}")
            fi
            if [[ ! -f "${teacher_dir}/metrics.json" ]]; then
                echo "[SKIP] cache_${teacher}_seed${seed}: missing teacher ${teacher_dir}"
                continue
            fi
            run_cmd "cache_${teacher}_seed${seed}" "$(cache_file "${teacher}" "${seed}")" \
                python3 scripts/cache_teacher_soft_targets.py \
                --teacher-run-dir "${teacher_dir}" \
                --output-dir "${OUT_ROOT}/cached_teacher_targets" \
                --cache-name "${teacher}_seed${seed}" \
                --seed "${seed}" --batch-size 8 --num-workers 2 \
                "${COMMON_ARGS[@]}" "${extra[@]}"
        done
    done
}

run_student_cached() {
    local stage_dir="$1"; local name="$2"; local seed="$3"; local cache="$4"
    shift 4
    local outdir="${OUT_ROOT}/${stage_dir}/${name}"
    run_cmd "${name}" "${outdir}/metrics.json" \
        python3 scripts/train_student_kd_cached_logits.py \
        --output-dir "${outdir}" \
        --cached-teacher-targets "${cache}" \
        --seed "${seed}" \
        "$@" \
        "${STUDENT_ARGS[@]}"
}

stage_s1() {
    local configs
    if [[ "${SMOKE}" == "1" ]]; then
        configs=("a0.10:T4:bs8:acc1:lr3e-4:wd1e-4:cosine")
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        configs=("a0.05:T4:bs8:acc1:lr2e-4:wd1e-4:cosine" "a0.10:T4:bs12:acc1:lr3e-4:wd1e-4:cosine" "a0.15:T6:bs8:acc2:lr2e-4:wd1e-5:warmup_cosine" "a0.20:T4:bs16:acc1:lr1e-4:wd1e-4:cosine")
    else
        configs=("a0.05:T4:bs12:acc1:lr2e-4:wd1e-4:cosine" "a0.10:T4:bs16:acc1:lr3e-4:wd1e-4:cosine" "a0.15:T6:bs8:acc2:lr2e-4:wd1e-5:warmup_cosine")
    fi
    for seed in "${SEEDS[@]}"; do
        cache="$(cache_file T0 "${seed}")"
        [[ -f "${cache}" || "${DRY_RUN}" == "1" ]] || { echo "[SKIP] s1 seed${seed}: missing cache ${cache}"; continue; }
        for cfg in "${configs[@]}"; do
            IFS=: read -r atag ttag bstag acctag lrtag wdtag sched <<<"${cfg}"
            alpha="${atag#a}"; temp="${ttag#T}"; bs="${bstag#bs}"; acc="${acctag#acc}"; lr="${lrtag#lr}"; wd="${wdtag#wd}"
            name="S1_cached_${atag}_${ttag}_${bstag}_${acctag}_${lrtag}_${wdtag}_${sched}_seed${seed}"
            run_student_cached "s1_ct_text_teacher_kd" "${name}" "${seed}" "${cache}" \
                --distillation-alpha "${alpha}" --distillation-temperature "${temp}" \
                --batch-size "${bs}" --grad-accum-steps "${acc}" --lr "${lr}" --weight-decay "${wd}" \
                --optimizer adamw --scheduler "${sched}" --epochs 50 --amp --early-stopping-patience 10
        done
    done
}

stage_s2() {
    local configs
    if [[ "${SMOKE}" == "1" ]]; then
        configs=("a0.05:T6:bs8:acc1:lr2e-4:wd1e-4:cosine")
    elif [[ "${RUN_MODE}" == "mini" ]]; then
        configs=("a0.05:T4:bs8:acc1:lr2e-4:wd1e-4:cosine" "a0.10:T6:bs12:acc1:lr2e-4:wd1e-4:cosine" "a0.15:T8:bs8:acc2:lr1e-4:wd1e-5:warmup_cosine")
    else
        configs=("a0.05:T6:bs12:acc1:lr2e-4:wd1e-4:cosine" "a0.10:T6:bs16:acc1:lr2e-4:wd1e-4:cosine" "a0.15:T8:bs8:acc2:lr1e-4:wd1e-5:warmup_cosine")
    fi
    for seed in "${SEEDS[@]}"; do
        cache="$(cache_file T1 "${seed}")"
        [[ -f "${cache}" || "${DRY_RUN}" == "1" ]] || { echo "[SKIP] s2 seed${seed}: missing cache ${cache}"; continue; }
        for cfg in "${configs[@]}"; do
            IFS=: read -r atag ttag bstag acctag lrtag wdtag sched <<<"${cfg}"
            alpha="${atag#a}"; temp="${ttag#T}"; bs="${bstag#bs}"; acc="${acctag#acc}"; lr="${lrtag#lr}"; wd="${wdtag#wd}"
            name="S2_gene_cached_${atag}_${ttag}_${bstag}_${acctag}_${lrtag}_${wdtag}_${sched}_seed${seed}"
            run_student_cached "s2_gene_teacher_kd" "${name}" "${seed}" "${cache}" \
                --distillation-alpha "${alpha}" --distillation-temperature "${temp}" \
                --batch-size "${bs}" --grad-accum-steps "${acc}" --lr "${lr}" --weight-decay "${wd}" \
                --optimizer adamw --scheduler "${sched}" --epochs 50 --amp --early-stopping-patience 10
        done
    done
}

stage_selective() {
    for seed in "${SEEDS[@]}"; do
        t1_cache="$(cache_file T1 "${seed}")"
        t0_cache="$(cache_file T0 "${seed}")"
        [[ -f "${t1_cache}" || "${DRY_RUN}" == "1" ]] || { echo "[SKIP] selective seed${seed}: missing cache ${t1_cache}"; continue; }
        for mode in confidence margin advantage; do
            ref_args=()
            if [[ "${mode}" == "advantage" ]]; then
                [[ -f "${t0_cache}" || "${DRY_RUN}" == "1" ]] || { echo "[SKIP] advantage seed${seed}: missing T0 cache ${t0_cache}"; continue; }
                ref_args=(--reference-teacher-targets "${t0_cache}")
            fi
            name="S2_selective_${mode}_a0.10_T6_bs12_lr2e-4_seed${seed}"
            run_student_cached "selective_kd" "${name}" "${seed}" "${t1_cache}" \
                --distillation-alpha 0.10 --distillation-temperature 6 \
                --batch-size 12 --grad-accum-steps 1 --lr 2e-4 --weight-decay 1e-4 \
                --optimizer adamw --scheduler cosine --epochs 50 --amp --early-stopping-patience 10 \
                --kd-weighting "${mode}" --kd-weight-floor 0.05 --kd-weight-max 1.0 "${ref_args[@]}"
        done
    done
}

stage_calibration() {
    run_cmd "analyze_best_student_kd_search" "${OUT_ROOT}/best_student_summary.md" \
        python3 experiments/analysis/analyze_best_student_kd_search.py --root "${OUT_ROOT}"
}

cat <<EOF
============================================================
Best student KD search
PROJECT_ROOT : ${PROJECT_ROOT}
RESULTS_ROOT : ${RESULTS_ROOT}
OUT_ROOT     : ${OUT_ROOT}
REF_MANIFEST : ${REF_MANIFEST}
STAGE        : ${STAGE}
RUN_MODE     : ${RUN_MODE}
SMOKE        : ${SMOKE}
DRY_RUN      : ${DRY_RUN}
SEEDS        : ${SEEDS[*]}
============================================================
EOF

if [[ "${STAGE}" == "cache" || "${STAGE}" == "all" ]]; then run_cache; fi
if [[ "${STAGE}" == "s1" || "${STAGE}" == "all" ]]; then stage_s1; fi
if [[ "${STAGE}" == "s2" || "${STAGE}" == "all" ]]; then stage_s2; fi
if [[ "${STAGE}" == "selective" || "${STAGE}" == "all" ]]; then stage_selective; fi
if [[ "${STAGE}" == "calibration" || "${STAGE}" == "all" ]]; then stage_calibration; fi

echo "[DONE] outputs: ${OUT_ROOT}"
