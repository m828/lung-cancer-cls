#!/usr/bin/env bash
# Shared guarded execution helpers for outputs0542. This file is sourced.

set -euo pipefail

ATTR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTR_PROJECT_ROOT="${ATTR_PROJECT_ROOT:-$(cd "${ATTR_SCRIPT_DIR}/.." && pwd)}"
ATTR_PARENT_ROOT="$(cd "${ATTR_PROJECT_ROOT}/.." && pwd)"

SMOKE="${SMOKE:-1}"
RUN_MODE="${RUN_MODE:-smoke}"
SEEDS="${SEEDS:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ATTR_PROJECT_ROOT}/outputs0542_privileged_genomic_attribution_suite}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

attribution_parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --root) ATTR_PROJECT_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
            --resume) RESUME=1; shift ;;
            --no-resume) RESUME=0; shift ;;
            --force) FORCE=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            *) ATTR_REMAINING_ARGS+=("$1"); shift ;;
        esac
    done
}

attribution_guard_mode() {
    if [[ "${RUN_MODE}" == "full" && "${SMOKE}" != "0" ]]; then
        echo "[FATAL] full training requires SMOKE=0 and RUN_MODE=full" >&2
        return 2
    fi
    if [[ "${SMOKE}" == "0" && "${RUN_MODE}" != "full" ]]; then
        echo "[FATAL] SMOKE=0 is accepted only with RUN_MODE=full" >&2
        return 2
    fi
    if [[ "${RUN_MODE}" != "full" ]]; then
        SMOKE=1
        RUN_MODE=smoke
    fi
    export SMOKE RUN_MODE OUTPUT_ROOT
}

attribution_seed_array() {
    local raw="${SEEDS//,/ }"
    read -r -a ATTR_SEED_ARRAY <<< "${raw}"
    if [[ "${RUN_MODE}" == "smoke" ]]; then
        ATTR_SEED_ARRAY=("${ATTR_SEED_ARRAY[0]:-42}")
    fi
}

attribution_results_root() {
    if [[ -n "${RESULTS_ROOT:-}" ]]; then
        printf '%s\n' "${RESULTS_ROOT}"
    elif [[ -d "${ATTR_PROJECT_ROOT}/outputs0531_teacher_homogeneous_gene_test" ]]; then
        printf '%s\n' "${ATTR_PROJECT_ROOT}"
    else
        printf '%s\n' "${ATTR_PARENT_ROOT}"
    fi
}

attribution_infer_paths() {
    local results_root="$1"
    local payload
    payload="$(python3 - "${results_root}" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
candidates=[
 root/'outputs0531_teacher_homogeneous_gene_test'/'densenet3d121_ct_cnv_text_teacher_strict_seed42'/'metrics.json',
 root/'outputs0531_teacher_homogeneous_gene_test'/'densenet3d121_ct_text_teacher_strict_seed42'/'metrics.json',
]
keys=['data_root','metadata_csv','ct_root','text_feature_tsv','gene_tsv','gene_id_col','gene_label_col']
out={key:'' for key in keys}
for path in candidates:
 if not path.is_file(): continue
 config=json.loads(path.read_text(encoding='utf-8')).get('config',{})
 for key in keys:
  if not out[key] and config.get(key): out[key]=str(config[key])
for key in keys: print(f'{key}\t{out[key]}')
PY
)"
    declare -gA ATTR_PATHS=()
    while IFS=$'\t' read -r key value; do ATTR_PATHS["${key}"]="${value}"; done <<< "${payload}"
    DATA_ROOT="${DATA_ROOT:-${ATTR_PATHS[data_root]:-}}"
    METADATA_CSV="${METADATA_CSV:-${ATTR_PATHS[metadata_csv]:-}}"
    CT_ROOT="${CT_ROOT:-${ATTR_PATHS[ct_root]:-}}"
    TEXT_FEATURE_TSV="${TEXT_FEATURE_TSV:-${ATTR_PATHS[text_feature_tsv]:-}}"
    GENE_TSV="${GENE_TSV:-${ATTR_PATHS[gene_tsv]:-}}"
    GENE_ID_COL="${GENE_ID_COL:-${ATTR_PATHS[gene_id_col]:-}}"
    GENE_LABEL_COL="${GENE_LABEL_COL:-${ATTR_PATHS[gene_label_col]:-}}"
    local local_manifest="${results_root}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed42/split_manifest.csv"
    REF_MANIFEST="${REF_MANIFEST:-${local_manifest}}"
    export DATA_ROOT METADATA_CSV CT_ROOT TEXT_FEATURE_TSV GENE_TSV GENE_ID_COL GENE_LABEL_COL REF_MANIFEST
}

attribution_validate_paths() {
    if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
    local name
    for name in DATA_ROOT METADATA_CSV CT_ROOT TEXT_FEATURE_TSV REF_MANIFEST; do
        if [[ -z "${!name:-}" || ! -e "${!name}" ]]; then
            echo "[FATAL] ${name} is unavailable: ${!name:-<unset>}" >&2
            return 2
        fi
    done
}

attribution_cache_for() {
    local results_root="$1" teacher="$2" seed="$3"
    local first="${results_root}/outputs0534_best_student_kd_search/cached_teacher_targets/${teacher}_seed${seed}.csv"
    local second="${results_root}/outputs0535_student_kd_refinement/cached_teacher_targets/${teacher}_seed${seed}.csv"
    if [[ -f "${first}" ]]; then printf '%s\n' "${first}"; else printf '%s\n' "${second}"; fi
}

attribution_teacher_dir() {
    local results_root="$1" teacher="$2" seed="$3"
    if [[ "${teacher}" == "T0" ]]; then
        printf '%s\n' "${results_root}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_text_teacher_strict_seed${seed}"
    else
        printf '%s\n' "${results_root}/outputs0531_teacher_homogeneous_gene_test/densenet3d121_ct_cnv_text_teacher_strict_seed${seed}"
    fi
}

attribution_prepare_output() {
    local outdir="$1"
    if [[ -f "${outdir}/run_complete.json" && "${FORCE}" != "1" ]]; then
        echo "[SKIP] complete run: ${outdir}"
        return 1
    fi
    if [[ -d "${outdir}" && "${FORCE}" == "1" ]]; then
        local archived="${outdir}.superseded.$(date -u +%Y%m%dT%H%M%SZ)"
        mv "${outdir}" "${archived}"
        echo "[ARCHIVE] ${outdir} -> ${archived}"
    fi
    mkdir -p "${outdir}"
    return 0
}

attribution_run() {
    local name="$1" outdir="$2"; shift 2
    if ! attribution_prepare_output "${outdir}"; then return 0; fi
    local logdir="${OUTPUT_ROOT}/logs"
    local logfile="${logdir}/${name}.log"
    mkdir -p "${logdir}" "${outdir}"
    printf '%q ' "$@" > "${outdir}/command.txt"; printf '\n' >> "${outdir}/command.txt"
    env | sort > "${outdir}/environment.txt"
    git -C "${ATTR_PROJECT_ROOT}" rev-parse HEAD > "${outdir}/git_commit.txt" 2>/dev/null || printf 'UNAVAILABLE\n' > "${outdir}/git_commit.txt"
    git -C "${ATTR_PROJECT_ROOT}" status --short > "${outdir}/source_status.txt" 2>/dev/null || true
    git -C "${ATTR_PROJECT_ROOT}" diff --binary > "${outdir}/source_diff.patch" 2>/dev/null || true
    python3 - "${ATTR_PROJECT_ROOT}" "${outdir}/source_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); target=Path(sys.argv[2])
suffixes={'.py','.sh','.yaml','.yml','.toml'}
files=[]
for relative in ('scripts','experiments','src','configs/experiments','tests'):
 base=root/relative
 if not base.exists(): continue
 for path in sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in suffixes):
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  files.append({'path':str(path.relative_to(root)),'sha256':digest})
aggregate=hashlib.sha256('\n'.join(f"{row['path']}\t{row['sha256']}" for row in files).encode()).hexdigest()
target.write_text(json.dumps({'source_tree_sha256':aggregate,'files':files},indent=2),encoding='utf-8')
PY
    printf '%s\n' "${RUN_MODE}" > "${outdir}/run_mode.txt"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY_RUN] ${name}: $(cat "${outdir}/command.txt")"
        return 0
    fi
    echo "[RUN] ${name}"
    set +e
    "$@" > >(tee "${logfile}") 2> >(tee -a "${logfile}" >&2)
    local rc=$?
    set -e
    if (( rc != 0 )); then
        printf '%s\n' "${rc}" > "${outdir}/failed.exit_code"
        echo "[FAIL] ${name}: exit=${rc}; other seeds may continue" >&2
        return "${rc}"
    fi
    if [[ ! -f "${outdir}/run_complete.json" ]]; then
        python3 - "${outdir}" "${RUN_MODE}" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
out=Path(sys.argv[1]); mode=sys.argv[2]
def digest(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='run_complete.json'}
(out/'run_complete.json').write_text(json.dumps({'status':'complete','run_mode':mode,'finished_unix':time.time(),'file_sha256':files},indent=2),encoding='utf-8')
PY
    fi
}

attribution_student_common_args() {
    ATTR_STUDENT_COMMON=(
        --task malignant_vs_normal --class-mode binary
        --strict-no-leakage --disable-text-numeric-features
        --split-manifest "${REF_MANIFEST}"
        --data-root "${DATA_ROOT}" --metadata-csv "${METADATA_CSV}"
        --ct-root "${CT_ROOT}" --text-feature-tsv "${TEXT_FEATURE_TSV}"
        --ct-model "${CT_MODEL:-densenet3d_121}" --modalities ct,text
        --depth-size "${DEPTH_SIZE:-128}" --volume-hw "${VOLUME_HW:-256}"
        --ct-feature-dim "${CT_FEATURE_DIM:-128}" --text-feature-dim "${TEXT_FEATURE_DIM:-256}" --fusion-hidden-dim "${FUSION_HIDDEN_DIM:-256}"
        --dropout 0.3 --loss ce --label-smoothing 0.05
        --sampling-strategy weighted --class-weight-strategy effective_num --effective-num-beta 0.999
        --optimizer adamw --scheduler cosine --learning-rate 1e-4 --weight-decay 1e-4
        --batch-size 12 --epochs 50 --early-stopping-patience 10
        --alpha 0.1 --temperature 8 --kd-weight-floor 0.05 --kd-weight-max 1.0
        --amp
        --student-checkpoint-metric composite --save-all-checkpoint-metrics
    )
    if [[ "${RUN_MODE}" == "smoke" ]]; then
        ATTR_STUDENT_COMMON+=(--smoke --epochs 2 --num-workers 0)
    else
        ATTR_STUDENT_COMMON+=(--num-workers "${NUM_WORKERS:-2}")
    fi
}
