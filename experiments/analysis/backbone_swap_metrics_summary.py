#!/usr/bin/env python3
"""Summarise the outputs0530_backbone_swap batch (strict rescue — backbone swap).

Reads every run directory under ``--root``:

  outputs0530_backbone_swap/
    resnet3d18_full_combo_strict_seed{43,44,45}/
    densenet3d_121_full_combo_strict_seed{42,43,44,45}/
    mc3_18_full_combo_strict_seed42/          (optional, RUN_MC3=1)
    swin3d_tiny_full_combo_strict_seed42/     (optional, RUN_SWIN3D=1)

Produces three artefacts in ``--root``:

  backbone_swap_metrics.csv  — per-run row with split / strict / metrics / status
  backbone_swap_metrics.md   — same as a markdown table + readable explanation
  backbone_swap_summary.md   — per-backbone multi-seed stability + main-candidate verdict

Context it pulls in (read-only):
  --baseline-dir (outputs0529)        : the strict ResNet3D18 full-combo single seed (42)
                                        and the R2Plus1D strict runs, for cross-backbone comparison.
  --rescue-dir   (outputs0530_strict_rescue)
                                        : the R2Plus1D lite-combo / logits-only rescue results,
                                          so the verdict can rank backbone-swap against them.

Decision thresholds: AUROC >= 0.95 (recovery), 4-seed std < 0.015 (stability).

Pure read-only — never re-trains. Safe to invoke independently if the bash
batch was interrupted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

EXPECTED = (1019, 652, 163, 204)
RECOVERY_AUROC = 0.95
STABILITY_STD = 0.015

CSV_FIELDS = [
    "run_name",
    "output_dir",
    "ct_model",
    "modalities",
    "distill_methods",
    "distillation_alpha",
    "distillation_temperature",
    "seed",
    "best_epoch",
    "test_num_samples",
    "split_total",
    "train",
    "val",
    "test",
    "test_pred_rows",
    "strict_no_leakage",
    "disable_text_numeric_features",
    "num_features_used",
    "has_text_feature_audit",
    "has_leakage_warnings",
    "warnings_count",
    "split_check_status",
    "auroc",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "precision",
    "accuracy",
    "ece",
    "brier",
    "notes",
]


# ---------- low-level readers (mirror rescue_metrics_summary.py) -----------

def _read_json(p: Path) -> dict[str, Any] | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _count_rows(p: Path) -> int:
    if not p.is_file():
        return -1
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f)
            try:
                next(r)
            except StopIteration:
                return 0
            return sum(1 for _ in r)
    except OSError:
        return -1


def _split_counts(p: Path) -> tuple[int, int, int, int] | None:
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return None
            col = None
            for c in ("assigned_split", "split", "Split"):
                if c in reader.fieldnames:
                    col = c
                    break
            if col is None:
                return None
            cnt = {"train": 0, "val": 0, "test": 0}
            total = 0
            for row in reader:
                total += 1
                v = (row.get(col) or "").strip().lower()
                if v in cnt:
                    cnt[v] += 1
            return total, cnt["train"], cnt["val"], cnt["test"]
    except OSError:
        return None


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4f}"
    return str(v)


def _md_cell(v: Any) -> str:
    if v is True:
        return "yes"
    if v is False:
        return "no"
    if v == "" or v is None:
        return "-"
    return str(v).replace("|", "\\|")


def analyse(run_dir: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {k: "" for k in CSV_FIELDS}
    rec["run_name"] = run_dir.name
    rec["output_dir"] = str(run_dir)
    notes: list[str] = []

    metrics = _read_json(run_dir / "metrics.json")
    if metrics:
        cfg = metrics.get("config") or {}
        tm = metrics.get("test_metrics") or {}
        rec["ct_model"] = cfg.get("ct_model", "")
        rec["modalities"] = ",".join(cfg.get("modalities") or [])
        rec["distill_methods"] = ",".join(cfg.get("distill_methods") or [])
        rec["distillation_alpha"] = cfg.get("distillation_alpha")
        rec["distillation_temperature"] = cfg.get("distillation_temperature")
        rec["seed"] = cfg.get("seed")
        rec["best_epoch"] = metrics.get("best_epoch")
        rec["test_num_samples"] = tm.get("num_samples")
        rec["strict_no_leakage"] = cfg.get("strict_no_leakage")
        rec["disable_text_numeric_features"] = cfg.get("disable_text_numeric_features")
        mfd = metrics.get("modality_feature_dims") or {}
        rec["num_features_used"] = mfd.get("text_num", "")

        rec["auroc"] = tm.get("auroc")
        rec["balanced_accuracy"] = tm.get("balanced_accuracy")
        rec["f1"] = tm.get("f1")
        rec["recall"] = tm.get("recall")
        rec["specificity"] = tm.get("specificity")
        rec["precision"] = tm.get("precision")
        rec["accuracy"] = tm.get("accuracy")
        rec["ece"] = tm.get("ece")
        rec["brier"] = tm.get("brier_score")
    else:
        notes.append("missing metrics.json")

    split_counts = _split_counts(run_dir / "split_manifest.csv")
    if split_counts is None:
        rec["split_check_status"] = "MISSING"
        notes.append("missing split_manifest.csv")
    else:
        rec["split_total"], rec["train"], rec["val"], rec["test"] = split_counts
        if split_counts == EXPECTED:
            rec["split_check_status"] = "OK"
        else:
            rec["split_check_status"] = "SPLIT_MISMATCH"
            notes.append(f"split counts {split_counts} != {EXPECTED}")

    pred = run_dir / "test_predictions.csv"
    rec["test_pred_rows"] = _count_rows(pred) if pred.is_file() else ""
    if pred.is_file():
        if rec["test_pred_rows"] != 204:
            notes.append(f"TEST_PRED_ROWS_NOT_204 ({rec['test_pred_rows']})")
    else:
        notes.append("missing test_predictions.csv")

    audit = run_dir / "text_feature_audit.json"
    warn = run_dir / "leakage_warnings.json"
    rec["has_text_feature_audit"] = audit.is_file()
    rec["has_leakage_warnings"] = warn.is_file()
    if not audit.is_file() and not warn.is_file():
        notes.append("STRICT_AUDIT_MISSING")
    if warn.is_file():
        wd = _read_json(warn) or {}
        warns = wd.get("warnings") or []
        rec["warnings_count"] = len(warns) if isinstance(warns, list) else ""
    else:
        rec["warnings_count"] = ""

    if rec.get("strict_no_leakage") is not True:
        notes.append("STRICT_NOT_ENABLED")
    if rec.get("disable_text_numeric_features") is not True:
        notes.append("NUM_FEATURES_NOT_DISABLED")
    if rec.get("num_features_used") not in (0, "", None):
        if rec["num_features_used"] != 0:
            notes.append(f"text_num={rec['num_features_used']} (should be 0)")

    rec["notes"] = "; ".join(notes)
    return rec


def discover_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and p.name not in {"logs", "scripts_used", "__pycache__"}
    ]


def write_csv(records: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def write_metrics_md(records: list[dict[str, Any]], out: Path, root: Path,
                     ref: Path | None, context_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# Backbone-swap metrics — outputs0530_backbone_swap\n"]
    lines.append(f"- Root: `{root}`")
    if ref is not None:
        lines.append(f"- Reference manifest: `{ref}`")
    lines.append("- Expected split: total=1019, train=652, val=163, test=204")
    lines.append("- All runs: strict-no-leakage, full-combo KD, ref-1019 split, ct+text modalities.")
    lines.append(f"- Runs discovered: {len(records)}")
    lines.append("")
    lines.append("## Backbone-swap runs")
    headers = ["run", "ct_model", "distill", "alpha", "T",
               "seed", "best_ep", "split", "test_rows", "strict",
               "AUROC", "BAcc", "F1", "Recall", "Spec", "ECE", "Brier",
               "warn_n", "status", "notes"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in records:
        split = f"{r.get('split_total','-')}/{r.get('train','-')}/{r.get('val','-')}/{r.get('test','-')}"
        row = [
            r["run_name"], r["ct_model"], r["distill_methods"],
            _fmt(r["distillation_alpha"]), _fmt(r["distillation_temperature"]),
            _fmt(r["seed"]), _fmt(r["best_epoch"]),
            split, _fmt(r["test_pred_rows"]),
            "yes" if r.get("strict_no_leakage") is True else "no",
            _fmt(r["auroc"]), _fmt(r["balanced_accuracy"]), _fmt(r["f1"]),
            _fmt(r["recall"]), _fmt(r["specificity"]),
            _fmt(r["ece"]), _fmt(r["brier"]),
            _fmt(r["warnings_count"]), r["split_check_status"], r["notes"],
        ]
        lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")

    if context_rows:
        lines.append("")
        lines.append("## Context runs (read-only: outputs0529 + outputs0530_strict_rescue)")
        lines.append("| run | source | ct_model | distill | alpha | seed | AUROC | BAcc | F1 | notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in context_rows:
            lines.append("| " + " | ".join(_md_cell(c) for c in [
                r["run_name"], r.get("source", ""), r.get("ct_model", ""),
                r.get("distill_methods", ""), _fmt(r.get("distillation_alpha")),
                _fmt(r.get("seed")), _fmt(r.get("auroc")),
                _fmt(r.get("balanced_accuracy")), _fmt(r.get("f1")), r.get("notes", ""),
            ]) + " |")

    lines.append("")
    lines.append("## Status flags")
    lines.append("- `SPLIT_MISMATCH` — split_manifest counts ≠ 1019/652/163/204.")
    lines.append("- `TEST_PRED_ROWS_NOT_204` — test_predictions.csv row count ≠ 204.")
    lines.append("- `STRICT_AUDIT_MISSING` — text_feature_audit.json / leakage_warnings.json absent.")
    lines.append("- `STRICT_NOT_ENABLED` / `NUM_FEATURES_NOT_DISABLED` — config flag did not take effect.")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_std(xs: Iterable[float]) -> tuple[float, float] | None:
    vals = [v for v in xs if isinstance(v, (int, float))]
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def _g(r: dict[str, Any], key: str) -> Any:
    v = r.get(key)
    return v if isinstance(v, (int, float)) else None


def _backbone_of(rec: dict[str, Any]) -> str:
    """Prefer config ct_model; fall back to a prefix of the run name."""
    bb = rec.get("ct_model")
    if isinstance(bb, str) and bb:
        return bb
    name = rec.get("run_name", "")
    for cand in ("resnet3d18", "densenet3d_121", "mc3_18", "swin3d_tiny", "r2plus1d_18"):
        if name.startswith(cand):
            return cand
    return "unknown"


def _seed_of(rec: dict[str, Any]) -> Any:
    s = rec.get("seed")
    if isinstance(s, int):
        return s
    name = rec.get("run_name", "")
    if "_seed" in name:
        tail = name.rsplit("_seed", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return None


def write_summary(records: list[dict[str, Any]],
                  context_rows: list[dict[str, Any]],
                  out: Path) -> None:
    lines: list[str] = ["# Backbone-swap summary — outputs0530_backbone_swap\n"]
    lines.append("策略：不改代码，仅切换 `--ct-model`，把 R2Plus1D 那套 strict + full-combo + "
                 "ref-1019 + bs=1 配方跑在其它 3D backbone 上，看是否能拿到稳定主候选。\n")

    # group backbone-swap runs by backbone, merging the matching outputs0529 single seed
    by_bb: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        bb = _backbone_of(r)
        by_bb.setdefault(bb, []).append(r)

    # pull the outputs0529 resnet3d18 strict single-seed (seed 42) into the resnet group
    ctx_by_name = {r["run_name"]: r for r in context_rows}
    resnet_base = next(
        (r for r in context_rows
         if r.get("source") == "outputs0529" and "resnet3d18_full_combo" in r["run_name"]),
        None,
    )
    if resnet_base is not None:
        merged = dict(resnet_base)
        merged.setdefault("seed", 42)
        by_bb.setdefault("resnet3d18", [])
        # only add if seed 42 not already present from this batch
        seeds_present = {_seed_of(x) for x in by_bb["resnet3d18"]}
        if 42 not in seeds_present:
            by_bb["resnet3d18"].append(merged)

    # ---- Q1: per-backbone multi-seed stability -----------------------------
    lines.append("## Q1. 每个 backbone 的多 seed 稳定性\n")
    lines.append(f"门槛：均值 AUROC ≥ {RECOVERY_AUROC}，且 std < {STABILITY_STD}。")
    lines.append("（resnet3d18 的 seed42 来自 outputs0529 strict 单 seed，已并入下表凑多 seed。）\n")

    backbone_stats: dict[str, dict[str, Any]] = {}
    for bb in sorted(by_bb):
        runs = by_bb[bb]
        lines.append(f"### {bb}")
        lines.append("| source | seed | AUROC | BAcc | F1 | best_ep | status |")
        lines.append("|---|---|---|---|---|---|---|")
        aucs: list[float] = []
        for r in sorted(runs, key=lambda x: (_seed_of(x) is None, _seed_of(x))):
            src = r.get("source") or "outputs0530_backbone_swap"
            auc = _g(r, "auroc")
            if auc is not None:
                aucs.append(auc)
            lines.append(
                f"| {src} | {_seed_of(r) if _seed_of(r) is not None else '-'} "
                f"| {_fmt(r.get('auroc'))} | {_fmt(r.get('balanced_accuracy'))} "
                f"| {_fmt(r.get('f1'))} | {_fmt(r.get('best_epoch'))} "
                f"| {r.get('split_check_status', '-') or '-'} |"
            )
        ms = _mean_std(aucs)
        if ms is None:
            lines.append("\n- AUROC: **无可读结果**")
            backbone_stats[bb] = {"mean": None, "std": None, "n": 0}
        else:
            mean, std = ms
            n = len(aucs)
            backbone_stats[bb] = {"mean": mean, "std": std, "n": n}
            lines.append(f"\n- AUROC: mean ± std = {mean:.4f} ± {std:.4f} (n={n})")
            if n < 2:
                verdict = "**单 seed，无法判断稳定性**"
            elif mean >= RECOVERY_AUROC and std < STABILITY_STD:
                verdict = "✅ **稳定且达标**"
            elif mean >= RECOVERY_AUROC:
                verdict = f"⚠️ 均值达标但 std {std:.4f} ≥ {STABILITY_STD}（偏抖）"
            else:
                verdict = f"❌ 均值 {mean:.4f} < {RECOVERY_AUROC}（未恢复）"
            lines.append(f"- 判定: {verdict}")
        lines.append("")

    # ---- Q2: cross-backbone ranking ---------------------------------------
    lines.append("## Q2. backbone 横向排名（按均值 AUROC）\n")
    ranked = sorted(
        ((bb, s) for bb, s in backbone_stats.items() if s["mean"] is not None),
        key=lambda kv: kv[1]["mean"],
        reverse=True,
    )
    if not ranked:
        lines.append("- 无可读结果。")
    else:
        lines.append("| rank | backbone | mean AUROC | std | n | 达标(≥0.95) | 稳定(std<0.015) |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, (bb, s) in enumerate(ranked, 1):
            hit = "yes" if s["mean"] >= RECOVERY_AUROC else "no"
            stab = "yes" if (s["n"] >= 2 and s["std"] < STABILITY_STD) else ("n/a" if s["n"] < 2 else "no")
            lines.append(
                f"| {i} | {bb} | {s['mean']:.4f} | {s['std']:.4f} | {s['n']} | {hit} | {stab} |"
            )
    lines.append("")

    # ---- Q3: vs R2Plus1D rescue (the thing this batch tries to replace) ----
    lines.append("## Q3. backbone-swap 是否比 R2Plus1D rescue 更好?\n")
    r2_lite = [r for r in context_rows
               if r.get("source") == "outputs0530_strict_rescue" and "lite_combo" in r["run_name"]]
    r2_logits = [r for r in context_rows
                 if "logits_only" in r["run_name"]
                 and (r.get("ct_model") == "r2plus1d_18" or "r2plus1d" in r["run_name"])]
    r2_lite_best = max((x for x in (_g(r, "auroc") for r in r2_lite) if x is not None), default=None)
    r2_logits_aucs = [x for x in (_g(r, "auroc") for r in r2_logits) if x is not None]
    r2_logits_ms = _mean_std(r2_logits_aucs) if r2_logits_aucs else None

    lines.append("R2Plus1D rescue 现状（来自 outputs0530_strict_rescue / outputs0529，read-only）：")
    lines.append(f"- R2Plus1D lite-combo best AUROC: {_fmt(r2_lite_best)}")
    if r2_logits_ms:
        lines.append(f"- R2Plus1D logits-only AUROC: {r2_logits_ms[0]:.4f} ± {r2_logits_ms[1]:.4f} "
                     f"(n={len(r2_logits_aucs)})")
    else:
        lines.append("- R2Plus1D logits-only AUROC: 无可读结果")
    lines.append("")
    best_swap = ranked[0] if ranked else None
    if best_swap is None:
        lines.append("- backbone-swap 暂无可读结果，无法比较。")
    else:
        bb, s = best_swap
        baseline = max(filter(None, [r2_lite_best,
                                     r2_logits_ms[0] if r2_logits_ms else None]),
                       default=None)
        if baseline is None:
            lines.append(f"- 最优 backbone-swap = **{bb}** (mean {s['mean']:.4f})，"
                         "但缺少 R2Plus1D rescue 基线，无法对比。")
        elif s["mean"] > baseline:
            lines.append(f"- 最优 backbone-swap = **{bb}** (mean {s['mean']:.4f}) "
                         f"> R2Plus1D rescue 最优 {baseline:.4f} → **换 backbone 有收益**。")
        else:
            lines.append(f"- 最优 backbone-swap = **{bb}** (mean {s['mean']:.4f}) "
                         f"≤ R2Plus1D rescue 最优 {baseline:.4f} → **换 backbone 无明显收益**。")
    lines.append("")

    # ---- Q4: strict main candidate verdict --------------------------------
    lines.append("## Q4. strict 论文主候选 backbone 选哪个?\n")
    # a backbone qualifies if mean>=0.95 and (n>=2 -> std<0.015)
    qualified = [
        (bb, s) for bb, s in ranked
        if s["mean"] >= RECOVERY_AUROC and (s["n"] < 2 or s["std"] < STABILITY_STD)
    ]
    # prefer ones with n>=3 (real multi-seed evidence)
    strong = [(bb, s) for bb, s in qualified if s["n"] >= 3]
    if strong:
        bb, s = strong[0]
        lines.append(f"### 决策: **{bb}**")
        lines.append(f"理由: 多 seed (n={s['n']}) 下 AUROC {s['mean']:.4f} ± {s['std']:.4f}，"
                     f"达标 (≥{RECOVERY_AUROC}) 且稳定 (std<{STABILITY_STD})，"
                     "可作为 strict 主候选 backbone。")
    elif qualified:
        bb, s = qualified[0]
        lines.append(f"### 决策: **{bb}（暂定，需补 seed）**")
        lines.append(f"理由: {bb} 均值 {s['mean']:.4f} 达标，但 seed 数 n={s['n']} < 3，"
                     "稳定性证据不足；建议补到 ≥3 seed 后再定。")
    else:
        lines.append("### 决策: **暂无合格 backbone（等价 D）**")
        lines.append(f"理由: 所有 backbone 要么均值 < {RECOVERY_AUROC}，要么多 seed std ≥ "
                     f"{STABILITY_STD}。strict 主结果仍无法靠换 backbone 拿下，"
                     "维持 non-strict 0521 主结果 + strict 补充验证的结论。")
    lines.append("")
    lines.append("### 备注")
    lines.append(f"- 门槛为自动判定（AUROC ≥ {RECOVERY_AUROC}，多 seed std < {STABILITY_STD}）；"
                 "属候选建议，投稿前以人工复核为准。")
    lines.append("- 本批不改任何代码，仅切换 `--ct-model`；KD 配方、split、bs 与 outputs0529 "
                 "ResNet3D18 strict run 对齐（bs 若用 BATCH_SIZE=2 则与 R2Plus1D bs=1 非严格可比）。")
    lines.append("- resnet3d18 的 seed42 复用 outputs0529 单 seed，未重训。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- context readers (read-only) ------------------------------------

BASELINE_RUNS = [
    "ct_text_student_kd_resnet3d18_full_combo_strict_ref1019",
    "ct_text_student_kd_r2plus1d_full_combo_strict_ref1019",
    "ct_text_student_kd_r2plus1d_logits_only_strict_ref1019",
]


def _row_from_metrics(name: str, d: dict[str, Any], source: str,
                      seed_fallback: Any = None) -> dict[str, Any]:
    cfg = d.get("config") or {}
    tm = d.get("test_metrics") or {}
    return {
        "run_name": name,
        "source": source,
        "ct_model": cfg.get("ct_model"),
        "modalities": ",".join(cfg.get("modalities") or []),
        "distill_methods": ",".join(cfg.get("distill_methods") or []),
        "distillation_alpha": cfg.get("distillation_alpha"),
        "seed": cfg.get("seed", seed_fallback),
        "auroc": tm.get("auroc"),
        "balanced_accuracy": tm.get("balanced_accuracy"),
        "f1": tm.get("f1"),
        "best_epoch": d.get("best_epoch"),
        "notes": f"{source} (read-only)",
    }


def read_context(baseline_dir: Path, rescue_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if baseline_dir.is_dir():
        for name in BASELINE_RUNS:
            d = _read_json(baseline_dir / name / "metrics.json")
            if d:
                out.append(_row_from_metrics(name, d, "outputs0529"))
    if rescue_dir.is_dir():
        for p in sorted(rescue_dir.iterdir()):
            if not p.is_dir() or p.name in {"logs", "scripts_used", "__pycache__"}:
                continue
            d = _read_json(p / "metrics.json")
            if d:
                out.append(_row_from_metrics(p.name, d, "outputs0530_strict_rescue"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0530_backbone_swap"))
    p.add_argument("--reference-manifest", type=Path,
                   default=Path("outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"))
    p.add_argument("--baseline-dir", type=Path, default=Path("outputs0529"))
    p.add_argument("--rescue-dir", type=Path, default=Path("outputs0530_strict_rescue"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 1

    ref_path: Path | None = args.reference_manifest.resolve() if args.reference_manifest else None
    if ref_path is not None and not ref_path.is_file():
        print(f"[WARN] reference manifest not found: {ref_path}", file=sys.stderr)
        ref_path = None

    runs = discover_runs(root)
    records = [analyse(r) for r in runs]
    context_rows = read_context(args.baseline_dir.resolve(), args.rescue_dir.resolve())

    csv_path = root / "backbone_swap_metrics.csv"
    md_path = root / "backbone_swap_metrics.md"
    summary_path = root / "backbone_swap_summary.md"

    write_csv(records, csv_path)
    write_metrics_md(records, md_path, root, ref_path, context_rows)
    write_summary(records, context_rows, summary_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(f"Runs analysed: {len(records)} (context rows: {len(context_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
