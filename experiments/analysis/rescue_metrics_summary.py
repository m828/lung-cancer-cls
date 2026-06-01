#!/usr/bin/env python3
"""Summarise the outputs0530_strict_rescue batch.

Reads every run directory under ``--root``:

  outputs0530_strict_rescue/
    r2plus1d_lite_combo_alpha02_strict_seed42/
    r2plus1d_lite_combo_alpha01_strict_seed42/
    r2plus1d_logits_only_strict_seed{43,44,45}/
    r2plus1d_lite_combo_alpha02_strict_seed{43,44,45}/   (conditional)

Produces three artefacts in ``--root``:

  rescue_metrics.csv     — per-run row with split / strict / metrics / status
  rescue_metrics.md      — same as a markdown table + readable explanation
  rescue_summary.md      — answers to the 5 rescue questions

Also reads matching strict baselines from --baseline-dir (default outputs0529)
to compare lite-combo against the failed full-combo strict seed 42 and the
existing strict logits-only seed 42 / strict ResNet3D full-combo.

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

METRIC_FIELDS = [
    "auroc",
    "balanced_accuracy",
    "f1",
    "recall",
    "specificity",
    "precision",
    "accuracy",
    "ece",
    "brier_score",
]

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
                     ref: Path | None, baseline_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# Rescue batch metrics — outputs0530_strict_rescue\n"]
    lines.append(f"- Root: `{root}`")
    if ref is not None:
        lines.append(f"- Reference manifest: `{ref}`")
    lines.append(f"- Expected split: total=1019, train=652, val=163, test=204")
    lines.append(f"- Runs discovered: {len(records)}")
    lines.append("")
    lines.append("## Rescue runs")
    headers = ["run", "ct_model", "modalities", "distill_methods", "alpha", "T",
               "seed", "best_ep", "split", "test_rows", "strict",
               "AUROC", "BAcc", "F1", "Recall", "Spec", "ECE", "Brier",
               "warn_n", "status", "notes"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in records:
        split = f"{r.get('split_total','-')}/{r.get('train','-')}/{r.get('val','-')}/{r.get('test','-')}"
        row = [
            r["run_name"], r["ct_model"], r["modalities"], r["distill_methods"],
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

    if baseline_rows:
        lines.append("")
        lines.append("## Baseline strict runs (from outputs0529, read-only)")
        lines.append("| run | ct_model | modalities | distill | alpha | seed | AUROC | BAcc | F1 | notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in baseline_rows:
            lines.append("| " + " | ".join(_md_cell(c) for c in [
                r["run_name"], r.get("ct_model",""), r.get("modalities",""),
                r.get("distill_methods",""), _fmt(r.get("distillation_alpha")),
                _fmt(r.get("seed")), _fmt(r.get("auroc")),
                _fmt(r.get("balanced_accuracy")), _fmt(r.get("f1")), r.get("notes",""),
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


def _pick(records: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [r for r in records if predicate(r)]


def _g(r: dict[str, Any], key: str) -> Any:
    v = r.get(key)
    return v if isinstance(v, (int, float)) else None


def write_summary(records: list[dict[str, Any]], baseline_rows: list[dict[str, Any]],
                  out: Path) -> None:
    lite02 = _pick(records, lambda r: r["run_name"].startswith("r2plus1d_lite_combo_alpha02_strict_seed"))
    lite01 = _pick(records, lambda r: r["run_name"].startswith("r2plus1d_lite_combo_alpha01_strict_seed"))
    logits = _pick(records, lambda r: r["run_name"].startswith("r2plus1d_logits_only_strict_seed"))

    # baseline references
    base_logits_seed42 = next(
        (r for r in baseline_rows if "logits_only" in r["run_name"]), None)
    base_full_combo_seed42 = next(
        (r for r in baseline_rows if "full_combo_strict_ref1019" in r["run_name"]
         and "resnet" not in r["run_name"] and "_seed" not in r["run_name"]),
        None,
    )
    base_resnet_full = next(
        (r for r in baseline_rows if "resnet3d18_full_combo" in r["run_name"]),
        None,
    )

    lines: list[str] = ["# Rescue summary — outputs0530_strict_rescue\n"]

    # Q1 — lite combo recovery
    lines.append("## Q1. lite-combo 是否恢复到 AUROC ≥ 0.95?\n")
    lite02_seed42 = next((r for r in lite02 if r["run_name"].endswith("seed42")), None)
    lite01_seed42 = next((r for r in lite01 if r["run_name"].endswith("seed42")), None)
    if lite02_seed42 is None and lite01_seed42 is None:
        lines.append("- **MISSING**: 找不到 lite-combo seed42 的任何一个 run。")
    else:
        if lite02_seed42:
            auc = _g(lite02_seed42, "auroc")
            ok = "✅" if auc and auc >= 0.95 else "❌"
            lines.append(f"- lite-combo α=0.2 seed42: AUROC = {_fmt(auc)} {ok}")
        else:
            lines.append("- lite-combo α=0.2 seed42: MISSING")
        if lite01_seed42:
            auc = _g(lite01_seed42, "auroc")
            ok = "✅" if auc and auc >= 0.95 else "❌"
            lines.append(f"- lite-combo α=0.1 seed42: AUROC = {_fmt(auc)} {ok}")
        else:
            lines.append("- lite-combo α=0.1 seed42: MISSING")
        if base_full_combo_seed42:
            lines.append(
                f"- Reference: outputs0529 full-combo strict seed42 AUROC = "
                f"{_fmt(base_full_combo_seed42.get('auroc'))} (the run that collapsed)")
    lines.append("")

    # Q2 — alpha 0.2 vs 0.1
    lines.append("## Q2. α=0.2 vs α=0.1 哪个更稳?\n")
    a02 = _g(lite02_seed42 or {}, "auroc")
    a01 = _g(lite01_seed42 or {}, "auroc")
    if a02 is None and a01 is None:
        lines.append("- **MISSING**: 没有 seed42 数据可比。")
    else:
        rows = [("α=0.2", lite02_seed42), ("α=0.1", lite01_seed42)]
        lines.append("| variant | AUROC | BAcc | F1 | best_ep |")
        lines.append("|---|---|---|---|---|")
        for tag, r in rows:
            if r is None:
                lines.append(f"| {tag} | MISSING | MISSING | MISSING | - |")
            else:
                lines.append(
                    f"| {tag} | {_fmt(r.get('auroc'))} | {_fmt(r.get('balanced_accuracy'))} "
                    f"| {_fmt(r.get('f1'))} | {_fmt(r.get('best_epoch'))} |"
                )
        if a02 is not None and a01 is not None:
            winner = "α=0.2" if a02 >= a01 else "α=0.1"
            lines.append(f"\n→ 单 seed 比较，更高 AUROC 的是 **{winner}**。单点比较只能作为方向参考，并非"
                          "稳定性结论。")
        elif a02 is not None:
            lines.append("\n→ 只有 α=0.2 可读，**α=0.1 缺失**，无法比较。")
        else:
            lines.append("\n→ 只有 α=0.1 可读，**α=0.2 缺失**，无法比较。")
    lines.append("")

    # Q3 — logits-only multi-seed stability
    lines.append("## Q3. logits-only 多 seed 是否稳定?\n")
    rows: list[dict[str, Any]] = []
    if base_logits_seed42 is not None:
        rows.append(base_logits_seed42 | {"seed": 42, "source": "outputs0529"})
    for r in logits:
        rows.append(r | {"source": "outputs0530"})
    if not rows:
        lines.append("- **MISSING**: 没有任何 logits-only run 可读。")
    else:
        seeds_seen = sorted({r.get("seed") for r in rows if r.get("seed") is not None})
        lines.append(f"- Seeds 收集: {seeds_seen}")
        lines.append("\n| source | seed | AUROC | BAcc | F1 | best_ep |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r.get('source','-')} | {r.get('seed','-')} "
                f"| {_fmt(r.get('auroc'))} | {_fmt(r.get('balanced_accuracy'))} "
                f"| {_fmt(r.get('f1'))} | {_fmt(r.get('best_epoch'))} |"
            )
        aucs = [_g(r, "auroc") for r in rows if _g(r, "auroc") is not None]
        baccs = [_g(r, "balanced_accuracy") for r in rows if _g(r, "balanced_accuracy") is not None]
        f1s = [_g(r, "f1") for r in rows if _g(r, "f1") is not None]
        if aucs:
            ms = _mean_std(aucs); ms_b = _mean_std(baccs); ms_f = _mean_std(f1s)
            lines.append(f"\n- AUROC: mean ± std = {ms[0]:.4f} ± {ms[1]:.4f} (n={len(aucs)})")
            if ms_b: lines.append(f"- BAcc : mean ± std = {ms_b[0]:.4f} ± {ms_b[1]:.4f} (n={len(baccs)})")
            if ms_f: lines.append(f"- F1   : mean ± std = {ms_f[0]:.4f} ± {ms_f[1]:.4f} (n={len(f1s)})")
            if ms[1] < 0.015 and ms[0] >= 0.95:
                lines.append("\n→ **稳定** — std < 0.015 且均值 ≥ 0.95。")
            elif len(aucs) < 3:
                lines.append("\n→ **判断不足** — seed 数 < 3，无法定论。")
            else:
                lines.append(
                    f"\n→ **不稳定** — std {ms[1]:.4f} ≥ 0.015 或均值 {ms[0]:.4f} < 0.95。")
    lines.append("")

    # Q4 — still need to rescue full-combo?
    lines.append("## Q4. 是否还需要继续救 full-combo?\n")
    if lite02_seed42 is None and lite01_seed42 is None:
        lines.append("- **MISSING** — 没有 rescue 结果，无法回答。")
    else:
        best_lite_auc = max(filter(None, [a02, a01]), default=None)
        if best_lite_auc is None:
            lines.append("- **MISSING** — lite-combo 结果不可读。")
        elif best_lite_auc >= 0.95:
            lines.append(
                f"- **建议放弃 R2Plus1D + full-combo strict** — lite-combo 已可恢复到 "
                f"AUROC {best_lite_auc:.4f} ≥ 0.95，无需再纠结 relation/attention 这两个"
                "梯度死的组件。把它们从 distill_methods 中移除即可。")
        else:
            lines.append(
                f"- **lite-combo 也未达 0.95** (best={best_lite_auc:.4f}) — 不是 KD 分量"
                "选择问题，而是 R2Plus1D strict 整体的优化问题。继续救 full-combo 风险高,"
                "优先考虑切换到 backbone (ResNet3D) 或 logits-only。")
    lines.append("")

    # Q5 — paper main candidate
    lines.append("## Q5. strict 论文主候选应该选择: A / B / C / D?\n")
    cand_lines: list[str] = []
    # collect best of each option
    A_auc = max(filter(None, [a02, a01]), default=None)  # lite combo
    B_aucs = []
    for r in rows:
        x = _g(r, "auroc")
        if x is not None: B_aucs.append(x)
    B_mean = statistics.mean(B_aucs) if B_aucs else None
    B_std = statistics.stdev(B_aucs) if len(B_aucs) > 1 else 0.0 if B_aucs else None
    C_auc = _g(base_resnet_full or {}, "auroc") if base_resnet_full else None

    cand_lines.append(f"- A. R2Plus1D lite-combo strict — best AUROC: {_fmt(A_auc)}")
    if B_mean is None:
        cand_lines.append("- B. R2Plus1D logits-only strict — AUROC: MISSING")
    else:
        cand_lines.append(
            f"- B. R2Plus1D logits-only strict — AUROC: {B_mean:.4f} ± "
            f"{B_std:.4f} (n={len(B_aucs)})")
    cand_lines.append(f"- C. ResNet3D18 full-combo strict — AUROC: {_fmt(C_auc)} (single seed in outputs0529)")
    cand_lines.append("- D. 暂不选择 strict 主候选")
    lines.extend(cand_lines)
    lines.append("")

    # decision logic
    pick = None
    reason = ""
    if A_auc is not None and A_auc >= 0.95 and (
            B_mean is None or A_auc >= B_mean):
        pick = "A"
        reason = (f"R2Plus1D lite-combo (drop relation/attention) 已恢复到 AUROC "
                  f"{A_auc:.4f} ≥ 0.95，证明 strict 设置下 R2Plus1D + KD 仍可工作；"
                  "结构与非 strict 主结果对齐，最易在论文中复用。")
    elif B_mean is not None and B_mean >= 0.95 and (B_std or 0) < 0.015:
        pick = "B"
        reason = (f"logits-only 在多 seed 下稳定 (AUROC {B_mean:.4f} ± "
                  f"{B_std:.4f})。lite-combo 未达标或未补齐多 seed。")
    elif C_auc is not None and C_auc >= 0.95:
        pick = "C"
        reason = (f"R2Plus1D 在 strict 下未恢复，ResNet3D18 full-combo (AUROC {C_auc:.4f})"
                  "是最稳的 strict 候选。")
    else:
        pick = "D"
        reason = "所有 strict 候选要么 < 0.95，要么 seed 不足，无法宣称主候选。"

    lines.append(f"### 决策: **{pick}**")
    lines.append(f"理由: {reason}")
    lines.append("")
    lines.append("### 备注")
    lines.append("- 决策基于自动门槛（AUROC ≥ 0.95 + 多 seed std < 0.015）。这只是 *候选* 建议,"
                 "实际投稿前应该补足 lite-combo 和 ResNet3D 的多 seed 稳定性。")
    lines.append("- D 等价于：strict 暂未具备主候选条件，论文继续以 non-strict 0521 主结果为主,"
                 "strict 作为补充验证 (即 outputs0529 + 当前 rescue 的实际结论)。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- baseline reader for outputs0529 -------------------------------

BASELINE_RUNS = [
    "ct_text_student_kd_r2plus1d_full_combo_strict_ref1019",
    "ct_text_student_kd_r2plus1d_logits_only_strict_ref1019",
    "ct_text_student_kd_resnet3d18_full_combo_strict_ref1019",
]


def read_baseline(baseline_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not baseline_dir.is_dir():
        return out
    for name in BASELINE_RUNS:
        p = baseline_dir / name / "metrics.json"
        d = _read_json(p)
        if not d:
            continue
        cfg = d.get("config") or {}
        tm = d.get("test_metrics") or {}
        out.append({
            "run_name": name,
            "ct_model": cfg.get("ct_model"),
            "modalities": ",".join(cfg.get("modalities") or []),
            "distill_methods": ",".join(cfg.get("distill_methods") or []),
            "distillation_alpha": cfg.get("distillation_alpha"),
            "seed": cfg.get("seed"),
            "auroc": tm.get("auroc"),
            "balanced_accuracy": tm.get("balanced_accuracy"),
            "f1": tm.get("f1"),
            "best_epoch": d.get("best_epoch"),
            "notes": "outputs0529 baseline (read-only)",
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs0530_strict_rescue"))
    p.add_argument("--reference-manifest", type=Path,
                   default=Path("outputs/ct_cnv_text_teacher_mvn_tvt/split_manifest.csv"))
    p.add_argument("--baseline-dir", type=Path, default=Path("outputs0529"))
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
    baseline_rows = read_baseline(args.baseline_dir.resolve())

    csv_path = root / "rescue_metrics.csv"
    md_path = root / "rescue_metrics.md"
    summary_path = root / "rescue_summary.md"

    write_csv(records, csv_path)
    write_metrics_md(records, md_path, root, ref_path, baseline_rows)
    write_summary(records, baseline_rows, summary_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(f"Runs analysed: {len(records)} (baseline rows: {len(baseline_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
