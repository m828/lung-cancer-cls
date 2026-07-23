#!/usr/bin/env python3
"""Create isolated synthetic artifacts for analysis-only 0542 smoke checks.

These files exercise identity alignment, contrasts, weight/permutation audits,
and checkpoint-summary parsing. They are fixtures, never scientific results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FACTORIAL_ARMS = (
    "S0_MATCHED",
    "KD_CT_TEXT_UNIFORM",
    "KD_CT_TEXT_CONFIDENCE",
    "KD_CT_TEXT_CNV_UNIFORM",
    "KD_CT_TEXT_CNV_CONFIDENCE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probabilities(count: int, offset: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        label = index % 2
        signed = 0.22 + 0.035 * (index % 5) + offset
        probability = 0.5 + signed if label else 0.5 - signed
        probability = min(0.98, max(0.02, probability))
        rows.append(
            {
                "sample_id": f"ANALYSIS_SMOKE_{index:03d}",
                "label": label,
                "prob_normal": 1.0 - probability,
                "prob_malignant": probability,
                "prediction": int(probability >= 0.5),
            }
        )
    return rows


def write_run(run_dir: Path, offset: float, *, arm: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "val_predictions.csv", probabilities(12, offset / 2.0))
    write_csv(run_dir / "test_predictions.csv", probabilities(18, offset))
    metrics = {
        "config": {"seed": 42},
        "cached_kd_config": {"run_mode": "smoke"},
        "analysis_smoke_fixture": True,
        "arm": arm,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "run_complete.json").write_text(
        json.dumps({"status": "complete", "run_mode": "smoke", "analysis_smoke_fixture": True}, indent=2),
        encoding="utf-8",
    )


def write_checkpoint_evaluations(run_dir: Path, arm_offset: float) -> None:
    records = []
    for criterion_index, criterion in enumerate(
        ("val_loss", "val_auroc", "val_f1", "val_bacc", "val_composite", "last")
    ):
        path = run_dir / "checkpoints" / ("last.pt" if criterion == "last" else f"best_{criterion}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"analysis-smoke:{criterion}".encode("ascii"))
        adjustment = arm_offset + criterion_index * 0.001
        records.append(
            {
                "checkpoint_criterion": criterion,
                "status": "complete",
                "checkpoint_path": str(path.resolve()),
                "epoch": 1,
                "test_metrics_used_for_selection": False,
                "test_metrics": {
                    "auroc": 0.80 + adjustment,
                    "balanced_accuracy": 0.72 + adjustment,
                    "f1": 0.73 + adjustment,
                    "recall": 0.74 + adjustment,
                    "specificity": 0.70 + adjustment,
                    "ece": 0.14 - adjustment,
                    "brier_score": 0.16 - adjustment,
                },
            }
        )
    (run_dir / "checkpoint_evaluations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def stable_hash(values: list[float]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def main() -> int:
    args = parse_args()
    marker = args.root / "ANALYSIS_SMOKE_FIXTURE_ONLY.json"
    if marker.is_file() and not args.force:
        print(f"[SKIP] analysis smoke fixture: {args.root}")
        return 0
    args.root.mkdir(parents=True, exist_ok=True)

    factorial_root = args.root / "01_binary_factorial"
    factorial_offsets = {
        "S0_MATCHED": 0.000,
        "KD_CT_TEXT_UNIFORM": 0.005,
        "KD_CT_TEXT_CONFIDENCE": 0.010,
        "KD_CT_TEXT_CNV_UNIFORM": 0.015,
        "KD_CT_TEXT_CNV_CONFIDENCE": 0.025,
    }
    for arm, offset in factorial_offsets.items():
        run_dir = factorial_root / arm / "smoke" / "seed42"
        write_run(run_dir, offset, arm=arm)
        write_checkpoint_evaluations(run_dir, offset)

    shuffle_root = args.root / "02_shuffled_confidence"
    for arm, offset in {"uniform": 0.015, "true_confidence": 0.025, "shuffled_confidence": 0.008}.items():
        write_run(shuffle_root / arm / "smoke" / "seed42", offset, arm=arm)
    confidence = [0.51 + index * 0.02 for index in range(18)]
    shuffled = confidence[1:] + confidence[:1]
    write_csv(
        shuffle_root / "shuffled_confidence" / "smoke" / "seed42" / "kd_weight_mapping.csv",
        [
            {
                "sample_id": f"ANALYSIS_SMOKE_{index:03d}",
                "donor_sample_id": f"ANALYSIS_SMOKE_{(index + 1) % len(confidence):03d}",
                "original_weight": confidence[index],
                "effective_weight": shuffled[index],
                "mode": "shuffled_confidence",
                "split": "train",
            }
            for index in range(len(confidence))
        ],
    )

    cnv_root = args.root / "03_cnv_permutation"
    for arm, offset in {"ct_text": 0.005, "permuted_cnv": 0.003, "real_cnv": 0.018}.items():
        write_run(cnv_root / "teachers" / arm / "smoke" / "seed42", offset, arm=arm)
    for arm, offset in {
        "ct_text_confidence": 0.010,
        "permuted_cnv_confidence": 0.007,
        "real_cnv_confidence": 0.025,
    }.items():
        write_run(cnv_root / "students" / arm / "smoke" / "seed42", offset, arm=arm)
    mapping_rows: list[dict[str, object]] = []
    index = 0
    for split in ("train", "val", "test"):
        ids = [f"{split.upper()}_{local:02d}" for local in range(4)]
        features = [[float(index + local), float(index + local + 100)] for local in range(4)]
        for local, sid in enumerate(ids):
            donor_local = (local + 1) % len(ids)
            donor_features = features[donor_local]
            mapping_rows.append(
                {
                    "split": split,
                    "sample_id": sid,
                    "donor_sample_id": ids[donor_local],
                    "target_index": index + local,
                    "donor_index": index + donor_local,
                    "is_fixed_point": 0,
                    "permutation_seed": 20042,
                    "target_feature_hash_before": stable_hash(features[local]),
                    "donor_feature_hash": stable_hash(donor_features),
                    "target_feature_hash_after": stable_hash(donor_features),
                }
            )
        index += len(ids)
    write_csv(cnv_root / "permutation_manifests" / "smoke_seed42.csv", mapping_rows)

    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": "analysis-only smoke fixture",
                "scientific_result": False,
                "contains_model_training": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] analysis smoke fixture: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
