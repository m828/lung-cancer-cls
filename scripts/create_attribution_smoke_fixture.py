#!/usr/bin/env python3
"""Create a small balanced multimodal fixture for 0542 execution smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows, fields, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    complete = args.root / "fixture_complete.json"
    if complete.is_file() and not args.force:
        print(f"[SKIP] existing smoke fixture: {args.root}")
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    ct_root = args.root / "ct"
    ct_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5402)
    plan = [("train", 24), ("val", 12), ("test", 12)]
    metadata = []
    genes = []
    text = []
    manifest = []
    index = 0
    for split, count in plan:
        for local_index in range(count):
            label = local_index % 2
            sample = f"SMOKE_{index:04d}"
            record = f"TEXT_{index:04d}"
            label_name = "肺癌" if label else "健康对照"
            volume = rng.normal(loc=0.7 * label, scale=1.0, size=(8, 20, 20)).astype("float32")
            np.save(ct_root / f"{sample}.npy", volume)
            metadata.append(
                {
                    "SampleID": sample,
                    "record_id": record,
                    "样本类型": label_name,
                    "CT_numpy_cloud路径": f"{sample}.npy",
                    "CT_train_val_split": split,
                }
            )
            genes.append(
                {
                    "SampleID": sample,
                    "label": label_name,
                    **{f"cnv_{feature:03d}": float(rng.normal(loc=0.4 * label, scale=1.0)) for feature in range(12)},
                }
            )
            text.append(
                {
                    "record_id": record,
                    **{f"bert_{feature:04d}": float(rng.normal(loc=0.25 * label, scale=1.0)) for feature in range(16)},
                }
            )
            manifest.append({"sample_id": sample, "record_id": record, "label_name": label_name, "label": label, "split": split, "assigned_split": split})
            index += 1
    write_csv(args.root / "metadata.csv", metadata, list(metadata[0]))
    write_csv(args.root / "gene.tsv", genes, list(genes[0]), delimiter="\t")
    write_csv(args.root / "text.tsv", text, list(text[0]), delimiter="\t")
    (args.root / "text.tsv.meta.json").write_text(json.dumps({"record_id_col": "record_id", "embedding_backend": "smoke_fixture"}, indent=2), encoding="utf-8")
    write_csv(args.root / "split_manifest.csv", manifest, list(manifest[0]))
    complete.write_text(json.dumps({"status": "complete", "samples": index, "split_counts": {split: count for split, count in plan}}, indent=2), encoding="utf-8")
    print(f"[OK] attribution smoke fixture: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
