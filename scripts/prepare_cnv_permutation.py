#!/usr/bin/env python3
"""Create a split-local, whole-row CNV permutation with a traceable mapping."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.utils.attribution_audit import (  # noqa: E402
    AttributionAuditError,
    atomic_write_json,
    load_split_manifest,
    sha256_file,
    split_local_donor_mapping,
    stable_json_hash,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-tsv", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--cnv-permutation-seed", type=int, required=True)
    parser.add_argument("--gene-id-col", type=str, default=None)
    parser.add_argument("--gene-label-col", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for output in (args.output_tsv, args.mapping_csv):
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite existing permutation artifact: {output}")

    split_by_id = load_split_manifest(args.split_manifest)
    with args.gene_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields or not rows:
        raise AttributionAuditError(f"empty CNV table: {args.gene_tsv}")

    id_col = args.gene_id_col or fields[0]
    label_col = args.gene_label_col or (fields[1] if len(fields) > 1 else None)
    if id_col not in fields:
        raise AttributionAuditError(f"gene ID column not found: {id_col}")
    feature_cols = [field for field in fields if field not in {id_col, label_col}]
    if not feature_cols:
        raise AttributionAuditError("CNV table has no feature columns to permute")

    row_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        sid = str(row.get(id_col, "")).strip()
        if not sid:
            continue
        if sid in row_by_id:
            raise AttributionAuditError(f"duplicate CNV sample ID: {sid}")
        row_by_id[sid] = row

    aligned_ids = [sid for sid in split_by_id if sid in row_by_id]
    missing = sorted(set(split_by_id) - set(row_by_id))
    if missing:
        raise AttributionAuditError(f"split manifest identities missing from CNV table: {missing[:20]}")
    mapping = split_local_donor_mapping(
        aligned_ids,
        [split_by_id[sid] for sid in aligned_ids],
        args.cnv_permutation_seed,
    )

    output_by_id = {sid: dict(row) for sid, row in row_by_id.items()}
    mapping_rows: list[dict[str, object]] = []
    for record in mapping:
        target_id = str(record["sample_id"])
        donor_id = str(record["donor_sample_id"])
        target = output_by_id[target_id]
        donor = row_by_id[donor_id]
        for feature in feature_cols:
            target[feature] = donor[feature]
        mapping_rows.append(
            {
                **record,
                "target_feature_hash_before": stable_json_hash([row_by_id[target_id][field] for field in feature_cols]),
                "donor_feature_hash": stable_json_hash([donor[field] for field in feature_cols]),
                "target_feature_hash_after": stable_json_hash([target[field] for field in feature_cols]),
            }
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_tsv.with_name(f".{args.output_tsv.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for source_row in rows:
            sid = str(source_row.get(id_col, "")).strip()
            writer.writerow(output_by_id.get(sid, source_row))
    temporary.replace(args.output_tsv)

    mapping_fields = [
        "split",
        "sample_id",
        "donor_sample_id",
        "target_index",
        "donor_index",
        "is_fixed_point",
        "permutation_seed",
        "target_feature_hash_before",
        "donor_feature_hash",
        "target_feature_hash_after",
    ]
    write_csv_rows(args.mapping_csv, mapping_rows, mapping_fields)
    metadata_path = args.metadata_json or args.output_tsv.with_suffix(args.output_tsv.suffix + ".metadata.json")
    atomic_write_json(
        metadata_path,
        {
            "source_gene_tsv": str(args.gene_tsv),
            "source_gene_tsv_sha256": sha256_file(args.gene_tsv),
            "split_manifest": str(args.split_manifest),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "output_tsv": str(args.output_tsv),
            "output_tsv_sha256": sha256_file(args.output_tsv),
            "mapping_csv": str(args.mapping_csv),
            "mapping_csv_sha256": sha256_file(args.mapping_csv),
            "gene_id_col": id_col,
            "gene_label_col": label_col,
            "feature_count": len(feature_cols),
            "permuted_identity_count": len(mapping_rows),
            "cnv_permutation_seed": args.cnv_permutation_seed,
            "permutation_scope": "independent within train/val/test",
            "permutation_unit": "complete CNV feature row",
            "labels_used_for_permutation": False,
            "fixed_point_count": sum(int(row["is_fixed_point"]) for row in mapping_rows),
        },
    )
    print(f"[OK] wrote split-local CNV permutation: {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
