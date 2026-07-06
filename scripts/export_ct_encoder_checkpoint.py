#!/usr/bin/env python3
"""Export CT encoder weights from a multimodal R3 student checkpoint.

The exported checkpoint intentionally contains only keys under ``ct_encoder.``
so the existing LIDC training entry can load it with:

    --init-checkpoint best_ct_encoder.pt --init-checkpoint-prefix ct_encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata-output", type=Path, required=True)
    p.add_argument("--source-run-dir", type=Path, default=None)
    p.add_argument("--lidc-model", default="densenet3d_121")
    p.add_argument("--lidc-num-classes", type=int, default=2)
    p.add_argument("--verify-lidc-load", action="store_true")
    return p.parse_args()


def unwrap_state(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                return nested
        if all(isinstance(k, str) for k in raw):
            return raw
    raise ValueError("Unsupported checkpoint format; expected a state_dict-like mapping.")


def normalize_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            continue
        clean = key[7:] if key.startswith("module.") else key
        if torch.is_tensor(value):
            out[clean] = value.detach().cpu()
    return out


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def verify_lidc_compatibility(
    prefixed_ct_state: dict[str, torch.Tensor],
    model_name: str,
    num_classes: int,
) -> dict[str, Any]:
    from lung_cancer_cls.model import build_model

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model_state = model.state_dict()
    stripped = {
        key[len("ct_encoder.") :]: value
        for key, value in prefixed_ct_state.items()
        if key.startswith("ct_encoder.")
    }
    compatible = {
        key: value
        for key, value in stripped.items()
        if key in model_state and getattr(value, "shape", None) == model_state[key].shape
    }
    unexpected = sorted(key for key in stripped if key not in model_state)
    incompatible = sorted(
        key
        for key, value in stripped.items()
        if key in model_state and getattr(value, "shape", None) != model_state[key].shape
    )
    missing = sorted(key for key in model_state if key not in compatible)

    model.load_state_dict(compatible, strict=False)
    return {
        "lidc_model": model_name,
        "lidc_num_classes": num_classes,
        "strict": False,
        "init_checkpoint_prefix": "ct_encoder.",
        "loaded_keys": len(compatible),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "incompatible_shape_keys": incompatible,
        "skipped_keys": len(stripped) - len(compatible),
    }


def main() -> int:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = normalize_state(unwrap_state(raw))
    ct_state = {key: value for key, value in state.items() if key.startswith("ct_encoder.")}
    if not ct_state:
        raise SystemExit(f"No ct_encoder.* keys found in checkpoint: {args.checkpoint}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ct_state, args.output)

    source_metrics = load_json(args.source_run_dir / "metrics.json" if args.source_run_dir else None)
    metadata: dict[str, Any] = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_run_dir": str(args.source_run_dir.resolve()) if args.source_run_dir else None,
        "output_checkpoint": str(args.output.resolve()),
        "contains_only_ct_encoder_prefixed_keys": True,
        "ct_encoder_key_count": len(ct_state),
        "ct_encoder_keys_preview": sorted(ct_state)[:20],
        "source_best_epoch": source_metrics.get("best_epoch") if source_metrics else None,
        "source_best_val_metrics": source_metrics.get("best_val_metrics") if source_metrics else None,
        "source_config": source_metrics.get("config") if source_metrics else None,
    }
    if args.verify_lidc_load:
        metadata["lidc_load_report"] = verify_lidc_compatibility(
            ct_state,
            args.lidc_model,
            args.lidc_num_classes,
        )

    args.metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] exported CT encoder keys: {len(ct_state)}")
    print(f"[OK] best_ct_encoder: {args.output}")
    print(f"[OK] metadata: {args.metadata_output}")
    if "lidc_load_report" in metadata:
        report = metadata["lidc_load_report"]
        print(
            "[OK] LIDC strict=False compatibility: "
            f"loaded={report['loaded_keys']} missing={len(report['missing_keys'])} "
            f"unexpected={len(report['unexpected_keys'])} skipped={report['skipped_keys']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
