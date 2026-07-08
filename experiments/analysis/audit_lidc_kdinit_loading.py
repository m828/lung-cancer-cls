#!/usr/bin/env python3
"""Audit LIDC KDInit checkpoint loading without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to audit an existing checkpoint.") from exc
    return torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--lidc-model", type=str, required=True)
    p.add_argument("--lidc-num-classes", type=int, default=2)
    p.add_argument("--init-prefix", type=str, default="ct_encoder.")
    p.add_argument("--source-run-dir", type=Path, default=None)
    p.add_argument("--source-backbone", type=str, default="")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    p.add_argument("--warn-ratio", type=float, default=0.50)
    return p.parse_args()


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def unwrap_state(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                return nested
        if all(isinstance(k, str) for k in raw):
            return raw
    raise ValueError("Unsupported checkpoint format; expected a state_dict-like mapping.")


def normalize_state(state: dict[str, Any], torch_mod: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch_mod.is_tensor(value):
            continue
        clean = key[7:] if key.startswith("module.") else key
        out[clean] = value.detach().cpu()
    return out


def infer_source_run_dir(checkpoint: Path) -> Path | None:
    parent = checkpoint.parent
    if (parent / "metrics.json").is_file():
        return parent
    metadata = read_json(parent / "checkpoint_metadata.json")
    if isinstance(metadata, dict) and metadata.get("source_run_dir"):
        return Path(str(metadata["source_run_dir"]))
    return None


def infer_source_backbone(checkpoint: Path, source_run_dir: Path | None, fallback: str) -> str:
    if fallback:
        return fallback
    metrics = read_json(source_run_dir / "metrics.json" if source_run_dir else None)
    config = (metrics or {}).get("config") or {}
    for key in ("ct_model", "model"):
        value = config.get(key)
        if value:
            return str(value)
    metadata = read_json(checkpoint.parent / "checkpoint_metadata.json")
    if isinstance(metadata, dict):
        source_config = metadata.get("source_config") or {}
        for key in ("ct_model", "model"):
            value = source_config.get(key)
            if value:
                return str(value)
    return "unknown"


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("-", "_")


def md_list(values: list[str]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- `{v}`" for v in values)


def is_head_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("classifier", "classification_head", "head", "fc", "logits"))


def is_text_fusion_head_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "text" in lowered
        or "fusion" in lowered
        or is_head_key(lowered)
    )


def write_outputs(report: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LIDC KDInit Loading Audit",
        "",
        f"- checkpoint path: `{report['checkpoint_path']}`",
        f"- LIDC backbone: `{report['lidc_backbone']}`",
        f"- source R3 backbone: `{report['source_r3_backbone']}`",
        f"- init prefix: `{report['init_prefix']}`",
        f"- strict: `{report['strict']}`",
        f"- loaded key count: `{report['loaded_key_count']}`",
        f"- missing key count: `{report['missing_key_count']}`",
        f"- unexpected key count: `{report['unexpected_key_count']}`",
        f"- incompatible shape key count: `{report['incompatible_shape_key_count']}`",
        f"- expected CT encoder key count: `{report['expected_ct_encoder_key_count']}`",
        f"- loaded CT encoder key count: `{report['loaded_ct_encoder_key_count']}`",
        f"- loaded key ratio: `{report['loaded_key_ratio']:.4f}`",
        f"- loads only CT encoder: `{report['loads_only_ct_encoder']}`",
        f"- text/fusion keys in checkpoint: `{report['text_or_fusion_key_count_in_checkpoint']}`",
        f"- loaded text/fusion keys: `{report['loaded_text_or_fusion_key_count']}`",
        f"- backbone mismatch: `{report['backbone_mismatch']}`",
        "",
        "## Warnings",
    ]
    warnings = report.get("warnings") or []
    lines.extend(f"- WARNING: {item}" for item in warnings)
    if not warnings:
        lines.append("- none")
    lines.extend([
        "",
        "## First 20 Loaded Keys",
        md_list(report.get("loaded_keys_preview", [])),
        "",
        "## First 20 Missing Keys",
        md_list(report.get("missing_keys_preview", [])),
        "",
        "## First 20 Unexpected Keys",
        md_list(report.get("unexpected_keys_preview", [])),
    ])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_run_dir = args.source_run_dir.expanduser().resolve() if args.source_run_dir else infer_source_run_dir(checkpoint)

    report: dict[str, Any] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "lidc_backbone": args.lidc_model,
        "source_r3_backbone": "unknown",
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "lidc_num_classes": args.lidc_num_classes,
        "init_prefix": args.init_prefix,
        "strict": False,
        "loaded_key_count": 0,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "incompatible_shape_key_count": 0,
        "expected_ct_encoder_key_count": 0,
        "loaded_ct_encoder_key_count": 0,
        "loaded_key_ratio": 0.0,
        "loaded_keys_preview": [],
        "missing_keys_preview": [],
        "unexpected_keys_preview": [],
        "loads_only_ct_encoder": False,
        "text_or_fusion_key_count_in_checkpoint": 0,
        "loaded_text_or_fusion_key_count": 0,
        "backbone_mismatch": None,
        "warnings": [],
    }

    if not checkpoint.is_file():
        report["warnings"].append("Checkpoint file is missing; KDInit cannot be audited.")
        write_outputs(report, args.output_json, args.output_md)
        print(f"[MISSING] checkpoint: {checkpoint}")
        return 2

    source_backbone = infer_source_backbone(checkpoint, source_run_dir, args.source_backbone)
    report["source_r3_backbone"] = source_backbone
    report["backbone_mismatch"] = normalize_name(args.lidc_model) != normalize_name(source_backbone)
    if report["backbone_mismatch"]:
        report["warnings"].append(
            f"Backbone mismatch: LIDC uses {args.lidc_model}, source R3 uses {source_backbone}. "
            "Use DenseNet3D121 LIDC backbone before comparing KDInit if the source R3 model is DenseNet3D121."
        )

    try:
        torch_mod = import_torch()
    except RuntimeError as exc:
        report["warnings"].append(str(exc))
        write_outputs(report, args.output_json, args.output_md)
        print(f"[ERROR] {exc}")
        return 2

    raw = torch_mod.load(checkpoint, map_location="cpu")
    normalized = normalize_state(unwrap_state(raw), torch_mod)
    text_or_fusion_keys = sorted(key for key in normalized if is_text_fusion_head_key(key))
    report["text_or_fusion_key_count_in_checkpoint"] = len(text_or_fusion_keys)

    if args.init_prefix:
        prefixed = {key: value for key, value in normalized.items() if key.startswith(args.init_prefix)}
        stripped = {key[len(args.init_prefix):]: value for key, value in prefixed.items()}
    else:
        prefixed = dict(normalized)
        stripped = dict(normalized)

    from lung_cancer_cls.model import build_model

    model = build_model(args.lidc_model, num_classes=args.lidc_num_classes, pretrained=False)
    model_state = model.state_dict()
    expected_ct_encoder_keys = sorted(key for key in model_state if not is_head_key(key))
    compatible = {
        key: value
        for key, value in stripped.items()
        if key in model_state and getattr(value, "shape", None) == model_state[key].shape
    }
    unexpected = sorted(key for key in stripped if key not in model_state)
    incompatible = sorted(
        key for key, value in stripped.items()
        if key in model_state and getattr(value, "shape", None) != model_state[key].shape
    )
    loaded_ct_encoder_keys = sorted(key for key in compatible if key in expected_ct_encoder_keys)
    missing = sorted(key for key in expected_ct_encoder_keys if key not in loaded_ct_encoder_keys)
    loaded_original_keys = sorted(
        f"{args.init_prefix}{key}" if args.init_prefix else key
        for key in compatible
    )
    loaded_text_or_fusion = [
        key for key in loaded_original_keys
        if is_text_fusion_head_key(key)
    ]
    loaded_non_ct_encoder = sorted(key for key in compatible if key not in expected_ct_encoder_keys)

    # Mirrors train.py loading behavior without mutating any training run.
    model.load_state_dict(compatible, strict=False)

    ratio = len(loaded_ct_encoder_keys) / max(1, len(expected_ct_encoder_keys))
    report.update({
        "checkpoint_key_count": len(normalized),
        "filtered_prefix_key_count": len(stripped),
        "lidc_model_key_count": len(model_state),
        "expected_ct_encoder_key_count": len(expected_ct_encoder_keys),
        "loaded_ct_encoder_key_count": len(loaded_ct_encoder_keys),
        "loaded_key_count": len(compatible),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "incompatible_shape_key_count": len(incompatible),
        "loaded_key_ratio": ratio,
        "loaded_keys_preview": loaded_original_keys[:20],
        "missing_keys_preview": missing[:20],
        "unexpected_keys_preview": unexpected[:20],
        "incompatible_shape_keys_preview": incompatible[:20],
        "loaded_non_ct_encoder_key_count": len(loaded_non_ct_encoder),
        "loaded_non_ct_encoder_keys_preview": loaded_non_ct_encoder[:20],
        "loads_only_ct_encoder": bool(
            (not args.init_prefix or all(key.startswith(args.init_prefix) for key in loaded_original_keys))
            and not loaded_text_or_fusion
            and not loaded_non_ct_encoder
        ),
        "loaded_text_or_fusion_key_count": len(loaded_text_or_fusion),
        "loaded_text_or_fusion_keys_preview": loaded_text_or_fusion[:20],
    })

    if len(compatible) == 0 or ratio < args.warn_ratio:
        report["warnings"].append("KDInit may not be a valid CT encoder initialization: too few CT encoder keys were loaded.")
    if not report["loads_only_ct_encoder"]:
        report["warnings"].append("KDInit did not prove CT-encoder-only loading. Check init prefix and checkpoint contents.")
    if loaded_text_or_fusion:
        report["warnings"].append("Text/fusion/head keys were loaded unexpectedly; KDInit should load CT encoder only.")

    write_outputs(report, args.output_json, args.output_md)
    print("[AUDIT] LIDC KDInit checkpoint loading")
    print(f"checkpoint path: {checkpoint}")
    print(f"LIDC backbone: {args.lidc_model}")
    print(f"source R3 backbone: {source_backbone}")
    print(f"loaded key count: {report['loaded_key_count']}")
    print(f"missing key count: {report['missing_key_count']}")
    print(f"unexpected key count: {report['unexpected_key_count']}")
    print(f"loaded key ratio: {ratio:.4f}")
    print("first loaded keys: " + ", ".join(report["loaded_keys_preview"][:20]))
    print("first missing keys: " + ", ".join(report["missing_keys_preview"][:20]))
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")
    print(f"[OK] audit json: {args.output_json}")
    print(f"[OK] audit md: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
