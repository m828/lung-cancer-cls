#!/usr/bin/env python3
"""Wrapper for KD training CLI.

Supported extensions (without editing `train_student_kd.py` / source code):
1) Gradient accumulation
2) Teacher-confidence-weighted KD loss
3) Calibration-aware teacher softening (teacher logits temperature scaling)
4) Wrapper-level warmup + cosine scheduling

Usage examples:
  python3 scripts/train_student_kd_suite_wrapper.py \
    --accumulation-steps 4 -- \
    python3 train_student_kd.py ...

  python3 scripts/train_student_kd_suite_wrapper.py \
    --confidence-mode soft --confidence-threshold 0.85 --confidence-floor 0.10 -- \
    python3 train_student_kd.py ...

  python3 scripts/train_student_kd_suite_wrapper.py \
    --wrapper-scheduler warmup_cosine -- \
    python3 train_student_kd.py ...
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import torch.nn.functional as F


def _normalize_passthrough(passthrough: list[str]) -> list[str]:
    args = list(passthrough)
    if args and args[0] == "--":
        args = args[1:]
    if args and Path(args[0]).name.startswith("python"):
        args = args[1:]
    if args and args[0].endswith(".py"):
        args = args[1:]
    return args


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="KD wrapper with extra runtime controls.")
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for student KD training.",
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        default="none",
        choices=["none", "soft", "hard"],
        help="Apply teacher-confidence weighting to KL distillation term.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Hard mode threshold for teacher confidence",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.0,
        help="Minimum confidence weight in confidence-weighted KD",
    )
    parser.add_argument(
        "--confidence-gamma",
        type=float,
        default=1.0,
        help="Exponent for soft confidence weighting",
    )
    parser.add_argument(
        "--wrapper-scheduler",
        type=str,
        default="",
        choices=["", "warmup_cosine", "cosine"],
        help="Runtime scheduler override without editing source code.",
    )
    parser.add_argument(
        "--wrapper-teacher-temperature-scale",
        type=float,
        default=1.0,
        help="Additional temperature scaling only applied to teacher logits "
             "(for calibration-aware KD).",
    )
    parser.add_argument(
        "--wrapper-calibration-aware",
        action="store_true",
        help=(
            "Enable calibration-aware mode: apply teacher-only temperature scaling "
            "before distillation."
        ),
    )
    args, passthrough = parser.parse_known_args()
    return args, _normalize_passthrough(passthrough)


def _patch_distillation_loss(
    module,
    mode: str,
    threshold: float,
    floor: float,
    gamma: float,
    teacher_scale: float = 1.0,
    calibration_aware: bool = False,
) -> None:
    original = module.distillation_loss

    if mode == "none" and not calibration_aware:
        return

    def confidence_weighted_kd_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-4)
        scaled_teacher_temp = temp
        if calibration_aware and teacher_scale > 0:
            scaled_teacher_temp = temp * float(teacher_scale)
        scaled_student = student_logits / temp
        scaled_teacher = teacher_logits / scaled_teacher_temp

        student_log_prob = F.log_softmax(scaled_student, dim=1)
        with torch.no_grad():
            teacher_prob = F.softmax(scaled_teacher, dim=1)
            conf = teacher_prob.max(dim=1).values
            if mode == "hard":
                weights = torch.where(
                    conf >= float(threshold),
                    torch.ones_like(conf),
                    torch.full_like(conf, float(floor)),
                )
            else:
                conf_power = conf.clamp(0.0, 1.0).pow(float(gamma))
                weights = float(floor) + (1.0 - float(floor)) * conf_power

        # KL per sample => shape [N], then weighted average.
        per_sample = F.kl_div(student_log_prob, teacher_prob, reduction="none", log_target=False).sum(dim=1) * (temp ** 2)
        denom = torch.clamp(weights.sum(), min=1e-12)
        return (per_sample * weights).sum() / denom

    module.distillation_loss = confidence_weighted_kd_loss
    # Keep reference for debugging if needed.
    module.__dict__["_wrapped_distillation_loss_original"] = original


def _patch_train_epoch(module, accumulation_steps: int) -> None:
    if accumulation_steps <= 1:
        return

    original_train_epoch = module.train_epoch_student_kd

    def patched_train_epoch_student_kd(
        student,
        teacher,
        loader,
        device,
        criterion,
        optimizer,
        alpha: float,
        temperature: float,
        distill_methods,
        distill_method_weights,
        distill_projectors,
        distillation_feature_loss_name: str = "smooth_l1",
        distillation_normalize_features: bool = False,
        scheduler=None,
        scheduler_step_per_batch: bool = False,
    ):
        student.train()
        teacher.eval()

        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0
        running_components = {method: 0.0 for method in module.normalize_distill_methods(distill_methods)}
        seen = 0

        # Original loop applies one optimizer step each batch.
        # Here we accumulate gradients over `accumulation_steps` mini-batches.
        micro_count = 0

        for inputs, labels in loader:
            inputs = module.move_inputs_to_device(inputs, device)
            labels = labels.to(device)

            student_outputs = student.forward_outputs(inputs)
            with torch.no_grad():
                teacher_outputs = teacher.forward_outputs(inputs)

            ce = criterion(student_outputs["logits"], labels)
            component_losses = {}
            for method in running_components:
                if method == "logits":
                    component_losses[method] = module.distillation_loss(
                        student_outputs["logits"],
                        teacher_outputs["logits"],
                        temperature=temperature,
                    )
                elif method == "fused":
                    component_losses[method] = module.feature_distillation_loss(
                        distill_projectors("fused", student_outputs["fused"]),
                        teacher_outputs["fused"],
                        loss_name=distillation_feature_loss_name,
                        normalize=distillation_normalize_features,
                    )
                elif method == "hint":
                    component_losses[method] = module.feature_distillation_loss(
                        distill_projectors("hint", student_outputs["fused_input"]),
                        teacher_outputs["fused_input"],
                        loss_name=distillation_feature_loss_name,
                        normalize=distillation_normalize_features,
                    )
                elif method == "relation":
                    component_losses[method] = module.relation_distillation_loss(
                        student_outputs["fused"],
                        teacher_outputs["fused"],
                    )
                elif method == "attention":
                    component_losses[method] = module.attention_transfer_loss(
                        distill_projectors("attention", student_outputs["fused"]),
                        teacher_outputs["fused"],
                    )
                elif method in {"ct", "text"}:
                    component_losses[method] = module.feature_distillation_loss(
                        distill_projectors(method, student_outputs["modal_features"][method]),
                        teacher_outputs["modal_features"][method],
                        loss_name=distillation_feature_loss_name,
                        normalize=distillation_normalize_features,
                    )
                else:
                    raise ValueError(f"Unsupported distillation method: {method}")

            kd = sum(
                distill_method_weights[method] * component_losses[method]
                for method in component_losses
            )
            loss = (1.0 - alpha) * ce + alpha * kd

            # Gradient accumulation.
            loss_scaled = loss / float(accumulation_steps)
            loss_scaled.backward()

            running_loss += loss.item() * labels.size(0)
            running_ce += ce.item() * labels.size(0)
            running_kd += kd.item() * labels.size(0)
            for method, component in component_losses.items():
                running_components[method] += component.item() * labels.size(0)
            seen += labels.size(0)
            micro_count += 1

            if micro_count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

        if micro_count % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and scheduler_step_per_batch:
                scheduler.step()

        denom = max(seen, 1)
        return {
            "loss": running_loss / denom,
            "ce_loss": running_ce / denom,
            "kd_loss": running_kd / denom,
            **{f"kd_{method}_loss": value / denom for method, value in running_components.items()},
        }

    module.train_epoch_student_kd = patched_train_epoch_student_kd
    module.__dict__["_wrapped_train_epoch_original"] = original_train_epoch


def _patch_scheduler(module, wrapper_scheduler: str) -> None:
    if not wrapper_scheduler:
        return
    if wrapper_scheduler not in {"warmup_cosine", "cosine"}:
        return

    original_create_scheduler = module.create_scheduler

    def wrapped_create_scheduler(
        scheduler_name: str,
        optimizer,
        epochs: int,
        steps_per_epoch: int,
    ):
        # Fallback to existing pipeline for non-cosine schedules.
        if wrapper_scheduler == "cosine" and scheduler_name == "cosine":
            return original_create_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch)

        if wrapper_scheduler == "warmup_cosine":
            if scheduler_name != "cosine":
                return original_create_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch)

            total_steps = max(1, int(epochs) * max(1, int(steps_per_epoch)))
            # Short warmup by default to avoid over-extending tiny local schedules.
            warmup_steps = max(1, int(total_steps * 0.10))

            def lr_lambda(step_index: int) -> float:
                if step_index < warmup_steps:
                    return (step_index + 1) / float(warmup_steps)
                progress = (step_index - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return (module.torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), True)

        return original_create_scheduler(scheduler_name, optimizer, epochs, steps_per_epoch)

    module.create_scheduler = wrapped_create_scheduler
    module.__dict__["_wrapped_create_scheduler_original"] = original_create_scheduler


def main() -> int:
    args, passthrough = parse_args()

    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be >= 1")
    if not (0.0 <= args.confidence_floor <= 1.0):
        raise ValueError("--confidence-floor must be in [0,1]")
    if args.confidence_gamma < 0:
        raise ValueError("--confidence-gamma must be >= 0")

    m = importlib.import_module("lung_cancer_cls.multimodal_teacher_student")
    # ensure torch alias available for scheduler wrapper.
    m.torch = __import__("torch")

    _patch_distillation_loss(
        m,
        mode=args.confidence_mode,
        threshold=args.confidence_threshold,
        floor=args.confidence_floor,
        gamma=args.confidence_gamma,
        teacher_scale=args.wrapper_teacher_temperature_scale,
        calibration_aware=args.wrapper_calibration_aware,
    )
    _patch_train_epoch(m, accumulation_steps=args.accumulation_steps)
    _patch_scheduler(m, args.wrapper_scheduler)

    # Forward only trainer arguments to the original entrypoint.
    sys.argv = ["train_student_kd.py"] + passthrough
    if args.accumulation_steps > 1:
        print(
            f"[WRAPPER] gradient accumulation enabled: accumulation_steps={args.accumulation_steps}",
            file=sys.stderr,
        )
    if args.confidence_mode != "none":
        print(
            f"[WRAPPER] confidence-weighted KD: mode={args.confidence_mode}, "
            f"threshold={args.confidence_threshold}, floor={args.confidence_floor}, "
            f"gamma={args.confidence_gamma}",
            file=sys.stderr,
        )
    if args.wrapper_scheduler:
        print(f"[WRAPPER] scheduler override: {args.wrapper_scheduler}", file=sys.stderr)
    if args.wrapper_calibration_aware:
        print(
            f"[WRAPPER] calibration-aware KD enabled: teacher_scale={args.wrapper_teacher_temperature_scale}",
            file=sys.stderr,
        )

    return m.main_student_kd()


if __name__ == "__main__":
    raise SystemExit(main())
