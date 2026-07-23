from __future__ import annotations

from pathlib import Path

import pytest

from experiments.utils.attribution_audit import atomic_torch_save, checkpoint_filename, checkpoint_payload, checkpoint_score


VAL_METRICS = {
    "loss": 0.4,
    "auroc": 0.9,
    "balanced_accuracy": 0.8,
    "f1": 0.75,
    "recall": 0.7,
    "ece": 0.1,
}


def test_multiple_validation_checkpoints_are_atomic_and_self_describing(tmp_path: Path):
    torch = pytest.importorskip("torch")
    paths = []
    for criterion in ("val_loss", "val_auroc", "val_f1", "val_bacc", "val_composite"):
        payload = checkpoint_payload(
            model_state={"weight": torch.tensor([1.0])},
            optimizer_state={"state": {}},
            scheduler_state={"last_epoch": 2},
            epoch=3,
            criterion=criterion,
            val_metrics=VAL_METRICS,
            threshold=0.47,
            configuration={"primary": "val_composite"},
        )
        path = tmp_path / checkpoint_filename(criterion)
        atomic_torch_save(payload, path)
        paths.append(path)
        loaded = torch.load(path, map_location="cpu")
        assert loaded["checkpoint_criterion"] == criterion
        assert loaded["test_metrics_used_for_selection"] is False
        assert loaded["validation_threshold"] == 0.47
    assert all(path.is_file() for path in paths)
    assert not list(tmp_path.glob(".*.tmp"))


def test_primary_composite_score_matches_protocol_and_test_is_ignored():
    expected = 0.9 + 0.5 * 0.8 + 0.5 * 0.75 + 0.25 * 0.7 - 0.25 * 0.1
    assert checkpoint_score("val_composite", {**VAL_METRICS, "test_auroc": 0.0}) == pytest.approx(expected)


def test_checkpoint_filenames_match_suite_contract():
    assert checkpoint_filename("val_loss") == "best_val_loss.pt"
    assert checkpoint_filename("val_auroc") == "best_val_auroc.pt"
    assert checkpoint_filename("val_f1") == "best_val_f1.pt"
    assert checkpoint_filename("val_bacc") == "best_val_bacc.pt"
    assert checkpoint_filename("val_composite") == "best_val_composite.pt"


def test_checkpoint_payload_can_preserve_resume_state_without_test_selection():
    payload = checkpoint_payload(
        model_state={"weight": 1},
        optimizer_state={},
        scheduler_state=None,
        epoch=4,
        criterion="val_composite",
        val_metrics=VAL_METRICS,
        threshold=0.5,
        configuration={"primary": "val_composite"},
        training_state={"best_epoch": 3, "best_score": 2.0, "stale_epochs": 1},
    )
    assert payload["training_state"]["best_epoch"] == 3
    assert payload["test_metrics_used_for_selection"] is False
