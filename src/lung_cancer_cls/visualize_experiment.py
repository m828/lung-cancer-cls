from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("matplotlib is required for experiment visualization. Install it with `pip install matplotlib`.") from exc

from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def _load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_prediction_table(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / f"{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    return pd.read_csv(path)


def _resolve_positive_probability(df: pd.DataFrame, metrics: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    class_names = metrics.get("class_names", {})
    positive_name = None
    if isinstance(class_names, dict):
        if "1" in class_names:
            positive_name = class_names["1"]
        elif 1 in class_names:
            positive_name = class_names[1]
    if positive_name is None:
        positive_name = "class_1"
    positive_col = f"prob_{positive_name}"
    if positive_col not in df.columns:
        prob_cols = [col for col in df.columns if col.startswith("prob_")]
        if len(prob_cols) < 2:
            raise ValueError("Could not infer positive probability column from prediction CSV.")
        positive_col = prob_cols[-1]
    return df[positive_col].to_numpy(dtype=np.float32), positive_col


def _plot_history(metrics: Dict[str, Any], output_dir: Path) -> None:
    history = metrics.get("history", [])
    if not history:
        return
    hist_df = pd.DataFrame(history)
    if hist_df.empty or "epoch" not in hist_df.columns:
        return

    epochs = hist_df["epoch"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss" in hist_df.columns:
        axes[0].plot(epochs, hist_df["train_loss"], label="train_loss")
    if "val_loss" in hist_df.columns:
        axes[0].plot(epochs, hist_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    metric_cols = [col for col in ["val_auroc", "val_auprc", "val_balanced_accuracy", "val_f1"] if col in hist_df.columns]
    for col in metric_cols:
        axes[1].plot(epochs, hist_df[col], label=col)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "history_curves.png", dpi=200)
    plt.close(fig)


def _plot_binary_curves(df: pd.DataFrame, metrics: Dict[str, Any], output_dir: Path) -> None:
    y_true = df["label"].to_numpy(dtype=np.int64)
    y_prob, prob_col = _resolve_positive_probability(df, metrics)
    y_pred = df["prediction"].to_numpy(dtype=np.int64)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].set_xlabel("FPR")
    axes[0, 0].set_ylabel("TPR")
    axes[0, 0].legend()

    axes[0, 1].plot(recall, precision, label=f"AUPRC={auc(recall, precision):.4f}")
    axes[0, 1].set_title("PR Curve")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()

    axes[0, 2].plot(prob_pred, prob_true, marker="o")
    axes[0, 2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 2].set_title("Calibration Curve")
    axes[0, 2].set_xlabel("Mean Predicted Prob.")
    axes[0, 2].set_ylabel("Fraction of Positives")

    im = axes[1, 0].imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    axes[1, 1].hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="negative")
    axes[1, 1].hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="positive")
    axes[1, 1].set_title(f"Probability Histogram ({prob_col})")
    axes[1, 1].set_xlabel("Predicted Probability")
    axes[1, 1].legend()

    sorted_idx = np.argsort(y_prob)
    axes[1, 2].plot(y_prob[sorted_idx], label="pred_prob")
    axes[1, 2].plot(y_true[sorted_idx], label="label", alpha=0.7)
    axes[1, 2].set_title("Sorted Score Trace")
    axes[1, 2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "binary_diagnostics.png", dpi=200)
    plt.close(fig)


def export_experiment_plots(run_dir: Path, output_dir: Path, split: str) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(run_dir / "metrics.json")
    _plot_history(metrics, output_dir)

    prediction_path = run_dir / f"{split}_predictions.csv"
    created: List[str] = []
    if prediction_path.exists():
        df = _load_prediction_table(run_dir, split)
        num_classes = len(metrics.get("class_names", {}))
        if num_classes == 2 and {"label", "prediction"}.issubset(df.columns):
            _plot_binary_curves(df, metrics, output_dir)
            created.append("binary_diagnostics.png")
    if (output_dir / "history_curves.png").exists():
        created.append("history_curves.png")

    summary = {
        "run_dir": str(run_dir),
        "split": split,
        "created_files": created,
    }
    with open(output_dir / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export training and prediction visualizations for an experiment run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Experiment run directory containing metrics.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated plots.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val", help="Prediction split to visualize.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    export_experiment_plots(args.run_dir, args.output_dir, args.split)


if __name__ == "__main__":
    main()
