import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from evaluate_basic_autoencoder import (
    ERRORS_PATH,
    METRICS_JSON_PATH,
    THRESHOLD_PATH,
)
from train_final_basic_autoencoder import HISTORY_PATH


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures" / "basic_autoencoder"
FIGURE_DPI = 220


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_training_curve() -> None:
    """
    Plot training loss and validation loss by epoch.
    """
    if not HISTORY_PATH.exists():
        print(f"Skipped training curve because training history was not found: {HISTORY_PATH}")
        return

    epochs = []
    train_losses = []
    validation_losses = []

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            validation_losses.append(float(row["validation_loss"]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="Training normal")
    plt.plot(epochs, validation_losses, label="Validation normal")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared reconstruction error")
    plt.title("Basic Autoencoder Training Curve")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURE_DIR / "training_curve.png"
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    print(f"Saved training curve to: {output_path}")


def save_error_distribution(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold_info: dict,
) -> None:
    """
    Plot normal and anomaly reconstruction error distributions.
    """
    primary_threshold = threshold_info["primary_threshold"]
    primary_key = threshold_info["primary_threshold_key"]

    positive_errors = np.concatenate([normal_errors, anomaly_errors])
    positive_errors = positive_errors[positive_errors > 0]
    epsilon = float(positive_errors.min() / 10)
    normal_log_errors = np.log10(normal_errors + epsilon)
    anomaly_log_errors = np.log10(anomaly_errors + epsilon)
    threshold_log = np.log10(primary_threshold + epsilon)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    log_min = min(normal_log_errors.min(), anomaly_log_errors.min())
    log_max = max(normal_log_errors.max(), anomaly_log_errors.max())
    log_bins = np.linspace(log_min, log_max, 90)

    axes[0].hist(normal_log_errors, bins=log_bins, alpha=0.75, density=True, label="Calibration normal")
    axes[0].hist(anomaly_log_errors, bins=log_bins, alpha=0.55, density=True, label="Anomaly reference")
    axes[0].axvline(
        threshold_log,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({primary_key})",
    )
    axes[0].set_xlabel("log10 reconstruction error")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Full Distribution")
    axes[0].legend()

    zoom_max = max(primary_threshold * 2.2, float(np.percentile(normal_errors, 99.8)))
    zoom_bins = np.linspace(0, zoom_max, 80)
    axes[1].hist(normal_errors, bins=zoom_bins, alpha=0.75, label="Calibration normal")
    axes[1].hist(anomaly_errors, bins=zoom_bins, alpha=0.55, label="Anomaly reference")
    axes[1].axvline(
        primary_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({primary_key})",
    )
    axes[1].set_xlim(0, zoom_max)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Mean squared reconstruction error")
    axes[1].set_ylabel("Record count (log scale)")
    axes[1].set_title("Threshold Region Zoom")
    axes[1].legend()

    fig.suptitle("Reconstruction Error Distribution")
    fig.tight_layout()
    output_path = FIGURE_DIR / "reconstruction_error_distribution.png"
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"Saved error distribution to: {output_path}")


def save_precision_recall_f1_vs_threshold(metrics: dict) -> None:
    """
    Plot precision, recall, and F1 score for each calibrated threshold.
    """
    results = metrics["threshold_results"]
    labels = [result["threshold_key"] for result in results]
    threshold_values = [result["threshold"] for result in results]
    precision = [result["precision"] for result in results]
    recall = [result["recall"] for result in results]
    f1 = [result["f1"] for result in results]

    x = np.arange(len(labels))
    all_scores = precision + recall + f1
    y_min = max(0.0, min(all_scores) - 0.01)
    y_max = min(1.005, max(all_scores) + 0.004)
    if y_max - y_min < 0.025:
        y_min = max(0.0, y_max - 0.025)

    plt.figure(figsize=(8.2, 5.0))
    plt.plot(x, precision, marker="o", linewidth=2, label="Precision")
    plt.plot(x, recall, marker="o", linewidth=2, label="Recall")
    plt.plot(x, f1, marker="o", linewidth=2, label="F1")

    for index, score in enumerate(f1):
        plt.text(index, score + 0.001, f"{score:.4f}", ha="center", va="bottom", fontsize=8)

    tick_labels = [
        f"{label}\n{threshold:.4f}"
        for label, threshold in zip(labels, threshold_values)
    ]
    plt.xticks(x, tick_labels)
    plt.ylim(y_min, y_max)
    plt.grid(axis="y", alpha=0.25)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs Threshold")
    plt.legend()
    plt.tight_layout()
    output_path = FIGURE_DIR / "precision_recall_f1_vs_threshold.png"
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    print(f"Saved precision/recall/F1 threshold plot to: {output_path}")


def save_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    metrics: dict,
) -> None:
    """
    Plot the precision-recall curve using reconstruction error as the anomaly score.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    threshold_results = metrics["threshold_results"]

    for axis_index, ax in enumerate(axes):
        ax.plot(recall, precision, linewidth=2, label=f"AP = {average_precision:.4f}")
        for result in threshold_results:
            ax.scatter(
                result["recall"],
                result["precision"],
                s=42,
                label=result["threshold_key"],
            )
            if axis_index == 1:
                ax.annotate(
                    result["threshold_key"],
                    (result["recall"], result["precision"]),
                    textcoords="offset points",
                    xytext=(5, -8),
                    fontsize=8,
                )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.25)

    axes[0].set_title("Full Curve")
    axes[0].set_xlim(0, 1.02)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower left")

    threshold_precisions = [result["precision"] for result in threshold_results]
    threshold_recalls = [result["recall"] for result in threshold_results]
    axes[1].set_title("Top-Right Zoom")
    axes[1].set_xlim(max(0.0, min(threshold_recalls) - 0.015), 1.002)
    axes[1].set_ylim(max(0.0, min(threshold_precisions) - 0.015), 1.002)
    axes[1].legend(loc="lower left")

    fig.suptitle("Precision-Recall Curve")
    fig.tight_layout()
    output_path = FIGURE_DIR / "precision_recall_curve.png"
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"Saved precision-recall curve to: {output_path}")


def save_confusion_matrix(metrics: dict) -> None:
    """
    Plot confusion matrix for the primary threshold.
    """
    primary = metrics["primary_result"]
    matrix = np.array(
        [
            [
                primary["true_normal_pred_normal"],
                primary["true_normal_pred_anomaly"],
            ],
            [
                primary["true_anomaly_pred_normal"],
                primary["true_anomaly_pred_anomaly"],
            ],
        ]
    )

    plt.figure(figsize=(5.2, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Pred normal", "Pred anomaly"])
    plt.yticks([0, 1], ["True normal", "True anomaly"])
    plt.colorbar(label="Count")

    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")

    plt.title(f"Confusion Matrix ({metrics['primary_threshold_key']})")
    plt.tight_layout()
    output_path = FIGURE_DIR / "confusion_matrix_primary_threshold.png"
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    print(f"Saved confusion matrix to: {output_path}")


def visualize() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    threshold_info = load_json(THRESHOLD_PATH)
    metrics = load_json(METRICS_JSON_PATH)
    errors = np.load(ERRORS_PATH, allow_pickle=False)

    normal_errors = errors["normal_errors"]
    anomaly_errors = errors["anomaly_errors"]
    scores = errors["scores"]
    y_true = errors["y_true"]

    save_training_curve()
    save_error_distribution(normal_errors, anomaly_errors, threshold_info)
    save_precision_recall_f1_vs_threshold(metrics)
    save_precision_recall_curve(y_true, scores, metrics)
    save_confusion_matrix(metrics)

    print(f"Saved figures under: {FIGURE_DIR}")


if __name__ == "__main__":
    visualize()
