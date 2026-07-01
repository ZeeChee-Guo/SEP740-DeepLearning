"""
Evaluate the basic autoencoder anomaly detector.
"""

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,)
from torch.utils.data import DataLoader, TensorDataset

from train_final_basic_autoencoder import DATA_PATH, MODEL_PATH, load_trained_model


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
THRESHOLD_PATH = ARTIFACTS_DIR / "thresholds" / "basic_threshold.json"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
METRICS_CSV_PATH = EVALUATION_DIR / "basic_metrics.csv"
METRICS_JSON_PATH = EVALUATION_DIR / "basic_metrics.json"
ERRORS_PATH = EVALUATION_DIR / "basic_reconstruction_errors.npz"


def reconstruction_errors(model: torch.nn.Module,x: np.ndarray, batch_size: int, device: torch.device, ) -> np.ndarray:
    """
    For each sample, calculate its mean squared reconstruction error.
    """
    tensor = torch.from_numpy(x)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Avoid divide by zero
    """
    return numerator / denominator if denominator != 0 else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """
    Calculate overall binary classification metrics
    Normal: 0; anomaly: 1
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "true_normal_pred_normal": int(tn),
        "true_normal_pred_anomaly": int(fp),
        "true_anomaly_pred_normal": int(fn),
        "true_anomaly_pred_anomaly": int(tp),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "specificity": safe_divide(tn, tn + fp),
    }


def collect_threshold_rows(threshold_info: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert percentile thresholds into rows for evaluation
    """
    primary_key = threshold_info["primary_threshold_key"]
    rows = []

    for key, value in threshold_info["percentile_thresholds"].items():
        rows.append(
            {
                "threshold_key": key,
                "method": "percentile",
                "percentile": value["percentile"],
                "threshold": value["threshold"],
                "is_primary": key == primary_key,
            }
        )

    return rows


def category_recalls(anomaly_categories: np.ndarray, anomaly_predictions: np.ndarray,) -> dict[str, float]:
    """
    Calculate recall for each attack category
    """
    recalls = {}
    for category in sorted(np.unique(anomaly_categories)):
        mask = anomaly_categories == category
        recalls[str(category)] = float(np.mean(anomaly_predictions[mask]))
    return recalls


def evaluate_thresholds(
    scores: np.ndarray, y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    anomaly_categories: np.ndarray,
    threshold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply each threshold and calculate its metrics
    """
    results = []

    for row in threshold_rows:
        threshold_ = float(row["threshold"])
        y_pred = (scores > threshold_).astype(np.int64)
        anomaly_predictions = (anomaly_scores > threshold_).astype(np.int64)

        result = dict(row)
        result.update(binary_metrics(y_true, y_pred))
        result["category_recall"] = category_recalls(anomaly_categories, anomaly_predictions,)
        results.append(result)

    return results


def save_metrics(results: list[dict[str, Any]], output_summary: dict[str, Any]) -> None:
    """
    Save metrics
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(output_summary, file, indent=2)

    category_names = sorted(output_summary["category_names"])
    fieldnames = [
        "threshold_key",
        "method",
        "percentile",
        "threshold",
        "is_primary",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_positive_rate",
        "specificity",
        "true_normal_pred_normal",
        "true_normal_pred_anomaly",
        "true_anomaly_pred_normal",
        "true_anomaly_pred_anomaly",
    ] + [f"recall_{category}" for category in category_names]

    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                key: value
                for key, value in result.items()
                if key != "category_recall"
            }
            for category in category_names:
                row[f"recall_{category}"] = result["category_recall"].get(category, 0.0)
            writer.writerow(row)


def save_errors(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    anomaly_categories: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """
    Save reconstruction errors
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ERRORS_PATH,
        normal_errors=normal_errors,
        anomaly_errors=anomaly_errors,
        anomaly_categories=anomaly_categories,
        scores=scores,
        y_true=y_true,
    )


def evaluate() -> None:
    # if using gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the final trained autoencoder. The checkpoint stores the selected
    # architecture, so evaluation always matches the tuned final model.
    model, checkpoint, config = load_trained_model(MODEL_PATH, device)

    # load calibrated thresholds
    with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
        threshold_info =  json.load(file)

    # load evaluation data
    # allow_pickle=True is needed because anomaly_categories is saved as an object array
    data = np.load(DATA_PATH, allow_pickle=True)
    x_normal = data["X_calibration_normal"].astype(np.float32)
    x_anomaly = data["X_anomaly_reference"].astype(np.float32)
    anomaly_categories = data["anomaly_categories"].astype(str)

    # compute reconstruction errors
    normal_errors = reconstruction_errors(model, x_normal, config.batch_size, device)
    anomaly_errors = reconstruction_errors(model, x_anomaly, config.batch_size, device)

    scores = np.concatenate([normal_errors, anomaly_errors])
    y_true = np.concatenate(
        [np.zeros(normal_errors.shape[0], dtype=np.int64), np.ones(anomaly_errors.shape[0], dtype=np.int64),])

    # compare all calibrated percentile thresholds
    threshold_rows = collect_threshold_rows(threshold_info)
    results = evaluate_thresholds(scores, y_true, anomaly_errors, anomaly_categories, threshold_rows,)

    # keep the primary threshold result easy to access
    primary_key = threshold_info["primary_threshold_key"]
    primary_result = next(result for result in results if result["threshold_key"] == primary_key)
    output_summary = {
        "model_path": str(MODEL_PATH),
        "model_config": checkpoint.get("config", {}),
        "threshold_path": str(THRESHOLD_PATH),
        "normal_source": "X_calibration_normal",
        "anomaly_source": "X_anomaly_reference",
        "normal_samples": int(normal_errors.shape[0]),
        "anomaly_samples": int(anomaly_errors.shape[0]),
        "category_names": sorted(str(value) for value in np.unique(anomaly_categories)),
        "primary_threshold_key": primary_key,
        "primary_result": primary_result,
        "threshold_results": results,
        "error_summary": {
            "normal_mean": float(normal_errors.mean()),
            "normal_std": float(normal_errors.std()),
            "normal_min": float(normal_errors.min()),
            "normal_max": float(normal_errors.max()),
            "anomaly_mean": float(anomaly_errors.mean()),
            "anomaly_std": float(anomaly_errors.std()),
            "anomaly_min": float(anomaly_errors.min()),
            "anomaly_max": float(anomaly_errors.max()),
        },
    }

    save_metrics(results, output_summary)
    save_errors(normal_errors, anomaly_errors, anomaly_categories, scores, y_true)


    print("Threshold comparison:")
    for result in results:
        primary_marker = " (primary)" if result["is_primary"] else ""
        print(
            f"  {result['threshold_key']}{primary_marker}: "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1']:.4f}, "
            f"fpr={result['false_positive_rate']:.4f}"
        )
    print(f"Saved metrics to: {METRICS_CSV_PATH}")
    print(f"Saved detailed metrics to: {METRICS_JSON_PATH}")
    print(f"Saved reconstruction errors to: {ERRORS_PATH}")


if __name__ == "__main__":
    evaluate()
