"""
Select the basic autoencoder configuration.

This script trains models as temporary candidate models for configuration selection. The final basic model
 is trained by train_final_basic_autoencoder.py.

"""

import csv
import json
from typing import Any
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from train_final_basic_autoencoder import (ARTIFACTS_DIR, DATA_PATH, AutoencoderConfig,
                                              config_to_dict, fit_autoencoder,)


TUNING_DIR = ARTIFACTS_DIR / "hyperparameter_search"
TUNING_RESULTS_CSV_PATH = TUNING_DIR / "basic_autoencoder_tuning_results.csv"
TUNING_RESULTS_JSON_PATH = TUNING_DIR / "basic_autoencoder_tuning_results.json"
BEST_CONFIG_PATH = TUNING_DIR / "basic_autoencoder_best_config.json"


TUNING_THRESHOLD_PERCENTILE = 95.0


# grid search
HIDDEN_DIM_OPTIONS = [("basic_width", (48, 24)), ("wider_width", (64, 32)),]
LATENT_DIM_OPTIONS = [8, 12, 16]
LEARNING_RATE_OPTIONS = [1e-3, 5e-4]


CANDIDATE_CONFIGS = [
    AutoencoderConfig(
        name=(
            f"{hidden_name}_latent{latent_dim}_"
            f"lr{str(learning_rate).replace('.', 'p')}"
        ),
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        batch_size=512,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        max_epochs=45,
        patience=8,
    )
    for hidden_name, hidden_dims in HIDDEN_DIM_OPTIONS
    for latent_dim in LATENT_DIM_OPTIONS
    for learning_rate in LEARNING_RATE_OPTIONS
]


def reconstruction_errors( model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device,) -> np.ndarray:
    """
    Calculate one mean squared reconstruction error per row.
    """
    tensor = torch.from_numpy(x)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    model.eval()
    errors = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)


def evaluate_candidate( model: torch.nn.Module, config: AutoencoderConfig, device: torch.device,) -> dict[str, float]:
    """
    Evaluate a trained candidate model
    """
    data = np.load(DATA_PATH, allow_pickle=False)
    x_calibration = data["X_calibration_normal"].astype(np.float32)
    x_anomaly = data["X_anomaly_reference"].astype(np.float32)

    # reconstruct error
    normal_errors = reconstruction_errors(model, x_calibration, config.batch_size, device,)
    anomaly_errors = reconstruction_errors(model, x_anomaly, config.batch_size, device,)

    threshold = float(np.percentile(normal_errors, TUNING_THRESHOLD_PERCENTILE))
    scores = np.concatenate([normal_errors, anomaly_errors])
    y_true = np.concatenate([np.zeros(normal_errors.shape[0], dtype=np.int64),
                             np.ones(anomaly_errors.shape[0], dtype=np.int64),])
    y_pred = (scores > threshold).astype(np.int64)

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "normal_error_mean": float(normal_errors.mean()),
        "anomaly_error_mean": float(anomaly_errors.mean()),
    }


def result_row(config: AutoencoderConfig, training_result: dict[str, Any],
               detection_result: dict[str, float],) -> dict[str, Any]:
    """
    Combine training, architecture, and detection metrics into one row.
    """
    row = {
        "name": config.name,
        "hidden_dims": "-".join(str(value) for value in config.hidden_dims),
        "latent_dim": config.latent_dim,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "best_epoch": int(training_result["best_epoch"]),
        "epochs_trained": int(training_result["epochs_trained"]),
        "validation_loss": float(training_result["best_validation_loss"]),
    }
    row.update(detection_result)
    return row


def choose_best_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Select the best candidate
    Rule: higher F1 -> normal-validation reconstruction loss
    """
    return sorted(rows, key=lambda row: (-float(row["f1"]),
                                         float(row["validation_loss"]), int(row["epochs_trained"]),),)[0]


def save_selection_outputs( rows: list[dict[str, Any]], best_row: dict[str, Any], best_config: AutoencoderConfig,):
    """
    Save the candidate comparison table and selected final training configuration.
    """
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "hidden_dims",
        "latent_dim",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "best_epoch",
        "epochs_trained",
        "validation_loss",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "normal_error_mean",
        "anomaly_error_mean",
    ]

    # Save the candidate comparison table
    with TUNING_RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tuning_payload = {
        "selection_rule": (
            "Choose the highest p95-calibrated F1. If tied, choose the lower "
            "normal-validation reconstruction loss."
        ),
        "threshold_percentile": TUNING_THRESHOLD_PERCENTILE,
        "candidate_results": rows,
        "best_result": best_row,
    }

    # Save candidate results in JSON format
    with TUNING_RESULTS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(tuning_payload, file, indent=2)

    best_config_payload = {
        "source": "select_basic_autoencoder_config.py",
        "selection_rule": tuning_payload["selection_rule"],
        "threshold_percentile": TUNING_THRESHOLD_PERCENTILE,
        "best_config": config_to_dict(best_config),
        "best_result": best_row,
    }

    # The selected configuration for final model training
    with BEST_CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(best_config_payload, file, indent=2)


def select_config() -> dict[str, Any]:
    """
    Train candidate models and save the selected basic configuration.
    """
    rows: list[dict[str, Any]] = []
    configs_by_name = {config.name: config for config in CANDIDATE_CONFIGS}


    for index, config in enumerate(CANDIDATE_CONFIGS, start=1):
        print(f"\n[{index}/{len(CANDIDATE_CONFIGS)}] Candidate: {config.name}")
        training_result = fit_autoencoder(config)
        detection_result = evaluate_candidate(training_result["model"], config, training_result["device"],)
        row = result_row(config, training_result, detection_result)
        rows.append(row)

        print(
            f"Validation_loss={row['validation_loss']:.8f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"f1={row['f1']:.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_row = choose_best_result(rows)
    best_config = configs_by_name[best_row["name"]]
    save_selection_outputs(rows, best_row, best_config)

    print("\nBest basic autoencoder configuration:")
    print(f"  name: {best_config.name}")
    print(f"  hidden_dims: {best_config.hidden_dims}")
    print(f"  latent_dim: {best_config.latent_dim}")
    print(f"  learning_rate: {best_config.learning_rate}")
    print(f"  weight_decay: {best_config.weight_decay}")
    print(f"  validation_loss: {best_row['validation_loss']:.8f}")
    print(f"  f1: {best_row['f1']:.4f}")
    print(f"Saved tuning results to: {TUNING_RESULTS_CSV_PATH}")
    print(f"Saved best config to: {BEST_CONFIG_PATH}")

    return {"best_config": best_config, "best_result": best_row, "candidate_results": rows,}


if __name__ == "__main__":
    select_config()
