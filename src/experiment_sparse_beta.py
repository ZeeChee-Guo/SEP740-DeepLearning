"""
Beta (sparsity weight) sweep for the sparse autoencoder.

This experiment trains the sparse autoencoder several times with different
SPARSITY_WEIGHT (beta) values while keeping every other setting identical
(same architecture, seed, epochs, learning rate, and sparsity target rho).
For each beta it calibrates a p95 threshold on the normal calibration split
and evaluates on the anomaly reference split, then records precision, recall,
F1, false-positive rate, and per-category recall.

beta = 0.0 corresponds to a plain (non-sparse) Sigmoid autoencoder and serves
as the reference point that isolates the effect of the sparsity penalty.

All outputs are written under artifacts/experiments so the main sparse model
artifacts produced by the standard pipeline are never overwritten.
"""

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# reuse the model and helpers from the standard sparse pipeline
from train_sparse_autoencoder import (
    BATCH_SIZE,
    DATA_PATH,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    SparseAutoencoder,
    load_data,
    make_loader,
    sparsity_penalty,
)
from evaluate_sparse_autoencoder import (
    binary_metrics,
    category_recalls,
    reconstruction_errors,
)

# beta values to compare (spread across several orders of magnitude)
BETA_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

# sparsity target is fixed while the sparsity weight beta is varied; this is
# defined explicitly here so the beta sweep does not depend on the current
# SPARSITY_TARGET value in the training module
SPARSITY_TARGET = 0.05

# threshold percentile used for the headline comparison
PRIMARY_PERCENTILE = 95.0

# paths (kept separate from the main pipeline artifacts)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "artifacts" / "experiments"
RESULTS_CSV_PATH = EXPERIMENT_DIR / "sparse_beta_sweep.csv"
RESULTS_JSON_PATH = EXPERIMENT_DIR / "sparse_beta_sweep.json"
FIGURE_PATH = EXPERIMENT_DIR / "sparse_beta_sweep.png"


def set_seed() -> None:
    """
    Reset all random seeds so every beta run starts from the same state.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def train_with_beta(
    beta: float,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    device: torch.device,
) -> tuple[SparseAutoencoder, float, float]:
    """
    Train the sparse autoencoder for a single beta value.

    Returns the model with the lowest validation reconstruction error, that
    best validation reconstruction, and the final validation sparsity penalty.
    """
    set_seed()

    train_loader = make_loader(x_train, BATCH_SIZE, shuffle=True)
    validation_loader = make_loader(x_validation, BATCH_SIZE, shuffle=False)

    input_dim = x_train.shape[1]
    model = SparseAutoencoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_validation_reconstruction = float("inf")
    best_state_dict = None
    last_validation_sparsity = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, activations = model(inputs)
            reconstruction_loss = criterion(outputs, targets)
            penalty = sparsity_penalty(activations, SPARSITY_TARGET)
            loss = reconstruction_loss + beta * penalty
            loss.backward()
            optimizer.step()

        # validation pass (reconstruction and sparsity tracked separately)
        model.eval()
        validation_reconstruction_sum = 0.0
        validation_sparsity_sum = 0.0
        validation_rows = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, activations = model(inputs)
                reconstruction_loss = criterion(outputs, targets)
                penalty = sparsity_penalty(activations, SPARSITY_TARGET)
                batch_size = inputs.size(0)
                validation_reconstruction_sum += reconstruction_loss.item() * batch_size
                validation_sparsity_sum += penalty.item() * batch_size
                validation_rows += batch_size

        validation_reconstruction = validation_reconstruction_sum / validation_rows
        last_validation_sparsity = validation_sparsity_sum / validation_rows

        if validation_reconstruction < best_validation_reconstruction:
            best_validation_reconstruction = validation_reconstruction
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state_dict)
    return model, best_validation_reconstruction, last_validation_sparsity


def evaluate_beta(
    model: SparseAutoencoder,
    x_calibration: np.ndarray,
    x_anomaly: np.ndarray,
    anomaly_categories: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Calibrate the p95 threshold on normal calibration errors and evaluate the
    model on the normal + anomaly mixture.
    """
    normal_errors = reconstruction_errors(model, x_calibration, BATCH_SIZE, device)
    anomaly_errors = reconstruction_errors(model, x_anomaly, BATCH_SIZE, device)

    threshold = float(np.percentile(normal_errors, PRIMARY_PERCENTILE))

    scores = np.concatenate([normal_errors, anomaly_errors])
    y_true = np.concatenate(
        [
            np.zeros(normal_errors.shape[0], dtype=np.int64),
            np.ones(anomaly_errors.shape[0], dtype=np.int64),
        ]
    )
    y_pred = (scores > threshold).astype(np.int64)
    anomaly_predictions = (anomaly_errors > threshold).astype(np.int64)

    metrics = binary_metrics(y_true, y_pred)
    metrics["threshold"] = threshold
    metrics["category_recall"] = category_recalls(
        anomaly_categories, anomaly_predictions
    )
    return metrics


def save_results(rows: list[dict]) -> None:
    """
    Save the sweep results to json and csv.
    """
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    category_names = sorted(rows[0]["category_recall"].keys())
    fieldnames = [
        "beta",
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "false_positive_rate",
        "specificity",
        "best_validation_reconstruction",
        "final_validation_sparsity",
    ] + [f"recall_{category}" for category in category_names]

    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fieldnames if key in row}
            for category in category_names:
                flat[f"recall_{category}"] = row["category_recall"].get(category, 0.0)
            writer.writerow(flat)


def plot_results(rows: list[dict]) -> None:
    """
    Plot overall F1/precision/recall and per-category recall against beta.
    """
    # beta = 0 is plotted at a small positive position so the log axis works
    betas = [row["beta"] for row in rows]
    plot_betas = [beta if beta > 0 else 1e-5 for beta in betas]
    tick_labels = ["0" if beta == 0 else f"{beta:g}" for beta in betas]

    category_names = sorted(rows[0]["category_recall"].keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(plot_betas, [row["precision"] for row in rows], marker="o", label="Precision")
    axes[0].plot(plot_betas, [row["recall"] for row in rows], marker="o", label="Recall")
    axes[0].plot(plot_betas, [row["f1"] for row in rows], marker="o", label="F1")
    axes[0].set_xscale("log")
    axes[0].set_xticks(plot_betas)
    axes[0].set_xticklabels(tick_labels)
    axes[0].set_xlabel("Sparsity weight beta")
    axes[0].set_ylabel("Score (p95 threshold)")
    axes[0].set_title("Overall Metrics vs Beta")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for category in category_names:
        axes[1].plot(
            plot_betas,
            [row["category_recall"][category] for row in rows],
            marker="o",
            label=category,
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks(plot_betas)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_xlabel("Sparsity weight beta")
    axes[1].set_ylabel("Recall (p95 threshold)")
    axes[1].set_title("Per-Category Recall vs Beta")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Sparse Autoencoder Beta Sweep")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def run_sweep() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data once and reuse across all beta runs
    x_train, x_validation = load_data()
    data = np.load(DATA_PATH, allow_pickle=True)
    x_calibration = data["X_calibration_normal"].astype(np.float32)
    x_anomaly = data["X_anomaly_reference"].astype(np.float32)
    anomaly_categories = data["anomaly_categories"].astype(str)

    print(f"Device: {device}")
    print(f"Sparsity target (rho): {SPARSITY_TARGET}")
    print(f"Beta values: {BETA_VALUES}")

    rows = []
    for beta in BETA_VALUES:
        print(f"\n=== Training with beta={beta:g} ===")
        model, best_validation_reconstruction, final_validation_sparsity = train_with_beta(
            beta, x_train, x_validation, device
        )
        metrics = evaluate_beta(
            model, x_calibration, x_anomaly, anomaly_categories, device
        )
        row = {
            "beta": beta,
            "best_validation_reconstruction": best_validation_reconstruction,
            "final_validation_sparsity": final_validation_sparsity,
            **metrics,
        }
        rows.append(row)
        print(
            f"beta={beta:g}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"fpr={metrics['false_positive_rate']:.4f}, "
            f"val_recon={best_validation_reconstruction:.6f}, "
            f"val_sparsity={final_validation_sparsity:.4f}"
        )
        print(f"  category_recall={json.dumps(metrics['category_recall'])}")

    save_results(rows)
    plot_results(rows)

    print("\n=== Beta sweep summary (p95) ===")
    header = f"{'beta':>8} {'precision':>10} {'recall':>8} {'f1':>8} {'fpr':>8}"
    print(header)
    for row in rows:
        print(
            f"{row['beta']:>8g} {row['precision']:>10.4f} {row['recall']:>8.4f} "
            f"{row['f1']:>8.4f} {row['false_positive_rate']:>8.4f}"
        )
    print(f"\nSaved results to: {RESULTS_CSV_PATH}")
    print(f"Saved results to: {RESULTS_JSON_PATH}")
    print(f"Saved figure to: {FIGURE_PATH}")


if __name__ == "__main__":
    run_sweep()
