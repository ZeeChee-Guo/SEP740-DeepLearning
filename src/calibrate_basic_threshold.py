"""
Calibrate anomaly thresholds for the basic autoencoder.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_final_basic_autoencoder import DATA_PATH, MODEL_PATH, load_trained_model

# threshold settings
PRIMARY_THRESHOLD_PERCENTILE = 95.0
THRESHOLD_PERCENTILES = [90.0, 95.0, 97.5, 99.0]

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
THRESHOLD_PATH = ARTIFACTS_DIR / "thresholds" / "basic_threshold.json"


def reconstruction_errors(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device,) -> np.ndarray:
    """
    For each sample, calculate the average squared error between the original input and the reconstructed
     outputs by the autoencoder.
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


def threshold_key(value: float) -> str:
    """
    Convert a percentile value into a readable json key
    """
    return f"p{value:g}".replace(".", "_")


def save_threshold(threshold_info: dict[str, object]) -> None:
    """
    Save threshold information for evaluation
    """
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    with THRESHOLD_PATH.open("w", encoding="utf-8") as file:
        json.dump(threshold_info, file, indent=2)


def calibrate_threshold() -> None:
    # if using gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the final trained autoencoder. The checkpoint stores the architecture
    # chosen during configuration selection, so calibration uses the final model.
    model, checkpoint, config = load_trained_model(MODEL_PATH, device)

    # load normal data
    x_calibration = np.load(DATA_PATH, allow_pickle=False)["X_calibration_normal"].astype(np.float32)

    # compute reconstruction errors on normal calibration samples
    errors = reconstruction_errors(model, x_calibration, config.batch_size, device)
    error_mean = float(errors.mean())
    error_std = float(errors.std())

    # percentile thresholds
    percentile_thresholds = {
        threshold_key(percentile): {"percentile": percentile, "threshold": float(np.percentile(errors, percentile)),}
        for percentile in THRESHOLD_PERCENTILES
    }

    # use p95 as the main threshold for the basic result
    primary_key = threshold_key(PRIMARY_THRESHOLD_PERCENTILE)
    primary_threshold = percentile_thresholds[primary_key]["threshold"]

    threshold_info = {
        "model_path": str(MODEL_PATH),
        "model_config": checkpoint.get("config", {}),
        "primary_threshold_method": "percentile",
        "primary_threshold_key": primary_key,
        "primary_threshold_percentile": PRIMARY_THRESHOLD_PERCENTILE,
        "primary_threshold": primary_threshold,
        "percentile_thresholds": percentile_thresholds,
        "calibration_samples": int(x_calibration.shape[0]),
        "calibration_error_summary": {
            "mean": error_mean,
            "std": error_std,
            "min": float(errors.min()),
            "max": float(errors.max()),
        },
    }
    save_threshold(threshold_info)


    print("Percentile thresholds:")
    for key, value in percentile_thresholds.items():
        print(f"  {key}: {value['threshold']:.8f}")
    print(f"Primary threshold: {primary_key} = {primary_threshold:.8f}")
    print(f"Saved threshold to: {THRESHOLD_PATH}")


if __name__ == "__main__":
    calibrate_threshold()
