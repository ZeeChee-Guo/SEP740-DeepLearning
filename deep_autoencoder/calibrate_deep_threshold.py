"""
Calibrate anomaly thresholds for the basic autoencoder.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from train_final_deep_autoencoder import (
    BATCH_SIZE,
    DATA_PATH,
    HIDDEN_DIMS,
    LATENT_DIM,
    MODEL_PATH,
    DeepAutoencoder,
)

# threshold settings
PRIMARY_THRESHOLD_PERCENTILE = 95.0
THRESHOLD_PERCENTILES = [90.0, 95.0, 97.5, 99.0]

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
THRESHOLD_PATH = ARTIFACTS_DIR / "thresholds" / "deep_threshold.json"


def reconstruction_errors( model: tf.keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Calculate one mean squared reconstruction error per row.
    """
    outputs = model.predict(x, batch_size=batch_size, verbose=0) #model predict the outputs from inputs 
    errors = np.mean(np.square(x - outputs), axis=1) #compare if input and output are matched 

    return errors


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
    # Load the final trained autoencoder directly, same pattern as
    # calibrate_sparse_threshold.py: read input_dim from the saved checkpoint,
    # rebuild the model with the predefined architecture, then load weights.
    config_path = MODEL_PATH.with_suffix(".json")
    with config_path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    model = DeepAutoencoder(input_dim=checkpoint["input_dim"], hidden_dims=HIDDEN_DIMS, latent_dim=LATENT_DIM)
    model.load_weights(MODEL_PATH)

    # load normal data
    x_calibration = np.load(DATA_PATH, allow_pickle=False)["X_calibration_normal"].astype(np.float32)

    # compute reconstruction errors on normal calibration samples
    errors = reconstruction_errors(model, x_calibration, BATCH_SIZE)
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
