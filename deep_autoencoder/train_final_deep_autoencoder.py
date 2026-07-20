"""
Train the final deep autoencoder

Model structure

Input features dim: 74
Encoder:
    Linear(74 -> 64) -> ReLU
    Linear(64 -> 48) -> ReLU
    Linear(48 -> 32) -> ReLU
    Linear(32 -> 24) -> ReLU
    Linear(24 -> 16) -> ReLU
Latent space:
    16 dims
Decoder:
    Linear(16 -> 24) -> ReLU
    Linear(24 -> 32) -> ReLU
    Linear(32 -> 48) -> ReLU
    Linear(48 -> 64) -> ReLU
    Linear(64 -> 74) -> Sigmoid

"""

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "kdd99_preprocessed_data.npz"
MODEL_PATH = ARTIFACTS_DIR / "models" / "deep_autoencoder.weights.h5"
HISTORY_PATH = ARTIFACTS_DIR / "training_history" / "deep_autoencoder_history.csv"


SEED = 42

# hyperparams (predefined, matching the architecture documented above)
HIDDEN_DIMS = (64, 48, 32, 24)
LATENT_DIM = 16
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
MIN_DELTA = 1e-6


def DeepAutoencoder(input_dim: int, hidden_dims: tuple[int, ...], latent_dim: int) -> tf.keras.Model:
    """
    Deep autoencoder
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

    # len of hidden_dims is the number of hidden layers
    # encoder side
    for hidden_dim in hidden_dims:
        model.add(tf.keras.layers.Dense(hidden_dim, activation="relu"))

    model.add(tf.keras.layers.Dense(latent_dim, activation="relu"))  # bottleneck

    # decoder side
    for hidden_dim in reversed(hidden_dims):
        model.add(tf.keras.layers.Dense(hidden_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(input_dim, activation="sigmoid"))  # reconstruction

    return model


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load only the normal train and validation splits data
    """
    data = np.load(DATA_PATH, allow_pickle=False)
    x_train = data["X_train_normal"].astype(np.float32)
    x_validation = data["X_validation_normal"].astype(np.float32)
    return x_train, x_validation


def save_history(history: list[dict[str, float | int]], history_path: Path) -> None:
    """
    Save train and validation losses
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "train_loss", "validation_loss", "is_best"]
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def fit_autoencoder(*, save_model_path: Path | None = None,
                    save_history_path: Path | None = None) -> dict[str, Any]:
    """
    Train the predefined deep autoencoder with early stopping.
    """
    # set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    #load the data (train + val)
    x_train, x_validation = load_data()

    input_dim = x_train.shape[1]
    model = DeepAutoencoder(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, latent_dim=LATENT_DIM)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                 loss = 'mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)

    print(f"Training predefined deep autoencoder")
    print(f"Architecture: input -> {list(HIDDEN_DIMS)} -> {LATENT_DIM}")

    model_history = model.fit(x_train, x_train,
                        validation_data=(x_validation, x_validation),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[early_stopping], verbose=1)

    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    best_epoch = int(np.argmin(val_loss)) + 1
    best_validation_loss = float(np.min(val_loss))

    history = [
        { "epoch": i + 1, "train_loss": trainLoss, "validation_loss": valLoss, "is_best": None}
        for i, (trainLoss, valLoss) in enumerate(zip(train_loss, val_loss))
    ]
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(save_model_path)
        config_path = save_model_path.with_suffix(".json")
        with config_path.open("w", encoding="utf-8") as file:
            json.dump({
                "input_dim": input_dim,
                "hidden_dims": list(HIDDEN_DIMS),
                "latent_dim": LATENT_DIM,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epoch": best_epoch,
                "validation_loss": best_validation_loss,
                "epochs_trained": len(history),
                "seed": SEED,
            }, file, indent=2)

    if save_history_path is not None:
        save_history(history, save_history_path)

    return {
        "model": model,
        "input_dim": input_dim,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "history": history,
    }


def train() -> dict[str, Any]:
    """
    Train and save the deep autoencoder with the predefined architecture.
    """
    result = fit_autoencoder(save_model_path=MODEL_PATH, save_history_path=HISTORY_PATH)
    print(f"Model saved")

    return result


if __name__ == "__main__":
    train()
