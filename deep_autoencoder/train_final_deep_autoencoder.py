"""
Train the final deep autoencoder

Model structure

Input features dim: 74
Encoder:
    Linear(74 -> 64) -> ReLU
    Linear(64 -> 32) -> ReLU
    Linear(32 -> 16) -> ReLU
Latent space:
    16 dims
Decoder:
    Linear(16 -> 32) -> ReLU
    Linear(32 -> 64) -> ReLU
    Linear(64 -> 74) -> Sigmoid

"""

import csv
import json
import random
from dataclasses import asdict, dataclass
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
BEST_CONFIG_PATH = (ARTIFACTS_DIR/ "hyperparameter_search"/ "deep_autoencoder_best_config.json")


SEED = 42


@dataclass(frozen=True)
class AutoencoderConfig:
    """
    Training and architecture settings for one deep autoencoder.
    """

    name: str = "default_deep"
    hidden_dims: tuple[int, ...] = (48, 24)
    latent_dim: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-6


# Keep these constants for other scripts that only need a default batch size.
DEFAULT_CONFIG = AutoencoderConfig()
BATCH_SIZE = DEFAULT_CONFIG.batch_size


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

def config_to_dict(config: AutoencoderConfig) -> dict[str, Any]:
    """
    Convert config to JSON
    """
    data = asdict(config)
    data["hidden_dims"] = list(config.hidden_dims)
    return data


def config_from_dict(data: dict[str, Any] | None) -> AutoencoderConfig:
    """
    Build an AutoencoderConfig
    """
    if not data:
        return DEFAULT_CONFIG

    values = dict(data)
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(int(value) for value in values["hidden_dims"])
    return AutoencoderConfig(**values)


def load_selected_config() -> AutoencoderConfig:
    """
    Load the best hyperparameter config
    """
    if not BEST_CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    with BEST_CONFIG_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return config_from_dict(payload["best_config"])


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


def fit_autoencoder(config: AutoencoderConfig,*, save_model_path: Path | None = None,
                    save_history_path: Path | None = None) -> dict[str, Any]:
    """
    Train one autoencoder configuration with early stopping.
    """
    # set random seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    #load the data (train + val)
    x_train, x_validation = load_data()

    input_dim = x_train.shape[1]
    model = DeepAutoencoder(input_dim=input_dim, hidden_dims=config.hidden_dims,
                             latent_dim=config.latent_dim,)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate, weight_decay=config.weight_decay),
                 loss = 'mse')
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.patience, min_delta=config.min_delta, restore_best_weights=True)

    print(f'Training config: {config.name}')
    print(f"Architecture: input -> {list(config.hidden_dims)} -> {config.latent_dim}")

    model_history = model.fit(x_train, x_train, 
                        validation_data=(x_validation, x_validation), 
                        batch_size=config.batch_size, 
                        epochs=config.max_epochs, 
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
                "config": config_to_dict(config),
                "epoch": best_epoch,
                "validation_loss": best_validation_loss,
                "epochs_trained": len(history),
                "seed": SEED,
            }, file, indent=2)

    if save_history_path is not None:
        save_history(history, save_history_path)

    return {
        "model": model,
        "config": config,
        "input_dim": input_dim,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "history": history,
    }


def load_trained_model(model_path: Path = MODEL_PATH,
) -> tuple[tf.keras.Model, dict[str, Any], AutoencoderConfig]:
    """
    Load the saved final deep model
    """
    config_path = model_path.with_suffix(".json")
    with config_path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    config = config_from_dict(checkpoint.get("config"))

    model = DeepAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    )
    model.load_weights(model_path)

    return model, checkpoint, config


def train(config: AutoencoderConfig | None = None) -> dict[str, Any]:
    """
    Train and save the deep autoencoder
    """
    selected_config = config or load_selected_config()
    result = fit_autoencoder(selected_config, save_model_path=MODEL_PATH, save_history_path=HISTORY_PATH)
    print(f"Model saved")

    return result


if __name__ == "__main__":
    train()
