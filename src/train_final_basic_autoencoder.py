"""
Train the final basic autoencoder

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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "kdd99_preprocessed_data.npz"
MODEL_PATH = ARTIFACTS_DIR / "models" / "basic_autoencoder.pt"
HISTORY_PATH = ARTIFACTS_DIR / "training_history" / "basic_autoencoder_history.csv"
BEST_CONFIG_PATH = (ARTIFACTS_DIR/ "hyperparameter_search"/ "basic_autoencoder_best_config.json")


SEED = 42


@dataclass(frozen=True)
class AutoencoderConfig:
    """
    Training and architecture settings for one basic autoencoder.
    """

    name: str = "default_basic"
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


class BasicAutoencoder(nn.Module):
    """
    Basic autoencoder
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = DEFAULT_CONFIG.hidden_dims,
        latent_dim: int = DEFAULT_CONFIG.latent_dim,
    ):
        super().__init__()

        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            previous_dim = hidden_dim


        encoder_layers.append(nn.Linear(previous_dim, latent_dim)) # Latent
        encoder_layers.append(nn.ReLU())

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


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


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(x)
    dataset = TensorDataset(tensor, tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss( model: nn.Module,loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Evaluate mean reconstruction loss without changing model weights.
    """
    model.eval()
    total_loss = 0.0
    total_rows = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_rows += batch_size

    return total_loss / total_rows


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Store the best epoch weights
    """
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


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
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # if using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_validation = load_data()

    #  data loaders
    train_loader = make_loader(x_train, config.batch_size, shuffle=True)
    validation_loader = make_loader(x_validation, config.batch_size, shuffle=False)

    input_dim = x_train.shape[1]
    model = BasicAutoencoder(input_dim=input_dim, hidden_dims=config.hidden_dims,
                             latent_dim=config.latent_dim,).to(device)

    criterion = nn.MSELoss() # reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,)

    best_validation_loss = float("inf") # tracking the best model and early stopping
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []


    print(f"Training config: {config.name}")
    print(f"Architecture: input -> {list(config.hidden_dims)} -> {config.latent_dim}")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0

        for inputs, targets in train_loader: # mini-batches
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss_sum += loss.item() * batch_size
            train_rows += batch_size

        train_loss = train_loss_sum / train_rows
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)

        improved = validation_loss < best_validation_loss - config.min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state_dict = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "is_best": int(improved),})

        print(
            f"Epoch {epoch:03d}/{config.max_epochs} "
            f"train_loss={train_loss:.8f} "
            f"validation_loss={validation_loss:.8f}"
        )

        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}." )
            break

    if best_state_dict is None:
        best_state_dict = clone_state_dict(model)
        best_epoch = len(history)

    model.load_state_dict(best_state_dict)

    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "config": config_to_dict(config),
                "epoch": best_epoch,
                "validation_loss": best_validation_loss,
                "epochs_trained": len(history),
                "seed": SEED,
            },
            save_model_path,
        )

    if save_history_path is not None:
        save_history(history, save_history_path)

    return {
        "model": model,
        "config": config,
        "device": device,
        "input_dim": input_dim,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "history": history,
    }


def load_trained_model(model_path: Path = MODEL_PATH, device: torch.device | None = None,
) -> tuple[BasicAutoencoder, dict[str, Any], AutoencoderConfig]:
    """
    Load the saved final basic model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    config = config_from_dict(checkpoint.get("config"))

    model = BasicAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, config


def train(config: AutoencoderConfig | None = None) -> dict[str, Any]:
    """
    Train and save the basic autoencoder
    """
    selected_config = config or load_selected_config()
    result = fit_autoencoder(selected_config, save_model_path=MODEL_PATH, save_history_path=HISTORY_PATH)
    print(f"Model saved")

    return result


if __name__ == "__main__":
    train()
