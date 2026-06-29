"""
Train basic autoencoder here

Autoencoder structure
Input
    ->
Linear(input_dim → 48)
    ->
ReLU
    ->
Linear(48 → 24)
    ->
ReLU
    ->
Linear(24 → 12)
    ->
ReLU
    ->
Latent Space (12)
    ->
Linear(12 → 24)
   ->
ReLU
    ->
Linear(24 → 48)
    ->
ReLU
    ->
Linear(48 → input_dim)
    ->
Sigmoid
    ->
Output
"""

from pathlib import Path
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "kdd99_preprocessed_data.npz"
MODEL_PATH = ARTIFACTS_DIR / "models" / "baseline_autoencoder.pt"
HISTORY_PATH = ARTIFACTS_DIR / "training_history" / "baseline_autoencoder_history.csv"

# seed
SEED = 42

# hyperparams
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3


class BaselineAutoencoder(nn.Module):
    """
    Basic autoencoder
    """
    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    load data from artifacts
    """
    data = np.load(DATA_PATH, allow_pickle=False)
    x_train = data["X_train_normal"].astype(np.float32)
    x_validation = data["X_validation_normal"].astype(np.float32)
    return x_train, x_validation


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(x)
    dataset = TensorDataset(tensor, tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,) -> float:
    """
    evaluate loss
    """
    model.eval()
    total_loss = 0.0
    total_rows = 0

    # do not update grad
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


# train
def train() -> None:
    # set seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # if using gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_validation = load_data()

    # load data
    train_loader = make_loader(x_train, BATCH_SIZE, shuffle=True)
    validation_loader = make_loader(x_validation, BATCH_SIZE, shuffle=False)

    # get model instance
    input_dim = x_train.shape[1]
    model = BaselineAutoencoder(input_dim=input_dim).to(device)

    criterion = nn.MSELoss() #  loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []

    # make model folder
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Training:")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward() # Backpropagation
            optimizer.step()

            batch_size = inputs.size(0)
            train_loss_sum += loss.item() * batch_size
            train_rows += batch_size

        train_loss = train_loss_sum / train_rows
        validation_loss = evaluate_loss(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,})

        if validation_loss < best_validation_loss: # the best model
            best_validation_loss = validation_loss
            torch.save({ "model_state_dict": model.state_dict(), "input_dim": input_dim, "epoch": epoch,
                  "validation_loss": validation_loss,},MODEL_PATH,)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"train_loss={train_loss:.8f}"
            f"validation_loss={validation_loss:.8f}"
        )

    print(f"Best validation loss: {best_validation_loss:.8f}")
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # save training history for plotting and evaluation
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "validation_loss"],)
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved model and train history")


if __name__ == "__main__":
    train()
