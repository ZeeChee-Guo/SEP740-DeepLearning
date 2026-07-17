"""
Train a sparse autoencoder for KDD Cup 1999 anomaly detection.

The layer sizes mirror the final basic autoencoder (74 -> 64 -> 32 -> 16 ->
32 -> 64 -> 74) so the two models can be compared fairly. The only structural
change is that the encoder uses Sigmoid activations (so each hidden unit output
can be read as an activation probability in the range 0..1) and the training
loss adds a KL-divergence sparsity penalty on the encoder activations.

Total loss = MSE(reconstruction) + SPARSITY_WEIGHT * KL_sparsity_penalty

Autoencoder structure
Input
    ->
Linear(input_dim -> 64) -> Sigmoid   (encoder activation a1, sparsity applied)
    ->
Linear(64 -> 32) -> Sigmoid          (encoder activation a2, sparsity applied)
    ->
Linear(32 -> 16) -> Sigmoid          (latent activation a3, sparsity applied)
    ->
Linear(16 -> 32) -> ReLU
    ->
Linear(32 -> 64) -> ReLU
    ->
Linear(64 -> input_dim) -> Sigmoid   (output in the same 0..1 range as inputs)
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
MODEL_PATH = ARTIFACTS_DIR / "models" / "sparse_autoencoder.pt"
HISTORY_PATH = ARTIFACTS_DIR / "training_history" / "sparse_autoencoder_history.csv"

# seed
SEED = 42

# hyperparams
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-3

# sparsity hyperparams
# SPARSITY_TARGET (rho): the desired average activation of each hidden unit.
#   A small value (0.05) means we want each unit to stay mostly inactive.
# SPARSITY_WEIGHT (beta): how strongly the KL sparsity penalty is applied.
SPARSITY_TARGET = 0.1
SPARSITY_WEIGHT = 1e-3


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder with Sigmoid encoder activations.

    forward() returns both the reconstruction and the list of encoder
    activations, because the sparsity penalty is computed from those
    activations during training.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # encoder layers are defined individually so the intermediate
        # activations can be collected for the sparsity penalty
        self.encoder_hidden_1 = nn.Linear(input_dim, 64)
        self.encoder_hidden_2 = nn.Linear(64, 32)
        self.encoder_latent = nn.Linear(32, 16)

        # decoder mirrors the basic decoder (ReLU hidden, Sigmoid output)
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

        # Sigmoid keeps encoder activations in 0..1 so they can be treated
        # as activation probabilities for the KL sparsity penalty
        self.encoder_activation = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        a1 = self.encoder_activation(self.encoder_hidden_1(x))
        a2 = self.encoder_activation(self.encoder_hidden_2(a1))
        latent = self.encoder_activation(self.encoder_latent(a2))
        # every encoder activation is pushed toward the sparsity target
        return latent, [a1, a2, latent]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        latent, activations = self.encode(x)
        reconstructed = self.decoder(latent)
        return reconstructed, activations


def kl_divergence(rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between the target activation rho (a scalar) and the observed
    mean activation rho_hat (one value per hidden unit).

    KL(rho || rho_hat) = rho * log(rho / rho_hat)
                         + (1 - rho) * log((1 - rho) / (1 - rho_hat))
    """
    # clamp to avoid log(0) / division by zero when a unit is fully on/off
    rho_hat = torch.clamp(rho_hat, 1e-6, 1.0 - 1e-6)
    return rho * torch.log(rho / rho_hat) + (1.0 - rho) * torch.log(
        (1.0 - rho) / (1.0 - rho_hat)
    )


def sparsity_penalty(activations: list[torch.Tensor], rho: float) -> torch.Tensor:
    """
    Sum the KL sparsity penalty across all encoder layers.

    For each layer we average the activations over the batch dimension to get
    the observed mean activation per unit (rho_hat), then sum the KL divergence
    over the units.
    """
    total = activations[0].new_zeros(())
    for activation in activations:
        rho_hat = torch.mean(activation, dim=0)
        total = total + torch.sum(kl_divergence(rho, rho_hat))
    return total


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the preprocessed normal train / validation splits from artifacts.
    """
    data = np.load(DATA_PATH, allow_pickle=False)
    x_train = data["X_train_normal"].astype(np.float32)
    x_validation = data["X_validation_normal"].astype(np.float32)
    return x_train, x_validation


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(x)
    dataset = TensorDataset(tensor, tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate the average reconstruction, sparsity, and total loss on a loader.
    """
    model.eval()
    total_reconstruction = 0.0
    total_sparsity = 0.0
    total_rows = 0

    # do not update grad
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, activations = model(inputs)
            reconstruction_loss = criterion(outputs, targets)
            penalty = sparsity_penalty(activations, SPARSITY_TARGET)

            batch_size = inputs.size(0)
            total_reconstruction += reconstruction_loss.item() * batch_size
            total_sparsity += penalty.item() * batch_size
            total_rows += batch_size

    mean_reconstruction = total_reconstruction / total_rows
    mean_sparsity = total_sparsity / total_rows
    mean_total = mean_reconstruction + SPARSITY_WEIGHT * mean_sparsity
    return mean_reconstruction, mean_sparsity, mean_total


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
    model = SparseAutoencoder(input_dim=input_dim).to(device)

    criterion = nn.MSELoss()  # reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # model selection is based on validation reconstruction error, so the
    # threshold calibration and evaluation stay comparable with the baseline
    best_validation_reconstruction = float("inf")
    history: list[dict[str, float]] = []

    # make model folder
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Sparsity target (rho): {SPARSITY_TARGET}, weight (beta): {SPARSITY_WEIGHT}")
    print(f"Training:")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_reconstruction_sum = 0.0
        train_sparsity_sum = 0.0
        train_rows = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, activations = model(inputs)
            reconstruction_loss = criterion(outputs, targets)
            penalty = sparsity_penalty(activations, SPARSITY_TARGET)
            loss = reconstruction_loss + SPARSITY_WEIGHT * penalty

            loss.backward()  # Backpropagation
            optimizer.step()

            batch_size = inputs.size(0)
            train_reconstruction_sum += reconstruction_loss.item() * batch_size
            train_sparsity_sum += penalty.item() * batch_size
            train_rows += batch_size

        train_reconstruction = train_reconstruction_sum / train_rows
        train_sparsity = train_sparsity_sum / train_rows
        train_total = train_reconstruction + SPARSITY_WEIGHT * train_sparsity

        (
            validation_reconstruction,
            validation_sparsity,
            validation_total,
        ) = evaluate_loss(model, validation_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_reconstruction": train_reconstruction,
                "train_sparsity": train_sparsity,
                "train_total": train_total,
                "validation_reconstruction": validation_reconstruction,
                "validation_sparsity": validation_sparsity,
                "validation_total": validation_total,
            }
        )

        # keep the model with the lowest validation reconstruction error
        if validation_reconstruction < best_validation_reconstruction:
            best_validation_reconstruction = validation_reconstruction
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "epoch": epoch,
                    "validation_reconstruction": validation_reconstruction,
                    "sparsity_target": SPARSITY_TARGET,
                    "sparsity_weight": SPARSITY_WEIGHT,
                },
                MODEL_PATH,
            )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} "
            f"train_recon={train_reconstruction:.8f} "
            f"val_recon={validation_reconstruction:.8f} "
            f"val_sparsity={validation_sparsity:.4f}"
        )

    print(f"Best validation reconstruction: {best_validation_reconstruction:.8f}")
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # save training history for plotting and evaluation
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_reconstruction",
                "train_sparsity",
                "train_total",
                "validation_reconstruction",
                "validation_sparsity",
                "validation_total",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved model and train history")


if __name__ == "__main__":
    train()
