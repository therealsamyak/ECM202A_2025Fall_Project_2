"""
Training Pipeline for POMDP Imitation Learning

Implements supervised imitation learning to minimize ℒ(θ) = 𝔼_{(o_t,a_t*)∼𝔻}[ℓ(π_θ(o_t), a_t*)]
where ℓ is cross-entropy loss between predicted and oracle actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
import os
import json
from pathlib import Path

# Try to import tensorboard, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Direct imports for simplicity

# Import local modules
from model import PolicyNetwork, get_device, save_model, load_model
from data_generator import (
    load_training_data,
    create_data_loaders,
    validate_data_format,
)


class ImitationLoss(nn.Module):
    """
    Combined loss for model selection and charging decisions.

    Loss = CrossEntropy(model_logits, model_target) + BCEWithLogits(charge_logit, charge_target)
    """

    def __init__(self):
        super(ImitationLoss, self).__init__()
        self.model_loss = nn.CrossEntropyLoss()
        self.charge_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        model_logits: torch.Tensor,
        charge_logit: torch.Tensor,
        model_target: torch.Tensor,
        charge_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate combined loss.

        Args:
            model_logits: Model selection logits (batch_size, 7)
            charge_logit: Charging decision logit (batch_size, 1)
            model_target: Oracle model selection indices (batch_size,)
            charge_target: Oracle charge decisions (batch_size, 1)

        Returns:
            total_loss: Combined loss
            model_loss: Model selection loss
            charge_loss: Charging decision loss
        """
        model_loss = self.model_loss(model_logits, model_target)
        charge_loss = self.charge_loss(charge_logit.squeeze(), charge_target.float())
        total_loss = model_loss + charge_loss

        return total_loss, model_loss, charge_loss


class EarlyStopping:
    """
    Early stopping based on validation loss improvement.
    Stops if validation loss doesn't improve by `min_delta` for `patience` epochs.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.001, verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if should stop training.

        Args:
            val_loss: Current validation loss

        Returns:
            True if should stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"  ✓ Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"  ✗ Validation loss did not improve. Counter: {self.counter}/{self.patience}"
                )

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"⚠ Early stopping triggered (patience={self.patience})")
            return True

        return False


def train_epoch(
    model: PolicyNetwork,
    train_loader: DataLoader,
    criterion: ImitationLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Train model for one epoch.

    Returns:
        avg_total_loss, avg_model_loss, avg_charge_loss
    """
    model.train()
    train_total_losses = []
    train_model_losses = []
    train_charge_losses = []

    for observations, actions in train_loader:
        observations = observations.to(device)
        model_targets = actions[:, 0].to(device)
        charge_targets = actions[:, 1].to(device)

        optimizer.zero_grad()

        model_logits, charge_logit = model(observations)
        total_loss, model_loss, charge_loss = criterion(
            model_logits, charge_logit, model_targets, charge_targets
        )

        total_loss.backward()
        optimizer.step()

        train_total_losses.append(total_loss.item())
        train_model_losses.append(model_loss.item())
        train_charge_losses.append(charge_loss.item())

    avg_total_loss = np.mean(train_total_losses)
    avg_model_loss = np.mean(train_model_losses)
    avg_charge_loss = np.mean(train_charge_losses)

    return float(avg_total_loss), float(avg_model_loss), float(avg_charge_loss)


def validate_epoch(
    model: PolicyNetwork,
    val_loader: DataLoader,
    criterion: ImitationLoss,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Validate model for one epoch.

    Returns:
        avg_total_loss, avg_model_loss, avg_charge_loss
    """
    model.eval()
    val_total_losses = []
    val_model_losses = []
    val_charge_losses = []

    with torch.no_grad():
        for observations, actions in val_loader:
            observations = observations.to(device)
            model_targets = actions[:, 0].to(device)
            charge_targets = actions[:, 1].to(device)

            model_logits, charge_logit = model(observations)
            total_loss, model_loss, charge_loss = criterion(
                model_logits, charge_logit, model_targets, charge_targets
            )

            val_total_losses.append(total_loss.item())
            val_model_losses.append(model_loss.item())
            val_charge_losses.append(charge_loss.item())

    avg_total_loss = np.mean(val_total_losses)
    avg_model_loss = np.mean(val_model_losses)
    avg_charge_loss = np.mean(val_charge_losses)

    return float(avg_total_loss), float(avg_model_loss), float(avg_charge_loss)


def train_imitation_learning(
    config: Dict,
    data_dir: str = "../data/training_data",
    log_dir: str = "./logs",
    model_save_dir: str = "./models",
) -> Tuple[PolicyNetwork, Dict]:
    """
    Train imitation learning policy network.

    Args:
        config: Training configuration dictionary
        data_dir: Directory containing training data
        log_dir: TensorBoard log directory
        model_save_dir: Directory to save trained models

    Returns:
        trained_model: Trained PolicyNetwork
        training_history: Dictionary with training metrics
    """
    # Setup device and directories
    device = get_device()
    Path(log_dir).mkdir(exist_ok=True)
    Path(model_save_dir).mkdir(exist_ok=True)

    print(f"Using device: {device}")

    # Load and validate data
    print("Loading training data...")
    datasets, metadata = load_training_data(data_dir)
    validate_data_format(datasets)

    train_loader, val_loader, test_loader = create_data_loaders(
        datasets,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
    )

    # Initialize model, loss, optimizer
    model = PolicyNetwork().to(device)
    criterion = ImitationLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Setup tensorboard
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Training setup
    epochs = config["training"]["epochs"]
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["early_stopping_min_delta"],
    )

    best_val_loss = float("inf")
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_model_loss": [],
        "val_model_loss": [],
        "train_charge_loss": [],
        "val_charge_loss": [],
    }

    print(f"Starting training for {epochs} epochs...")
    print("-" * 60)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_total_loss, train_model_loss, train_charge_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_total_loss, val_model_loss, val_charge_loss = validate_epoch(
            model, val_loader, criterion, device
        )

        # Log metrics
        training_history["train_loss"].append(train_total_loss)
        training_history["val_loss"].append(val_total_loss)
        training_history["train_model_loss"].append(train_model_loss)
        training_history["val_model_loss"].append(val_model_loss)
        training_history["train_charge_loss"].append(train_charge_loss)
        training_history["val_charge_loss"].append(val_charge_loss)

        # Tensorboard logging
        if writer is not None:
            writer.add_scalar("Loss/Train_Total", train_total_loss, epoch)
            writer.add_scalar("Loss/Val_Total", val_total_loss, epoch)
            writer.add_scalar("Loss/Train_Model", train_model_loss, epoch)
            writer.add_scalar("Loss/Val_Model", val_model_loss, epoch)
            writer.add_scalar("Loss/Train_Charge", train_charge_loss, epoch)
            writer.add_scalar("Loss/Val_Charge", val_charge_loss, epoch)

        print(
            f"  Train Loss: {train_total_loss:.6f} (Model: {train_model_loss:.6f}, Charge: {train_charge_loss:.6f})"
        )
        print(
            f"  Val Loss:   {val_total_loss:.6f} (Model: {val_model_loss:.6f}, Charge: {val_charge_loss:.6f})"
        )

        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_path = os.path.join(model_save_dir, "best_model.pth")

            model_metadata = {
                "config": config,
                "metadata": metadata,
                "training_history": training_history,
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
            }

            save_model(model, best_model_path, model_metadata)
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.6f})")

        print("-" * 60)

        # Early stopping check
        if early_stopping(val_total_loss):
            break

    # Test evaluation
    print("Evaluating on test set...")
    test_total_loss, test_model_loss, test_charge_loss = validate_epoch(
        model, test_loader, criterion, device
    )
    print(
        f"Test Loss: {test_total_loss:.6f} (Model: {test_model_loss:.6f}, Charge: {test_charge_loss:.6f})"
    )

    training_history["test_loss"] = test_total_loss
    training_history["test_model_loss"] = test_model_loss
    training_history["test_charge_loss"] = test_charge_loss

    if writer is not None:
        writer.close()

    # Load best model for return
    best_model_path = os.path.join(model_save_dir, "best_model.pth")
    best_model, _ = load_model(best_model_path, device)

    return best_model, training_history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train POMDP Imitation Learning")
    parser.add_argument(
        "--config",
        type=str,
        default="training/training.config.json",
        help="Path to training configuration",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/training_data",
        help="Directory containing training data",
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 60)
    print("Starting POMDP Imitation Learning training...")
    print("=" * 60)

    config_path = args.config

    with open(config_path, "r") as f:
        config = json.load(f)

    # Train model
    print("=" * 60)
    print("Starting POMDP Imitation Learning training...")
    model, history = train_imitation_learning(
        config=config,
        data_dir=args.data_dir,
        log_dir="./logs",
        model_save_dir="training/models",
    )

    print("Training completed successfully!")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"Test loss: {history['test_loss']:.6f}")
    print("=" * 60)
