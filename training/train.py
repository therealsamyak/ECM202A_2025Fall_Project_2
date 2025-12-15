#!/usr/bin/env python3
"""
Training script for POMDP imitation learning.

Simple, clean, configuration-driven training.
Works from project root: 'uv run training/train.py'

All configuration in training.config.json - no CLI arguments needed.
"""

import copy
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import training utilities (same directory, no path manipulation needed)
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """3-layer policy network for POMDP imitation learning."""

    def __init__(self, input_dim: int = 3, num_models: int = 7):
        super(PolicyNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models + 1),
        )
        self.num_models = num_models

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_output = self.shared_layers(x)
        model_logits = shared_output[:, : self.num_models]
        charge_logit = shared_output[:, self.num_models :]
        return model_logits, charge_logit


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_model(model: torch.nn.Module, filepath: str, metadata: dict = None) -> None:
    """Save model with metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata or {}}, filepath
    )


class ImitationLoss(nn.Module):
    """Combined loss for model selection and charging decisions."""

    def __init__(self):
        super(ImitationLoss, self).__init__()
        self.model_loss = nn.CrossEntropyLoss()
        self.charge_loss = nn.BCEWithLogitsLoss()

    def forward(self, model_logits, charge_logit, model_target, charge_target):
        model_loss = self.model_loss(model_logits, model_target)
        charge_loss = self.charge_loss(charge_logit.squeeze(), charge_target.float())
        total_loss = model_loss + charge_loss
        return total_loss, model_loss, charge_loss


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights from early stopping")
            return True
        return False


def load_config(config_path: str):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration...")
        return {
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
            },
            "data": {
                "train_data_path": "data/training_data/train.npy",
                "val_data_path": "data/training_data/val.npy",
                "test_data_path": "data/training_data/test.npy",
                "train_actions_path": "data/training_data/train_actions.npy",
                "val_actions_path": "data/training_data/val_actions.npy",
                "test_actions_path": "data/training_data/test_actions.npy",
            },
            "paths": {
                "model_save_path": "training/models/best_model.pth",
            },
        }

    with open(config_path, "r") as f:
        return json.load(f)


def calculate_accuracy(model_logits, charge_logit, model_targets, charge_targets):
    """Calculate accuracy metrics for model and charge predictions."""
    # Model accuracy
    model_preds = torch.argmax(model_logits, dim=1)
    model_acc = (model_preds == model_targets).float().mean().item()

    # Charge accuracy (binary classification)
    charge_preds = (torch.sigmoid(charge_logit.squeeze()) > 0.5).float()
    charge_acc = (charge_preds == charge_targets).float().mean().item()

    return model_acc, charge_acc


def evaluate_model(model, data_loader, device, criterion):
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0
    model_accuracies = []
    charge_accuracies = []

    with torch.no_grad():
        for observations, actions in data_loader:
            observations = observations.to(device)
            model_targets = actions[:, 0].to(device)
            charge_targets = actions[:, 1].to(device)

            model_logits, charge_logit = model(observations)
            total_loss_batch, model_loss, charge_loss = criterion(
                model_logits, charge_logit, model_targets, charge_targets
            )

            total_loss += total_loss_batch.item()

            # Calculate accuracies
            model_acc, charge_acc = calculate_accuracy(
                model_logits, charge_logit, model_targets, charge_targets
            )
            model_accuracies.append(model_acc)
            charge_accuracies.append(charge_acc)

    avg_loss = total_loss / len(data_loader)
    avg_model_acc = np.mean(model_accuracies)
    avg_charge_acc = np.mean(charge_accuracies)

    return avg_loss, avg_model_acc, avg_charge_acc


def main():
    """Main training function."""
    # Setup logging first
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("🚀 POMDP Imitation Learning Training")
    logger.info("=" * 60)

    # Load configuration from fixed file path
    config_path = "training/training.config.json"
    config = load_config(config_path)

    # Display configuration
    logger.info(f"📄 Configuration: {config_path}")

    training_config = config.get("training", {})
    logger.info(f"🏋 Epochs: {training_config.get('epochs', 'N/A')}")
    logger.info(f"📦 Batch size: {training_config.get('batch_size', 'N/A')}")
    logger.info(f"⚡ Learning rate: {training_config.get('learning_rate', 'N/A')}")
    logger.info("-" * 60)

    # Setup
    device = get_device()
    logger.info(f"💻 Using device: {device}")

    model = PolicyNetwork().to(device)
    criterion = ImitationLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=training_config.get("learning_rate", 0.001)
    )

    # Load real oracle training data
    logger.info("Loading oracle training data...")

    data_config = config.get("data", {})
    train_data_path = data_config.get("train_data_path", "data/training_data/train.npy")
    train_actions_path = data_config.get(
        "train_actions_path", "data/training_data/train_actions.npy"
    )
    val_data_path = data_config.get("val_data_path", "data/training_data/val.npy")
    val_actions_path = data_config.get(
        "val_actions_path", "data/training_data/val_actions.npy"
    )
    test_data_path = data_config.get("test_data_path", "data/training_data/test.npy")
    test_actions_path = data_config.get(
        "test_actions_path", "data/training_data/test_actions.npy"
    )

    # Load numpy arrays
    train_obs = np.load(train_data_path)
    train_actions = np.load(train_actions_path)
    val_obs = np.load(val_data_path)
    val_actions = np.load(val_actions_path)
    test_obs = np.load(test_data_path)
    test_actions = np.load(test_actions_path)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_obs), torch.LongTensor(train_actions)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_obs), torch.LongTensor(val_actions)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_obs), torch.LongTensor(test_actions)
    )

    logger.info(f"Loaded training data: {len(train_dataset)} samples")
    logger.info(f"Loaded validation data: {len(val_dataset)} samples")
    logger.info(f"Loaded test data: {len(test_dataset)} samples")

    batch_size = training_config.get("batch_size", 32)
    epochs = training_config.get("epochs", 100)
    model_save_path = config.get("paths", {}).get(
        "model_save_path", "training/models/best_model.pth"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("📊 Data loaded successfully (using real oracle data)")
    logger.info(f"🚀 Starting training for {epochs} epochs...")
    logger.info("-" * 60)

    # Setup early stopping
    early_stopping_config = training_config.get("early_stopping", {})
    if early_stopping_config.get("enabled", True):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.001),
            restore_best_weights=True,
        )
        logger.info(
            f"🛑 Early stopping enabled: patience={early_stopping_config.get('patience', 10)}"
        )
    else:
        early_stopping = None
        logger.info("🛑 Early stopping disabled")

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        logger.info(f"📈 Epoch {epoch + 1}/{epochs}")

        # Training
        model.train()
        train_total_losses = []

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

        # Validation
        val_loss, val_model_acc, val_charge_acc = evaluate_model(
            model, val_loader, device, criterion
        )

        avg_train_loss = np.mean(train_total_losses) if train_total_losses else 0

        logger.info(f"  🏋 Train Loss: {avg_train_loss:.6f}")
        logger.info(
            f"  ✅ Val Loss: {val_loss:.6f} (Model Acc: {val_model_acc:.4f}, Charge Acc: {val_charge_acc:.4f})"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

            model_metadata = {
                "config": config,
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
                "val_model_accuracy": val_model_acc,
                "val_charge_accuracy": val_charge_acc,
            }

            save_model(model, model_save_path, model_metadata)
            logger.info(f"  💾 New best model saved (val_loss: {best_val_loss:.6f})")

        # Check early stopping
        if early_stopping:
            if early_stopping(val_loss, model):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("-" * 60)

    # Test set evaluation
    logger.info("🧪 Evaluating on test set...")
    test_loss, test_model_acc, test_charge_acc = evaluate_model(
        model, test_loader, device, criterion
    )
    logger.info("📊 Test Results:")
    logger.info(f"  Test Loss: {test_loss:.6f}")
    logger.info(f"  Test Model Accuracy: {test_model_acc:.4f}")
    logger.info(f"  Test Charge Accuracy: {test_charge_acc:.4f}")

    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training completed successfully!")
    logger.info(f"📊 Best validation loss: {best_val_loss:.6f}")
    logger.info(f"💾 Best model saved to: {model_save_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
