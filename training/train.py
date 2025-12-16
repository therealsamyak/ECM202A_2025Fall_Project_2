#!/usr/bin/env python3
"""
Training script for POMDP imitation learning.

Simple, clean, configuration-driven training.
Works from project root: 'uv run training/train.py'

All configuration in training.config.json - no CLI arguments needed.
"""

import copy
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import training utilities (same directory, no path manipulation needed)
from utils.logging_config import setup_logging


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


def save_model(
    model: torch.nn.Module, filepath: str, metadata: dict | None = None
) -> None:
    """Save model with metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata if metadata is not None else {},
        },
        filepath,
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
        total_loss = 0.5 * model_loss + 0.5 * charge_loss
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
                print("Restored best model weights from early stopping")
            return True
        return False


def load_config(config_path: str):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration...")
        return {
            "training": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "early_stopping": {"enabled": True, "patience": 10, "min_delta": 0.001},
            },
            "data_directory": "data/training_data/",
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


class ControllerDiscovery:
    """Discovers and parses controller-specific training datasets."""

    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.pattern = r"controller_acc(?P<acc>[\d.]+)_lat(?P<lat>[\d.]+)_succ(?P<succ>\d+)_small(?P<small>\d+)_large(?P<large>\d+)_carb(?P<carb>\d+)_cap(?P<cap>\d+)_rate(?P<rate>[\d.]+)"

    def discover_controllers(self) -> list[dict]:
        """Discover all controller configurations in data directory."""
        controllers = []
        combined_files = glob.glob(f"{self.data_directory}/controller_*_combined.npz")

        for filepath in combined_files:
            filename = os.path.basename(filepath)
            match = re.match(self.pattern, filename)
            if match:
                controller_name = match.group(0).replace("_combined.npz", "")
                params = match.groupdict()

                controller = {
                    "name": controller_name,
                    "parameters": params,
                    "combined_file": filepath,
                    "train_file": filepath.replace("_combined.npz", "_train.npy"),
                    "actions_file": filepath.replace(
                        "_combined.npz", "_train_actions.npy"
                    ),
                    "val_file": filepath.replace("_combined.npz", "_val.npy"),
                    "val_actions_file": filepath.replace(
                        "_combined.npz", "_val_actions.npy"
                    ),
                    "test_file": filepath.replace("_combined.npz", "_test.npy"),
                    "test_actions_file": filepath.replace(
                        "_combined.npz", "_test_actions.npy"
                    ),
                }
                controllers.append(controller)

        return controllers


def load_controller_data(controller: dict) -> tuple:
    """Load train/val/test data for specific controller."""
    train_obs = np.load(controller["train_file"])
    train_actions = np.load(controller["actions_file"])
    val_obs = np.load(controller["val_file"])
    val_actions = np.load(controller["val_actions_file"])
    test_obs = np.load(controller["test_file"])
    test_actions = np.load(controller["test_actions_file"])

    return train_obs, train_actions, val_obs, val_actions, test_obs, test_actions


def train_single_controller(
    model,
    train_obs,
    train_actions,
    val_obs,
    val_actions,
    optimizer,
    criterion,
    controller,
    config,
    device,
    logger,
):
    """Train single controller model."""
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_obs), torch.LongTensor(train_actions)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_obs), torch.LongTensor(val_actions)
    )

    batch_size = config.get("training", {}).get("batch_size", 32)
    epochs = config.get("training", {}).get("epochs", 100)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping
    early_stopping_config = config.get("training", {}).get("early_stopping", {})
    if early_stopping_config.get("enabled", True):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.001),
            restore_best_weights=True,
        )
    else:
        early_stopping = None

    best_val_loss = float("inf")
    final_val_model_acc = 0.0
    final_val_charge_acc = 0.0

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

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_val_model_acc = val_model_acc
            final_val_charge_acc = val_charge_acc

        # Check early stopping
        if early_stopping:
            if early_stopping(val_loss, model):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

        logger.info("-" * 60)

    return best_val_loss, final_val_model_acc, final_val_charge_acc


def log_training_summary(results: list[dict], logger):
    """Log summary of all controller training results."""
    logger.info("\n" + "=" * 80)
    logger.info("🏆 TRAINING SUMMARY - ALL CONTROLLERS")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"\n🎯 {result['controller_name']}")
        logger.info(f"   Parameters: {result['parameters']}")
        logger.info(f"   Test Model Acc: {result['test_model_acc']:.4f}")
        logger.info(f"   Test Charge Acc: {result['test_charge_acc']:.4f}")
        logger.info(f"   Test Loss: {result['test_loss']:.6f}")

    # Average performance
    avg_model_acc = np.mean([r["test_model_acc"] for r in results])
    avg_charge_acc = np.mean([r["test_charge_acc"] for r in results])

    logger.info("\n📊 AVERAGE PERFORMANCE:")
    logger.info(f"   Model Accuracy: {avg_model_acc:.4f}")
    logger.info(f"   Charge Accuracy: {avg_charge_acc:.4f}")
    logger.info("=" * 80)


def train_per_controller_mode(config, device, logger) -> int:
    """Train separate model for each controller configuration."""

    # Discover all controllers
    discovery = ControllerDiscovery(config["data_directory"])
    controllers = discovery.discover_controllers()

    logger.info(f"Found {len(controllers)} controllers to train")

    results = []

    for i, controller in enumerate(controllers, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"🎯 Training Controller {i}/{len(controllers)}: {controller['name']}"
        )
        logger.info(f"📋 Parameters: {controller['parameters']}")
        logger.info(f"{'=' * 60}")

        # Load controller-specific data
        train_obs, train_actions, val_obs, val_actions, test_obs, test_actions = (
            load_controller_data(controller)
        )

        # Create fresh model for this controller
        model = PolicyNetwork().to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("training", {}).get("learning_rate", 0.001),
        )
        criterion = ImitationLoss()

        # Train this controller
        best_val_loss, val_model_acc, val_charge_acc = train_single_controller(
            model,
            train_obs,
            train_actions,
            val_obs,
            val_actions,
            optimizer,
            criterion,
            controller,
            config,
            device,
            logger,
        )

        # Test evaluation
        test_dataset = TensorDataset(
            torch.FloatTensor(test_obs), torch.LongTensor(test_actions)
        )
        batch_size = config.get("training", {}).get("batch_size", 32)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_loss, test_model_acc, test_charge_acc = evaluate_model(
            model, test_loader, device, criterion
        )

        # Save model with controller-specific naming
        model_save_path = f"training/models/C{i}_{controller['name']}_best_model.pth"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        model_metadata = {
            "config": config,
            "controller_name": controller["name"],
            "controller_parameters": controller["parameters"],
            "best_val_loss": best_val_loss,
            "val_model_accuracy": val_model_acc,
            "val_charge_accuracy": val_charge_acc,
            "test_loss": test_loss,
            "test_model_accuracy": test_model_acc,
            "test_charge_accuracy": test_charge_acc,
        }

        save_model(model, model_save_path, model_metadata)
        logger.info(f"  💾 Model saved: {model_save_path}")

        # Store results
        result = {
            "controller_name": controller["name"],
            "parameters": controller["parameters"],
            "best_val_loss": best_val_loss,
            "val_model_acc": val_model_acc,
            "val_charge_acc": val_charge_acc,
            "test_loss": test_loss,
            "test_model_acc": test_model_acc,
            "test_charge_acc": test_charge_acc,
        }
        results.append(result)

        logger.info(f"✅ Controller {controller['name']} completed")
        logger.info(
            f"   Test Results: Model Acc: {test_model_acc:.4f}, Charge Acc: {test_charge_acc:.4f}"
        )

    # Final summary
    log_training_summary(results, logger)

    # Write metadata to training.models/training.metadata.json
    metadata = {
        "controllers": [],
        "average_performance": {
            "model_accuracy": np.mean([r["test_model_acc"] for r in results]),
            "charge_accuracy": np.mean([r["test_charge_acc"] for r in results]),
        },
    }

    # Format controller data for metadata
    for result in results:
        params = result["parameters"]
        controller_data = {
            "acc": float(params["acc"]),
            "lat": float(params["lat"]),
            "succ": int(params["succ"]),
            "small": int(params["small"]),
            "large": int(params["large"]),
            "carb": int(params["carb"]),
            "cap": int(params["cap"]),
            "rate": float(params["rate"]),
            "test_model_acc": result["test_model_acc"],
            "test_charge_acc": result["test_charge_acc"],
            "test_loss": result["test_loss"],
        }
        metadata["controllers"].append(controller_data)

    # Write metadata file
    metadata_path = "training/models/training.metadata.json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"📄 Metadata written to: {metadata_path}")

    return 0


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
    logger.info("🎯 Training Mode: per_controller")
    logger.info("-" * 60)

    # Setup
    device = get_device()
    logger.info(f"💻 Using device: {device}")

    # Always use per_controller mode
    return train_per_controller_mode(config, device, logger)


if __name__ == "__main__":
    sys.exit(main())
