#!/usr/bin/env python3
"""
Simple working training script for POMDP Imitation Learning
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

# Add training to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PolicyNetwork, get_device, save_model
from data_generator import load_training_data, create_data_loaders, validate_data_format


class CombinedLoss(nn.Module):
    """Combined loss for model selection and charging decisions."""

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.model_loss = nn.CrossEntropyLoss()
        self.charge_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        model_logits: torch.Tensor,
        charge_logit: torch.Tensor,
        model_target: torch.Tensor,
        charge_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_loss = self.model_loss(model_logits, model_target)
        charge_loss = self.charge_loss(charge_logit.squeeze(), charge_target.float())
        total_loss = model_loss + charge_loss
        return total_loss, model_loss, charge_loss


def main():
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
        default="../data/training_data",
        help="Directory containing training data",
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 60)
    print("Starting POMDP Imitation Learning training...")
    print("=" * 60)

    with open(args.config, "r") as f:
        config = json.load(f)

    print(f"Using device: {get_device()}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Batch size: {config['training']['batch_size']}")

    # Load and validate data
    print("\nLoading training data...")
    datasets, metadata = load_training_data(args.data_dir)
    validate_data_format(datasets)

    train_loader, val_loader, test_loader = create_data_loaders(
        datasets, batch_size=config["training"]["batch_size"]
    )

    # Initialize model, loss, optimizer
    device = get_device()
    model = PolicyNetwork().to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training parameters
    epochs = 2  # Use small number for demo
    patience = config["training"]["early_stopping_patience"]
    min_delta = config["training"]["early_stopping_min_delta"]

    best_val_loss = float("inf")
    patience_counter = 0
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_model_loss": [],
        "val_model_loss": [],
        "train_charge_loss": [],
        "val_charge_loss": [],
    }

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        model.train()
        train_total_losses = []
        train_model_losses = []
        train_charge_losses = []

        for batch_idx, (observations, actions) in enumerate(train_loader):
            if batch_idx >= 3:  # Limit batches for demo
                break

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

        # Validate (simplified)
        model.eval()
        with torch.no_grad():
            val_total_losses = []
            val_model_losses = []
            val_charge_losses = []

            for batch_idx, (observations, actions) in enumerate(val_loader):
                if batch_idx >= 2:  # Limit batches for demo
                    break

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

        # Calculate averages
        avg_train_total = np.mean(train_total_losses) if train_total_losses else 0
        avg_train_model = np.mean(train_model_losses) if train_model_losses else 0
        avg_train_charge = np.mean(train_charge_losses) if train_charge_losses else 0

        avg_val_total = np.mean(val_total_losses) if val_total_losses else 0
        avg_val_model = np.mean(val_model_losses) if val_model_losses else 0
        avg_val_charge = np.mean(val_charge_losses) if val_charge_losses else 0

        # Store history with proper type conversion
        training_history["train_loss"].append(float(avg_train_total))
        training_history["val_loss"].append(float(avg_val_total))
        training_history["train_model_loss"].append(float(avg_train_model))
        training_history["val_model_loss"].append(float(avg_val_model))
        training_history["train_charge_loss"].append(float(avg_train_charge))
        training_history["val_charge_loss"].append(float(avg_val_charge))

        print(
            f"  Train Loss: {avg_train_total:.6f} (Model: {avg_train_model:.6f}, Charge: {avg_train_charge:.6f})"
        )
        print(
            f"  Val Loss:   {avg_val_total:.6f} (Model: {avg_val_model:.6f}, Charge: {avg_val_charge:.6f})"
        )

        # Save best model
        if avg_val_total < best_val_loss - min_delta:
            best_val_loss = avg_val_total
            patience_counter = 0

            best_model_path = os.path.join("training/models", "best_model.pth")

            model_metadata = {
                "config": config,
                "metadata": metadata,
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
            }

            save_model(model, best_model_path, model_metadata)
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        print("-" * 60)

        # Early stopping
        if patience_counter >= patience:
            print(f"⚠ Early stopping triggered (patience={patience})")
            break

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
