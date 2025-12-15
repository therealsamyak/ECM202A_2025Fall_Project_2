"""
3-Layer Policy Network π_θ(o_t) for POMDP Imitation Learning

Implements neural network that maps observations o_t = (B_t, d_t, Δd_t)
to action distribution over (M ∪ {∅}) × {0,1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    3-layer policy network for imitation learning.

    Architecture:
    - Input: o_t = (B_t, d_t, Δd_t) ∈ [0,1] × [0,1] × [-1,1]
    - Shared backbone: Linear(64) → ReLU → Linear(32) → ReLU
    - Output heads:
      - Model selection: Linear(7) + Softmax → P(m_t | o_t)
      - Charging decision: Linear(1) + Sigmoid → P(c_t = 1 | o_t)
    """

    def __init__(self, input_dim: int = 3, num_models: int = 7):
        super(PolicyNetwork, self).__init__()

        # Shared backbone (3 layers including output splits)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models + 1),  # 7 for models + 1 for charging
        )

        self.num_models = num_models

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.

        Args:
            x: Observation tensor of shape (batch_size, 3)
               Each observation is (B_t, d_t, Δd_t)

        Returns:
            model_logits: Model selection logits (batch_size, 7)
            charge_logit: Charging decision logit (batch_size, 1)
        """
        # Pass through shared backbone
        shared_output = self.shared_layers(x)

        # Split into model logits and charge logit
        model_logits = shared_output[:, : self.num_models]  # [:7]
        charge_logit = shared_output[:, self.num_models :]  # [7:]

        return model_logits, charge_logit

    def get_action_distribution(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action probabilities from observation.

        Args:
            x: Observation tensor of shape (batch_size, 3)

        Returns:
            model_probs: Model selection probabilities (batch_size, 7)
            charge_prob: Charging probability (batch_size, 1)
        """
        model_logits, charge_logit = self.forward(x)
        model_probs = F.softmax(model_logits, dim=1)
        charge_prob = torch.sigmoid(charge_logit)

        return model_probs, charge_prob


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    Prioritizes Apple Silicon MPS for M2 Max acceleration.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def save_model(
    model: PolicyNetwork, filepath: str, metadata: dict | None = None
) -> None:
    """
    Save model with optional metadata.

    Args:
        model: Trained policy network
        filepath: Path to save model
        metadata: Optional training metadata
    """
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata or {}}, filepath
    )


def load_model(
    filepath: str, device: torch.device | None = None
) -> Tuple[PolicyNetwork, dict]:
    """
    Load model from file.

    Args:
        filepath: Path to model file
        device: Target device (auto-detected if None)

    Returns:
        model: Loaded policy network
        metadata: Training metadata
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)

    model = PolicyNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = checkpoint.get("metadata", {})

    return model, metadata
