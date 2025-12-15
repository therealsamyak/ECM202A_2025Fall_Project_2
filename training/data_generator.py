"""
Data Generator for POMDP Imitation Learning

Loads oracle demonstration dataset 𝔻 = {(o_t, a_t*)} and creates PyTorch Dataset.
Observations: o_t = (B_t, d_t, Δd_t) ∈ [0,1] × [0,1] × [-1,1]
Actions: a_t* = (m_t*, c_t*) ∈ (M ∪ {∅}) × {0,1}
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any
import json
import os


class ImitationDataset(Dataset):
    """
    PyTorch Dataset for oracle demonstration data.

    Each sample contains:
    - observation: (battery_level, carbon_intensity, carbon_change)
    - action: (model_type_index, charge_decision)
    """

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        """
        Args:
            observations: Array of shape (N, 3) with [battery, carbon, change]
            actions: Array of shape (N, 2) with [model_type, charge]
        """
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)

        assert len(self.observations) == len(self.actions), (
            f"Observations ({len(self.observations)}) and actions ({len(self.actions)}) must have same length"
        )

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            observation: Tensor of shape (3,)
            action: Tensor of shape (2,) with [model_type, charge]
        """
        return self.observations[idx], self.actions[idx]


def load_training_data(
    data_dir: str = "../data/training_data",
) -> Tuple[Dict[str, ImitationDataset], dict]:
    """
    Load oracle demonstration data from existing training files.

    Args:
        data_dir: Directory containing training data files

    Returns:
        datasets: Dict with 'train', 'val', 'test' ImitationDataset objects
        metadata: Training data metadata
    """
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load numpy arrays
    datasets = {}

    for split in ["train", "val", "test"]:
        obs_path = os.path.join(data_dir, f"{split}.npy")
        actions_path = os.path.join(data_dir, f"{split}_actions.npy")

        if os.path.exists(obs_path) and os.path.exists(actions_path):
            observations = np.load(obs_path)
            actions = np.load(actions_path)

            datasets[split] = ImitationDataset(observations, actions)
            print(f"Loaded {split}: {len(datasets[split])} samples")
        else:
            print(f"Warning: Missing files for {split} split")

    return datasets, metadata


def create_data_loaders(
    datasets: Dict[str, ImitationDataset], batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training.

    Args:
        datasets: Dict with 'train', 'val', 'test' datasets
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
    """
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # No pin memory for MPS
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # No pin memory for MPS
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=False,  # No pin memory for MPS
    )

    return train_loader, val_loader, test_loader


def get_data_statistics(datasets: Dict[str, ImitationDataset]) -> Dict[str, Any]:
    """
    Calculate statistics for loaded datasets.

    Args:
        datasets: Dict with training datasets

    Returns:
        Statistics dictionary with observation and action distributions
    """
    # Concatenate all data for statistics
    all_obs = []
    all_actions = []

    for split, dataset in datasets.items():
        obs_np = dataset.observations.numpy()
        actions_np = dataset.actions.numpy()

        all_obs.append(obs_np)
        all_actions.append(actions_np)

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # Observation statistics
    obs_stats = {
        "total_samples": len(all_obs),
        "observations": {
            "battery_level": {
                "min": float(np.min(all_obs[:, 0])),
                "max": float(np.max(all_obs[:, 0])),
                "mean": float(np.mean(all_obs[:, 0])),
                "std": float(np.std(all_obs[:, 0])),
            },
            "carbon_intensity": {
                "min": float(np.min(all_obs[:, 1])),
                "max": float(np.max(all_obs[:, 1])),
                "mean": float(np.mean(all_obs[:, 1])),
                "std": float(np.std(all_obs[:, 1])),
            },
            "carbon_change": {
                "min": float(np.min(all_obs[:, 2])),
                "max": float(np.max(all_obs[:, 2])),
                "mean": float(np.mean(all_obs[:, 2])),
                "std": float(np.std(all_obs[:, 2])),
            },
        },
        "actions": {
            "model_type_distribution": {
                str(i): int(np.sum(all_actions[:, 0] == i)) for i in range(7)
            },
            "charge_distribution": {
                "charge": int(np.sum(all_actions[:, 1] == 1)),
                "no_charge": int(np.sum(all_actions[:, 1] == 0)),
            },
        },
    }

    return obs_stats


def validate_data_format(datasets: Dict[str, ImitationDataset]) -> bool:
    """
    Validate that data format matches POMDP specifications.

    Args:
        datasets: Loaded datasets

    Returns:
        True if format is valid, raises ValueError otherwise
    """
    for split, dataset in datasets.items():
        obs_np = dataset.observations.numpy()
        actions_np = dataset.actions.numpy()

        # Check observation shape: (N, 3)
        if obs_np.shape[1] != 3:
            raise ValueError(
                f"Observations must have 3 features, got {obs_np.shape[1]}"
            )

        # Check action shape: (N, 2)
        if actions_np.shape[1] != 2:
            raise ValueError(f"Actions must have 2 features, got {actions_np.shape[1]}")

        # Check observation ranges
        battery_levels = obs_np[:, 0]
        carbon_intensities = obs_np[:, 1]
        carbon_changes = obs_np[:, 2]

        if not (0 <= battery_levels).all() or not (battery_levels <= 1).all():
            raise ValueError("Battery levels must be in [0, 1]")

        if not (0 <= carbon_intensities).all() or not (carbon_intensities <= 1).all():
            raise ValueError("Carbon intensities must be in [0, 1]")

        if not (-1 <= carbon_changes).all() or not (carbon_changes <= 1).all():
            raise ValueError("Carbon changes must be in [-1, 1]")

        # Check action ranges
        model_types = actions_np[:, 0]
        charge_decisions = actions_np[:, 1]

        if not (0 <= model_types).all() or not (model_types < 7).all():
            raise ValueError("Model types must be in [0, 6]")

        if not (0 <= charge_decisions).all() or not (charge_decisions < 2).all():
            raise ValueError("Charge decisions must be 0 or 1")

    print("✓ Data format validation passed")
    return True
