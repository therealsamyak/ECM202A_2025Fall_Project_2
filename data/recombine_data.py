#!/usr/bin/env python3
"""
Final dataset assembly script.
Combines chunk files, performs data splitting, and generates final training dataset.
"""

import os
import json
import numpy as np
import shutil
from datetime import datetime
from typing import Dict, List, Any, Tuple


class DataRecombiner:
    """Combines chunk files into final training dataset"""

    def __init__(
        self, temp_dir: str = "data/temp", output_dir: str = "data/training_data"
    ):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_all_chunks(self) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Load all chunk files and combine data

        Returns:
            Tuple of (observations, actions, metadata_list)
        """
        chunk_files = [
            f
            for f in os.listdir(self.temp_dir)
            if f.startswith("chunk_") and f.endswith(".npz")
        ]
        chunk_files.sort(
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )  # Sort by combination ID

        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.temp_dir}")

        print(f"Loading {len(chunk_files)} chunk files...")

        all_observations = []
        all_actions = []
        all_metadata = []

        total_timesteps = 0

        for chunk_file in chunk_files:
            chunk_path = os.path.join(self.temp_dir, chunk_file)

            try:
                # Load chunk data
                data = np.load(chunk_path)
                metadata = json.loads(str(data["metadata"]))
                detailed_results = json.loads(str(data["detailed_results"]))

                # Extract arrays
                chunk_observations = data["observations"]
                chunk_actions = data["actions"]

                # Validate shapes
                if chunk_observations.shape[0] != chunk_actions.shape[0]:
                    print(f"Warning: Shape mismatch in {chunk_file}")
                    continue

                # Add to combined data
                all_observations.append(chunk_observations)
                all_actions.append(chunk_actions)

                # Extend metadata for each timestep
                for result in detailed_results:
                    if result["success"]:
                        training_data = result["training_data"]
                        for _ in range(len(training_data["observations"])):
                            all_metadata.append(
                                {
                                    "combination_id": result["combination_id"],
                                    "total_reward": result["metadata"]["total_reward"],
                                    "optimal_value": result["metadata"][
                                        "optimal_value"
                                    ],
                                    "execution_time": result["metadata"][
                                        "execution_time_seconds"
                                    ],
                                    "chunk_file": chunk_file,
                                }
                            )

                total_timesteps += metadata["total_timesteps"]
                print(f"  Loaded {chunk_file}: {metadata['total_timesteps']} timesteps")

            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
                continue

        if not all_observations:
            raise ValueError("No valid chunk data found")

        # Combine all arrays
        combined_observations = np.vstack(all_observations)
        combined_actions = np.vstack(all_actions)

        print(f"Combined dataset: {combined_observations.shape[0]} timesteps")

        return combined_observations, combined_actions, all_metadata

    def split_data(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Split data into train/val/test sets

        Args:
            observations: Observation array
            actions: Action array
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Dictionary with train/val/test splits
        """
        total_samples = observations.shape[0]

        # Calculate split indices
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        # Split data
        train_obs = observations[:train_end]
        train_actions = actions[:train_end]

        val_obs = observations[train_end:val_end]
        val_actions = actions[train_end:val_end]

        test_obs = observations[val_end:]
        test_actions = actions[val_end:]

        splits = {
            "train_obs": train_obs,
            "train_actions": train_actions,
            "val_obs": val_obs,
            "val_actions": val_actions,
            "test_obs": test_obs,
            "test_actions": test_actions,
        }

        print("Data split:")
        print(
            f"  Train: {train_obs.shape[0]} samples ({train_obs.shape[0] / total_samples:.1%})"
        )
        print(
            f"  Val:   {val_obs.shape[0]} samples ({val_obs.shape[0] / total_samples:.1%})"
        )
        print(
            f"  Test:  {test_obs.shape[0]} samples ({test_obs.shape[0] / total_samples:.1%})"
        )

        # Display sample data for validation
        print("\nSample data validation:")
        self.display_samples("Train", train_obs, train_actions, max_samples=3)
        self.display_samples("Val", val_obs, val_actions, max_samples=2)
        self.display_samples("Test", test_obs, test_actions, max_samples=2)

        return splits

    def save_final_dataset(
        self, splits: Dict[str, Any], metadata: List[Dict[str, Any]]
    ):
        """
        Save final dataset to output directory

        Args:
            splits: Dictionary with train/val/test data
            metadata: List of metadata for each sample
        """
        print("Saving final dataset...")

        # Save individual split files
        train_path = os.path.join(self.output_dir, "train.npy")
        val_path = os.path.join(self.output_dir, "val.npy")
        test_path = os.path.join(self.output_dir, "test.npy")

        train_actions_path = os.path.join(self.output_dir, "train_actions.npy")
        val_actions_path = os.path.join(self.output_dir, "val_actions.npy")
        test_actions_path = os.path.join(self.output_dir, "test_actions.npy")

        np.save(train_path, splits["train_obs"])
        np.save(train_actions_path, splits["train_actions"])
        np.save(val_path, splits["val_obs"])
        np.save(val_actions_path, splits["val_actions"])
        np.save(test_path, splits["test_obs"])
        np.save(test_actions_path, splits["test_actions"])

        print(f"  Saved train: {train_path}")
        print(f"  Saved train_actions: {train_actions_path}")
        print(f"  Saved val: {val_path}")
        print(f"  Saved val_actions: {val_actions_path}")
        print(f"  Saved test: {test_path}")
        print(f"  Saved test_actions: {test_actions_path}")

        # Save combined dataset
        all_obs = np.vstack(
            [splits["train_obs"], splits["val_obs"], splits["test_obs"]]
        )
        all_actions = np.vstack(
            [splits["train_actions"], splits["val_actions"], splits["test_actions"]]
        )
        combined_path = os.path.join(self.output_dir, "combined_dataset.npz")

        np.savez_compressed(
            combined_path,
            observations=all_obs,
            actions=all_actions,
            train_indices=(0, splits["train_obs"].shape[0]),
            val_indices=(
                splits["train_obs"].shape[0],
                splits["train_obs"].shape[0] + splits["val_obs"].shape[0],
            ),
            test_indices=(
                splits["train_obs"].shape[0] + splits["val_obs"].shape[0],
                all_obs.shape[0],
            ),
        )
        print(f"  Saved combined: {combined_path}")

        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")

        # Generate statistics
        stats = self.generate_statistics(all_obs, all_actions, metadata)

        final_metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_samples": all_obs.shape[0],
            "observation_features": 3,
            "action_features": 2,
            "data_split": {
                "train": splits["train_obs"].shape[0],
                "val": splits["val_obs"].shape[0],
                "test": splits["test_obs"].shape[0],
            },
            "statistics": stats,
            "sample_metadata": metadata[:100],  # Save first 100 samples as examples
        }

        with open(metadata_path, "w") as f:
            json.dump(final_metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

    def generate_statistics(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate dataset statistics"""
        stats = {}

        # Observation statistics
        stats["observations"] = {
            "battery_level": {
                "min": float(observations[:, 0].min()),
                "max": float(observations[:, 0].max()),
                "mean": float(observations[:, 0].mean()),
                "std": float(observations[:, 0].std()),
            },
            "carbon_intensity": {
                "min": float(observations[:, 1].min()),
                "max": float(observations[:, 1].max()),
                "mean": float(observations[:, 1].mean()),
                "std": float(observations[:, 1].std()),
            },
            "carbon_change": {
                "min": float(observations[:, 2].min()),
                "max": float(observations[:, 2].max()),
                "mean": float(observations[:, 2].mean()),
                "std": float(observations[:, 2].std()),
            },
        }

        # Action statistics
        stats["actions"] = {
            "model_type_distribution": {
                str(i): int(np.sum(actions[:, 0] == i))
                for i in range(int(actions[:, 0].max()) + 1)
            },
            "charge_distribution": {
                "no_charge": int(np.sum(actions[:, 1] == 0)),
                "charge": int(np.sum(actions[:, 1] == 1)),
            },
        }

        # Metadata statistics
        if metadata:
            rewards = [m["total_reward"] for m in metadata]
            stats["simulation"] = {
                "total_combinations": len(set(m["combination_id"] for m in metadata)),
                "reward_stats": {
                    "min": min(rewards),
                    "max": max(rewards),
                    "mean": sum(rewards) / len(rewards),
                },
            }

        return stats

    def display_samples(
        self,
        split_name: str,
        observations: np.ndarray,
        actions: np.ndarray,
        max_samples: int = 3,
    ):
        """Display sample data for validation"""
        num_samples = min(len(observations), max_samples)
        if num_samples == 0:
            print(f"  {split_name}: No samples")
            return

        print(f"  {split_name} samples (showing first {num_samples}):")
        for i in range(num_samples):
            obs = observations[i]
            action = actions[i]
            print(
                f"    Sample {i + 1}: obs=[{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}], action=[{action[0]}, {action[1]}]"
            )

            # Validate ranges
            warnings = []
            if not (0 <= obs[0] <= 1):
                warnings.append(f"battery {obs[0]:.3f} out of [0,1]")
            if not (0 <= obs[1] <= 1):
                warnings.append(f"carbon {obs[1]:.3f} out of [0,1]")
            if not (-1 <= obs[2] <= 1):
                warnings.append(f"carbon_change {obs[2]:.3f} out of [-1,1]")
            if not (0 <= action[0] <= 6):  # 7 model types (0-6)
                warnings.append(f"model_type {action[0]} out of [0,6]")
            if not (0 <= action[1] <= 1):
                warnings.append(f"charge {action[1]} out of [0,1]")

            if warnings:
                print(f"      ⚠️  {', '.join(warnings)}")
            else:
                print("      ✓ Valid ranges")

    def cleanup_temp_files(self):
        """Remove temporary chunk files"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to clean up temp files: {e}")

    def validate_final_dataset(self) -> bool:
        """Validate the final dataset"""
        print("Validating final dataset...")

        try:
            # Check combined dataset
            combined_path = os.path.join(self.output_dir, "combined_dataset.npz")
            if not os.path.exists(combined_path):
                print("✗ Combined dataset file missing")
                return False

            data = np.load(combined_path)
            obs = data["observations"]
            actions = data["actions"]

            # Check shapes
            if obs.shape[0] != actions.shape[0]:
                print("✗ Observation and action count mismatch")
                return False

            if obs.shape[1] != 3:
                print(f"✗ Expected 3 observation features, got {obs.shape[1]}")
                return False

            if actions.shape[1] != 2:
                print(f"✗ Expected 2 action features, got {actions.shape[1]}")
                return False

            # Check ranges
            if np.any(obs[:, 0] < 0) or np.any(obs[:, 0] > 1):
                print("✗ Battery level out of range [0,1]")
                return False

            if np.any(obs[:, 1] < 0) or np.any(obs[:, 1] > 1):
                print("✗ Carbon intensity out of range [0,1]")
                return False

            if np.any(obs[:, 2] < -1) or np.any(obs[:, 2] > 1):
                print("✗ Carbon change out of range [-1,1]")
                return False

            # Check split files
            for split in ["train", "val", "test"]:
                split_path = os.path.join(self.output_dir, f"{split}.npy")
                if not os.path.exists(split_path):
                    print(f"✗ Split file missing: {split}.npy")
                    return False

            print("✓ Final dataset validation passed")
            return True

        except Exception as e:
            print(f"✗ Dataset validation failed: {e}")
            return False

    def recombine_all(
        self, train_ratio: float = 0.7, val_ratio: float = 0.1, cleanup: bool = True
    ) -> bool:
        """
        Complete recombination process

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            cleanup: Whether to clean up temporary files

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load all chunks
            observations, actions, metadata = self.load_all_chunks()

            # Split data
            splits = self.split_data(observations, actions, train_ratio, val_ratio)

            # Save final dataset
            self.save_final_dataset(splits, metadata)

            # Validate
            if not self.validate_final_dataset():
                return False

            # Cleanup
            if cleanup:
                self.cleanup_temp_files()

            print("✓ Dataset recombination completed successfully")
            return True

        except Exception as e:
            print(f"✗ Dataset recombination failed: {e}")
            return False


def main():
    """Main entry point"""
    recombiner = DataRecombiner()

    try:
        print("Starting dataset recombination...")
        success = recombiner.recombine_all()

        if success:
            print("✓ Dataset recombination completed")
            return 0
        else:
            print("✗ Dataset recombination failed")
            return 1

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
