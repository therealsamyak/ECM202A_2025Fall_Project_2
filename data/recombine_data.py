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
from typing import Dict, List, Any


class DataRecombiner:
    """Combines chunk files into final training dataset"""

    def __init__(
        self, temp_dir: str = "data/temp", output_dir: str = "data/training_data"
    ):
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _create_controller_signature(self, result: Dict[str, Any]) -> str:
        """
        Create unique controller signature from parameter combination

        Args:
            result: Simulation result dictionary

        Returns:
            Controller signature string with parameter values
        """
        combination = result["combination"]
        user_params = combination["user_parameters"]
        reward_weights = combination["reward_weights"]
        battery_config = combination["battery_config"]

        signature = f"acc{user_params['accuracy_threshold']}_lat{user_params['latency_threshold_seconds']}_succ{reward_weights['success_weight']}_small{reward_weights['small_miss_weight']}_large{reward_weights['large_miss_weight']}_carb{reward_weights['carbon_weight']}_cap{battery_config['battery_capacity_mwh']}_rate{battery_config['charge_rate_mwh_per_second']}"

        return signature

    def load_all_chunks(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all chunk files and group data by controller

        Returns:
            Dictionary mapping controller signatures to their data
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

        controller_data = {}

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

                # Process each result in the chunk
                for result in detailed_results:
                    if result["success"]:
                        training_data = result["training_data"]

                        # Create controller signature
                        controller_sig = self._create_controller_signature(result)

                        # Initialize controller data if not exists
                        if controller_sig not in controller_data:
                            controller_data[controller_sig] = {
                                "observations": [],
                                "actions": [],
                                "metadata": [],
                                "combination": result["combination"],
                            }

                        # Add data to controller group
                        controller_data[controller_sig]["observations"].append(
                            training_data["observations"]
                        )
                        controller_data[controller_sig]["actions"].append(
                            training_data["actions"]
                        )

                        # Add metadata for each timestep
                        for _ in range(len(training_data["observations"])):
                            controller_data[controller_sig]["metadata"].append(
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
                                    "date": result["combination"]["date"],
                                    "location": result["combination"]["location"],
                                }
                            )

                total_timesteps = metadata["total_timesteps"]
                print(f"  Loaded {chunk_file}: {total_timesteps} timesteps")

            except Exception as e:
                print(f"Error loading {chunk_file}: {e}")
                continue

        if not controller_data:
            raise ValueError("No valid chunk data found")

        # Combine arrays for each controller
        for controller_sig, data in controller_data.items():
            data["observations"] = np.vstack(data["observations"])
            data["actions"] = np.vstack(data["actions"])

        print(f"Found {len(controller_data)} unique controllers:")
        for controller_sig, data in controller_data.items():
            print(f"  {controller_sig}: {data['observations'].shape[0]} timesteps")

        return controller_data

    def split_data(
        self,
        controller_data: Dict[str, Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Split controller data into train/val/test sets

        Args:
            controller_data: Dictionary mapping controller signatures to their data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Dictionary mapping controller signatures to their splits
        """
        controller_splits = {}

        for controller_sig, data in controller_data.items():
            observations = data["observations"]
            actions = data["actions"]
            metadata = data["metadata"]

            total_samples = observations.shape[0]

            # Calculate split indices
            train_end = int(total_samples * train_ratio)
            val_end = int(total_samples * (train_ratio + val_ratio))

            # Split data
            train_obs = observations[:train_end]
            train_actions = actions[:train_end]
            train_metadata = metadata[:train_end]

            val_obs = observations[train_end:val_end]
            val_actions = actions[train_end:val_end]
            val_metadata = metadata[train_end:val_end]

            test_obs = observations[val_end:]
            test_actions = actions[val_end:]
            test_metadata = metadata[val_end:]

            splits = {
                "train_obs": train_obs,
                "train_actions": train_actions,
                "train_metadata": train_metadata,
                "val_obs": val_obs,
                "val_actions": val_actions,
                "val_metadata": val_metadata,
                "test_obs": test_obs,
                "test_actions": test_actions,
                "test_metadata": test_metadata,
            }

            controller_splits[controller_sig] = splits

            print(f"Data split for {controller_sig}:")
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
            print(f"Sample data validation for {controller_sig}:")
            self.display_samples("Train", train_obs, train_actions, max_samples=2)
            self.display_samples("Val", val_obs, val_actions, max_samples=1)
            self.display_samples("Test", test_obs, test_actions, max_samples=1)

        return controller_splits

    def save_final_dataset(
        self,
        controller_splits: Dict[str, Dict[str, Any]],
        controller_data: Dict[str, Dict[str, Any]],
    ):
        """
        Save controller-specific datasets to output directory

        Args:
            controller_splits: Dictionary mapping controller signatures to their splits
            controller_data: Dictionary mapping controller signatures to their full data
        """
        print("Saving controller-specific datasets...")

        all_metadata = []
        all_controllers_info = {}

        for controller_sig, splits in controller_splits.items():
            print(f"\nSaving datasets for {controller_sig}...")

            # Save individual split files for this controller
            controller_prefix = f"controller_{controller_sig}"

            train_path = os.path.join(self.output_dir, f"{controller_prefix}_train.npy")
            val_path = os.path.join(self.output_dir, f"{controller_prefix}_val.npy")
            test_path = os.path.join(self.output_dir, f"{controller_prefix}_test.npy")

            train_actions_path = os.path.join(
                self.output_dir, f"{controller_prefix}_train_actions.npy"
            )
            val_actions_path = os.path.join(
                self.output_dir, f"{controller_prefix}_val_actions.npy"
            )
            test_actions_path = os.path.join(
                self.output_dir, f"{controller_prefix}_test_actions.npy"
            )

            np.save(train_path, splits["train_obs"])
            np.save(train_actions_path, splits["train_actions"])
            np.save(val_path, splits["val_obs"])
            np.save(val_actions_path, splits["val_actions"])
            np.save(test_path, splits["test_obs"])
            np.save(test_actions_path, splits["test_actions"])

            print(f"  Saved {controller_prefix}_train: {train_path}")
            print(f"  Saved {controller_prefix}_train_actions: {train_actions_path}")
            print(f"  Saved {controller_prefix}_val: {val_path}")
            print(f"  Saved {controller_prefix}_val_actions: {val_actions_path}")
            print(f"  Saved {controller_prefix}_test: {test_path}")
            print(f"  Saved {controller_prefix}_test_actions: {test_actions_path}")

            # Save combined dataset for this controller
            all_obs = np.vstack(
                [splits["train_obs"], splits["val_obs"], splits["test_obs"]]
            )
            all_actions = np.vstack(
                [splits["train_actions"], splits["val_actions"], splits["test_actions"]]
            )
            combined_path = os.path.join(
                self.output_dir, f"{controller_prefix}_combined.npz"
            )

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
            print(f"  Saved {controller_prefix}_combined: {combined_path}")

            # Collect all metadata for statistics
            all_metadata.extend(controller_data[controller_sig]["metadata"])

            # Store controller information
            all_controllers_info[controller_sig] = {
                "combination": controller_data[controller_sig]["combination"],
                "total_samples": all_obs.shape[0],
                "data_split": {
                    "train": splits["train_obs"].shape[0],
                    "val": splits["val_obs"].shape[0],
                    "test": splits["test_obs"].shape[0],
                },
            }

        # Generate global statistics across all controllers
        all_obs_combined = []
        all_actions_combined = []
        for controller_sig, data in controller_data.items():
            all_obs_combined.append(data["observations"])
            all_actions_combined.append(data["actions"])

        all_obs_combined = np.vstack(all_obs_combined)
        all_actions_combined = np.vstack(all_actions_combined)

        stats = self.generate_statistics(
            all_obs_combined, all_actions_combined, all_metadata
        )

        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        final_metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "total_controllers": len(controller_splits),
            "total_samples": all_obs_combined.shape[0],
            "observation_features": 3,
            "action_features": 2,
            "controllers": all_controllers_info,
            "statistics": stats,
            "sample_metadata": all_metadata[:100],  # Save first 100 samples as examples
        }

        with open(metadata_path, "w") as f:
            json.dump(final_metadata, f, indent=2)
        print(f"\nSaved metadata: {metadata_path}")
        print(f"Total controllers: {len(controller_splits)}")
        print(f"Total samples across all controllers: {all_obs_combined.shape[0]}")

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
        """Validate the controller-specific datasets"""
        print("Validating controller-specific datasets...")

        try:
            # Check metadata file exists
            metadata_path = os.path.join(self.output_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                print("✗ Metadata file missing")
                return False

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Validate each controller's dataset
            for controller_sig, controller_info in metadata["controllers"].items():
                controller_prefix = f"controller_{controller_sig}"

                # Check combined dataset
                combined_path = os.path.join(
                    self.output_dir, f"{controller_prefix}_combined.npz"
                )
                if not os.path.exists(combined_path):
                    print(f"✗ Combined dataset file missing for {controller_sig}")
                    return False

                data = np.load(combined_path)
                obs = data["observations"]
                actions = data["actions"]

                # Check shapes
                if obs.shape[0] != actions.shape[0]:
                    print(
                        f"✗ Observation and action count mismatch for {controller_sig}"
                    )
                    return False

                if obs.shape[1] != 3:
                    print(
                        f"✗ Expected 3 observation features, got {obs.shape[1]} for {controller_sig}"
                    )
                    return False

                if actions.shape[1] != 2:
                    print(
                        f"✗ Expected 2 action features, got {actions.shape[1]} for {controller_sig}"
                    )
                    return False

                # Check ranges
                if np.any(obs[:, 0] < 0) or np.any(obs[:, 0] > 1):
                    print(f"✗ Battery level out of range [0,1] for {controller_sig}")
                    return False

                if np.any(obs[:, 1] < 0) or np.any(obs[:, 1] > 1):
                    print(f"✗ Carbon intensity out of range [0,1] for {controller_sig}")
                    return False

                if np.any(obs[:, 2] < -1) or np.any(obs[:, 2] > 1):
                    print(f"✗ Carbon change out of range [-1,1] for {controller_sig}")
                    return False

                # Check split files
                for split in ["train", "val", "test"]:
                    split_path = os.path.join(
                        self.output_dir, f"{controller_prefix}_{split}.npy"
                    )
                    if not os.path.exists(split_path):
                        print(f"✗ Split file missing: {split}.npy for {controller_sig}")
                        return False

                    actions_path = os.path.join(
                        self.output_dir, f"{controller_prefix}_{split}_actions.npy"
                    )
                    if not os.path.exists(actions_path):
                        print(
                            f"✗ Actions split file missing: {split}_actions.npy for {controller_sig}"
                        )
                        return False

                print(f"✓ {controller_sig} validation passed")

            print("✓ All controller datasets validation passed")
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
            # Load all chunks and group by controller
            controller_data = self.load_all_chunks()

            # Split data by controller
            controller_splits = self.split_data(controller_data, train_ratio, val_ratio)

            # Save controller-specific datasets
            self.save_final_dataset(controller_splits, controller_data)

            # Validate
            if not self.validate_final_dataset():
                return False

            # Cleanup
            if cleanup:
                # self.cleanup_temp_files()
                print("Cleanup)")

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
