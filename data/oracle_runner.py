#!/usr/bin/env python3
"""
Single simulation executor for Oracle controller.
Handles individual simulation runs and extracts training data.
"""

import os
import sys
import json
import numpy as np
import traceback
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ruff: noqa: E402
from simulation.controllers.oracle import OracleController
from simulation.utils.core import ModelType


class OracleRunner:
    """Execute single Oracle simulation and extract training data"""

    def __init__(self, temp_dir: str = "data/temp"):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def run_simulation(self, combination: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run single Oracle simulation with given parameter combination

        Args:
            combination: Parameter combination dictionary

        Returns:
            Dictionary with training data and metadata
        """
        start_time = datetime.now()
        print(
            f"    [OracleRunner] Starting simulation for combination {combination['combination_id']}"
        )

        try:
            # Create simulation config
            config_start = datetime.now()
            config = self._create_simulation_config(combination)
            config_time = (datetime.now() - config_start).total_seconds()
            print(f"    [OracleRunner] Config creation: {config_time:.3f}s")

            # Initialize Oracle controller
            init_start = datetime.now()
            oracle = OracleController(config)
            init_time = (datetime.now() - init_start).total_seconds()
            print(f"    [OracleRunner] Oracle initialization: {init_time:.3f}s")

            # Solve MDP
            solve_start = datetime.now()
            path = oracle.solve()
            solve_time = (datetime.now() - solve_start).total_seconds()
            print(
                f"    [OracleRunner] MDP solving: {solve_time:.3f}s, path length: {len(path)}"
            )

            # Extract training data
            extract_start = datetime.now()
            training_data = oracle.export_training_data(path)
            extract_time = (datetime.now() - extract_start).total_seconds()
            print(f"    [OracleRunner] Training data extraction: {extract_time:.3f}s")

            # Calculate additional metadata
            calc_start = datetime.now()
            total_reward = oracle._calculate_path_reward(path)
            optimal_value = oracle.get_optimal_value()
            calc_time = (datetime.now() - calc_start).total_seconds()
            print(f"    [OracleRunner] Metadata calculation: {calc_time:.3f}s")

            # Prepare result
            total_time = (datetime.now() - start_time).total_seconds()
            result = {
                "success": True,
                "combination_id": combination["combination_id"],
                "combination": combination,
                "training_data": training_data,
                "metadata": {
                    "total_timesteps": len(path),
                    "total_reward": total_reward,
                    "optimal_value": optimal_value,
                    "execution_time_seconds": total_time,
                    "timestamp": start_time.isoformat(),
                    "timing_breakdown": {
                        "config_creation": config_time,
                        "oracle_init": init_time,
                        "mdp_solving": solve_time,
                        "data_extraction": extract_time,
                        "metadata_calc": calc_time,
                    },
                },
            }

            print(f"    [OracleRunner] Total simulation time: {total_time:.3f}s")

            return result

        except Exception as e:
            # Detailed error reporting
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"    [OracleRunner] Simulation failed after {total_time:.3f}s: {e}")
            error_result = {
                "success": False,
                "combination_id": combination["combination_id"],
                "combination": combination,
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "execution_time_seconds": total_time,
                    "timestamp": start_time.isoformat(),
                },
            }

            return error_result

    def _create_simulation_config(self, combination: Dict[str, Any]) -> Dict[str, Any]:
        """Create simulation config from parameter combination"""
        # Base configuration using system constants from data config
        config = {
            "system": combination["system_constants"].copy(),
            "simulation": {"start_date": combination["date"], "start_time": "00:00:00"},
            "data_paths": {
                "model_profiles": "../model-profiler/power_profiles.json",
                "energy_data": f"../energy-data/{combination['location']}",
            },
        }

        # Merge combination parameters
        config["system"].update(combination["battery_config"])
        config["system"]["max_workers"] = combination.get(
            "max_workers", 90
        )  # Add max_workers to system
        config["user_requirements"] = combination["user_parameters"]
        config["reward_weights"] = combination["reward_weights"]

        return config

    def save_chunk(self, chunk_results: List[Dict[str, Any]], chunk_id: int) -> str:
        """
        Save chunk results to file

        Args:
            chunk_results: List of simulation results
            chunk_id: Chunk identifier

        Returns:
            Path to saved file
        """
        save_start = datetime.now()
        print(f"    [OracleRunner] Starting chunk save for chunk {chunk_id}")

        # Prepare data for saving
        prep_start = datetime.now()
        observations = []
        actions = []
        metadata = []

        successful_simulations = 0
        failed_simulations = 0

        for result in chunk_results:
            if result["success"]:
                # Extract training data
                training_data = result["training_data"]
                observations.extend(training_data["observations"])
                actions.extend(training_data["actions"])

                # Add metadata for each timestep
                for _ in range(len(training_data["observations"])):
                    metadata.append(
                        {
                            "combination_id": result["combination_id"],
                            "total_reward": result["metadata"]["total_reward"],
                            "optimal_value": result["metadata"]["optimal_value"],
                            "execution_time": result["metadata"][
                                "execution_time_seconds"
                            ],
                        }
                    )

                successful_simulations += 1
            else:
                failed_simulations += 1

        prep_time = (datetime.now() - prep_start).total_seconds()
        print(f"    [OracleRunner] Data preparation: {prep_time:.3f}s")

        # Convert to numpy arrays
        convert_start = datetime.now()
        observations_array = np.array(observations, dtype=np.float32)
        actions_array = np.array(actions, dtype=np.int32)
        convert_time = (datetime.now() - convert_start).total_seconds()
        print(f"    [OracleRunner] Array conversion: {convert_time:.3f}s")

        # Get combination ID for filename (since chunk_size=1, first result is the only one)
        combination_id = chunk_results[0]["combination_id"]

        # Save to compressed numpy file with combination ID
        save_file_start = datetime.now()
        filename = f"chunk_{combination_id:03d}.npz"
        filepath = os.path.join(self.temp_dir, filename)

        np.savez_compressed(
            filepath,
            observations=observations_array,
            actions=actions_array,
            metadata=json.dumps(
                {
                    "chunk_id": chunk_id,
                    "successful_simulations": successful_simulations,
                    "failed_simulations": failed_simulations,
                    "total_timesteps": len(observations),
                    "combination_ids": [r["combination_id"] for r in chunk_results],
                    "generation_timestamp": datetime.now().isoformat(),
                }
            ),
            detailed_results=json.dumps(chunk_results),
        )
        save_file_time = (datetime.now() - save_file_start).total_seconds()
        total_save_time = (datetime.now() - save_start).total_seconds()

        print(f"    [OracleRunner] File save: {save_file_time:.3f}s")
        print(
            f"    [OracleRunner] Total chunk save: {total_save_time:.3f}s, saved {len(observations)} timesteps"
        )

        return filepath

    def validate_chunk_data(self, filepath: str) -> bool:
        """
        Validate saved chunk data

        Args:
            filepath: Path to chunk file

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Load and validate data
            data = np.load(filepath)

            observations = data["observations"]
            actions = data["actions"]
            metadata = json.loads(str(data["metadata"]))

            # Check shapes
            if observations.shape[0] != actions.shape[0]:
                print(
                    f"✗ Shape mismatch: {observations.shape[0]} obs vs {actions.shape[0]} actions"
                )
                return False

            # Check observation ranges
            if observations.shape[1] != 3:
                print(f"✗ Expected 3 observation features, got {observations.shape[1]}")
                return False

            # Check battery level range [0,1]
            battery_levels = observations[:, 0]
            if np.any(battery_levels < 0) or np.any(battery_levels > 1):
                print(
                    f"✗ Battery level out of range: [{battery_levels.min():.3f}, {battery_levels.max():.3f}]"
                )
                return False

            # Check carbon intensity range [0,1]
            carbon_intensity = observations[:, 1]
            if np.any(carbon_intensity < 0) or np.any(carbon_intensity > 1):
                print(
                    f"✗ Carbon intensity out of range: [{carbon_intensity.min():.3f}, {carbon_intensity.max():.3f}]"
                )
                return False

            # Check carbon change range [-1,1]
            carbon_change = observations[:, 2]
            if np.any(carbon_change < -1) or np.any(carbon_change > 1):
                print(
                    f"✗ Carbon change out of range: [{carbon_change.min():.3f}, {carbon_change.max():.3f}]"
                )
                return False

            # Check action ranges
            model_types = actions[:, 0]
            if np.any(model_types < 0) or np.any(model_types >= len(ModelType)):
                print(
                    f"✗ Model type out of range: [{model_types.min()}, {model_types.max()}]"
                )
                return False

            charge_decisions = actions[:, 1]
            if np.any(charge_decisions < 0) or np.any(charge_decisions > 1):
                print(
                    f"✗ Charge decision out of range: [{charge_decisions.min()}, {charge_decisions.max()}]"
                )
                return False

            print(f"✓ Chunk validation passed: {metadata['total_timesteps']} timesteps")
            return True

        except Exception as e:
            print(f"✗ Chunk validation failed: {e}")
            return False


def main():
    """Test oracle runner with single simulation"""
    runner = OracleRunner()

    # Test with minimal configuration
    test_combination = {
        "combination_id": 0,
        "user_param_idx": 0,
        "reward_weights_idx": 0,
        "battery_config_idx": 0,
        "date_idx": 0,
        "location_idx": 0,
        "user_parameters": {
            "accuracy_threshold": 0.8,
            "latency_threshold_seconds": 0.006,
        },
        "reward_weights": {
            "success_weight": 10,
            "small_miss_weight": 5,
            "large_miss_weight": 8,
            "carbon_weight": 7,
        },
        "battery_config": {
            "battery_capacity_mwh": 300,
            "charge_rate_mwh_per_second": 0.00001,
        },
        "date": "2024-01-15",
        "location": "US-CAL-LDWP_2024_5_minute.csv",
    }

    try:
        print("Running test simulation...")
        result = runner.run_simulation(test_combination)

        if result["success"]:
            print("✓ Test simulation successful")
            print(f"  - Timesteps: {result['metadata']['total_timesteps']}")
            print(f"  - Total reward: {result['metadata']['total_reward']:.3f}")
            print(
                f"  - Execution time: {result['metadata']['execution_time_seconds']:.2f}s"
            )

            # Test chunk saving
            chunk_path = runner.save_chunk([result], 0)
            print(f"✓ Chunk saved to: {chunk_path}")

            # Test validation
            if runner.validate_chunk_data(chunk_path):
                print("✓ Chunk validation passed")
            else:
                print("✗ Chunk validation failed")

        else:
            print("✗ Test simulation failed")
            print(f"  - Error: {result['error']['message']}")
            print(f"  - Type: {result['error']['type']}")

        return 0 if result["success"] else 1

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
