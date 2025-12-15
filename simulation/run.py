#!/usr/bin/env python3
"""
Simulation Runner for Oracle and Naive Controllers
"""

import json
import os
import sys
from typing import Dict, List, Tuple

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ruff: noqa: E402
from simulation.controllers.oracle import OracleController
from simulation.controllers.naive import NaiveController
from simulation.controllers.ml_controller import MLController
from simulation.utils.core import State, Action, validate_config


def load_config() -> Dict:
    """Load configuration from simulation.config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "simulation.config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def print_path_summary(
    controller_name: str,
    path: List[Tuple[State, Action]],
    total_reward: float,
    path_num: int | None = None,
):
    """Print summary of controller path"""
    prefix = f"Path {path_num} - " if path_num is not None else ""
    print(f"\n=== {prefix}{controller_name} Controller Results ===")
    print(f"Total timesteps: {len(path)}")
    print(f"Total reward: {total_reward:.7f}")

    # Count actions
    model_counts = {}
    charge_count = 0

    for state, action in path:
        model_name = action.model.value
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
        if action.charge:
            charge_count += 1

    print(f"Charging actions: {charge_count}/{len(path)}")
    print("Model usage:")
    for model, count in model_counts.items():
        print(f"  {model}: {count}")

    # Print first few actions
    print("\nFirst 5 actions:")
    for i, (state, action) in enumerate(path[:5]):
        charge_str = " + charge" if action.charge else ""
        print(
            f"  t={i}: {action.model.value}{charge_str} (battery: {state.battery_level:.7f} mWh)"
        )


def main():
    """Main simulation runner"""
    print("Loading configuration...")
    config = load_config()

    # Validate configuration
    validate_config(config)

    print("Initializing controllers...")

    # Run Oracle Controller
    print("\nSolving Oracle Controller (this may take a moment)...")
    oracle = OracleController(config)
    oracle_path = oracle.solve()
    oracle_reward = oracle._calculate_path_reward(oracle_path)

    # Oracle returns single optimal path

    # Run Naive Controller
    print("Running Naive Controller...")
    naive = NaiveController(config)
    naive_path = naive.run()
    naive_reward = naive._calculate_path_reward(naive_path)

    # Run ML Controller
    print("Running ML Controller...")
    ml = MLController(config)
    ml_path = ml.run()
    ml_reward = ml._calculate_path_reward(ml_path)

    # Print Oracle results
    print_path_summary("Oracle", oracle_path, oracle_reward)

    # Print Naive results
    print_path_summary("Naive", naive_path, naive_reward)

    # Print ML results
    print_path_summary("ML", ml_path, ml_reward)

    # Comparison
    print("\n=== Comparison ===")
    print(f"Oracle reward: {oracle_reward:.7f}")
    print(f"Naive reward: {naive_reward:.7f}")
    print(f"ML reward: {ml_reward:.7f}")
    print(
        f"Oracle vs Naive: {oracle_reward - naive_reward:.7f} ({((oracle_reward / naive_reward - 1) * 100):.1f}%)"
    )
    print(
        f"ML vs Naive: {ml_reward - naive_reward:.7f} ({((ml_reward / naive_reward - 1) * 100):.1f}%)"
    )
    print(
        f"Oracle vs ML: {oracle_reward - ml_reward:.7f} ({((oracle_reward / ml_reward - 1) * 100):.1f}%)"
    )

    # Save detailed results
    results = {
        "oracle": {
            "total_reward": oracle_reward,
            "path_length": len(oracle_path),
            "actions": [
                {
                    "timestep": i,
                    "model": action.model.value,
                    "charge": action.charge,
                    "battery_level": round(state.battery_level, 7),
                }
                for i, (state, action) in enumerate(oracle_path)
            ],
        },
        "naive": {
            "total_reward": naive_reward,
            "path_length": len(naive_path),
            "actions": [
                {
                    "timestep": i,
                    "model": action.model.value,
                    "charge": action.charge,
                    "battery_level": round(state.battery_level, 7),
                }
                for i, (state, action) in enumerate(naive_path)
            ],
        },
        "ml": {
            "total_reward": ml_reward,
            "path_length": len(ml_path),
            "actions": [
                {
                    "timestep": i,
                    "model": action.model.value,
                    "charge": action.charge,
                    "battery_level": round(state.battery_level, 7),
                }
                for i, (state, action) in enumerate(ml_path)
            ],
        },
        "config": config,
    }

    # Only save if output_dir is specified
    output_dir = config.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename from key parameters
        battery = config["system"]["battery_capacity_mwh"]
        accuracy = config["user_requirements"]["accuracy_threshold"]
        latency = config["user_requirements"]["latency_threshold_seconds"]
        start_date = config["simulation"]["start_date"].replace("-", "")

        filename = (
            f"{output_dir}/sim_b{battery}_a{accuracy}_l{latency}_{start_date}.json"
        )

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to {filename}")
    else:
        print("\nNo output directory specified - results not saved")


if __name__ == "__main__":
    main()
