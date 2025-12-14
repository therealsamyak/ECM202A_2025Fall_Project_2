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

    # Print Oracle results
    print_path_summary("Oracle", oracle_path, oracle_reward)

    # Print Naive results
    print_path_summary("Naive", naive_path, naive_reward)

    # Comparison
    print("\n=== Comparison ===")
    print(f"Oracle reward: {oracle_reward:.7f}")
    print(f"Naive reward: {naive_reward:.7f}")
    print(
        f"Improvement: {oracle_reward - naive_reward:.7f} ({((oracle_reward / naive_reward - 1) * 100):.1f}%)"
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
        "config": config,
    }

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    with open("simulation/data/simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to simulation/data/simulation_results.json")


if __name__ == "__main__":
    main()
