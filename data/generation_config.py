#!/usr/bin/env python3
"""
Configuration loader and validator for training data generation.
Handles parameter combinations and chunking logic.
"""

import json
import os
import itertools
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GenerationConfig:
    """Configuration for data generation"""

    system_constants: Dict[str, Any]
    user_parameters: List[Dict[str, float]]
    reward_weights: List[Dict[str, float]]
    battery_configs: List[Dict[str, float]]
    seasonal_dates: List[str]
    locations: List[str]
    output: Dict[str, Any]


class ConfigLoader:
    """Load and validate generation configuration"""

    def __init__(self, config_path: str = "data/data.config.json"):
        self.config_path = config_path
        self.config = None

    def load_config(self) -> GenerationConfig:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_data = json.load(f)

        self.config = GenerationConfig(
            system_constants=config_data["system_constants"],
            user_parameters=config_data["user_parameters"],
            reward_weights=config_data["reward_weights"],
            battery_configs=config_data["battery_configs"],
            seasonal_dates=config_data["seasonal_dates"],
            locations=config_data["locations"],
            output=config_data["output"],
        )

        self._validate_config()
        return self.config

    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.config:
            raise ValueError("Configuration not loaded")

        # Validate user parameters
        for i, user_param in enumerate(self.config.user_parameters):
            if not (0 <= user_param["accuracy_threshold"] <= 1):
                raise ValueError(
                    f"User parameter {i}: accuracy_threshold must be [0,1]"
                )
            if user_param["latency_threshold_seconds"] <= 0:
                raise ValueError(
                    f"User parameter {i}: latency_threshold must be positive"
                )

        # Validate reward weights
        for i, weights in enumerate(self.config.reward_weights):
            for key, value in weights.items():
                if value < 0:
                    raise ValueError(f"Reward weights {i}: {key} must be non-negative")

        # Validate battery configs
        for i, battery in enumerate(self.config.battery_configs):
            if battery["battery_capacity_mwh"] <= 0:
                raise ValueError(f"Battery config {i}: capacity must be positive")
            if battery["charge_rate_mwh_per_second"] < 0:
                raise ValueError(
                    f"Battery config {i}: charge_rate must be non-negative"
                )

        # Validate dates
        for date in self.config.seasonal_dates:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {date}")

        # Validate output config
        split = self.config.output["data_split"]
        if not abs(sum(split.values()) - 1.0) < 1e-6:
            raise ValueError("Data split ratios must sum to 1.0")

        if self.config.output["max_workers"] <= 0:
            raise ValueError("max_workers must be positive")

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations using Cartesian product"""
        if not self.config:
            raise ValueError("Configuration not loaded")

        combinations = []

        # Generate Cartesian product of all parameters
        for (user_idx, user_param), (reward_idx, reward_weights), (
            battery_idx,
            battery_config,
        ), (date_idx, date), (location_idx, location) in itertools.product(
            enumerate(self.config.user_parameters),
            enumerate(self.config.reward_weights),
            enumerate(self.config.battery_configs),
            enumerate(self.config.seasonal_dates),
            enumerate(self.config.locations),
        ):
            combination = {
                "combination_id": len(combinations),
                "user_param_idx": user_idx,
                "reward_weights_idx": reward_idx,
                "battery_config_idx": battery_idx,
                "date_idx": date_idx,
                "location_idx": location_idx,
                "system_constants": self.config.system_constants.copy(),
                "user_parameters": user_param.copy(),
                "reward_weights": reward_weights.copy(),
                "battery_config": battery_config.copy(),
                "date": date,
                "location": location,
                "max_workers": self.config.output["max_workers"],
            }

            combinations.append(combination)

        return combinations

    def calculate_chunks(
        self, combinations: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Create individual chunks - one combination per chunk"""
        if not self.config:
            raise ValueError("Configuration not loaded")

        # Each combination becomes its own chunk
        chunks = [[combination] for combination in combinations]
        return chunks

    def get_base_simulation_config(self) -> Dict[str, Any]:
        """Get base simulation config template"""
        if not self.config:
            raise ValueError("Configuration not loaded")
        return {
            "system": self.config.system_constants.copy(),
            "simulation": {"start_time": "00:00:00"},
            "data_paths": {"model_profiles": "../model-profiler/power_profiles.json"},
        }

    def merge_config_for_simulation(
        self, combination: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge combination parameters with base simulation config"""
        base_config = self.get_base_simulation_config()

        # Merge system parameters
        base_config["system"].update(combination["battery_config"])

        # Merge user requirements
        base_config["user_requirements"] = combination["user_parameters"]

        # Merge reward weights
        base_config["reward_weights"] = combination["reward_weights"]

        # Merge simulation parameters
        base_config["simulation"]["start_date"] = combination["date"]

        # Set energy data path
        base_config["data_paths"]["energy_data"] = (
            f"../energy-data/{combination['location']}"
        )

        return base_config


def main():
    """Test configuration loading and validation"""
    loader = ConfigLoader()

    try:
        config = loader.load_config()
        print("✓ Configuration loaded and validated successfully")

        combinations = loader.generate_parameter_combinations()
        print(f"✓ Generated {len(combinations)} parameter combinations")

        chunks = loader.calculate_chunks(combinations)
        print(
            f"✓ Created {len(chunks)} chunks (chunk size: {config.output['chunk_size']})"
        )

        # Test first combination merge
        if combinations:
            loader.merge_config_for_simulation(combinations[0])
            print("✓ Configuration merging works correctly")

        print("\nConfiguration summary:")
        print(f"- User parameters: {len(config.user_parameters)}")
        print(f"- Reward weight sets: {len(config.reward_weights)}")
        print(f"- Battery configs: {len(config.battery_configs)}")
        print(f"- Seasonal dates: {len(config.seasonal_dates)}")
        print(f"- Locations: {len(config.locations)}")
        print(f"- Total combinations: {len(combinations)}")
        print(f"- Chunks: {len(chunks)}")
        print(f"- Max workers: {config.output['max_workers']}")

    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
