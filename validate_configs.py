#!/usr/bin/env python3
"""Validate configuration files against report.html requirements"""

import sys
import json
import os
from datetime import datetime
import csv


def validate_model_profiles(path: str) -> dict:
    """Validate model profile JSON structure and value ranges"""
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))

    # Handle relative paths from simulation directory for simulation.config.json
    if path.startswith(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "simulation"))
    ):
        path = os.path.join(os.path.dirname(__file__), path.replace("../", ""))

    if not os.path.exists(path):
        raise ValueError(f"Model profiles file not found: {path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model profiles: {e}")

    profiles = {}
    for model_name, model_data in data.items():
        # Validate required fields
        required_fields = [
            "accuracy",
            "avg_inference_time_seconds",
            "energy_per_inference_mwh",
        ]
        for field in required_fields:
            if field not in model_data:
                raise ValueError(f"Model {model_name} missing required field: {field}")

        # Validate accuracy range [0, 1]
        accuracy = model_data["accuracy"]
        if not isinstance(accuracy, (int, float)) or not (0 <= accuracy <= 1):
            raise ValueError(
                f"Model {model_name} accuracy {accuracy} must be in [0, 1]"
            )

        # Validate latency > 0
        latency = model_data["avg_inference_time_seconds"]
        if not isinstance(latency, (int, float)) or latency <= 0:
            raise ValueError(f"Model {model_name} latency {latency} must be > 0")

        # Validate energy > 0
        energy = model_data["energy_per_inference_mwh"]
        if not isinstance(energy, (int, float)) or energy <= 0:
            raise ValueError(f"Model {model_name} energy {energy} must be > 0")

        profiles[model_name] = {
            "accuracy": accuracy,
            "latency": latency,
            "energy": energy,
        }

    return profiles


def validate_carbon_data(path: str) -> None:
    """Validate carbon data CSV file exists and has required columns"""
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))

    if not os.path.exists(path):
        raise ValueError(f"Carbon data file not found: {path}")

    # Check filename contains "2024"
    filename = os.path.basename(path)
    if "2024" not in filename:
        raise ValueError(f"Carbon data file {filename} must be from year 2024")

    # Validate required columns
    try:
        with open(path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read only header for validation
            required_columns = [
                "Datetime (UTC)",
                "Carbon-free energy percentage (CFE%)",
            ]
            missing_columns = [col for col in required_columns if col not in header]
            if missing_columns:
                raise ValueError(
                    f"Carbon data file {filename} missing required columns: {missing_columns}"
                )
    except Exception as e:
        raise ValueError(f"Error reading carbon data file {filename}: {e}")


def validate_user_requirements(user_req: dict, model_profiles: dict) -> None:
    """Validate user requirements are feasible with available models"""
    # Calculate MAX_ACC and MIN_LAT from available models
    max_acc = max(profile["accuracy"] for profile in model_profiles.values())
    min_lat = min(profile["latency"] for profile in model_profiles.values())

    # Validate accuracy threshold
    if "accuracy_threshold" not in user_req:
        raise ValueError("User requirements missing accuracy_threshold")

    u_acc = user_req["accuracy_threshold"]
    if not isinstance(u_acc, (int, float)) or not (0 <= u_acc <= 1):
        raise ValueError(f"Accuracy threshold {u_acc} must be in [0, 1]")

    if u_acc >= max_acc:
        raise ValueError(f"Accuracy threshold {u_acc} must be < MAX_ACC ({max_acc})")

    # Validate latency threshold
    if "latency_threshold_seconds" not in user_req:
        raise ValueError("User requirements missing latency_threshold_seconds")

    u_lat = user_req["latency_threshold_seconds"]
    if not isinstance(u_lat, (int, float)) or u_lat <= 0:
        raise ValueError(f"Latency threshold {u_lat} must be > 0")

    if u_lat <= min_lat:
        raise ValueError(f"Latency threshold {u_lat} must be > MIN_LAT ({min_lat})")


def validate_system_parameters(system: dict) -> None:
    """Validate system parameters according to report specifications"""
    # Validate battery capacity (optional for system_constants in data.config.json)
    if "battery_capacity_mwh" in system:
        if system["battery_capacity_mwh"] <= 0:
            raise ValueError("Battery capacity must be positive")

    # Validate task interval
    if "task_interval_seconds" not in system:
        raise ValueError("System missing task_interval_seconds")
    if system["task_interval_seconds"] <= 0:
        raise ValueError("Task interval must be positive")

    # Validate horizon
    if "horizon_seconds" not in system:
        raise ValueError("System missing horizon_seconds")
    if system["horizon_seconds"] <= 0:
        raise ValueError("Horizon must be positive")

    # Validate charge rate (optional for system_constants in data.config.json)
    if "charge_rate_mwh_per_second" in system:
        if system["charge_rate_mwh_per_second"] < 0:
            raise ValueError("Charge rate must be non-negative")

    # Validate battery discretization step
    if "battery_discretization_step" in system:
        step = system["battery_discretization_step"]
        if not isinstance(step, (int, float)) or step <= 0:
            raise ValueError("Battery discretization step must be positive")

    # Validate nearest neighbor K
    if "nearest_neighbor_k" in system:
        k = system["nearest_neighbor_k"]
        if not isinstance(k, int) or k <= 0 or k % 2 == 0:
            raise ValueError("nearest_neighbor_k must be an odd positive integer")

    # Check if horizon is multiple of task interval
    if system["horizon_seconds"] % system["task_interval_seconds"] != 0:
        raise ValueError("Horizon must be multiple of task interval")


def validate_reward_weights(weights: dict) -> None:
    """Validate reward weights are all positive"""
    required_weights = [
        "success_weight",
        "small_miss_weight",
        "large_miss_weight",
        "carbon_weight",
    ]

    for weight_name in required_weights:
        if weight_name not in weights:
            raise ValueError(f"Reward weights missing {weight_name}")

        weight_value = weights[weight_name]
        if not isinstance(weight_value, (int, float)) or weight_value < 0:
            raise ValueError(f"Reward weight {weight_name} must be non-negative")


def validate_date_format(date_str: str) -> None:
    """Validate date is in YYYY-MM-DD format"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Date {date_str} must be in YYYY-MM-DD format")


def validate_time_format(time_str: str) -> None:
    """Validate time is in HH:MM:SS format"""
    try:
        datetime.strptime(time_str, "%H:%M:%S")
    except ValueError:
        raise ValueError(f"Time {time_str} must be in HH:MM:SS format")


def validate_data_config():
    """Validate data.config.json with comprehensive checks"""
    with open("data/data.config.json", "r") as f:
        data_config = json.load(f)

    print("Validating data.config.json...")

    try:
        # Load model profiles once for all user requirement checks
        model_profiles = validate_model_profiles("model-profiler/power_profiles.json")

        # Validate system constants
        validate_system_parameters(data_config["system_constants"])

        # Validate each battery config
        for i, battery_config in enumerate(data_config["battery_configs"]):
            if "battery_capacity_mwh" not in battery_config:
                raise ValueError(f"battery_configs[{i}]: missing battery_capacity_mwh")
            if battery_config["battery_capacity_mwh"] <= 0:
                raise ValueError(
                    f"battery_configs[{i}]: battery_capacity_mwh must be positive"
                )
            if "charge_rate_mwh_per_second" not in battery_config:
                raise ValueError(
                    f"battery_configs[{i}]: missing charge_rate_mwh_per_second"
                )
            if battery_config["charge_rate_mwh_per_second"] < 0:
                raise ValueError(
                    f"battery_configs[{i}]: charge_rate_mwh_per_second must be non-negative"
                )

        # Validate each user parameter against model profiles
        for i, user_param in enumerate(data_config["user_parameters"]):
            try:
                validate_user_requirements(user_param, model_profiles)
            except ValueError as e:
                raise ValueError(f"user_parameters[{i}]: {e}")

        # Validate each reward weights set
        for i, weights in enumerate(data_config["reward_weights"]):
            try:
                validate_reward_weights(weights)
            except ValueError as e:
                raise ValueError(f"reward_weights[{i}]: {e}")

        # Validate seasonal dates
        for i, date in enumerate(data_config["seasonal_dates"]):
            try:
                validate_date_format(date)
            except ValueError as e:
                raise ValueError(f"seasonal_dates[{i}]: {e}")

        # Validate carbon data files exist and are 2024
        for i, location in enumerate(data_config["locations"]):
            try:
                validate_carbon_data(f"energy-data/{location}")
            except ValueError as e:
                raise ValueError(f"locations[{i}]: {e}")

        print("✓ data.config.json validation passed")
        return True

    except Exception as e:
        print(f"✗ data.config.json validation failed: {e}")
        return False


def validate_simulation_config():
    """Validate simulation.config.json with comprehensive checks"""
    with open("simulation/simulation.config.json", "r") as f:
        sim_config = json.load(f)

    print("Validating simulation.config.json...")

    try:
        # Validate system parameters
        validate_system_parameters(sim_config["system"])

        # Validate user requirements against model profiles
        model_profiles_path = sim_config["data_paths"]["model_profiles"]
        # Handle relative path from simulation directory
        if model_profiles_path.startswith("../"):
            model_profiles_path = model_profiles_path.replace("../", "")
        model_profiles = validate_model_profiles(model_profiles_path)
        validate_user_requirements(sim_config["user_requirements"], model_profiles)

        # Validate reward weights
        validate_reward_weights(sim_config["reward_weights"])

        # Validate date/time formats
        validate_date_format(sim_config["simulation"]["start_date"])
        validate_time_format(sim_config["simulation"]["start_time"])

        # Validate carbon data file
        energy_data_path = sim_config["data_paths"]["energy_data"]
        # Handle relative path from simulation directory
        if energy_data_path.startswith("../"):
            energy_data_path = energy_data_path.replace("../", "")
        validate_carbon_data(energy_data_path)

        print("✓ simulation.config.json validation passed")
        return True

    except Exception as e:
        print(f"✗ simulation.config.json validation failed: {e}")
        return False


if __name__ == "__main__":
    data_valid = validate_data_config()
    sim_valid = validate_simulation_config()

    if data_valid and sim_valid:
        print("\n✓ All configurations are valid")
        sys.exit(0)
    else:
        print("\n✗ Some configurations failed validation")
        sys.exit(1)
