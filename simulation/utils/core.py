import json
import os
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    YOLOv10_N = "YOLOv10_N"
    YOLOv10_S = "YOLOv10_S"
    YOLOv10_M = "YOLOv10_M"
    YOLOv10_B = "YOLOv10_B"
    YOLOv10_L = "YOLOv10_L"
    YOLOv10_X = "YOLOv10_X"
    NO_MODEL = "NO_MODEL"


@dataclass
class ModelProfile:
    name: str
    accuracy: float
    latency: float
    energy_per_inference: float


@dataclass
class Action:
    model: ModelType
    charge: bool


@dataclass
class State:
    timestep: int
    battery_level: float


class DataLoader:
    @staticmethod
    def load_model_profiles(path: str) -> Dict[ModelType, ModelProfile]:
        # Handle relative paths from simulation directory
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
        with open(path, "r") as f:
            data = json.load(f)

        profiles = {}
        for model_name, model_data in data.items():
            model_type = ModelType(model_name)
            profiles[model_type] = ModelProfile(
                name=model_name,
                accuracy=model_data["accuracy"],
                latency=model_data["avg_inference_time_seconds"],
                energy_per_inference=model_data["energy_per_inference_mwh"],
            )

        # Add NO_MODEL profile manually
        profiles[ModelType.NO_MODEL] = ModelProfile(
            name="NO_MODEL",
            accuracy=0.0,
            latency=0.0,
            energy_per_inference=0.0,
        )

        return profiles

    @staticmethod
    def load_carbon_data(
        path: str,
        start_date: str = "2024-01-01",
        start_time: str = "00:00:00",
        num_timesteps: int = 10,
    ) -> List[float]:
        # Handle relative paths from simulation directory
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
        df = pd.read_csv(path, low_memory=False)

        # Parse start datetime
        start_datetime = f"{start_date} {start_time}"

        # Find row index for the start datetime
        df["datetime"] = pd.to_datetime(df["Datetime (UTC)"])
        matching_rows = df[df["datetime"] >= start_datetime]
        start_idx = matching_rows.index[0]

        # Calculate dirty energy fraction from carbon-free energy percentage
        # Column 7 is "Carbon-free energy percentage (CFE%)"
        dirty_fractions = []

        for i in range(int(start_idx), min(int(start_idx) + num_timesteps, len(df))):
            cfe_percentage = df.iloc[i]["Carbon-free energy percentage (CFE%)"]
            dirty_fraction = 1.0 - (cfe_percentage / 100.0)
            dirty_fractions.append(max(0.0, min(1.0, dirty_fraction)))

        return dirty_fractions


class RewardCalculator:
    def __init__(self, weights: Dict[str, float], user_req: Dict[str, float]):
        self.W = weights["success_weight"]
        self.X = weights["small_miss_weight"]
        self.Y = weights["large_miss_weight"]
        self.Z = weights["carbon_weight"]
        self.accuracy_threshold = user_req["accuracy_threshold"]
        self.latency_threshold = user_req["latency_threshold_seconds"]

    def calculate_reward(
        self,
        action: Action,
        model_profile: ModelProfile | None,
        dirty_energy_fraction: float,
        feasible_models: List[ModelType] | None = None,
        model_profiles: Dict[ModelType, ModelProfile] | None = None,
    ) -> float:
        # Success indicators
        success = 0
        small_miss = 0
        large_miss = 0

        if (
            action.model != ModelType.NO_MODEL and model_profile is not None
        ):  # Using NO_MODEL as "no model" placeholder
            meets_accuracy = model_profile.accuracy >= self.accuracy_threshold
            meets_latency = model_profile.latency <= self.latency_threshold

            if meets_accuracy and meets_latency:
                success = 1
            else:
                small_miss = 1
        else:
            large_miss = 1

        # Carbon cost
        energy_cost = (
            model_profile.energy_per_inference
            if (action.model != ModelType.NO_MODEL and model_profile is not None)
            else 0
        )
        # Charging cost: charge_rate * 300 seconds * dirty_energy_fraction
        charging_cost = (
            action.charge * 0.00001 * 300 * dirty_energy_fraction
        )  # charge_rate from config
        delta_d = energy_cost * dirty_energy_fraction + charging_cost

        # Reward calculation
        reward = (
            self.W * success
            - self.X * small_miss
            - self.Y * large_miss
            - self.Z * delta_d
        )

        return reward


class TransitionDynamics:
    def __init__(self, battery_capacity: float, charge_rate: float):
        self.battery_capacity = battery_capacity
        self.charge_rate = charge_rate

    def transition(
        self,
        state: State,
        action: Action,
        model_profiles: Dict[ModelType, ModelProfile],
    ) -> State:
        # Energy consumption
        if action.model == ModelType.NO_MODEL:  # No model
            energy_cost = 0
        else:
            energy_cost = model_profiles[action.model].energy_per_inference

        # Energy gain from charging
        energy_gain = (
            action.charge * self.charge_rate * 300
        )  # 300 seconds per task interval

        # New battery level
        new_battery = state.battery_level + energy_gain - energy_cost
        new_battery = max(0, min(self.battery_capacity, new_battery))

        return State(timestep=state.timestep + 1, battery_level=round(new_battery, 7))

    def is_feasible(
        self,
        state: State,
        action: Action,
        model_profiles: Dict[ModelType, ModelProfile],
    ) -> bool:
        if action.model == ModelType.NO_MODEL:  # No model
            return True

        energy_cost = model_profiles[action.model].energy_per_inference
        return state.battery_level >= energy_cost


def validate_config(config: Dict) -> None:
    """Validate configuration parameters according to report specifications"""
    system = config["system"]
    user_req = config["user_requirements"]
    weights = config["reward_weights"]

    # Validate system parameters
    if system["battery_capacity_mwh"] <= 0:
        raise ValueError("Battery capacity must be positive")
    if system["task_interval_seconds"] <= 0:
        raise ValueError("Task interval must be positive")
    if system["horizon_seconds"] <= 0:
        raise ValueError("Horizon must be positive")
    if system["charge_rate_mwh_per_second"] < 0:
        raise ValueError("Charge rate must be non-negative")

    # Validate user requirements
    if not (0 <= user_req["accuracy_threshold"] <= 1):
        raise ValueError("Accuracy threshold must be between 0 and 1")
    if user_req["latency_threshold_seconds"] <= 0:
        raise ValueError("Latency threshold must be positive")

    # Validate reward weights
    if any(w < 0 for w in weights.values()):
        raise ValueError("All reward weights must be non-negative")

    # Validate K parameter
    if "nearest_neighbor_k" in system:
        k = system["nearest_neighbor_k"]
        if not isinstance(k, int) or k <= 0 or k % 2 == 0:
            raise ValueError("nearest_neighbor_k must be an odd positive integer")

    # Check if horizon is multiple of task interval
    if system["horizon_seconds"] % system["task_interval_seconds"] != 0:
        raise ValueError("Horizon must be multiple of task interval")

    # Validate date format
    sim = config["simulation"]
    try:
        from datetime import datetime

        datetime.strptime(sim["start_date"], "%Y-%m-%d")
        datetime.strptime(sim["start_time"], "%H:%M:%S")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid date/time format: {e}")
