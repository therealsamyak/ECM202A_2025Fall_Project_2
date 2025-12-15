import sys
import os
from typing import Dict, List, Tuple

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ruff: noqa: E402
from simulation.utils.core import (
    State,
    Action,
    ModelType,
    TransitionDynamics,
    RewardCalculator,
    DataLoader,
)


class NaiveController:
    def __init__(self, config: Dict):
        self.config = config
        self.model_profiles = DataLoader.load_model_profiles(
            config["data_paths"]["model_profiles"]
        )
        self.carbon_data = DataLoader.load_carbon_data(
            config["data_paths"]["energy_data"],
            config["simulation"]["start_date"],
            config["simulation"]["start_time"],
            num_timesteps=config["system"]["horizon_seconds"]
            // config["system"]["task_interval_seconds"],
        )

        self.transition = TransitionDynamics(
            config["system"]["battery_capacity_mwh"],
            config["system"]["charge_rate_mwh_per_second"],
            config["system"]["task_interval_seconds"],
        )

        self.reward_calc = RewardCalculator(
            config["reward_weights"], config["user_requirements"], config["system"]
        )

        self.num_timesteps = len(self.carbon_data)
        self.accuracy_threshold = config["user_requirements"]["accuracy_threshold"]
        self.latency_threshold = config["user_requirements"][
            "latency_threshold_seconds"
        ]

    def _find_lowest_energy_model_meeting_requirements(self) -> ModelType | None:
        """Find the lowest energy model that meets user requirements"""
        candidates = []

        for model_type, profile in self.model_profiles.items():
            if (
                profile.accuracy >= self.accuracy_threshold
                and profile.latency <= self.latency_threshold
            ):
                candidates.append((model_type, profile.energy_per_inference))

        if candidates:
            # Sort by energy and return the lowest
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def _find_lowest_energy_model(self) -> ModelType:
        """Find the lowest energy model regardless of requirements"""
        candidates = [
            (model_type, profile.energy_per_inference)
            for model_type, profile in self.model_profiles.items()
        ]
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def run(self) -> List[Tuple[State, Action]]:
        """Run the naive controller policy"""
        path = []
        current_battery = self.config["system"]["battery_capacity_mwh"]

        for t in range(self.num_timesteps):
            state = State(timestep=t, battery_level=current_battery)

            # Naive policy logic
            best_model = self._find_lowest_energy_model_meeting_requirements()

            if best_model is not None:
                # Check if we have enough energy
                energy_needed = self.model_profiles[best_model].energy_per_inference
                if current_battery >= energy_needed:
                    # Execute model without charging
                    action = Action(model=best_model, charge=False)
                else:
                    # Not enough energy, charge with lowest energy model
                    lowest_model = self._find_lowest_energy_model()
                    action = Action(model=lowest_model, charge=True)
            else:
                # No model meets requirements, try to run lowest energy model
                lowest_model = self._find_lowest_energy_model()
                energy_needed = self.model_profiles[lowest_model].energy_per_inference

                if current_battery >= energy_needed:
                    # Run lowest energy model without charging
                    action = Action(model=lowest_model, charge=False)
                else:
                    # Not enough energy for any model, just charge
                    action = Action(model=ModelType.NO_MODEL, charge=True)

            path.append((state, action))

            # Update battery level
            next_state = self.transition.transition(state, action, self.model_profiles)
            current_battery = next_state.battery_level

        return path

    def _calculate_path_reward(self, path: List[Tuple[State, Action]]) -> float:
        """Calculate total reward for the path"""
        total_reward = 0.0

        for t, (state, action) in enumerate(path):
            if action.model == ModelType.NO_MODEL:
                model_profile = None
            else:
                model_profile = self.model_profiles[action.model]

            reward = self.reward_calc.calculate_reward(
                action, model_profile, self.carbon_data[t]
            )
            total_reward += reward

        return total_reward
