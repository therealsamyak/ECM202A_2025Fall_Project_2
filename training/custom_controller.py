"""
CustomController for POMDP Imitation Learning

Deployable π_θ(o_t) policy network that matches OracleController interface exactly.
Drop-in replacement for oracle in existing simulation infrastructure.
"""

import sys
import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Any

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from simulation.utils.core import (  # noqa: E402
    State,
    Action,
    ModelType,
    TransitionDynamics,
    RewardCalculator,
    DataLoader as CoreDataLoader,
)
from training.model import get_device, load_model  # noqa: E402


class CustomController:
    """
    POMDP controller using trained policy network π_θ(o_t).

    Exact same interface as OracleController for seamless integration.
    Operates with limited observability: o_t = (B_t, d_t, Δd_t)
    """

    def __init__(self, config: Dict):
        """
        Match OracleController constructor pattern.

        Args:
            config: Controller configuration dictionary
        """
        self.config = config

        # Load trained policy network
        model_path = config.get("model_path", "./models/best_model.pth")
        self.device = get_device()
        self.model, self.model_metadata = load_model(model_path, self.device)

        # Initialize core simulation components
        self.transition = TransitionDynamics(
            config["system_parameters"]["battery_capacity_mwh"],
            config["system_parameters"]["charge_rate_mwh_per_second"],
            config["system_parameters"]["task_interval_seconds"],
        )

        # Load model profiles and carbon data
        self.model_profiles = CoreDataLoader.load_model_profiles(
            config["data_paths"]["model_profiles"]
        )
        self.carbon_data = CoreDataLoader.load_carbon_data(
            config["data_paths"]["energy_data"]
        )

        # Calculate number of timesteps
        self.num_timesteps = (
            config["system_parameters"]["horizon_seconds"]
            // config["system_parameters"]["task_interval_seconds"]
        )

        # Store user requirements for action selection
        self.user_requirements = config.get(
            "user_requirements",
            {"accuracy_threshold": 0.9, "latency_threshold_seconds": 0.006},
        )

        print(f"CustomController initialized with model: {model_path}")
        print(f"Using device: {self.device}")
        print(f"Simulation horizon: {self.num_timesteps} timesteps")

    def solve(self, scenario: Any = None) -> List[Tuple[State, Action]]:
        """
        Exact same interface as OracleController.solve().

        Args:
            scenario: Simulation scenario (compatible with oracle interface)

        Returns:
            List of (state, action) tuples representing optimal trajectory
        """
        path = []

        # Initialize starting state
        if scenario and hasattr(scenario, "initial_state"):
            current_state = scenario.initial_state
        else:
            # Default initial state (same as oracle)
            current_state = State(
                timestep=0,
                battery_level=self.config.get(
                    "B_max", self.config["system_parameters"]["battery_capacity_mwh"]
                ),
            )

        for t in range(self.num_timesteps):
            # Extract observation o_t = (B_t, d_t, Δd_t)
            observation = self._extract_observation(current_state, t)

            # Get action from policy network π_θ(o_t)
            action = self.get_action(observation)

            # Append to trajectory
            path.append((current_state, action))

            # Transition to next state (same dynamics as oracle)
            next_state = self.transition.transition(
                current_state, action, self.model_profiles
            )
            current_state = next_state

            # Stop if we've reached the horizon
            if current_state.timestep >= self.num_timesteps - 1:
                break

        return path

    def _extract_observation(self, state: State, timestep: int) -> List[float]:
        """
        Exact observation space from Section 3.4.

        o_t = (B_t, d_t, Δd_t) ∈ [0,1] × [0,1] × [-1,1]

        Args:
            state: Current State object
            timestep: Current timestep index

        Returns:
            Observation vector [battery_level, carbon_intensity, carbon_change]
        """
        # Battery level (normalized to [0,1])
        battery_capacity = self.config.get(
            "B_max", self.config["system_parameters"]["battery_capacity_mwh"]
        )
        battery_level = state.battery_level / battery_capacity

        # Carbon intensity (already normalized in data)
        if timestep < len(self.carbon_data):
            carbon_intensity = self.carbon_data[timestep]
        else:
            # Use last available value if we exceed data length
            carbon_intensity = (
                self.carbon_data[-1] if len(self.carbon_data) > 0 else 0.5
            )

        # Carbon change Δd_t = d_t - d_{t-1}
        if timestep > 0 and timestep - 1 < len(self.carbon_data):
            carbon_change = carbon_intensity - self.carbon_data[timestep - 1]
        else:
            carbon_change = 0.0

        return [battery_level, carbon_intensity, carbon_change]

    def get_action(self, observation: List[float]) -> Action:
        """
        Map π_θ(o_t) to action space (M ∪ {∅}) × {0,1}.

        Args:
            observation: [battery_level, carbon_intensity, carbon_change]

        Returns:
            Action with model and charge decisions
        """
        self.model.eval()

        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            # Get model predictions
            model_logits, charge_logit = self.model(obs_tensor)

            # Sample from policy distribution
            model_probs = torch.softmax(model_logits, dim=1)
            charge_prob = torch.sigmoid(charge_logit)

            # Select action based on probabilities
            model_idx = torch.multinomial(model_probs, 1).item()
            charge_decision = torch.bernoulli(charge_prob).item() == 1

            # Convert model index to ModelType
            model_type = ModelType(model_idx)

            return Action(model=model_type, charge=charge_decision)

    def get_action_distribution(
        self, observation: List[float]
    ) -> Dict[str, np.ndarray]:
        """
        Get action probabilities for given observation.
        Useful for analysis and debugging.

        Args:
            observation: [battery_level, carbon_intensity, carbon_change]

        Returns:
            Dictionary with model_probs and charge_prob
        """
        self.model.eval()

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            model_probs, charge_prob = self.model.get_action_distribution(obs_tensor)

            return {
                "model_probs": model_probs.cpu().numpy().flatten(),
                "charge_prob": charge_prob.cpu().numpy().flatten(),
            }

    def get_optimal_value(self, scenario: Any = None) -> float:
        """
        Get the optimal value achievable (same interface as OracleController).

        For imitation controller, this returns the expected value based on policy.

        Args:
            scenario: Simulation scenario

        Returns:
            Expected cumulative reward
        """
        trajectory = self.solve(scenario)

        # Calculate cumulative reward using same RewardCalculator as oracle

        reward_calculator = RewardCalculator(
            self.config.get("reward_weights", {}),
            self.config.get("user_requirements", {}),
            self.config.get("system_parameters", {}),
        )

        # Calculate reward manually since no cumulative method exists
        total_reward = 0.0
        for t, (state, action) in enumerate(trajectory):
            # Get model profile
            model_profile = self.model_profiles.get(action.model)

            # Get carbon data for this timestep
            dirty_energy_fraction = (
                self.carbon_data[min(t, len(self.carbon_data) - 1)]
                if self.carbon_data
                else 0.0
            )

            # Calculate single step reward
            step_reward = reward_calculator.calculate_reward(
                action, model_profile, dirty_energy_fraction, None, self.model_profiles
            )
            total_reward += step_reward

        return total_reward

    def set_model_path(self, model_path: str) -> None:
        """
        Load a different trained model.

        Args:
            model_path: Path to model file
        """
        self.model, self.model_metadata = load_model(model_path, self.device)
        self.config["model_path"] = model_path
        print(f"Loaded new model from: {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model metadata dictionary
        """
        info = {
            "model_path": self.config.get("model_path", "./models/best_model.pth"),
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

        # Add training metadata if available
        if self.model_metadata:
            info.update(
                {
                    "training_config": self.model_metadata.get("config", {}),
                    "best_val_loss": self.model_metadata.get("best_val_loss"),
                    "epochs_trained": self.model_metadata.get("epochs_trained"),
                }
            )

        return info
