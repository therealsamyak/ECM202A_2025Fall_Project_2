import os
import sys
import torch
import torch.nn as nn
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


class PolicyNetwork(nn.Module):
    """3-layer policy network for POMDP imitation learning."""

    def __init__(self, input_dim: int = 3, num_models: int = 7):
        super(PolicyNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_models + 1),
        )
        self.num_models = num_models

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_output = self.shared_layers(x)
        model_logits = shared_output[:, : self.num_models]
        charge_logit = shared_output[:, self.num_models :]
        return model_logits, charge_logit


class MLController:
    """Machine Learning controller using trained PolicyNetwork."""

    def __init__(self, config: Dict, model_path: str | None = None):
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
        self.battery_capacity = config["system"]["battery_capacity_mwh"]

        # Load trained model
        if model_path is None:
            model_path = os.path.join(
                project_root, "training", "models", "best_model.pth"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = torch.device("mps")  # Apple Silicon
        self.model = PolicyNetwork().to(self.device)

        # Load model weights and metadata
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded ML model from {model_path}")
        print(f"Model metadata: {checkpoint.get('metadata', {})}")

    def _create_observation(self, state: State, timestep: int) -> torch.Tensor:
        """Create observation tensor matching training data format."""
        # Normalize battery level to [0,1]
        normalized_battery = state.battery_level / self.battery_capacity

        # Get current carbon intensity
        carbon_intensity = self.carbon_data[timestep]

        # Calculate carbon change from previous timestep
        carbon_change = (
            self.carbon_data[timestep] - self.carbon_data[timestep - 1]
            if timestep > 0
            else 0
        )

        # Create observation tensor: [battery_level, carbon_intensity, carbon_change]
        observation = torch.FloatTensor(
            [
                float(normalized_battery),  # [0,1]
                float(carbon_intensity),  # [0,1]
                float(carbon_change),  # [-1,1]
            ]
        ).unsqueeze(0)  # Add batch dimension

        return observation.to(self.device)

    def _model_index_to_type(self, model_index: int) -> ModelType:
        """Map model index to ModelType enum (exact order from training)."""
        model_mapping = {
            0: ModelType.YOLOv10_N,  # 1139 occurrences in training
            1: ModelType.YOLOv10_S,  # 1139 occurrences in training
            2: ModelType.YOLOv10_M,  # 0 occurrences in training
            3: ModelType.YOLOv10_B,  # 1139 occurrences in training
            4: ModelType.YOLOv10_L,  # 0 occurrences in training
            5: ModelType.YOLOv10_X,  # 0 occurrences in training
            6: ModelType.NO_MODEL,  # 39 occurrences in training
        }
        return model_mapping.get(int(model_index), ModelType.NO_MODEL)

    def _predict_action(self, state: State, timestep: int) -> Action:
        """Predict action using trained network."""
        observation = self._create_observation(state, timestep)

        with torch.no_grad():
            model_logits, charge_logit = self.model(observation)

            # Convert model logits to action
            model_index = int(torch.argmax(model_logits, dim=1).item())
            model_type = self._model_index_to_type(model_index)

            # Convert charge logit to boolean decision
            charge_prob = torch.sigmoid(charge_logit).item()
            charge_decision = charge_prob > 0.5

            return Action(model=model_type, charge=charge_decision)

    def run(self) -> List[Tuple[State, Action]]:
        """Run the ML controller policy."""
        path = []
        current_battery = self.config["system"]["battery_capacity_mwh"]

        for t in range(self.num_timesteps):
            state = State(timestep=t, battery_level=current_battery)

            # Predict action using ML model
            action = self._predict_action(state, t)

            # Ensure action is feasible
            if not self.transition.is_feasible(state, action, self.model_profiles):
                # Fallback: try to find a feasible action
                feasible_action = self._find_feasible_fallback(state)
                action = (
                    feasible_action
                    if feasible_action
                    else Action(model=ModelType.NO_MODEL, charge=False)
                )

            path.append((state, action))

            # Update battery level
            next_state = self.transition.transition(state, action, self.model_profiles)
            current_battery = next_state.battery_level

        return path

    def _find_feasible_fallback(self, state: State) -> Action:
        """Find a feasible fallback action when ML prediction is invalid."""
        # Try to run the lowest energy model that meets requirements
        accuracy_threshold = self.config["user_requirements"]["accuracy_threshold"]
        latency_threshold = self.config["user_requirements"][
            "latency_threshold_seconds"
        ]

        candidates = []
        for model_type, profile in self.model_profiles.items():
            if (
                profile.accuracy >= accuracy_threshold
                and profile.latency <= latency_threshold
                and state.battery_level >= profile.energy_per_inference
            ):
                candidates.append((model_type, profile.energy_per_inference))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            return Action(model=candidates[0][0], charge=False)

        # Fallback to lowest energy model without charging
        all_candidates = [
            (model_type, profile.energy_per_inference)
            for model_type, profile in self.model_profiles.items()
            if state.battery_level >= profile.energy_per_inference
        ]

        if all_candidates:
            all_candidates.sort(key=lambda x: x[1])
            return Action(model=all_candidates[0][0], charge=False)

        # Last resort: charge without model
        return Action(model=ModelType.NO_MODEL, charge=True)

    def _calculate_path_reward(self, path: List[Tuple[State, Action]]) -> float:
        """Calculate total reward for the path."""
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
