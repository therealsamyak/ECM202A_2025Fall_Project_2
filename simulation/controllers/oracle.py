import numpy as np
import sys
import os
import math
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

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
    ModelProfile,
)


def _process_battery_levels_worker(args):
    """Worker function for parallel battery level processing"""
    (
        timestep,
        battery_keys_chunk,
        config_data,
        carbon_intensity,
        model_profiles_data,
        next_timestep_values,
    ) = args

    # Recreate necessary objects from serialized data
    transition = TransitionDynamics(
        config_data["battery_capacity"],
        config_data["charge_rate"],
        config_data["task_interval"],
    )
    reward_calc = RewardCalculator(
        config_data["reward_weights"],
        config_data["user_requirements"],
        {
            "charge_rate_mwh_per_second": config_data["charge_rate"],
            "task_interval_seconds": config_data.get("task_interval", 300),
        },
    )

    # Convert model profiles back to ModelProfile objects
    model_profiles = {}
    for model_str, profile_data in model_profiles_data.items():
        try:
            model_profiles[ModelType(model_str)] = ModelProfile(**profile_data)
        except ValueError as e:
            raise ValueError(f"Invalid model type '{model_str}' in worker: {e}")

    results = {}
    for battery_key in battery_keys_chunk:
        # Exact same logic as original lines 248-295
        battery_float = float(battery_key)
        state = State(timestep=timestep, battery_level=battery_float)

        best_value = -np.inf
        best_action = None

        # All possible actions (recreate in worker)
        all_actions = []
        for model in list(ModelType):
            for charge in [False, True]:
                all_actions.append(Action(model=model, charge=charge))
        for charge in [False, True]:
            all_actions.append(Action(model=ModelType.NO_MODEL, charge=charge))

        for action in all_actions:
            if not transition.is_feasible(state, action, model_profiles):
                continue

            model_profile = (
                None
                if action.model == ModelType.NO_MODEL
                else model_profiles[action.model]
            )
            next_state = transition.transition(state, action, model_profiles)
            next_battery_key = str(round(next_state.battery_level, 7))

            reward = round(
                reward_calc.calculate_reward(action, model_profile, carbon_intensity), 7
            )

            future_value = next_timestep_values.get(next_battery_key, 0)
            value = reward + future_value

            if value > best_value:
                best_value = value
                best_action = action

        if best_action is not None:
            results[battery_key] = (
                best_value,
                (best_action.model.value, best_action.charge),
            )

    return results


class OracleController:
    def __init__(self, config: Dict):
        self.config = config
        self.model_profiles = DataLoader.load_model_profiles(
            config["data_paths"]["model_profiles"]
        )
        # Calculate actual timesteps needed for horizon
        actual_timesteps = (
            config["system"]["horizon_seconds"]
            // config["system"]["task_interval_seconds"]
        )

        # Load carbon data with one extra timestep for DP completion
        # This is handled internally by the oracle
        extra_carbon_data = DataLoader.load_carbon_data(
            config["data_paths"]["energy_data"],
            config["simulation"]["start_date"],
            config["simulation"]["start_time"],
            num_timesteps=actual_timesteps + 1,  # Request one extra
        )

        # Handle edge case: if insufficient data available, repeat last value
        if len(extra_carbon_data) < actual_timesteps + 1:
            while len(extra_carbon_data) < actual_timesteps + 1:
                extra_carbon_data.append(extra_carbon_data[-1])

        # Store only the actual needed data length for num_timesteps
        self.carbon_data = extra_carbon_data[:actual_timesteps]
        # Store extra value for DP computation
        self.extra_carbon_value = (
            extra_carbon_data[actual_timesteps]
            if len(extra_carbon_data) > actual_timesteps
            else extra_carbon_data[-1]
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
        self.discretization_step = config["system"]["battery_discretization_step"]

        # Use continuous battery values with dictionary lookup
        # Value function: V[t][battery_key] = optimal value from state (t, battery_level)
        self.V = [{} for _ in range(self.num_timesteps + 2)]  # +1 for extra timestep
        self.V[
            self.num_timesteps
        ] = {}  # Original terminal timestep (now has lookahead)
        self.V[self.num_timesteps + 1] = {}  # New terminal timestep (true end)

        # Policy: pi[t][battery_key] = optimal action from state (t, battery_level)
        self.pi = [
            {} for _ in range(self.num_timesteps + 1)
        ]  # +1 for policy at original terminal

        # All possible actions
        self.all_actions = []
        for model in list(ModelType):
            for charge in [False, True]:
                self.all_actions.append(Action(model=model, charge=charge))

        # Add NO_MODEL option (no model execution)
        for charge in [False, True]:
            self.all_actions.append(Action(model=ModelType.NO_MODEL, charge=charge))

        # Cache for feasible actions per battery level to avoid repeated checks
        self.feasible_actions_cache = {}

        # Load K parameter from config
        self.k_neighbors = config["system"]["nearest_neighbor_k"]

        # Cache for battery key conversions to avoid repeated string operations
        self.battery_key_cache = {}

        # Load max_workers from config
        self.max_workers = config["system"]["max_workers"]

        # Lock for atomic DP table updates
        self._merge_lock = Lock()

    def _get_optimal_worker_count(self, num_tasks: int) -> int:
        """Calculate optimal worker count based on task count"""
        return min(self.max_workers, num_tasks)

    def _merge_worker_results(self, t: int, results: Dict):
        """Merge worker results into DP matrices"""
        with self._merge_lock:
            for battery_key, (value, (model_str, charge_bool)) in results.items():
                model_enum = ModelType(model_str)
                action = Action(model=model_enum, charge=charge_bool)
                self.V[t][battery_key] = value
                self.pi[t][battery_key] = action

    def _prepare_worker_data(self, t: int) -> Tuple:
        """Prepare data for worker processes"""
        config_data = {
            "battery_capacity": self.battery_capacity,
            "charge_rate": self.config["system"]["charge_rate_mwh_per_second"],
            "task_interval": self.config["system"]["task_interval_seconds"],
            "reward_weights": self.config["reward_weights"],
            "user_requirements": self.config["user_requirements"],
        }

        model_profiles_data = {}
        for model_type, profile in self.model_profiles.items():
            model_profiles_data[model_type.value] = {
                "name": profile.name,
                "accuracy": profile.accuracy,
                "latency": profile.latency,
                "energy_per_inference": profile.energy_per_inference,
            }

        next_timestep_values = self.V[t + 1] if t <= self.num_timesteps else {}

        # Get carbon intensity for current timestep (use extra value for final computation)
        carbon_intensity = (
            self.carbon_data[t]
            if t < len(self.carbon_data)
            else self.extra_carbon_value
        )

        return (
            config_data,
            carbon_intensity,
            model_profiles_data,
            next_timestep_values,
        )

    def _process_timestep_parallel(self, t: int, battery_keys_to_evaluate: set):
        """Process single timestep with parallel workers and progress tracking"""
        num_tasks = len(battery_keys_to_evaluate)
        if num_tasks == 0:
            return

        actual_workers = self._get_optimal_worker_count(num_tasks)
        chunk_size = math.ceil(num_tasks / actual_workers)

        print(
            f"Timestep {t}: Processing {num_tasks} battery levels with {actual_workers} workers"
        )

        # Prepare data for workers
        config_data, carbon_intensity, model_profiles_data, next_timestep_values = (
            self._prepare_worker_data(t)
        )

        # Create chunks
        battery_keys_list = list(battery_keys_to_evaluate)
        battery_chunks = [
            battery_keys_list[i : i + chunk_size]
            for i in range(0, len(battery_keys_list), chunk_size)
        ]

        # Prepare arguments for each worker
        worker_args = [
            (
                t,
                chunk,
                config_data,
                carbon_intensity,
                model_profiles_data,
                next_timestep_values,
            )
            for chunk in battery_chunks
        ]

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = [
                executor.submit(_process_battery_levels_worker, args)
                for args in worker_args
            ]
            completed = 0

            for future in as_completed(futures):
                try:
                    results = future.result()
                    self._merge_worker_results(t, results)
                    completed += 1
                    progress = (completed / len(futures)) * 100
                    print(
                        f"  Progress: {progress:.1f}% ({completed}/{len(futures)} chunks)"
                    )
                except Exception as e:
                    raise RuntimeError(f"Worker process failed at timestep {t}: {e}")

    def _battery_to_key(self, battery_level: float) -> str:
        """Convert continuous battery level to discretized key for lookup"""
        if battery_level not in self.battery_key_cache:
            discretized_value = (
                round(battery_level / self.discretization_step)
                * self.discretization_step
            )
            self.battery_key_cache[battery_level] = str(round(discretized_value, 7))
        return self.battery_key_cache[battery_level]

    def _has_feasible_actions_for_battery(self, battery_level: float) -> bool:
        """Check if any feasible actions exist for battery level"""
        battery_key = self._battery_to_key(battery_level)

        # Check cache first
        if battery_key in self.feasible_actions_cache:
            return len(self.feasible_actions_cache[battery_key]) > 0

        # Compute if not cached
        state = State(timestep=0, battery_level=battery_level)
        for action in self.all_actions:
            if self.transition.is_feasible(state, action, self.model_profiles):
                return True
        return False

    def _expand_search_until_k_valid(
        self, battery_level: float, available_keys: set, k: int
    ) -> str:
        """Expand search until K valid options found"""
        search_radius = 1
        max_radius = 50  # Prevent infinite loop

        while search_radius <= max_radius:
            # Find all candidates within current radius
            candidates = [
                float(key)
                for key in available_keys
                if abs(float(key) - battery_level) <= search_radius
            ]

            if len(candidates) >= k:
                # Sort by distance and collect K valid options
                candidates.sort(key=lambda x: abs(x - battery_level))
                valid_count = 0
                for candidate in candidates:
                    if self._has_feasible_actions_for_battery(candidate):
                        valid_count += 1
                        if valid_count == k:
                            # Return the closest of the K valid options
                            return self._battery_to_key(candidate)
            search_radius += 1

        # Last resort: return any feasible key
        for key in available_keys:
            if self._has_feasible_actions_for_battery(float(key)):
                return self._battery_to_key(float(key))

        return self._battery_to_key(battery_level)  # Ultimate fallback

    def _find_k_nearest_keys_with_validation(
        self, battery_level: float, available_keys: set, k: int
    ) -> str:
        """Find K nearest discretized keys with feasibility validation"""
        # Validate K parameter
        if k <= 0 or k % 2 == 0:
            raise ValueError("K must be an odd positive integer")

        if not available_keys:
            return self._battery_to_key(battery_level)

        # Convert to float values and calculate distances
        available_items = [
            (float(key), abs(float(key) - battery_level)) for key in available_keys
        ]
        available_items.sort(key=lambda x: x[1])  # Sort by distance

        # Collect K nearest valid options
        valid_options = []
        for i in range(min(k, len(available_items))):
            nearest_key = str(round(available_items[i][0], 7))

            # Check if this battery level has feasible actions
            if self._has_feasible_actions_for_battery(float(nearest_key)):
                valid_options.append(nearest_key)
            else:
                # Debug: show why this option is invalid
                print(
                    f"DEBUG: Battery {float(nearest_key):.7f} has no feasible actions"
                )

        # If we found K valid options, return the closest one
        if len(valid_options) == k:
            return valid_options[0]

        # Otherwise, expand search until we get K valid options
        result = self._expand_search_until_k_valid(battery_level, available_keys, k)
        print(
            f"DEBUG: Expanded search result for battery {battery_level:.7f}: {result}"
        )
        return result

    def _find_nearest_discretized_key(
        self, battery_level: float, available_keys: set
    ) -> str:
        """Find the nearest discretized key to the actual battery level"""
        if not available_keys:
            return self._battery_to_key(battery_level)

        # Convert available keys back to float values for comparison
        available_values = [float(key) for key in available_keys]
        nearest_value = min(available_values, key=lambda x: abs(x - battery_level))
        return str(round(nearest_value, 7))

    def _get_feasible_actions(self, battery_level: float) -> List[Action]:
        """Get cached feasible actions for a battery level"""
        battery_key = self._battery_to_key(battery_level)

        if battery_key not in self.feasible_actions_cache:
            state = State(
                timestep=0, battery_level=battery_level
            )  # timestep doesn't matter for feasibility
            feasible = []
            for action in self.all_actions:
                if self.transition.is_feasible(state, action, self.model_profiles):
                    feasible.append(action)
            self.feasible_actions_cache[battery_key] = feasible

        return self.feasible_actions_cache[battery_key]

    def solve(self) -> List[Tuple[State, Action]]:
        """Solve MDP using backward induction with continuous battery levels"""
        # Initialize with full battery capacity
        initial_battery_key = self._battery_to_key(self.battery_capacity)

        # Backward induction
        for t in range(
            self.num_timesteps, -1, -1
        ):  # Include extra timestep in DP computation
            # Determine which battery levels to evaluate at this timestep
            battery_keys_to_evaluate = set()

            # Always include full battery capacity and zero battery level
            battery_keys_to_evaluate.add(initial_battery_key)
            battery_keys_to_evaluate.add(self._battery_to_key(0.0))

            if t == self.num_timesteps - 1:
                # Last timestep: already have full battery from above
                pass
            else:
                # For other timesteps, work backwards from t+1 states
                battery_keys_to_evaluate.update(self.V[t + 1].keys())

                # Pre-compute reachable battery levels from t+1 states
                # This avoids the double nested loop
                reachable_battery_keys = set()
                for next_battery_key in self.V[t + 1].keys():
                    battery_float = float(next_battery_key)
                    feasible_actions = self._get_feasible_actions(battery_float)

                    for action in feasible_actions:
                        state = State(timestep=t, battery_level=battery_float)
                        next_state = self.transition.transition(
                            state, action, self.model_profiles
                        )
                        reachable_battery_key = self._battery_to_key(
                            next_state.battery_level
                        )
                        reachable_battery_keys.add(reachable_battery_key)

                battery_keys_to_evaluate.update(reachable_battery_keys)

            # Process battery levels in parallel
            self._process_timestep_parallel(t, battery_keys_to_evaluate)

        # Extract optimal path
        return self._extract_optimal_path()

    def _calculate_path_reward(self, path: List[Tuple[State, Action]]) -> float:
        """Calculate total reward for a path"""
        total_reward = 0.0
        for t, (state, action) in enumerate(path):
            model_profile = (
                None
                if action.model == ModelType.NO_MODEL
                else self.model_profiles[action.model]
            )
            reward = round(
                self.reward_calc.calculate_reward(
                    action, model_profile, self.carbon_data[t]
                ),
                7,
            )
            total_reward += reward
        return total_reward

    def _extract_optimal_path(self) -> List[Tuple[State, Action]]:
        """Extract the optimal path from the solved policy"""
        path = []
        current_battery = self.battery_capacity
        self._battery_to_key(current_battery)

        for t in range(self.num_timesteps):
            state = State(timestep=t, battery_level=current_battery)

            # Use nearest neighbor lookup for actual battery values
            available_keys = set(self.pi[t].keys())
            nearest_key = self._find_k_nearest_keys_with_validation(
                current_battery, available_keys, self.k_neighbors
            )
            action = self.pi[t].get(nearest_key)

            if action is None:
                # Fallback: try exact key match
                exact_key = self._battery_to_key(current_battery)
                action = self.pi[t].get(exact_key)

                if action is None:
                    # Debug: check what battery keys are available at this timestep
                    available_keys_list = list(self.pi[t].keys())[:5]  # Show first 5
                    print(
                        f"t={t}: No policy found for battery {current_battery:.7f} (nearest key: {nearest_key}, exact key: {exact_key})"
                    )
                    print(f"  Available battery keys (first 5): {available_keys_list}")

                    # Default to NO_MODEL, no charge if no policy found
                    action = Action(model=ModelType.NO_MODEL, charge=False)

            path.append((state, action))

            # Move to next state
            next_state = self.transition.transition(state, action, self.model_profiles)
            current_battery = next_state.battery_level
            self._battery_to_key(current_battery)

        return path

    def get_optimal_value(self) -> float:
        """Get the optimal value from the initial state"""
        initial_battery_key = self._battery_to_key(self.battery_capacity)
        return self.V[0].get(initial_battery_key, 0)

    def export_training_data(self, path: List[Tuple[State, Action]]) -> Dict[str, Any]:
        """
        Export oracle trajectory as training data for imitation learning

        Args:
            path: List of (state, action) tuples from oracle solution

        Returns:
            Dictionary with observations and actions arrays
        """
        observations = []
        actions = []

        for t, (state, action) in enumerate(path):
            # Get carbon data for this timestep
            carbon_intensity = self.carbon_data[t]
            carbon_change = (
                self.carbon_data[t] - self.carbon_data[t - 1] if t > 0 else 0
            )

            # Normalize battery level to [0,1]
            normalized_battery = state.battery_level / self.battery_capacity

            # Create observation (minimal required features)
            observation = {
                "battery_level": float(normalized_battery),  # [0,1]
                "carbon_intensity": float(carbon_intensity),  # [0,1]
                "carbon_change": float(carbon_change),  # [-1,1]
            }

            # Create action (target for imitation learning)
            action_data = {
                "model_type": int(
                    list(ModelType).index(action.model)
                    if action.model in list(ModelType)
                    else 6  # Default to NO_MODEL index
                ),  # 0-6 for models
                "charge_decision": int(action.charge),  # 0 or 1
            }

            observations.append(
                [
                    observation["battery_level"],
                    observation["carbon_intensity"],
                    observation["carbon_change"],
                ]
            )

            actions.append([action_data["model_type"], action_data["charge_decision"]])

        return {
            "observations": observations,
            "actions": actions,
            "metadata": {
                "total_timesteps": len(path),
                "battery_capacity": self.battery_capacity,
                "task_interval": self.config["system"]["task_interval_seconds"],
                "horizon_seconds": self.config["system"]["horizon_seconds"],
                "model_types": [model.value for model in ModelType],
            },
        }
