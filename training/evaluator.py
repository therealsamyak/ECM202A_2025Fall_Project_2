"""
Evaluation Framework for POMDP Imitation Learning

Implements exact metrics from Section 1.6:
- Accuracy classification: success, small_miss, failure rates
- Utility comparison: R_imitation - R_oracle
- Feasibility-normalized effective uptime: a(m_t)/a_t*
- Action distribution comparison: KL divergence analysis
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import entropy
import json

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


class Evaluator:
    """
    Evaluates POMDP controller against oracle policy using Section 1.6 metrics.
    """

    def __init__(self, controller, oracle_controller, config: Dict):
        """
        Initialize evaluator with controllers and configuration.

        Args:
            controller: Trained CustomController (π_θ)
            oracle_controller: OracleController (π*)
            config: Evaluation configuration
        """
        self.controller = controller
        self.oracle_controller = oracle_controller
        self.config = config

        # Initialize core components from simulation
        self.transition = TransitionDynamics(
            config["system_parameters"]["battery_capacity_mwh"],
            config["system_parameters"]["charge_rate_mwh_per_second"],
            config["system_parameters"]["task_interval_seconds"],
        )
        self.reward_calculator = RewardCalculator(
            config["reward_weights"],
            config["user_requirements"],
            config["system_parameters"],
        )

        # Load model profiles and data
        self.model_profiles = CoreDataLoader.load_model_profiles(
            config["data_paths"]["model_profiles"]
        )
        self.carbon_data = CoreDataLoader.load_carbon_data(
            config["data_paths"]["energy_data"]
        )

    def _calculate_trajectory_reward(
        self, trajectory: List[Tuple[State, Action]]
    ) -> float:
        """Calculate cumulative reward for a trajectory."""
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
            step_reward = self.reward_calculator.calculate_reward(
                action, model_profile, dirty_energy_fraction, None, self.model_profiles
            )
            total_reward += step_reward

        return float(total_reward)

    def evaluate_accuracy_classification(
        self, test_scenarios: List[Any]
    ) -> Dict[str, float]:
        """
        Exact metrics from Section 1.6

        Classification based on:
        - Energy availability via transition.is_feasible()
        - Model requirements (accuracy >= 0.9, latency <= 0.006s)

        Returns:
            {success_rate, small_miss_rate, failure_rate}
        """
        success_count = small_miss_count = failure_count = 0
        total = 0

        for scenario in test_scenarios:
            trajectory = self.controller.solve(scenario)

            for t, (state, action) in enumerate(trajectory):
                # Check energy availability
                energy_available = self.transition.is_feasible(
                    state, action, self.model_profiles
                )

                if not energy_available:
                    failure_count += 1
                elif action.model != ModelType.NO_MODEL:
                    model_profile = self.model_profiles[action.model]
                    meets_reqs = (
                        model_profile.accuracy
                        >= self.config["user_requirements"]["accuracy_threshold"]
                        and model_profile.latency
                        <= self.config["user_requirements"]["latency_threshold_seconds"]
                    )

                    if meets_reqs:
                        success_count += 1
                    else:
                        small_miss_count += 1
                else:
                    failure_count += 1

                total += 1

        total = success_count + small_miss_count + failure_count
        return {
            "success_rate": success_count / total if total > 0 else 0.0,
            "small_miss_rate": small_miss_count / total if total > 0 else 0.0,
            "failure_rate": failure_count / total if total > 0 else 0.0,
        }

    def evaluate_utility_gap(self, test_scenarios: List[Any]) -> Dict[str, float]:
        """
        Calculate utility gap: R_imitation - R_oracle

        Runs both controllers on same scenarios and compares cumulative rewards.

        Returns:
            {imitation_reward, oracle_reward, utility_gap}
        """
        imitation_rewards = []
        oracle_rewards = []

        for scenario in test_scenarios:
            # Run imitation controller
            imitation_trajectory = self.controller.solve(scenario)
            imitation_reward = self._calculate_trajectory_reward(imitation_trajectory)
            imitation_rewards.append(imitation_reward)

            # Run oracle controller
            oracle_trajectory = self.oracle_controller.solve(scenario)
            oracle_reward = self._calculate_trajectory_reward(oracle_trajectory)
            oracle_rewards.append(oracle_reward)

        avg_imitation = np.mean(imitation_rewards)
        avg_oracle = np.mean(oracle_rewards)
        utility_gap = avg_imitation - avg_oracle

        return {
            "imitation_reward": float(avg_imitation),
            "oracle_reward": float(avg_oracle),
            "utility_gap": float(utility_gap),
            "performance_ratio": float(
                avg_imitation / avg_oracle if avg_oracle != 0 else 0.0
            ),
        }

    def calculate_uptime_metric(self, test_scenarios: List[Any]) -> Dict[str, float]:
        """
        Exact formula: score = a(m_t)/a_t* for each timestep

        a_t* = max_{m ∈ F_t} a(m) - best achievable accuracy in feasible set
        F_t = feasible models at timestep t

        Returns:
            {uptime_score, average_best_accuracy, average_selected_accuracy}
        """
        scores = []
        best_accuracies = []
        selected_accuracies = []

        for scenario in test_scenarios:
            trajectory = self.controller.solve(scenario)

            for t, (state, action) in enumerate(trajectory):
                # Get feasible models F_t
                feasible_models = []
                for model_type, model_profile in self.model_profiles.items():
                    if model_type == ModelType.NO_MODEL:
                        continue

                    test_action = Action(model=model_type, charge=False)
                    if self.transition.is_feasible(
                        state, test_action, self.model_profiles
                    ):
                        feasible_models.append(model_profile)

                if feasible_models:
                    # a_t* = max_{m ∈ F_t} a(m) - best achievable accuracy
                    best_accuracy = max([m.accuracy for m in feasible_models])
                    best_accuracies.append(best_accuracy)

                    # Get selected model accuracy
                    if action.model != ModelType.NO_MODEL:
                        selected_accuracy = self.model_profiles[action.model].accuracy
                        selected_accuracies.append(selected_accuracy)

                        score = selected_accuracy / best_accuracy  # a(m_t) / a_t*
                        scores.append(score)
                    else:
                        scores.append(0.0)
                        selected_accuracies.append(0.0)
                else:
                    scores.append(0.0)
                    best_accuracies.append(0.0)
                    selected_accuracies.append(0.0)

        return {
            "uptime_score": float(np.mean(scores) if scores else 0.0),
            "average_best_accuracy": float(
                np.mean(best_accuracies) if best_accuracies else 0.0
            ),
            "average_selected_accuracy": float(
                np.mean(selected_accuracies) if selected_accuracies else 0.0
            ),
            "total_timesteps": len(scores),
        }

    def analyze_action_distributions(self, test_scenarios: List[Any]) -> Dict[str, Any]:
        """
        Compare action distributions between imitation and oracle using KL divergence.
        Analyzes temporal patterns across timesteps.

        Returns:
            {model_kl_divergence, charge_kl_divergence, distribution_stats}
        """
        # Collect actions from both controllers
        imitation_model_actions = []
        imitation_charge_actions = []
        oracle_model_actions = []
        oracle_charge_actions = []

        for scenario in test_scenarios:
            # Imitation controller actions
            imitation_trajectory = self.controller.solve(scenario)
            for state, action in imitation_trajectory:
                imitation_model_actions.append(
                    action.model.value
                    if hasattr(action.model, "value")
                    else action.model
                )
                imitation_charge_actions.append(1 if action.charge else 0)

            # Oracle controller actions
            oracle_trajectory = self.oracle_controller.solve(scenario)
            for state, action in oracle_trajectory:
                oracle_model_actions.append(
                    action.model.value
                    if hasattr(action.model, "value")
                    else action.model
                )
                oracle_charge_actions.append(1 if action.charge else 0)

        # Calculate distributions
        def calculate_distribution(actions, num_classes):
            counts = np.zeros(num_classes)
            for action in actions:
                if 0 <= action < num_classes:
                    counts[action] += 1
            return counts / len(actions) if len(actions) > 0 else counts

        # Model distributions (7 models)
        imitation_model_dist = calculate_distribution(imitation_model_actions, 7)
        oracle_model_dist = calculate_distribution(oracle_model_actions, 7)

        # Charge distributions (2 classes)
        imitation_charge_dist = calculate_distribution(imitation_charge_actions, 2)
        oracle_charge_dist = calculate_distribution(oracle_charge_actions, 2)

        # KL divergences
        model_kl = entropy(oracle_model_dist, qk=imitation_model_dist + 1e-10)
        charge_kl = entropy(oracle_charge_dist, qk=imitation_charge_dist + 1e-10)

        # Per-timestep analysis
        def per_timestep_analysis(imitation_actions, oracle_actions, num_classes):
            timestep_kls = []

            for t in range(min(len(imitation_actions), len(oracle_actions))):
                # Create distributions from current timestep only
                # For single timestep, we use indicator distributions
                imitation_dist = np.zeros(num_classes)
                oracle_dist = np.zeros(num_classes)

                if 0 <= imitation_actions[t] < num_classes:
                    imitation_dist[imitation_actions[t]] = 1.0
                if 0 <= oracle_actions[t] < num_classes:
                    oracle_dist[oracle_actions[t]] = 1.0

                # KL for single timestep (0 if same, log(num_classes) if different)
                timestep_kl = entropy(oracle_dist, qk=imitation_dist + 1e-10)
                timestep_kls.append(timestep_kl)

            return timestep_kls

        model_timestep_kls = per_timestep_analysis(
            imitation_model_actions, oracle_model_actions, 7
        )
        charge_timestep_kls = per_timestep_analysis(
            imitation_charge_actions, oracle_charge_actions, 2
        )

        return {
            "model_kl_divergence": float(model_kl),
            "charge_kl_divergence": float(charge_kl),
            "model_distribution": imitation_model_dist.tolist(),
            "oracle_model_distribution": oracle_model_dist.tolist(),
            "charge_distribution": imitation_charge_dist.tolist(),
            "oracle_charge_distribution": oracle_charge_dist.tolist(),
            "model_timestep_kl_mean": np.mean(model_timestep_kls)
            if model_timestep_kls
            else 0.0,
            "charge_timestep_kl_mean": np.mean(charge_timestep_kls)
            if charge_timestep_kls
            else 0.0,
            "total_actions": len(imitation_model_actions),
        }

    def comprehensive_evaluation(self, test_scenarios: List[Any]) -> Dict[str, Any]:
        """
        Run all evaluation metrics and return comprehensive report.

        Returns:
            Complete evaluation dictionary with all Section 1.6 metrics
        """
        print("Running comprehensive evaluation...")
        print(f"Test scenarios: {len(test_scenarios)}")

        # Accuracy classification
        print("Evaluating accuracy classification...")
        accuracy_results = self.evaluate_accuracy_classification(test_scenarios)

        # Utility gap
        print("Evaluating utility gap...")
        utility_results = self.evaluate_utility_gap(test_scenarios)

        # Uptime metric
        print("Evaluating uptime metric...")
        uptime_results = self.calculate_uptime_metric(test_scenarios)

        # Action distributions
        print("Analyzing action distributions...")
        distribution_results = self.analyze_action_distributions(test_scenarios)

        # Compile results
        results = {
            "test_scenarios_count": len(test_scenarios),
            "accuracy_classification": accuracy_results,
            "utility_comparison": utility_results,
            "uptime_metric": uptime_results,
            "action_distribution_analysis": distribution_results,
            "summary": {
                "success_rate": accuracy_results["success_rate"],
                "utility_gap": utility_results["utility_gap"],
                "performance_ratio": utility_results["performance_ratio"],
                "uptime_score": uptime_results["uptime_score"],
                "model_kl_divergence": distribution_results["model_kl_divergence"],
                "charge_kl_divergence": distribution_results["charge_kl_divergence"],
            },
        }

        return results

    def save_evaluation_report(self, results: Dict[str, Any], filepath: str) -> None:
        """
        Save comprehensive evaluation report to JSON file.

        Args:
            results: Evaluation results dictionary
            filepath: Path to save report
        """
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation report saved to: {filepath}")

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary in readable format.

        Args:
            results: Evaluation results dictionary
        """
        summary = results["summary"]

        print("\n" + "=" * 60)
        print("POMDP Imitation Learning Evaluation Summary")
        print("=" * 60)

        print(f"Test Scenarios: {results['test_scenarios_count']}")
        print()

        print("Accuracy Classification:")
        print(f"  Success Rate:     {summary['success_rate']:.3f}")
        print(
            f"  Small Miss Rate:  {results['accuracy_classification']['small_miss_rate']:.3f}"
        )
        print(
            f"  Failure Rate:     {results['accuracy_classification']['failure_rate']:.3f}"
        )
        print()

        print("Utility Comparison:")
        print(
            f"  Imitation Reward: {results['utility_comparison']['imitation_reward']:.2f}"
        )
        print(
            f"  Oracle Reward:    {results['utility_comparison']['oracle_reward']:.2f}"
        )
        print(f"  Utility Gap:       {summary['utility_gap']:.2f}")
        print(f"  Performance Ratio: {summary['performance_ratio']:.3f}")
        print()

        print("Uptime Metric:")
        print(f"  Uptime Score:           {summary['uptime_score']:.3f}")
        print(
            f"  Avg Best Accuracy:      {results['uptime_metric']['average_best_accuracy']:.3f}"
        )
        print(
            f"  Avg Selected Accuracy:  {results['uptime_metric']['average_selected_accuracy']:.3f}"
        )
        print()

        print("Action Distribution Analysis:")
        print(f"  Model KL Divergence:    {summary['model_kl_divergence']:.4f}")
        print(f"  Charge KL Divergence:   {summary['charge_kl_divergence']:.4f}")
        print()

        print("=" * 60)
