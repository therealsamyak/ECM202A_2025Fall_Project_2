#!/usr/bin/env python3
"""
Batch Simulation Runner for All ML Models with Comprehensive Metrics

This script runs all trained models against Oracle and Naive controllers across
multiple test dates, collecting success metrics for graph generation.

Test Dates: 2024-02-20, 2024-05-20, 2024-08-20, 2024-11-20
Metrics: Accuracy (Success/Small Miss/Failure), Utility, Uptime
"""

import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ruff: noqa: E402
from simulation.controllers.oracle import OracleController
from simulation.controllers.naive import NaiveController
from simulation.controllers.ml_controller import MLController
from simulation.utils.core import State, Action, ModelType, validate_config


def discover_model_files() -> List[Dict[str, Any]]:
    """Discover all model files and extract their parameters."""
    models_dir = os.path.join(project_root, "training", "models")
    model_files = []

    for filename in os.listdir(models_dir):
        if filename.endswith("_best_model.pth"):
            model_path = os.path.join(models_dir, filename)
            params = extract_params_from_filename(filename)
            model_files.append(
                {"filename": filename, "path": model_path, "params": params}
            )

    return model_files


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    """Extract parameters from model filename pattern."""
    # Pattern: controller_controller_acc{acc}_lat{lat}_succ{succ}_small{small}_large{large}_carb{carb}_cap{cap}_rate{rate}_best_model.pth
    pattern = r"acc(\d+\.\d+)_lat(\d+\.\d+)_succ(\d+)_small(\d+)_large(\d+)_carb(\d+)_cap(\d+)_rate(\d+\.\d+)"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"Cannot parse parameters from filename: {filename}")

    return {
        "accuracy_threshold": float(match.group(1)),
        "latency_threshold": float(match.group(2)),
        "success_weight": int(match.group(3)),
        "small_miss_weight": int(match.group(4)),
        "large_miss_weight": int(match.group(5)),
        "carbon_weight": int(match.group(6)),
        "battery_capacity": float(match.group(7)),  # Use float for consistency
        "charge_rate": float(match.group(8)),
    }


def create_config_for_model(params: Dict[str, Any], test_date: str) -> Dict:
    """Create configuration for a model with specific test date."""
    base_config_path = os.path.join(os.path.dirname(__file__), "simulation.config.json")

    with open(base_config_path, "r") as f:
        config = json.load(f)

    # Override with model-specific parameters
    config["user_requirements"]["accuracy_threshold"] = float(
        params["accuracy_threshold"]
    )
    config["user_requirements"]["latency_threshold_seconds"] = float(
        params["latency_threshold"]
    )

    config["reward_weights"]["success_weight"] = params["success_weight"]
    config["reward_weights"]["small_miss_weight"] = params["small_miss_weight"]
    config["reward_weights"]["large_miss_weight"] = params["large_miss_weight"]
    config["reward_weights"]["carbon_weight"] = params["carbon_weight"]

    config["system"]["battery_capacity_mwh"] = int(params["battery_capacity"])
    config["system"]["charge_rate_mwh_per_second"] = params["charge_rate"]

    config["simulation"]["start_date"] = test_date

    return config


def setup_batch_directories():
    """Create batch output directories in simulation_data/"""
    # Get simulation directory regardless of where script is run from
    sim_dir = os.path.dirname(__file__)
    batch_results_dir = os.path.join(sim_dir, "simulation_data", "batch_results")
    batch_summaries_dir = os.path.join(sim_dir, "simulation_data", "batch_summaries")

    os.makedirs(batch_results_dir, exist_ok=True)
    os.makedirs(batch_summaries_dir, exist_ok=True)

    return batch_results_dir, batch_summaries_dir


def calculate_success_metrics(
    path: List[Tuple[State, Action]], config: Dict, model_profiles: Dict[ModelType, Any]
) -> Dict[str, Any]:
    """Calculate accuracy metrics: Success/Small Miss/Failure counts."""
    accuracy_threshold = config["user_requirements"]["accuracy_threshold"]
    latency_threshold = config["user_requirements"]["latency_threshold_seconds"]

    metrics = {
        "success_count": 0,
        "small_miss_count": 0,
        "failure_count": 0,
        "total_timesteps": len(path),
    }

    for state, action in path:
        if action.model == ModelType.NO_MODEL:
            # Failure: no model selected due to insufficient energy
            metrics["failure_count"] += 1
        else:
            # Check if model meets requirements
            model_profile = model_profiles[action.model]
            meets_accuracy = model_profile.accuracy >= accuracy_threshold
            meets_latency = model_profile.latency <= latency_threshold

            if meets_accuracy and meets_latency:
                metrics["success_count"] += 1
            else:
                metrics["small_miss_count"] += 1

    # Calculate rates
    total = float(metrics["total_timesteps"])
    metrics["success_rate"] = float(metrics["success_count"]) / total  # type: ignore
    metrics["small_miss_rate"] = float(metrics["small_miss_count"]) / total  # type: ignore
    metrics["failure_rate"] = float(metrics["failure_count"]) / total  # type: ignore

    return metrics


def calculate_uptime_metric(
    path: List[Tuple[State, Action]], config: Dict, model_profiles: Dict[ModelType, Any]
) -> float:
    """Calculate Feasibility-Normalized Effective Uptime."""
    uptime_scores = []

    for state, action in path:
        if action.model == ModelType.NO_MODEL:
            # Failure: score 0
            uptime_scores.append(0.0)
        else:
            # Find feasible models under current energy constraint
            feasible_models = []
            for model_type, profile in model_profiles.items():
                if model_type == ModelType.NO_MODEL:
                    continue
                if state.battery_level >= profile.energy_per_inference:
                    feasible_models.append((model_type, profile.accuracy))

            if feasible_models:
                # Best achievable accuracy among feasible models
                best_accuracy = max(acc for _, acc in feasible_models)

                # Selected model accuracy
                selected_profile = model_profiles[action.model]
                selected_accuracy = selected_profile.accuracy

                # Score: selected_accuracy / best_accuracy
                uptime_scores.append(selected_accuracy / best_accuracy)
            else:
                # No feasible models
                uptime_scores.append(0.0)

    # Average over horizon
    return sum(uptime_scores) / len(uptime_scores)


def run_single_batch(
    model_info: Dict[str, str], test_date: str, batch_results_dir: str
) -> Dict[str, Any] | None:
    """Run single model against all controllers for one test date."""
    try:
        print(f"\n=== Running {model_info['filename'][:50]}... on {test_date} ===")

        # Create config for this model and test date
        config = create_config_for_model(model_info["params"], test_date)  # type: ignore

        # Validate configuration
        validate_config(config)

        # Initialize controllers
        oracle = OracleController(config)
        naive = NaiveController(config)
        ml = MLController(config, model_path=model_info["path"])

        # Run simulations
        print("  Running Oracle Controller...")
        oracle_path = oracle.solve()
        oracle_reward = oracle._calculate_path_reward(oracle_path)

        print("  Running Naive Controller...")
        naive_path = naive.run()
        naive_reward = naive._calculate_path_reward(naive_path)

        print("  Running ML Controller...")
        ml_path = ml.run()
        ml_reward = ml._calculate_path_reward(ml_path)

        # Calculate metrics for all controllers
        model_profiles = ml.model_profiles

        results = {
            "model_filename": model_info["filename"],
            "test_date": test_date,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "controllers": {},
        }

        for controller_name, path, reward in [
            ("oracle", oracle_path, oracle_reward),
            ("naive", naive_path, naive_reward),
            ("ml", ml_path, ml_reward),
        ]:
            print(f"  Calculating metrics for {controller_name}...")

            success_metrics = calculate_success_metrics(path, config, model_profiles)
            uptime_metric = calculate_uptime_metric(path, config, model_profiles)

            results["controllers"][controller_name] = {
                "total_reward": reward,
                "path_length": len(path),
                "success_metrics": success_metrics,
                "uptime_metric": uptime_metric,
                "actions": [
                    {
                        "timestep": i,
                        "model": action.model.value,
                        "charge": action.charge,
                        "battery_level": round(state.battery_level, 7),
                    }
                    for i, (state, action) in enumerate(path)
                ],
            }

        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_params = sanitize_filename_string(model_info["filename"][:30])
        filename = f"model_{safe_params}_{test_date}_{timestamp}.json"
        filepath = os.path.join(batch_results_dir, filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Saved: {filename}")
        return results

    except Exception as e:
        print(f"✗ Failed: {model_info['filename'][:30]} on {test_date}: {e}")
        return None


def sanitize_filename_string(s: str) -> str:
    """Sanitize string for use in filename."""
    return re.sub(r"[^\w\-_\.]", "_", s)


def generate_summary_and_graph_data(
    all_results: List[Dict[str, Any]], batch_summaries_dir: str
):
    """Generate comprehensive summary with graph-ready data."""
    print("\n=== Generating Summary and Graph Data ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group results by model
    models_grouped = {}
    for result in all_results:
        model_key = result["model_filename"]
        if model_key not in models_grouped:
            models_grouped[model_key] = []
        models_grouped[model_key].append(result)

    # Initialize graph data structure
    graph_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_models": len(models_grouped),
            "test_dates": ["2024-02-20", "2024-05-20", "2024-08-20", "2024-11-20"],
            "controllers": ["oracle", "naive", "ml"],
        },
        "accuracy_metrics": {
            "models": [],
            "success_rates": {"oracle": [], "naive": [], "ml": []},
            "small_miss_rates": {"oracle": [], "naive": [], "ml": []},
            "failure_rates": {"oracle": [], "naive": [], "ml": []},
        },
        "utility_comparison": {
            "models": [],
            "total_rewards": {"oracle": [], "naive": [], "ml": []},
        },
        "uptime_metrics": {
            "models": [],
            "uptime_scores": {"oracle": [], "naive": [], "ml": []},
        },
    }

    # Process each model's results
    for model_key, model_results in models_grouped.items():
        graph_data["accuracy_metrics"]["models"].append(model_key)
        graph_data["utility_comparison"]["models"].append(model_key)
        graph_data["uptime_metrics"]["models"].append(model_key)

        # Aggregate across test dates for this model
        for controller in ["oracle", "naive", "ml"]:
            success_rates = []
            small_miss_rates = []
            failure_rates = []
            total_rewards = []
            uptime_scores = []

            for result in model_results:
                controller_data = result["controllers"][controller]
                success_metrics = controller_data["success_metrics"]

                success_rates.append(success_metrics["success_rate"])
                small_miss_rates.append(success_metrics["small_miss_rate"])
                failure_rates.append(success_metrics["failure_rate"])
                total_rewards.append(controller_data["total_reward"])
                uptime_scores.append(controller_data["uptime_metric"])

            # Store averages across test dates
            graph_data["accuracy_metrics"]["success_rates"][controller].append(
                sum(success_rates) / len(success_rates)
            )
            graph_data["accuracy_metrics"]["small_miss_rates"][controller].append(
                sum(small_miss_rates) / len(small_miss_rates)
            )
            graph_data["accuracy_metrics"]["failure_rates"][controller].append(
                sum(failure_rates) / len(failure_rates)
            )
            graph_data["utility_comparison"]["total_rewards"][controller].append(
                sum(total_rewards) / len(total_rewards)
            )
            graph_data["uptime_metrics"]["uptime_scores"][controller].append(
                sum(uptime_scores) / len(uptime_scores)
            )

    # Save comprehensive summary
    summary_data = {
        "summary": {
            "total_runs": len(all_results),
            "models_tested": list(models_grouped.keys()),
            "test_dates_used": ["2024-02-20", "2024-05-20", "2024-08-20", "2024-11-20"],
            "generation_timestamp": timestamp,
        },
        "graph_data": graph_data,
        "detailed_results": all_results,
    }

    summary_filename = f"batch_summary_{timestamp}.json"
    summary_filepath = os.path.join(batch_summaries_dir, summary_filename)

    with open(summary_filepath, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary saved: {summary_filename}")
    print("Graph data ready for visualization")

    return summary_data


def main():
    """Main batch execution runner."""
    print("=== Batch Simulation Runner ===")
    print("Testing all models across multiple dates with comprehensive metrics")

    # Setup directories
    batch_results_dir, batch_summaries_dir = setup_batch_directories()
    print("Output directories ready:")
    print(f"  Results: {batch_results_dir}")
    print(f"  Summaries: {batch_summaries_dir}")

    # Discover models
    model_files = discover_model_files()
    print(f"\nFound {len(model_files)} model files:")
    for model in model_files:
        print(f"  - {model['filename']}")

    # Test dates for seasonal coverage
    test_dates = ["2024-02-20", "2024-05-20", "2024-08-20", "2024-11-20"]
    print(f"\nTest dates: {test_dates}")

    # Track all results
    all_results = []
    total_runs = len(model_files) * len(test_dates)

    print(f"\nStarting {total_runs} total runs...")
    print("Each run processes 3 controllers (Oracle, Naive, ML)")
    print(
        "Note: Oracle controller uses its own internal multiprocessing for optimization"
    )

    # Execute batch runs sequentially
    current_run = 0
    for model_info in model_files:
        for test_date in test_dates:
            current_run += 1
            print(f"\n--- Progress: {current_run}/{total_runs} ---")

            try:
                result = run_single_batch(model_info, test_date, batch_results_dir)
                if result is not None:
                    all_results.append(result)
                    print(f"✓ Completed: {model_info['filename'][:30]} on {test_date}")
                else:
                    print(f"✗ Failed: {model_info['filename'][:30]} on {test_date}")
            except Exception as e:
                print(f"✗ Failed: {model_info['filename'][:30]} on {test_date}: {e}")
                continue

    # Generate summary and graph data
    print("\n=== Batch Complete ===")
    print(f"Successfully completed {len(all_results)}/{total_runs} runs")

    if all_results:
        generate_summary_and_graph_data(all_results, batch_summaries_dir)
        print("\nAll results saved to simulation_data/")
        print("Graph data ready for visualization")
    else:
        print("No successful runs to summarize")


if __name__ == "__main__":
    main()
