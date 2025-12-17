#!/usr/bin/env python3
"""
Results Analysis Script for Optimal Charge Security Camera

Implements GRAPH.md requirements:
- Dynamic ablation pair identification
- Consistent 3-panel horizontal layout (Accuracy | Utility | Uptime)
- Targeted parameter isolation studies
- Simple averages (no confidence intervals)

"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List
import warnings
import sys

warnings.filterwarnings("ignore")

# Set up matplotlib and seaborn styling
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


class ResultsAnalyzer:
    """Main class for analyzing batch results and generating targeted ablation charts."""

    def __init__(self, config_path: str = "results/results.config.json"):
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()

        # Override with latest batch summary
        self.batch_summary_path = self._find_latest_batch_summary()
        print(
            f"Using latest batch summary: {os.path.basename(self.batch_summary_path)}"
        )

        # Set paths from config
        self.batch_results_dir = self.config["batch_results_dir"]
        self.output_dir = self.config["output_dir"]

        # Initialize data containers
        self.batch_data = {}
        self.summary_data = {}
        self.model_configs = {}
        self.controller_configs = {}

        # Validate paths exist
        self._validate_paths()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _find_latest_batch_summary(self) -> str:
        """Find most recent batch summary file by timestamp."""
        import re

        summary_dir = os.path.dirname(self.config["batch_summary_path"])
        pattern = re.compile(r"batch_summary_(\d{8}_\d{6})\.json")

        latest_file = None
        latest_timestamp = 0

        for file in os.listdir(summary_dir):
            match = pattern.match(file)
            if match:
                timestamp_str = match.group(1)
                timestamp = int(timestamp_str.replace("_", ""))
                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_file = os.path.join(summary_dir, file)

        if latest_file is None:
            raise FileNotFoundError(f"No batch summary files found in {summary_dir}")

        return latest_file

    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            if "batch_summary_path" not in config:
                raise ValueError("batch_summary_path is required in config")

            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def _validate_paths(self):
        """Validate that required paths exist."""
        if not os.path.exists(self.batch_summary_path):
            raise FileNotFoundError(
                f"Batch summary file not found: {self.batch_summary_path}"
            )

        if not os.path.exists(self.batch_results_dir):
            raise FileNotFoundError(
                f"Batch results directory not found: {self.batch_results_dir}"
            )

    def load_data(self):
        """Load batch results and summary data."""
        print("Loading batch results data...")

        # Load summary data
        with open(self.batch_summary_path, "r") as f:
            self.summary_data = json.load(f)

        # Parse model configurations
        self._parse_model_configurations()

        # Load controller configurations from training metadata
        self.controller_configs = self.load_controller_configs()

        # Load individual batch results for detailed analysis
        self._load_individual_results()

        print(f"Loaded data for {len(self.model_configs)} model configurations")

    def load_controller_configs(self) -> Dict:
        """Load controller configurations from training.metadata.json."""
        config_path = os.path.join("..", "training", "models", "training.metadata.json")

        if not os.path.exists(config_path):
            # Fallback to relative path
            config_path = "training/models/training.metadata.json"

        try:
            with open(config_path, "r") as f:
                metadata = json.load(f)

            # Map configurations to model names
            configs = {}
            for i, config in enumerate(metadata["controllers"]):
                # Generate model filename for this config
                model_name = f"C{i + 1}_controller_acc{config['acc']}_lat{config['lat']}_succ{config['succ']}_small{config['small']}_large{config['large']}_carb{config['carb']}_cap{config['cap']}_rate{config['rate']}_best_model.pth"
                configs[model_name] = config

            print(f"Loaded {len(configs)} controller configurations")
            return configs

        except FileNotFoundError:
            print(f"Warning: Could not find training metadata at {config_path}")
            return {}

    def _parse_model_configurations(self):
        """Parse model filenames to extract configuration parameters."""
        if not self.summary_data:
            raise ValueError("summary_data is None - load_data() must be called first")

        model_files = self.summary_data["graph_data"]["accuracy_metrics"]["models"]

        for model_file in model_files:
            # Extract parameters from filename
            parts = model_file.replace(".pth", "").split("_")
            config = {}

            for part in parts:
                if "acc" in part:
                    config["accuracy_threshold"] = float(part.replace("acc", ""))
                elif "lat" in part:
                    config["latency_threshold"] = float(part.replace("lat", ""))
                elif "succ" in part:
                    config["success_weight"] = int(part.replace("succ", ""))
                elif "small" in part:
                    config["small_miss_weight"] = int(part.replace("small", ""))
                elif "large" in part:
                    config["large_miss_weight"] = int(part.replace("large", ""))
                elif "carb" in part:
                    config["carbon_weight"] = int(part.replace("carb", ""))
                elif "cap" in part:
                    config["battery_capacity"] = int(part.replace("cap", ""))
                elif "rate" in part:
                    config["charge_rate"] = float(part.replace("rate", ""))

            # Determine configuration type
            config["config_type"] = self._get_config_type(config)

            # Store the original model file for controller mapping
            config["model_file"] = model_file
            config["controller_name"] = self._extract_controller_name(model_file)
            config["short_name"] = config["controller_name"]

            self.model_configs[model_file] = config

    def _get_config_type(self, config: Dict) -> str:
        """Determine configuration type for grouping."""
        if config["success_weight"] == 20:
            return "performance_focused"
        elif config["carbon_weight"] == 15:
            return "carbon_focused"
        else:
            return "mixed"

    def _extract_controller_name(self, model_filename: str) -> str:
        """Extract controller name (C1, C2, etc.) from model filename."""
        if "_controller_" not in model_filename:
            raise ValueError(
                f"Invalid model filename format: {model_filename}. Expected pattern: 'C[1-9]_controller_...'"
            )
        return model_filename.split("_controller_")[0]

    def _load_individual_results(self):
        """Load individual batch result files for detailed analysis."""
        self.batch_data = {}

        batch_files = glob.glob(os.path.join(self.batch_results_dir, "*.json"))

        for batch_file in batch_files:
            with open(batch_file, "r") as f:
                data = json.load(f)
                model_file = data["model_filename"]
                test_date = data["test_date"]

                key = f"{model_file}_{test_date}"
                self.batch_data[key] = data

    def find_ablation_pairs(self, configs: Dict, param_type: str) -> Dict:
        """Find clean ablation pairs differing by exactly one parameter."""
        pairs = {}
        models = list(configs.keys())

        if param_type == "accuracy":
            # Find models differing only in accuracy threshold
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:
                        continue

                    config1, config2 = configs[model1], configs[model2]

                    # Check if all parameters except accuracy are the same
                    same_params = (
                        config1["lat"] == config2["lat"]
                        and config1["succ"] == config2["succ"]
                        and config1["small"] == config2["small"]
                        and config1["large"] == config2["large"]
                        and config1["carb"] == config2["carb"]
                        and config1["cap"] == config2["cap"]
                        and config1["rate"] == config2["rate"]
                        and config1["acc"] != config2["acc"]
                    )

                    if same_params:
                        pairs[param_type] = (model1, model2)
                        return pairs

        elif param_type == "latency":
            # Find models differing only in latency threshold
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:
                        continue

                    config1, config2 = configs[model1], configs[model2]

                    same_params = (
                        config1["acc"] == config2["acc"]
                        and config1["succ"] == config2["succ"]
                        and config1["small"] == config2["small"]
                        and config1["large"] == config2["large"]
                        and config1["carb"] == config2["carb"]
                        and config1["cap"] == config2["cap"]
                        and config1["rate"] == config2["rate"]
                        and config1["lat"] != config2["lat"]
                    )

                    if same_params:
                        pairs[param_type] = (model1, model2)
                        return pairs

        elif param_type == "battery":
            # Find models differing only in battery capacity/rate
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:
                        continue

                    config1, config2 = configs[model1], configs[model2]

                    same_params = (
                        config1["acc"] == config2["acc"]
                        and config1["lat"] == config2["lat"]
                        and config1["succ"] == config2["succ"]
                        and config1["small"] == config2["small"]
                        and config1["large"] == config2["large"]
                        and config1["carb"] == config2["carb"]
                        and (
                            config1["cap"] != config2["cap"]
                            or config1["rate"] != config2["rate"]
                        )
                    )

                    if same_params:
                        pairs[param_type] = (model1, model2)
                        return pairs

        elif param_type == "reward":
            # Find models differing only in reward weights
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:
                        continue

                    config1, config2 = configs[model1], configs[model2]

                    same_params = (
                        config1["acc"] == config2["acc"]
                        and config1["lat"] == config2["lat"]
                        and config1["cap"] == config2["cap"]
                        and config1["rate"] == config2["rate"]
                        and (
                            config1["carb"] != config2["carb"]
                            or config1["succ"] != config2["succ"]
                            or config1["small"] != config2["small"]
                            or config1["large"] != config2["large"]
                        )
                    )

                    if same_params:
                        pairs[param_type] = (model1, model2)
                        return pairs

        return pairs

    def format_percentage_axis(self, ax):
        """Format y-axis as percentage."""
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

    def create_three_panel_horizontal_layout(
        self, title: str, model_data: List, save_path: str
    ):
        """Create consistent horizontal 3-panel layout with controllers as groups."""
        if not self.summary_data:
            print("Warning: No summary data available for figure generation")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Get all unique models for this figure
        all_models = []
        for model_tuple in model_data:
            if isinstance(model_tuple, tuple):
                all_models.extend(model_tuple)
            else:
                all_models.append(model_tuple)

        # Use all provided models
        models_to_show = all_models

        # Algorithm color mapping
        algorithm_colors = {
            "oracle": "#2E86AB",  # Blue
            "ml": "#A23B72",  # Purple
            "naive": "#F18F01",  # Orange
        }

        # Calculate bar positions for controller grouping
        bar_width = 0.25

        # Get unique controller names for x-axis labels
        controller_names = []
        for model_file in models_to_show:
            controller_names.append(self._extract_controller_name(model_file))

        # Keep the order as provided (no sorting)
        controller_names = controller_names[: len(models_to_show)]
        models_to_show = models_to_show[: len(controller_names)]

        # Create positions for each controller group
        x_positions = list(range(len(controller_names)))

        # Plot data for each algorithm type
        for algorithm in ["oracle", "ml", "naive"]:
            accuracy_values = []
            utility_values = []
            uptime_values = []

            for model_file in models_to_show:
                try:
                    model_index = self.summary_data["graph_data"]["accuracy_metrics"][
                        "models"
                    ].index(model_file)

                    # Get metrics for this algorithm
                    accuracy_values.append(
                        self.summary_data["graph_data"]["accuracy_metrics"][
                            "success_rates"
                        ][algorithm][model_index]
                    )
                    utility_values.append(
                        self.summary_data["graph_data"]["utility_comparison"][
                            "total_rewards"
                        ][algorithm][model_index]
                        / 4  # Average across 4 seasons
                    )
                    uptime_values.append(
                        self.summary_data["graph_data"]["uptime_metrics"][
                            "uptime_scores"
                        ][algorithm][model_index]
                    )

                except (ValueError, IndexError, TypeError):
                    print(f"Warning: Could not process model {model_file}")
                    accuracy_values.append(0)
                    utility_values.append(0)
                    uptime_values.append(0)

            # Calculate x positions for this algorithm's bars (offset within each controller group)
            # Oracle: -0.25, ML: 0, Naive: +0.25
            algorithm_index = list(algorithm_colors.keys()).index(algorithm)
            offset = (algorithm_index - 1) * bar_width
            x_algo = [x + offset for x in x_positions]

            # Panel 1: Accuracy Metrics
            ax1.bar(
                x_algo,
                accuracy_values,
                bar_width,
                label=algorithm.capitalize(),
                color=algorithm_colors[algorithm],
                alpha=0.8,
            )

            # Panel 2: Utility Metrics
            ax2.bar(
                x_algo,
                utility_values,
                bar_width,
                color=algorithm_colors[algorithm],
                alpha=0.8,
            )

            # Panel 3: Uptime Metrics
            ax3.bar(
                x_algo,
                uptime_values,
                bar_width,
                color=algorithm_colors[algorithm],
                alpha=0.8,
            )

        # Configure Panel 1: Accuracy Metrics
        ax1.set_title("Accuracy Metrics", fontweight="bold")
        ax1.set_ylabel("Average Success Rate")
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(controller_names)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        self.format_percentage_axis(ax1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Configure Panel 2: Utility Metrics
        ax2.set_title("Utility Metrics", fontweight="bold")
        ax2.set_ylabel("Average Total Reward")
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(controller_names)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Configure Panel 3: Uptime Metrics
        ax3.set_title("Uptime Metrics", fontweight="bold")
        ax3.set_ylabel("Average Uptime Score")
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(controller_names)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        self.format_percentage_axis(ax3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

    # ========== GRAPH.MD IMPLEMENTATION ==========

    def create_figure_4_1(self):
        """Figure 4.1: Overall Controller Performance Comparison (Section 4.3)."""
        print("Creating Figure 4.1: Overall Controller Performance Comparison...")

        if not self.summary_data:
            self.load_data()

        # Average across all 32 runs (all models x all dates)
        models = self.summary_data["graph_data"]["accuracy_metrics"]["models"]

        # Calculate averages across all models
        avg_success_oracle = sum(
            self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                "oracle"
            ]
        ) / len(models)
        avg_success_ml = sum(
            self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"]["ml"]
        ) / len(models)
        avg_success_naive = sum(
            self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                "naive"
            ]
        ) / len(models)

        avg_reward_oracle = sum(
            self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                "oracle"
            ]
        ) / len(models)
        avg_reward_ml = sum(
            self.summary_data["graph_data"]["utility_comparison"]["total_rewards"]["ml"]
        ) / len(models)
        avg_reward_naive = sum(
            self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                "naive"
            ]
        ) / len(models)

        avg_uptime_oracle = sum(
            self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"]["oracle"]
        ) / len(models)
        avg_uptime_ml = sum(
            self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"]["ml"]
        ) / len(models)
        avg_uptime_naive = sum(
            self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"]["naive"]
        ) / len(models)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Figure 4.1: Overall Controller Performance Comparison",
            fontsize=16,
            fontweight="bold",
        )

        # Panel 1: Accuracy Metrics
        x = [0, 1, 2]
        success_rates = [avg_success_oracle, avg_success_ml, avg_success_naive]
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        labels = ["Oracle", "ML", "Naive"]

        ax1.bar(x, success_rates, color=colors, alpha=0.7)
        ax1.set_title("Accuracy Metrics", fontweight="bold")
        ax1.set_ylabel("Average Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        self.format_percentage_axis(ax1)

        # Panel 2: Utility Metrics
        rewards = [avg_reward_oracle, avg_reward_ml, avg_reward_naive]
        ax2.bar(x, rewards, color=colors, alpha=0.7)
        ax2.set_title("Utility Metrics", fontweight="bold")
        ax2.set_ylabel("Average Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Panel 3: Uptime Metrics
        uptimes = [avg_uptime_oracle, avg_uptime_ml, avg_uptime_naive]
        ax3.bar(x, uptimes, color=colors, alpha=0.7)
        ax3.set_title("Uptime Metrics", fontweight="bold")
        ax3.set_ylabel("Average Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        self.format_percentage_axis(ax3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir, "figure_4_1_overall_performance_comparison.png"
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_2_acc_lat_comparison(self):
        """Figure 4.2: Accuracy vs Latency Comparison."""
        print("Creating Figure 4.2: Accuracy vs Latency Comparison...")

        if not self.summary_data:
            self.load_data()

        # Accuracy vs Latency Comparison: C1, C8, C6, C7 (numerical order)
        acc_lat_models = [
            "C1_controller_acc0.95_lat0.015_succ20_small5_large8_carb7_cap105_rate0.001598_best_model.pth",
            "C8_controller_acc0.95_lat0.015_succ5_small7_large10_carb15_cap105_rate0.001598_best_model.pth",
            "C6_controller_acc0.819_lat0.006_succ20_small5_large8_carb7_cap105_rate0.001598_best_model.pth",
            "C7_controller_acc0.819_lat0.006_succ5_small7_large10_carb15_cap105_rate0.001598_best_model.pth",
        ]

        title = "Figure 4.2: Accuracy vs Latency Comparison"
        self.create_three_panel_horizontal_layout(
            title,
            acc_lat_models,
            os.path.join(self.output_dir, "figure_4_2_accuracy_latency_impact.png"),
        )

    def create_figure_4_3_reward_weight_comparison(self):
        """Figure 4.3: Reward Weight Comparison."""
        print("Creating Figure 4.3: Reward Weight Comparison...")

        if not self.summary_data:
            self.load_data()

        # Reward Weight Comparison: C2, C3, C6, C7 (numerical order)
        reward_weight_models = [
            "C2_controller_acc0.95_lat0.015_succ20_small5_large8_carb7_cap610_rate0.000269_best_model.pth",
            "C3_controller_acc0.819_lat0.006_succ5_small7_large10_carb15_cap610_rate0.000269_best_model.pth",
            "C6_controller_acc0.819_lat0.006_succ20_small5_large8_carb7_cap105_rate0.001598_best_model.pth",
            "C7_controller_acc0.819_lat0.006_succ5_small7_large10_carb15_cap105_rate0.001598_best_model.pth",
        ]

        title = "Figure 4.3: Reward Weight Comparison"
        self.create_three_panel_horizontal_layout(
            title,
            reward_weight_models,
            os.path.join(self.output_dir, "figure_4_3_reward_structure_impact.png"),
        )

    def create_figure_4_4_battery_comparison(self):
        """Figure 4.4: Battery Configuration Comparison."""
        print("Creating Figure 4.4: Battery Configuration Comparison...")

        if not self.summary_data:
            self.load_data()

        # Battery Configuration Comparison: C4, C6, C5, C8 (special order)
        battery_models = [
            "C4_controller_acc0.819_lat0.006_succ20_small5_large8_carb7_cap610_rate0.000269_best_model.pth",
            "C6_controller_acc0.819_lat0.006_succ20_small5_large8_carb7_cap105_rate0.001598_best_model.pth",
            "C5_controller_acc0.95_lat0.015_succ5_small7_large10_carb15_cap610_rate0.000269_best_model.pth",
            "C8_controller_acc0.95_lat0.015_succ5_small7_large10_carb15_cap105_rate0.001598_best_model.pth",
        ]

        title = "Figure 4.4: Battery Configuration Comparison"
        self.create_three_panel_horizontal_layout(
            title,
            battery_models,
            os.path.join(
                self.output_dir, "figure_4_4_battery_configuration_impact.png"
            ),
        )

    def create_figure_4_5_seasonal_average(self):
        """Figure 4.5: Seasonal Performance Analysis - Averaged across all controllers."""
        print("Creating Figure 4.5: Seasonal Performance Analysis...")

        if not self.summary_data:
            self.load_data()

        # Get all models and seasonal dates
        all_models = self.summary_data["graph_data"]["accuracy_metrics"]["models"]
        seasonal_dates = self.summary_data["graph_data"]["metadata"]["test_dates"]

        # Calculate seasonal averages across all controllers for each algorithm
        seasonal_success_rates = {"oracle": [], "ml": [], "naive": []}
        seasonal_rewards = {"oracle": [], "ml": [], "naive": []}
        seasonal_uptime = {"oracle": [], "ml": [], "naive": []}

        # For each season, average across all controllers
        for season_idx in range(len(seasonal_dates)):
            for algorithm in ["oracle", "ml", "naive"]:
                # Sum across all models for this season and algorithm
                season_success_total = sum(
                    self.summary_data["graph_data"]["accuracy_metrics"][
                        "success_rates"
                    ][algorithm][model_idx]
                    for model_idx in range(len(all_models))
                )
                season_reward_total = sum(
                    self.summary_data["graph_data"]["utility_comparison"][
                        "total_rewards"
                    ][algorithm][model_idx]
                    for model_idx in range(len(all_models))
                )
                season_uptime_total = sum(
                    self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                        algorithm
                    ][model_idx]
                    for model_idx in range(len(all_models))
                )

                # Calculate averages
                num_models = len(all_models)
                seasonal_success_rates[algorithm].append(
                    season_success_total / num_models
                )
                seasonal_rewards[algorithm].append(season_reward_total / num_models)
                seasonal_uptime[algorithm].append(season_uptime_total / num_models)

        # Create seasonal visualization with bar charts (similar to other figures)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Figure 4.5: Seasonal Performance Analysis", fontsize=16, fontweight="bold"
        )

        # X-axis labels (seasons)
        x_positions = range(len(seasonal_dates))
        season_labels = [
            date.replace("2024-", "") for date in seasonal_dates
        ]  # "02-20", "05-20", etc.

        # Algorithm colors
        algorithm_colors = {
            "oracle": "#2E86AB",  # Blue
            "ml": "#A23B72",  # Purple
            "naive": "#F18F01",  # Orange
        }

        bar_width = 0.25

        # Plot data for each algorithm type
        for algorithm in ["oracle", "ml", "naive"]:
            # Calculate x positions for this algorithm's bars
            algorithm_index = list(algorithm_colors.keys()).index(algorithm)
            offset = (algorithm_index - 1) * bar_width  # -0.25, 0, +0.25
            x_algo = [x + offset for x in x_positions]

            # Panel 1: Success Rate by Season
            ax1.bar(
                x_algo,
                seasonal_success_rates[algorithm],
                bar_width,
                label=algorithm.capitalize(),
                color=algorithm_colors[algorithm],
                alpha=0.8,
            )

            # Panel 2: Total Reward by Season
            ax2.bar(
                x_algo,
                seasonal_rewards[algorithm],
                bar_width,
                label=algorithm.capitalize(),
                color=algorithm_colors[algorithm],
                alpha=0.8,
            )

            # Panel 3: Uptime Score by Season
            ax3.bar(
                x_algo,
                seasonal_uptime[algorithm],
                bar_width,
                label=algorithm.capitalize(),
                color=algorithm_colors[algorithm],
                alpha=0.8,
            )

        # Configure Panel 1: Success Rate by Season
        ax1.set_title("Average Success Rate", fontweight="bold")
        ax1.set_ylabel("Average Success Rate")
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(season_labels)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        self.format_percentage_axis(ax1)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Configure Panel 2: Total Reward by Season
        ax2.set_title("Average Total Reward", fontweight="bold")
        ax2.set_ylabel("Average Total Reward")
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(season_labels)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Configure Panel 3: Uptime Score by Season
        ax3.set_title("Average Uptime Score", fontweight="bold")
        ax3.set_ylabel("Average Uptime Score")
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(season_labels)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        self.format_percentage_axis(ax3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_5_seasonal_variation.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_seasonal_variation_chart(
        self, controller_name: str, seasonal_dates: List, save_path: str
    ):
        """Create seasonal variation chart showing performance across 4 dates."""
        # Extract model index from the full model list
        models = self.summary_data["graph_data"]["accuracy_metrics"]["models"]
        try:
            model_index = models.index(controller_name)
        except ValueError:
            print(f"Warning: Model {controller_name} not found in summary data")
            return

        # Use summary data to get seasonal performance directly
        models = self.summary_data["graph_data"]["accuracy_metrics"]["models"]
        try:
            model_index = models.index(controller_name)
        except ValueError:
            print(f"Warning: Model {controller_name} not found in summary data")
            return

        # Extract seasonal data from summary for this controller
        seasonal_success_rates = {"oracle": [], "ml": [], "naive": []}

        # Get seasonal dates and extract corresponding data
        seasonal_dates = self.summary_data["graph_data"]["metadata"]["test_dates"]

        # Get success rates for each seasonal date for this controller
        for algorithm in ["oracle", "ml", "naive"]:
            seasonal_success_rates[algorithm] = [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    algorithm
                ][model_index]
                for _ in seasonal_dates  # Same model, shown across seasons
            ]

        # Create seasonal visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Figure 4.5: Seasonal Variation Analysis", fontsize=16, fontweight="bold"
        )

        x = range(len(seasonal_dates))
        colors = {"oracle": "#2E86AB", "ml": "#A23B72", "naive": "#F18F01"}
        labels = [
            date.replace("2024-", "") for date in seasonal_dates
        ]  # "02-20", "05-20", etc.

        # Panel 1: Success Rate by Season
        for algorithm in ["oracle", "ml", "naive"]:
            ax1.plot(
                x,
                seasonal_success_rates[algorithm],
                marker="o",
                color=colors[algorithm],
                label=algorithm.capitalize(),
                linewidth=2,
            )
        ax1.set_title("Success Rate by Season", fontweight="bold")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        self.format_percentage_axis(ax1)
        ax1.legend()

        # Panel 2: Total Reward by Season
        seasonal_rewards = {"oracle": [], "ml": [], "naive": []}

        for algorithm in ["oracle", "ml", "naive"]:
            seasonal_rewards[algorithm] = [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    algorithm
                ][model_index]
                for _ in seasonal_dates  # Same model, shown across seasons
            ]

        for algorithm in ["oracle", "ml", "naive"]:
            ax2.plot(
                x,
                seasonal_rewards[algorithm],
                marker="o",
                color=colors[algorithm],
                label=algorithm.capitalize(),
                linewidth=2,
            )
        ax2.set_title("Total Reward by Season", fontweight="bold")
        ax2.set_ylabel("Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Panel 3: Uptime Score by Season
        seasonal_uptime = {"oracle": [], "ml": [], "naive": []}

        for algorithm in ["oracle", "ml", "naive"]:
            seasonal_uptime[algorithm] = [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    algorithm
                ][model_index]
                for _ in seasonal_dates  # Same model, shown across seasons
            ]

        for algorithm in ["oracle", "ml", "naive"]:
            ax3.plot(
                x,
                seasonal_uptime[algorithm],
                marker="o",
                color=colors[algorithm],
                label=algorithm.capitalize(),
                linewidth=2,
            )
        ax3.set_title("Uptime Score by Season", fontweight="bold")
        ax3.set_ylabel("Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        self.format_percentage_axis(ax3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

    def create_figure_4_6(self):
        """Figure 4.6: Controller Performance Gaps."""
        print("Creating Figure 4.6: Controller Performance Gaps...")

        if not self.summary_data:
            self.load_data()

        # Use all models to show comprehensive performance gaps
        all_models = self.summary_data["graph_data"]["accuracy_metrics"]["models"]

        title = "Figure 4.6: Controller Performance Gaps (Oracle vs ML vs Naive)"
        self.create_three_panel_horizontal_layout(
            title,
            all_models,
            os.path.join(self.output_dir, "figure_4_6_performance_gaps.png"),
        )

    def generate_consolidated_charts(self):
        """Generate all consolidated charts for Section 4 with new structure."""
        print("Generating consolidated charts for Section 4...")

        # Ensure data is loaded first
        if not self.summary_data:
            self.load_data()

        # Section 4.3: Overall Comparison
        self.create_figure_4_1()

        # Section 4.4: Targeted Ablation Studies
        self.create_figure_4_2_acc_lat_comparison()  # Accuracy vs Latency Comparison
        self.create_figure_4_3_reward_weight_comparison()  # Reward Weight Comparison
        self.create_figure_4_4_battery_comparison()  # Battery Configuration Comparison
        self.create_figure_4_5_seasonal_average()  # Seasonal Performance Analysis
        self.create_figure_4_6()  # Performance Gaps (All Controllers)

        print(f"All consolidated charts saved to {self.output_dir}")

    def generate_all_charts(self):
        """Generate all required charts for Section 4 (GRAPH.md implementation)."""
        print("Generating all charts for Section 4...")

        # Ensure data is loaded first
        if not self.summary_data:
            self.load_data()

        # Section 4.3: Overall Comparison
        self.create_figure_4_1()

        # Section 4.4: Targeted Ablation Studies
        self.create_figure_4_2_acc_lat_comparison()  # Accuracy vs Latency Comparison
        self.create_figure_4_3_reward_weight_comparison()  # Reward Weight Comparison
        self.create_figure_4_4_battery_comparison()  # Battery Configuration Comparison
        self.create_figure_4_5_seasonal_average()  # Seasonal Performance Analysis
        self.create_figure_4_6()  # Performance Gaps (All Controllers)

        print(f"All charts saved to {self.output_dir}")


def main():
    """Main function to run analysis."""
    try:
        # Create analyzer and generate charts
        analyzer = ResultsAnalyzer()
        analyzer.load_data()
        analyzer.generate_consolidated_charts()
        print("All charts generated successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
