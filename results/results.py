#!/usr/bin/env python3
"""
Results Analysis Script for Optimal Charge Security Camera

Generates all charts required for Section 4 (Evaluation & Results) of report.
This script processes batch simulation results and creates comprehensive performance visualizations.

"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict
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
    """Main class for analyzing batch results and generating charts."""

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
        self.controller_mapping = self.config["controller_mapping"]

        # Initialize data containers
        self.batch_data = None
        self.summary_data = None
        self.model_configs = {}

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

        # Load individual batch results for detailed analysis
        self._load_individual_results()

        print(f"Loaded data for {len(self.model_configs)} model configurations")

    def _parse_model_configurations(self):
        """Parse model filenames to extract configuration parameters."""
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
            config["short_name"] = self._get_short_name(config)

            self.model_configs[model_file] = config

    def _get_config_type(self, config: Dict) -> str:
        """Determine configuration type for grouping."""
        if config["success_weight"] == 20:
            return "performance_focused"
        elif config["carbon_weight"] == 15:
            return "carbon_focused"
        else:
            return "mixed"

    def _get_short_name(self, config: Dict) -> str:
        """Extract controller short name from prefixed model filename."""
        model_file = config.get("model_file", "")

        # Extract C1, C2, etc. from beginning of filename
        if model_file.startswith("C"):
            # Format: "C1_controller_acc..." -> "C1"
            return model_file.split("_")[0]

        # Fallback to original logic for backward compatibility
        if model_file.startswith("controller_controller_"):
            mapping_key = model_file.replace("controller_controller_", "")
        else:
            mapping_key = model_file

        # Look up in controller mapping
        return self.controller_mapping.get(mapping_key, "Unknown")

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

    def generate_all_charts(self):
        """Generate all required charts for Section 4."""
        print("Generating all charts for Section 4...")

        # Figure 4.1: Controller Performance Comparison
        self.create_figure_4_1()

        # Figures 4.2-4.3: Accuracy/Latency Threshold Studies
        self.create_figure_4_2()
        self.create_figure_4_3()

        # Figures 4.4-4.5: Reward Weight Analysis
        self.create_figure_4_4()
        self.create_figure_4_5()

        # Figures 4.6-4.7: Battery Configuration Impact
        self.create_figure_4_6()
        self.create_figure_4_7()

        print(f"All charts saved to {self.output_dir}")

    def format_percentage_axis(self, ax):
        """Format y-axis as percentage."""
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

    def create_figure_4_1(self):
        """Figure 4.1: Controller Performance Comparison across all model configurations."""
        print("Creating Figure 4.1: Controller Performance Comparison...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.1: Controller Performance Comparison Across All Models",
            fontsize=16,
            fontweight="bold",
        )

        models = self.summary_data["graph_data"]["accuracy_metrics"]["models"]

        # Ensure model_configs is populated
        if not self.model_configs:
            self._parse_model_configurations()

        short_names = [self.model_configs[m]["short_name"] for m in models]

        # Success Rates
        success_rates = {
            "Oracle": self.summary_data["graph_data"]["accuracy_metrics"][
                "success_rates"
            ]["oracle"],
            "ML": self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                "ml"
            ],
            "Naive": self.summary_data["graph_data"]["accuracy_metrics"][
                "success_rates"
            ]["naive"],
        }

        x = list(range(len(short_names)))
        width = 0.25

        # Oracle bars
        ax1.bar(
            [i - width for i in x],
            success_rates["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )

        # ML bars with a single label but individual controller names on x-axis
        ax1.bar(x, success_rates["ML"], width, label="ML Controllers", color="#A23B72")

        # Naive bars
        ax1.bar(
            [i + width for i in x],
            success_rates["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax1.set_title("Success Rates by Model Configuration")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        self.format_percentage_axis(ax1)

        # Total Rewards
        rewards = {
            "Oracle": self.summary_data["graph_data"]["utility_comparison"][
                "total_rewards"
            ]["oracle"],
            "ML": self.summary_data["graph_data"]["utility_comparison"][
                "total_rewards"
            ]["ml"],
            "Naive": self.summary_data["graph_data"]["utility_comparison"][
                "total_rewards"
            ]["naive"],
        }

        ax2.bar(
            [i - width for i in x],
            rewards["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )

        # ML bars with single label but individual controller names on x-axis
        ax2.bar(x, rewards["ML"], width, label="ML Controllers", color="#A23B72")

        ax2.bar(
            [i + width for i in x],
            rewards["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax2.set_title("Total Utility Rewards")
        ax2.set_ylabel("Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha="right")
        ax2.legend()
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Uptime Metrics
        uptime = {
            "Oracle": self.summary_data["graph_data"]["uptime_metrics"][
                "uptime_scores"
            ]["oracle"],
            "ML": self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                "ml"
            ],
            "Naive": self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                "naive"
            ],
        }

        ax3.bar(
            [i - width for i in x],
            uptime["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )

        # ML bars with specific controller names
        for i, short_name in enumerate(short_names):
            ax3.bar(i, uptime["ML"][i], width, label=short_name, color="#A23B72")

        ax3.bar(
            [i + width for i in x],
            uptime["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax3.set_title("Uptime Metrics")
        ax3.set_ylabel("Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha="right")
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        self.format_percentage_axis(ax3)

        # Performance Gap Analysis (ML vs Oracle)
        ml_success = list(success_rates["ML"])
        oracle_success = list(success_rates["Oracle"])
        performance_gap = [
            oracle_success[i] - ml_success[i] for i in range(len(ml_success))
        ]

        ax4.bar(x, performance_gap, color="#E63946", alpha=0.7)
        ax4.set_title("ML Controllers vs Oracle Performance Gap")
        ax4.set_ylabel("Gap (Oracle - ML)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(short_names, rotation=45, ha="right")
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir, "figure_4_1_controller_performance_comparison.png"
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_2(self):
        """Figure 4.2: Success rates by accuracy threshold."""
        print("Creating Figure 4.2: Success Rates by Accuracy Threshold...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.2: Success Rates by Accuracy Threshold",
            fontsize=16,
            fontweight="bold",
        )

        # Group by accuracy threshold
        models = list(self.model_configs.keys())
        acc_819_indices = [
            i
            for i, m in enumerate(models)
            if self.model_configs[m]["accuracy_threshold"] == 0.819
        ]
        acc_950_indices = [
            i
            for i, m in enumerate(models)
            if self.model_configs[m]["accuracy_threshold"] == 0.95
        ]

        # Calculate average success rates by accuracy threshold
        success_rates_oracle = self.summary_data["graph_data"]["accuracy_metrics"][
            "success_rates"
        ]["oracle"]
        success_rates_ml = self.summary_data["graph_data"]["accuracy_metrics"][
            "success_rates"
        ]["ml"]
        success_rates_naive = self.summary_data["graph_data"]["accuracy_metrics"][
            "success_rates"
        ]["naive"]

        acc_819_oracle = [success_rates_oracle[i] for i in acc_819_indices]
        acc_950_oracle = [success_rates_oracle[i] for i in acc_950_indices]
        acc_819_ml = [success_rates_ml[i] for i in acc_819_indices]
        acc_950_ml = [success_rates_ml[i] for i in acc_950_indices]
        acc_819_naive = [success_rates_naive[i] for i in acc_819_indices]
        acc_950_naive = [success_rates_naive[i] for i in acc_950_indices]

        # Plot 1: Success rates by accuracy threshold
        thresholds = ["81.9%", "95.0%"]
        oracle_avg = [
            sum(acc_819_oracle) / len(acc_819_oracle),
            sum(acc_950_oracle) / len(acc_950_oracle),
        ]
        ml_avg = [sum(acc_819_ml) / len(acc_819_ml), sum(acc_950_ml) / len(acc_950_ml)]
        naive_avg = [
            sum(acc_819_naive) / len(acc_819_naive),
            sum(acc_950_naive) / len(acc_950_naive),
        ]

        x = list(range(len(thresholds)))
        width = 0.25

        ax1.bar(
            [i - width for i in x], oracle_avg, width, label="Oracle", color="#2E86AB"
        )
        ax1.bar(x, ml_avg, width, label="ML Controllers", color="#A23B72")
        ax1.bar(
            [i + width for i in x], naive_avg, width, label="Naive", color="#F18F01"
        )
        ax1.set_title("Average Success Rates by Accuracy Threshold")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(thresholds)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        self.format_percentage_axis(ax1)

        # Plot 2: Performance gap by accuracy threshold
        gap_819 = sum(acc_819_oracle) / len(acc_819_oracle) - sum(acc_819_ml) / len(
            acc_819_ml
        )
        gap_950 = sum(acc_950_oracle) / len(acc_950_oracle) - sum(acc_950_ml) / len(
            acc_950_ml
        )

        ax2.bar(thresholds, [gap_819, gap_950], color=["#2E86AB", "#A23B72"])
        ax2.set_title("Oracle-ML Performance Gap by Accuracy Threshold")
        ax2.set_ylabel("Performance Gap")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Plot 3: Success rate distribution
        ax3.boxplot([acc_819_oracle, acc_950_oracle], labels=["81.9%", "95.0%"])
        ax3.set_title("Oracle Success Rate Distribution")
        ax3.set_ylabel("Success Rate")
        self.format_percentage_axis(ax3)

        # Plot 4: Threshold impact on all controllers
        ax4.plot(
            thresholds, oracle_avg, "o-", label="Oracle", color="#2E86AB", linewidth=2
        )
        ax4.plot(
            thresholds,
            ml_avg,
            "s-",
            label="ML Controllers",
            color="#A23B72",
            linewidth=2,
        )
        ax4.plot(
            thresholds, naive_avg, "^-", label="Naive", color="#F18F01", linewidth=2
        )
        ax4.set_title("Success Rate vs Accuracy Threshold")
        ax4.set_ylabel("Success Rate")
        ax4.set_xlabel("Accuracy Threshold")
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_2_accuracy_threshold_analysis.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_3(self):
        """Figure 4.3: Performance metrics by latency threshold."""
        print("Creating Figure 4.3: Performance Metrics by Latency Threshold...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.3: Performance Metrics by Latency Threshold",
            fontsize=16,
            fontweight="bold",
        )

        # Group by latency threshold
        models = list(self.model_configs.keys())
        lat_006_indices = [
            i
            for i, m in enumerate(models)
            if self.model_configs[m]["latency_threshold"] == 0.006
        ]
        lat_012_indices = [
            i
            for i, m in enumerate(models)
            if self.model_configs[m]["latency_threshold"] == 0.015
        ]

        # Get data
        success_rates = self.summary_data["graph_data"]["accuracy_metrics"][
            "success_rates"
        ]["oracle"]
        rewards = self.summary_data["graph_data"]["utility_comparison"][
            "total_rewards"
        ]["oracle"]
        uptime = self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
            "oracle"
        ]

        lat_006_success = [success_rates[i] for i in lat_006_indices]
        lat_012_success = [success_rates[i] for i in lat_012_indices]
        lat_006_rewards = [rewards[i] for i in lat_006_indices]
        lat_012_rewards = [rewards[i] for i in lat_012_indices]
        lat_006_uptime = [uptime[i] for i in lat_006_indices]
        lat_012_uptime = [uptime[i] for i in lat_012_indices]

        # Plot 1: Success Rate by Latency
        latency_labels = ["6ms", "12ms"]
        success_avg = [
            sum(lat_006_success) / len(lat_006_success),
            sum(lat_012_success) / len(lat_012_success),
        ]

        ax1.bar(latency_labels, success_avg, color=["#2E86AB", "#A23B72"])
        ax1.set_title("Success Rate by Latency Threshold")
        ax1.set_ylabel("Success Rate")
        self.format_percentage_axis(ax1)

        # Plot 2: Total Reward by Latency
        reward_avg = [
            sum(lat_006_rewards) / len(lat_006_rewards),
            sum(lat_012_rewards) / len(lat_012_rewards),
        ]

        ax2.bar(latency_labels, reward_avg, color=["#2E86AB", "#A23B72"])
        ax2.set_title("Total Reward by Latency Threshold")
        ax2.set_ylabel("Total Reward")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Plot 3: Uptime by Latency
        uptime_avg = [
            sum(lat_006_uptime) / len(lat_006_uptime),
            sum(lat_012_uptime) / len(lat_012_uptime),
        ]

        ax3.bar(latency_labels, uptime_avg, color=["#2E86AB", "#A23B72"])
        ax3.set_title("Uptime by Latency Threshold")
        ax3.set_ylabel("Uptime Score")
        self.format_percentage_axis(ax3)

        # Plot 4: Combined performance radar-like plot
        categories = ["Success Rate", "Reward Score", "Uptime Score"]
        lat_006_normalized = [
            success_avg[0],
            (reward_avg[0] - min(reward_avg)) / (max(reward_avg) - min(reward_avg))
            if max(reward_avg) != min(reward_avg)
            else 0.5,
            uptime_avg[0],
        ]
        lat_012_normalized = [
            success_avg[1],
            (reward_avg[1] - min(reward_avg)) / (max(reward_avg) - min(reward_avg))
            if max(reward_avg) != min(reward_avg)
            else 0.5,
            uptime_avg[1],
        ]

        x = list(range(len(categories)))
        width = 0.35

        ax4.bar(
            [i - width / 2 for i in x],
            lat_006_normalized,
            width,
            label="6ms",
            color="#2E86AB",
            alpha=0.7,
        )
        ax4.bar(
            [i + width / 2 for i in x],
            lat_012_normalized,
            width,
            label="12ms",
            color="#A23B72",
            alpha=0.7,
        )
        ax4.set_title("Normalized Performance Comparison")
        ax4.set_ylabel("Normalized Score (0-1)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_3_latency_threshold_analysis.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_4(self):
        """Figure 4.4: Performance-focused reward weights results."""
        print("Creating Figure 4.4: Performance-Focused Reward Weights...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.4: Performance-Focused Reward Weights (Success=20, Small Miss=5, Large Miss=8, Carbon=7)",
            fontsize=16,
            fontweight="bold",
        )

        # Filter performance-focused models
        models = list(self.model_configs.keys())
        perf_models = [
            m
            for m, config in self.model_configs.items()
            if config["success_weight"] == 20
        ]
        perf_indices = [models.index(m) for m in perf_models]

        # Success rates
        perf_success = {
            "Oracle": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "oracle"
                ][i]
                for i in perf_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "ml"
                ][i]
                for i in perf_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "naive"
                ][i]
                for i in perf_indices
            ],
        }

        short_names = [self.model_configs[m]["short_name"] for m in perf_models]
        x = list(range(len(short_names)))
        width = 0.25

        ax1.bar(
            [i - width for i in x],
            perf_success["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax1.bar(x, perf_success["ML"], width, label="ML Controllers", color="#A23B72")
        ax1.bar(
            [i + width for i in x],
            perf_success["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax1.set_title("Success Rates")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        self.format_percentage_axis(ax1)

        # Total rewards
        perf_rewards = {
            "Oracle": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "oracle"
                ][i]
                for i in perf_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "ml"
                ][i]
                for i in perf_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "naive"
                ][i]
                for i in perf_indices
            ],
        }

        ax2.bar(
            [i - width for i in x],
            perf_rewards["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax2.bar(x, perf_rewards["ML"], width, label="ML Controllers", color="#A23B72")
        ax2.bar(
            [i + width for i in x],
            perf_rewards["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax2.set_title("Total Utility Rewards")
        ax2.set_ylabel("Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha="right")
        ax2.legend()
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Uptime metrics
        perf_uptime = {
            "Oracle": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "oracle"
                ][i]
                for i in perf_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "ml"
                ][i]
                for i in perf_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "naive"
                ][i]
                for i in perf_indices
            ],
        }

        ax3.bar(
            [i - width for i in x],
            perf_uptime["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax3.bar(x, perf_uptime["ML"], width, label="ML Controllers", color="#A23B72")
        ax3.bar(
            [i + width for i in x],
            perf_uptime["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax3.set_title("Uptime Metrics")
        ax3.set_ylabel("Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha="right")
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        self.format_percentage_axis(ax3)

        # Performance efficiency (combined metric)
        oracle_efficiency = [
            (s + u) / 2 for s, u in zip(perf_success["Oracle"], perf_uptime["Oracle"])
        ]
        ml_efficiency = [
            (s + u) / 2 for s, u in zip(perf_success["ML"], perf_uptime["ML"])
        ]
        naive_efficiency = [
            (s + u) / 2 for s, u in zip(perf_success["Naive"], perf_uptime["Naive"])
        ]

        ax4.bar(
            [i - width for i in x],
            oracle_efficiency,
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax4.bar(x, ml_efficiency, width, label="ML Controllers", color="#A23B72")
        ax4.bar(
            [i + width for i in x],
            naive_efficiency,
            width,
            label="Naive",
            color="#F18F01",
        )
        ax4.set_title("Performance Efficiency (Success + Uptime)/2")
        ax4.set_ylabel("Efficiency Score")
        ax4.set_xticks(x)
        ax4.set_xticklabels(short_names, rotation=45, ha="right")
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        self.format_percentage_axis(ax4)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_4_performance_focused_rewards.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_5(self):
        """Figure 4.5: Carbon-focused reward weights results."""
        print("Creating Figure 4.5: Carbon-Focused Reward Weights...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.5: Carbon-Focused Reward Weights (Success=5, Small Miss=7, Large Miss=10, Carbon=15)",
            fontsize=16,
            fontweight="bold",
        )

        # Filter carbon-focused models
        models = list(self.model_configs.keys())
        carbon_models = [
            m
            for m, config in self.model_configs.items()
            if config["carbon_weight"] == 15
        ]
        carbon_indices = [models.index(m) for m in carbon_models]

        # Success rates
        carbon_success = {
            "Oracle": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "oracle"
                ][i]
                for i in carbon_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "ml"
                ][i]
                for i in carbon_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "naive"
                ][i]
                for i in carbon_indices
            ],
        }

        short_names = [self.model_configs[m]["short_name"] for m in carbon_models]
        x = list(range(len(short_names)))
        width = 0.25

        ax1.bar(
            [i - width for i in x],
            carbon_success["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax1.bar(x, carbon_success["ML"], width, label="ML Controllers", color="#A23B72")
        ax1.bar(
            [i + width for i in x],
            carbon_success["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax1.set_title("Success Rates")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        self.format_percentage_axis(ax1)

        # Total rewards
        carbon_rewards = {
            "Oracle": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "oracle"
                ][i]
                for i in carbon_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "ml"
                ][i]
                for i in carbon_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "naive"
                ][i]
                for i in carbon_indices
            ],
        }

        ax2.bar(
            [i - width for i in x],
            carbon_rewards["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax2.bar(x, carbon_rewards["ML"], width, label="ML Controllers", color="#A23B72")
        ax2.bar(
            [i + width for i in x],
            carbon_rewards["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax2.set_title("Total Utility Rewards")
        ax2.set_ylabel("Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha="right")
        ax2.legend()
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Uptime metrics
        carbon_uptime = {
            "Oracle": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "oracle"
                ][i]
                for i in carbon_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "ml"
                ][i]
                for i in carbon_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "naive"
                ][i]
                for i in carbon_indices
            ],
        }

        ax3.bar(
            [i - width for i in x],
            carbon_uptime["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax3.bar(x, carbon_uptime["ML"], width, label="ML Controllers", color="#A23B72")
        ax3.bar(
            [i + width for i in x],
            carbon_uptime["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax3.set_title("Uptime Metrics")
        ax3.set_ylabel("Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha="right")
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        self.format_percentage_axis(ax3)

        # Carbon efficiency (inverse of negative rewards)
        oracle_carbon_eff = [max(0, -r / 100) for r in carbon_rewards["Oracle"]]
        ml_carbon_eff = [max(0, -r / 100) for r in carbon_rewards["ML"]]
        naive_carbon_eff = [max(0, -r / 100) for r in carbon_rewards["Naive"]]

        ax4.bar(
            [i - width for i in x],
            oracle_carbon_eff,
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax4.bar(x, ml_carbon_eff, width, label="ML Controllers", color="#A23B72")
        ax4.bar(
            [i + width for i in x],
            naive_carbon_eff,
            width,
            label="Naive",
            color="#F18F01",
        )
        ax4.set_title("Carbon Efficiency Score")
        ax4.set_ylabel("Efficiency Score")
        ax4.set_xticks(x)
        ax4.set_xticklabels(short_names, rotation=45, ha="right")
        ax4.legend()
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.1%}"))

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_5_carbon_focused_rewards.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_6(self):
        """Figure 4.6: Large battery configuration results."""
        print("Creating Figure 4.6: Large Battery Configuration...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.6: Large Battery Configuration (610 mWh, 0.000269 mWh/s)",
            fontsize=16,
            fontweight="bold",
        )

        # Filter large battery models
        models = list(self.model_configs.keys())
        large_battery_models = [
            m
            for m, config in self.model_configs.items()
            if config["battery_capacity"] == 610
        ]
        large_battery_indices = [models.index(m) for m in large_battery_models]

        # Success rates
        large_success = {
            "Oracle": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "oracle"
                ][i]
                for i in large_battery_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "ml"
                ][i]
                for i in large_battery_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "naive"
                ][i]
                for i in large_battery_indices
            ],
        }

        short_names = [
            self.model_configs[m]["short_name"] for m in large_battery_models
        ]
        x = list(range(len(short_names)))
        width = 0.25

        ax1.bar(
            [i - width for i in x],
            large_success["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax1.bar(x, large_success["ML"], width, label="ML Controllers", color="#A23B72")
        ax1.bar(
            [i + width for i in x],
            large_success["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax1.set_title("Success Rates")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        self.format_percentage_axis(ax1)

        # Total rewards
        large_rewards = {
            "Oracle": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "oracle"
                ][i]
                for i in large_battery_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "ml"
                ][i]
                for i in large_battery_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "naive"
                ][i]
                for i in large_battery_indices
            ],
        }

        ax2.bar(
            [i - width for i in x],
            large_rewards["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax2.bar(x, large_rewards["ML"], width, label="ML Controllers", color="#A23B72")
        ax2.bar(
            [i + width for i in x],
            large_rewards["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax2.set_title("Total Utility Rewards")
        ax2.set_ylabel("Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha="right")
        ax2.legend()
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Uptime metrics
        large_uptime = {
            "Oracle": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "oracle"
                ][i]
                for i in large_battery_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "ml"
                ][i]
                for i in large_battery_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "naive"
                ][i]
                for i in large_battery_indices
            ],
        }

        ax3.bar(
            [i - width for i in x],
            large_uptime["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax3.bar(x, large_uptime["ML"], width, label="ML Controllers", color="#A23B72")
        ax3.bar(
            [i + width for i in x],
            large_uptime["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax3.set_title("Uptime Metrics")
        ax3.set_ylabel("Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha="right")
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        self.format_percentage_axis(ax3)

        # Energy utilization (based on uptime and rewards)
        oracle_util = [
            (s + u) / 2 for s, u in zip(large_success["Oracle"], large_uptime["Oracle"])
        ]
        ml_util = [(s + u) / 2 for s, u in zip(large_success["ML"], large_uptime["ML"])]
        naive_util = [
            (s + u) / 2 for s, u in zip(large_success["Naive"], large_uptime["Naive"])
        ]

        ax4.bar(
            [i - width for i in x],
            oracle_util,
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax4.bar(x, ml_util, width, label="ML Controllers", color="#A23B72")
        ax4.bar(
            [i + width for i in x],
            naive_util,
            width,
            label="Naive",
            color="#F18F01",
        )
        ax4.set_title("Energy Utilization Efficiency")
        ax4.set_ylabel("Utilization Score")
        ax4.set_xticks(x)
        ax4.set_xticklabels(short_names, rotation=45, ha="right")
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        self.format_percentage_axis(ax4)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_6_large_battery_config.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_figure_4_7(self):
        """Figure 4.7: Small battery configuration results."""
        print("Creating Figure 4.7: Small Battery Configuration...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Figure 4.7: Small Battery Configuration (105 mWh, 0.001598 mWh/s)",
            fontsize=16,
            fontweight="bold",
        )

        # Filter small battery models
        models = list(self.model_configs.keys())
        small_battery_models = [
            m
            for m, config in self.model_configs.items()
            if config["battery_capacity"] == 105
        ]
        small_battery_indices = [models.index(m) for m in small_battery_models]

        # Success rates
        small_success = {
            "Oracle": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "oracle"
                ][i]
                for i in small_battery_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "ml"
                ][i]
                for i in small_battery_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["accuracy_metrics"]["success_rates"][
                    "naive"
                ][i]
                for i in small_battery_indices
            ],
        }

        short_names = [
            self.model_configs[m]["short_name"] for m in small_battery_models
        ]
        x = list(range(len(short_names)))
        width = 0.25

        ax1.bar(
            [i - width for i in x],
            small_success["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax1.bar(x, small_success["ML"], width, label="ML Controllers", color="#A23B72")
        ax1.bar(
            [i + width for i in x],
            small_success["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax1.set_title("Success Rates")
        ax1.set_ylabel("Success Rate")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        self.format_percentage_axis(ax1)

        # Total rewards
        small_rewards = {
            "Oracle": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "oracle"
                ][i]
                for i in small_battery_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "ml"
                ][i]
                for i in small_battery_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["utility_comparison"]["total_rewards"][
                    "naive"
                ][i]
                for i in small_battery_indices
            ],
        }

        ax2.bar(
            [i - width for i in x],
            small_rewards["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax2.bar(x, small_rewards["ML"], width, label="ML Controllers", color="#A23B72")
        ax2.bar(
            [i + width for i in x],
            small_rewards["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax2.set_title("Total Utility Rewards")
        ax2.set_ylabel("Total Reward")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha="right")
        ax2.legend()
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.0f}"))

        # Uptime metrics
        small_uptime = {
            "Oracle": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "oracle"
                ][i]
                for i in small_battery_indices
            ],
            "ML": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "ml"
                ][i]
                for i in small_battery_indices
            ],
            "Naive": [
                self.summary_data["graph_data"]["uptime_metrics"]["uptime_scores"][
                    "naive"
                ][i]
                for i in small_battery_indices
            ],
        }

        ax3.bar(
            [i - width for i in x],
            small_uptime["Oracle"],
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax3.bar(x, small_uptime["ML"], width, label="ML Controllers", color="#A23B72")
        ax3.bar(
            [i + width for i in x],
            small_uptime["Naive"],
            width,
            label="Naive",
            color="#F18F01",
        )
        ax3.set_title("Uptime Metrics")
        ax3.set_ylabel("Uptime Score")
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha="right")
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        self.format_percentage_axis(ax3)

        # Power efficiency (inversely related to charging frequency)
        oracle_efficiency = [
            s * u for s, u in zip(small_success["Oracle"], small_uptime["Oracle"])
        ]
        ml_efficiency = [s * u for s, u in zip(small_success["ML"], small_uptime["ML"])]
        naive_efficiency = [
            s * u for s, u in zip(small_success["Naive"], small_uptime["Naive"])
        ]

        ax4.bar(
            [i - width for i in x],
            oracle_efficiency,
            width,
            label="Oracle",
            color="#2E86AB",
        )
        ax4.bar(x, ml_efficiency, width, label="ML Controllers", color="#A23B72")
        ax4.bar(
            [i + width for i in x],
            naive_efficiency,
            width,
            label="Naive",
            color="#F18F01",
        )
        ax4.set_title("Power Efficiency (Success × Uptime)")
        ax4.set_ylabel("Efficiency Score")
        ax4.set_xticks(x)
        ax4.set_xticklabels(short_names, rotation=45, ha="right")
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        self.format_percentage_axis(ax4)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "figure_4_7_small_battery_config.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def main():
    """Main function to run the analysis."""
    try:
        # Create analyzer and generate charts
        analyzer = ResultsAnalyzer()
        analyzer.load_data()
        analyzer.generate_all_charts()
        print("All charts generated successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
