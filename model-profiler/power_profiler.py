import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")


class PowerProfiler:
    """Comprehensive power measurement system using macOS powermetrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.power_profiles: Dict[str, Dict] = {}
        self.profiles_file = Path("model-data/power_profiles.json")

        # Require powermetrics - no fallback
        if not self._check_powermetrics_available():
            raise RuntimeError(
                "powermetrics is required and not available. This tool only works on macOS with powermetrics."
            )

    def _check_powermetrics_available(self) -> bool:
        """Check if powermetrics is available (macOS only)."""
        try:
            result = subprocess.run(
                ["which", "powermetrics"], capture_output=True, text=True
            )
            if result.returncode == 0:
                self.logger.info("Using powermetrics for accurate power measurement")
                return True
            else:
                self.logger.error("powermetrics not found - powermetrics is required")
                return False
        except Exception:
            self.logger.error(
                "Failed to check powermetrics availability - powermetrics is required"
            )
            return False

    def _sample_powermetrics(
        self, samples: List[Tuple[float, str]], interval: float = 0.1
    ):
        """Sample powermetrics output in a separate thread."""
        try:
            proc = subprocess.Popen(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power",
                    "-i",
                    str(int(interval * 1000)),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            start = time.time()

            if proc.stdout:
                for line in proc.stdout:
                    if "CPU Power" in line or "GPU Power" in line:
                        samples.append((time.time() - start, line.strip()))

            proc.terminate()
        except Exception as e:
            self.logger.error(f"Powermetrics sampling failed: {e}")

    def _parse_power_line(self, line: str) -> float:
        """Parse power from powermetrics output line."""
        try:
            # Extract power value from lines like "CPU Power: 112 mW"
            # or "Combined Power (CPU + GPU + ANE): 132 mW"
            match = re.search(r":\s*(\d+\.?\d*)\s*mW", line, re.IGNORECASE)
            if match:
                return float(match.group(1))

            # Handle cases with different units (W instead of mW)
            match = re.search(r":\s*(\d+\.?\d*)\s*W", line, re.IGNORECASE)
            if match:
                return float(match.group(1)) * 1000.0  # Convert W to mW

        except Exception as e:
            self.logger.error(f"Failed to parse power line '{line}': {e}")

        return 0.0

    def _measure_with_powermetrics(
        self, func, *args, **kwargs
    ) -> Tuple[float, List[Tuple[float, str]]]:
        """Measure power consumption during function execution using powermetrics."""
        import threading
        import time

        power_samples = []
        sampling = True

        def sample_power():
            try:
                # Get sudo password from environment
                sudo_password = os.getenv("SUDO_PASSWORD")
                if not sudo_password:
                    raise RuntimeError("SUDO_PASSWORD environment variable not set")

                # Start powermetrics with password from stdin
                proc = subprocess.Popen(
                    [
                        "sudo",
                        "-S",  # Read password from stdin
                        "powermetrics",
                        "--samplers",
                        "cpu_power,gpu_power",
                        "-i",
                        "500",  # 500ms interval for more stable readings
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                )

                # Send password to stdin
                if proc.stdin:
                    proc.stdin.write(sudo_password + "\n")
                    proc.stdin.flush()

                # Read all output and process line by line
                while sampling and proc.poll() is None:
                    try:
                        if proc.stdout:
                            line = proc.stdout.readline()
                            if not line:
                                break
                            line = line.strip()
                        else:
                            continue

                        # Look for Combined Power line which is most comprehensive
                        if "Combined Power (CPU + GPU + ANE):" in line:
                            power_value = self._parse_power_line(line)
                            if power_value > 0:
                                power_samples.append(power_value)
                                self.logger.debug(
                                    f"Combined Power sample: {power_value} mW"
                                )

                        # Also capture individual CPU and GPU power as fallback
                        elif "CPU Power:" in line:
                            power_value = self._parse_power_line(line)
                            if power_value > 0:
                                power_samples.append(power_value)
                                self.logger.debug(f"CPU Power sample: {power_value} mW")

                        elif "GPU Power:" in line:
                            power_value = self._parse_power_line(line)
                            if power_value > 0:
                                power_samples.append(power_value)
                                self.logger.debug(f"GPU Power sample: {power_value} mW")

                    except Exception as e:
                        self.logger.debug(f"Error reading line: {e}")
                        continue

                # Terminate process gracefully
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

            except Exception as e:
                self.logger.error(f"Powermetrics sampling failed: {e}")
                # Check if it's a sudo permission issue
                if "permission denied" in str(e).lower() or "sudo" in str(e).lower():
                    self.logger.error(
                        "Sudo permission issue. Please run 'sudo powermetrics --samplers cpu_power,gpu_power -i 1000' once to cache credentials."
                    )

        # Start sampling thread
        sampler_thread = threading.Thread(target=sample_power, daemon=True)
        sampler_thread.start()

        # Give powermetrics time to start and get first reading
        time.sleep(1.0)

        # Clear initial samples to get fresh baseline
        power_samples.clear()

        # Run the function
        func(*args, **kwargs)

        # Continue sampling for a bit more to capture power draw
        time.sleep(0.5)

        # Stop sampling
        sampling = False
        sampler_thread.join(timeout=3)

        # Log sample count for debugging
        self.logger.info(f"Collected {len(power_samples)} power samples")

        # Calculate median power
        if power_samples:
            sorted_samples = sorted(power_samples)
            mid = len(sorted_samples) // 2
            if len(sorted_samples) % 2 == 0:
                median_power = (sorted_samples[mid - 1] + sorted_samples[mid]) / 2
            else:
                median_power = sorted_samples[mid]
        else:
            raise RuntimeError(
                "No power samples collected - check powermetrics output format"
            )

        return median_power, []

    def benchmark_model_power(
        self, model_name: str, model_version: str, image_path: str, iterations: int = 50
    ) -> Dict:
        """
        Benchmark power consumption for a specific model.

        Args:
            model_name: Name of the model (e.g., "YOLOv10")
            model_version: Model version (e.g., "N", "S", "M", "B", "L", "X")
            image_path: Path to test image
            iterations: Number of inference iterations

        Returns:
            Dictionary with power profiling results
        """
        from yolo_model import YOLOModel

        self.logger.info(
            f"Benchmarking {model_name} v{model_version} power consumption"
        )

        # Create model instance
        model = YOLOModel(model_name, model_version)

        # Take baseline measurements using powermetrics
        def baseline_task():
            time.sleep(2.0)  # Longer baseline for stable measurement

        baseline_power, _ = self._measure_with_powermetrics(baseline_task)

        time.sleep(1)  # Let system stabilize

        # Load model
        self.logger.info(f"Loading {model_name} v{model_version} model...")
        model.load_model()
        self.logger.info(f"Model {model_name} v{model_version} loaded successfully")

        # Measure idle power using powermetrics
        def idle_task():
            time.sleep(2.0)  # Longer idle measurement for stability

        idle_power, _ = self._measure_with_powermetrics(idle_task)

        # Run inference benchmark
        start_time = time.time()
        inference_powers = []
        successful_inferences = 0

        self.logger.info(
            f"Starting {iterations} inference iterations for {model_name} v{model_version} on {image_path}"
        )

        for i in range(iterations):
            # Use powermetrics for each inference
            def inference_task():
                return model.run_inference(image_path)

            avg_power, samples = self._measure_with_powermetrics(inference_task)
            # Need to actually run inference to get result
            success, detections = model.run_inference(image_path)
            inference_powers.append(avg_power)

            if success:
                successful_inferences += 1
            else:
                self.logger.error(f"Inference failed on iteration {i + 1}")

            # Progress indicator for each iteration
            self.logger.info(
                f"[{model_name} v{model_version}] Iteration {i + 1}/{iterations} - Power: {avg_power:.2f} mW"
            )

            # Longer delay between iterations for system stabilization
            time.sleep(4.0)

        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate statistics with outlier removal for more stable results
        sorted_powers = sorted(inference_powers)
        # Remove top and bottom 20% as outliers for consistency
        if iterations >= 5:
            trim_count = max(1, iterations // 5)
            trimmed_powers = sorted_powers[trim_count:-trim_count]
        else:
            trimmed_powers = sorted_powers

        # Use median instead of average for inference power
        mid = len(trimmed_powers) // 2
        if len(trimmed_powers) % 2 == 0:
            avg_inference_power = (trimmed_powers[mid - 1] + trimmed_powers[mid]) / 2
        else:
            avg_inference_power = trimmed_powers[mid]
        max_inference_power = max(trimmed_powers)
        min_inference_power = min(trimmed_powers)

        # Calculate model-specific power (difference from baseline)
        # Use absolute difference to get actual additional power consumption
        model_power_mw = abs(avg_inference_power - baseline_power)

        # Calculate energy per inference using actual inference time (not total duration)
        avg_inference_time_seconds = total_duration / iterations
        energy_per_inference_mwh = (
            (model_power_mw * avg_inference_time_seconds) / 3600.0
            if model_power_mw > 0
            else 0.0
        )

        profile = {
            "model_name": model_name,
            "model_version": model_version,
            "baseline_power_mw": baseline_power,
            "idle_power_mw": idle_power,
            "avg_inference_power_mw": avg_inference_power,
            "max_inference_power_mw": max_inference_power,
            "min_inference_power_mw": min_inference_power,
            "model_power_mw": model_power_mw,
            "energy_per_inference_mwh": energy_per_inference_mwh,
            "iterations": iterations,
            "total_duration_seconds": total_duration,
            "avg_inference_time_seconds": avg_inference_time_seconds,
            "success_rate": successful_inferences / iterations,
            "outliers_removed": (
                iterations - len(trimmed_powers) if iterations >= 20 else 0
            ),
            "measurement_method": "powermetrics",
        }

        # Store profile
        profile_key = f"{model_name}_{model_version}"
        self.power_profiles[profile_key] = profile

        self.logger.info(
            f"Power benchmark complete for {profile_key}: {model_power_mw:.2f} mW ({profile['measurement_method']})"
        )

        return profile

    def benchmark_all_models(
        self, image_path: str, iterations: int = 50
    ) -> Dict[str, Dict]:
        """
        Benchmark all YOLOv10 models.

        Args:
            image_path: Path to test image
            iterations: Number of iterations per model

        Returns:
            Dictionary of all model profiles
        """
        models = [
            ("YOLOv10", "N"),
            ("YOLOv10", "S"),
            ("YOLOv10", "M"),
            ("YOLOv10", "B"),
            ("YOLOv10", "L"),
            ("YOLOv10", "X"),
        ]

        for model_name, model_version in models:
            try:
                self.benchmark_model_power(
                    model_name, model_version, image_path, iterations
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to benchmark {model_name} v{model_version}: {e}"
                )

        self.save_profiles()
        return self.power_profiles

    def save_profiles(self):
        """Save power profiles to file."""
        try:
            self.profiles_file.parent.mkdir(exist_ok=True)
            with open(self.profiles_file, "w") as f:
                json.dump(self.power_profiles, f, indent=2)
            self.logger.info(f"Power profiles saved to {self.profiles_file}")
        except Exception as e:
            self.logger.error(f"Failed to save profiles: {e}")

    def load_profiles(self) -> Dict[str, Dict]:
        """Load power profiles from file."""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, "r") as f:
                    self.power_profiles = json.load(f)
                self.logger.info(f"Loaded {len(self.power_profiles)} power profiles")
            else:
                self.logger.info("No existing power profiles found")
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}")

        return self.power_profiles

    def get_model_power(self, model_name: str, model_version: str) -> float:
        """
        Get power consumption for a specific model.

        Args:
            model_name: Name of the model
            model_version: Model version

        Returns:
            Power consumption in milliwatts
        """
        profile_key = f"{model_name}_{model_version}"
        if profile_key in self.power_profiles:
            return self.power_profiles[profile_key]["model_power_mw"]

        raise RuntimeError(
            f"No power profile found for {profile_key}. Run benchmark_power.py first."
        )

    def get_all_models_data(self) -> Dict[str, Dict]:
        """Get all loaded models data."""
        return self.power_profiles


def load_power_profiles() -> Dict[str, Dict]:
    """Load power profiles from file."""
    profiles_file = Path("model-data/power_profiles.json")
    try:
        if profiles_file.exists():
            with open(profiles_file, "r") as f:
                power_profiles = json.load(f)
            logging.info(f"Loaded {len(power_profiles)} power profiles")
            return power_profiles
        else:
            logging.info("No existing power profiles found")
            return {}
    except Exception as e:
        logging.error(f"Failed to load profiles: {e}")
        return {}
