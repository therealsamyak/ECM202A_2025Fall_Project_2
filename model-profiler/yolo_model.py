import logging
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import psutil
from ultralytics import YOLO


class YOLOModel:
    """Wrapper for YOLOv10 models with power profiling capabilities."""

    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.power_profile_mw: Optional[float] = None
        self.logger = logging.getLogger(__name__)

        # Load model specifications from model-data
        self.model_specs = self._load_model_specs()

    def _load_model_specs(self) -> Dict[str, float]:
        """Load model specifications from model-data CSV."""
        try:
            df = pd.read_csv("model-data/model-data.csv")
            model_row = df[
                (df["model"] == self.model_name) & (df["version"] == self.model_version)
            ]

            if not model_row.empty:
                row_data = model_row.iloc[0]
                return {
                    "latency_ms": float(row_data["Latency T4 TensorRT10 FP16(ms/img)"]),
                    "accuracy_map": float(row_data["COCO mAP 50-95"]),
                }
        except Exception as e:
            self.logger.error(f"Failed to load model specs: {e}")

        return {"latency_ms": 0.0, "accuracy_map": 0.0}

    def load_model(self):
        """Load the YOLO model."""
        try:
            model_path = f"yolov10{self.model_version.lower()}.pt"
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded {self.model_name} v{self.model_version}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def benchmark_power_consumption(
        self, image_path: str, iterations: int = 10
    ) -> float:
        """
        Benchmark power consumption for this model.

        Args:
            image_path: Path to test image
            iterations: Number of iterations to run

        Returns:
            Average power consumption in milliwatts
        """
        if self.model is None:
            self.load_model()

        # Get initial power state
        _ = psutil.cpu_percent(interval=1)

        # Run inference multiple times
        start_time = time.time()

        for _ in range(iterations):
            try:
                results = self.model(image_path)
                # Process results to ensure actual computation
                if results and len(results) > 0:
                    _ = len(results[0].boxes)
                else:
                    _ = 0
            except Exception as e:
                self.logger.error(f"Inference failed: {e}")
                continue

        end_time = time.time()
        _ = end_time - start_time

        # Get power consumption during inference
        # This is a simplified approach - real power measurement would need more sophisticated tools
        avg_cpu_percent = psutil.cpu_percent(interval=1)

        # Estimate power based on CPU usage and system power
        # This is a rough approximation - real measurement would use hardware-specific APIs
        cpu_freq = psutil.cpu_freq()
        system_power_watts = cpu_freq.current if cpu_freq else 65  # Default estimate
        estimated_power_mw = (avg_cpu_percent / 100.0) * system_power_watts * 1000

        self.power_profile_mw = estimated_power_mw
        self.logger.info(
            f"Power benchmark for {self.model_name} v{self.model_version}: {estimated_power_mw:.2f} mW"
        )

        return estimated_power_mw

    def run_inference(self, image_path: str) -> Tuple[bool, int]:
        """
        Run inference on an image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (success, detection_count)
        """
        if self.model is None:
            self.load_model()

        try:
            results = self.model(image_path)
            if results and len(results) > 0:
                detection_count = len(results[0].boxes)
            else:
                detection_count = 0
            return True, detection_count
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return False, 0

    def get_power_consumption_mw(self) -> float:
        """Get power consumption in milliwatts."""
        return self.power_profile_mw if self.power_profile_mw else 0.0

    def get_specs(self) -> Dict[str, float]:
        """Get model specifications."""
        return self.model_specs.copy()
