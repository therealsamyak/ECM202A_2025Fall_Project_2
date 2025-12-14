#!/usr/bin/env python3
"""
Power benchmarking script for all YOLOv10 models.
"""

import json
import logging
import sys
from pathlib import Path

from utils.logging_config import setup_logging
from power_profiler import PowerProfiler


def main():
    """Run power benchmarking for all models."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    with open("model-profiler/model.config.json", "r") as f:
        config = json.load(f)
    iterations = config.get("iterations", 1)
    iter_sleep_sec = config.get("iter_sleep_sec", 4.0)

    # Initialize power profiler
    profiler = PowerProfiler()

    # Check if benchmark images exist
    image1 = Path("model-profiler/benchmark-images/image1.png")
    image2 = Path("model-profiler/benchmark-images/image2.jpeg")

    if not image1.exists():
        logger.error(f"Benchmark image not found: {image1}")
        return 1

    if not image2.exists():
        logger.error(f"Benchmark image not found: {image2}")
        return 1

    # Load existing profiles if any
    profiler.load_profiles()

    # Benchmark all models on image2 only
    logger.info(f"Benchmarking models on {image2} with {iterations} iterations")
    profiles = profiler.benchmark_all_models(
        str(image2), iterations=iterations, iter_sleep_sec=iter_sleep_sec
    )

    # Print results
    print(f"\nPower Benchmark Results for {image2.name}:")
    print("-" * 60)
    for model_key, profile in profiles.items():
        print(f"{model_key}:")
        print(f"  Model Power: {profile['model_power_mw']:.2f} mW")
        print(f"  Energy per Inference: {profile['energy_per_inference_mwh']:.6f} mWh")
        print(f"  Avg Inference Time: {profile['avg_inference_time_seconds']:.3f} s")
        print(f"  Success Rate: {profile['success_rate']:.2%}")
        print()

    logger.info("Power benchmarking complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
