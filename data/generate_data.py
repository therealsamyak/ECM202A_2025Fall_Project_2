#!/usr/bin/env python3
"""
Main entry point for training data generation.
Processes all parameter combinations sequentially.
"""

import traceback
import os
import time
from datetime import datetime

from generation_config import ConfigLoader
from oracle_runner import OracleRunner
from recombine_data import DataRecombiner


def main():
    """Generate training data by processing all combinations sequentially"""
    print("=" * 60)
    print("TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Load configuration and generate combinations
        print("Loading configuration...")
        config_loader = ConfigLoader("data/data.config.json")
        config = config_loader.load_config()
        combinations = config_loader.generate_parameter_combinations()

        print("✓ Configuration loaded")
        print(f"  - Total combinations: {len(combinations)}")
        print()

        # Initialize oracle runner
        oracle_runner = OracleRunner(temp_dir)

        # Process all combinations sequentially
        print("Processing combinations...")
        successful_combinations = 0

        for i, combination in enumerate(combinations):
            print(f"\nProcessing combination {i + 1}/{len(combinations)}")
            print(
                f"  Parameters: accuracy={combination['user_parameters']['accuracy_threshold']}, "
                f"latency={combination['user_parameters']['latency_threshold_seconds']}"
            )

            try:
                # Run simulation
                result = oracle_runner.run_simulation(combination)

                if result["success"]:
                    # Save chunk
                    chunk_path = oracle_runner.save_chunk([result], i)

                    # Validate chunk
                    if not oracle_runner.validate_chunk_data(chunk_path):
                        print("  ✗ Chunk validation failed")
                        continue

                    timesteps = result["metadata"]["total_timesteps"]
                    reward = result["metadata"]["total_reward"]
                    exec_time = result["metadata"]["execution_time_seconds"]

                    print(
                        f"  ✓ Completed - {timesteps} steps, reward: {reward:.2f}, time: {exec_time:.1f}s"
                    )
                    successful_combinations += 1

                else:
                    print(
                        f"  ✗ Failed - {result['error']['type']}: {result['error']['message']}"
                    )

            except Exception as e:
                print(f"  ✗ Exception: {e}")
                continue

        print(
            f"\n✓ {successful_combinations}/{len(combinations)} combinations completed successfully"
        )
        print()

        # Recombine data
        print("Creating final dataset...")
        recombiner = DataRecombiner()

        if not recombiner.recombine_all(
            train_ratio=config.output["data_split"]["train"],
            val_ratio=config.output["data_split"]["val"],
            cleanup=True,
        ):
            print("✗ Dataset recombination failed")
            return False

        print("✓ Final dataset created successfully")
        print()

        # Summary
        total_time = time.time() - start_time
        print("=" * 60)
        print("🎉 TRAINING DATA GENERATION COMPLETED!")
        print("=" * 60)
        print(f"Total time: {total_time / 3600:.1f} hours")
        print(f"Successful combinations: {successful_combinations}/{len(combinations)}")
        print("Output directory: data/training_data/")
        print()

        return True

    except KeyboardInterrupt:
        print("\n✗ Generation interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
