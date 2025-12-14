#!/usr/bin/env python3
"""
Main entry point for training data generation.
Processes all parameter combinations sequentially.
"""

import traceback
import os
import time
from datetime import datetime
import concurrent.futures

from generation_config import ConfigLoader
from oracle_runner import OracleRunner
from recombine_data import DataRecombiner


def process_combination(args):
    """Process a single combination in a worker process"""
    combination, temp_dir, i, total = args

    print(f"\nProcessing combination {i + 1}/{total}")
    print(
        f"  Parameters: accuracy={combination['user_parameters']['accuracy_threshold']}, "
        f"latency={combination['user_parameters']['latency_threshold_seconds']}"
    )

    try:
        # Initialize oracle runner in this worker process
        oracle_runner = OracleRunner(temp_dir)

        # Run simulation
        result = oracle_runner.run_simulation(combination)

        if result["success"]:
            # Save chunk
            chunk_path = oracle_runner.save_chunk([result], i)

            # Validate chunk
            if not oracle_runner.validate_chunk_data(chunk_path):
                print("  ✗ Chunk validation failed")
                return {
                    "success": False,
                    "combination_id": combination["combination_id"],
                    "error": {
                        "message": "Chunk validation failed",
                        "type": "ValidationError",
                    },
                }

            timesteps = result["metadata"]["total_timesteps"]
            reward = result["metadata"]["total_reward"]
            exec_time = result["metadata"]["execution_time_seconds"]

            print(
                f"  ✓ Completed - {timesteps} steps, reward: {reward:.2f}, time: {exec_time:.1f}s"
            )

            return {
                "success": True,
                "combination_id": combination["combination_id"],
                "result": result,
            }

        else:
            print(
                f"  ✗ Failed - {result['error']['type']}: {result['error']['message']}"
            )

            return {
                "success": False,
                "combination_id": combination["combination_id"],
                "error": result["error"],
            }

    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return {
            "success": False,
            "combination_id": combination["combination_id"],
            "error": {"message": str(e), "type": type(e).__name__},
        }


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

        # Process all combinations with multiprocessing
        print("Processing combinations with 10 workers...")
        successful_combinations = 0

        # Prepare arguments for worker processes
        worker_args = [
            (combination, temp_dir, i, len(combinations))
            for i, combination in enumerate(combinations)
        ]

        # Use ProcessPoolExecutor with 10 workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_combination = {
                executor.submit(process_combination, args): args[0]["combination_id"]
                for args in worker_args
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_combination):
                try:
                    worker_result = future.result()

                    if worker_result["success"]:
                        successful_combinations += 1
                    else:
                        # Crash everything on any failure as requested
                        raise RuntimeError(
                            f"Worker failed for combination {worker_result['combination_id']}: "
                            f"{worker_result['error']['type']}: {worker_result['error']['message']}"
                        )

                except Exception as e:
                    # Raise any exception to crash everything
                    raise RuntimeError(f"Worker process failed: {e}")

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
