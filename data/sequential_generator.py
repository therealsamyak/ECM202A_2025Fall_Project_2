#!/usr/bin/env python3
"""
Sequential data generator for training data generation.
Processes combinations one by one with detailed logging.
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, Any
import shutil

from generation_config import ConfigLoader
from oracle_runner import OracleRunner


class SequentialGenerator:
    """Orchestrates sequential data generation with detailed logging"""

    def __init__(self, config_path: str = "data/data.config.json"):
        # Fix path for running from project root
        if not os.path.isabs(config_path):
            config_path = os.path.join("data", os.path.basename(config_path))
        self.config_loader = ConfigLoader(config_path)
        self.oracle_runner = OracleRunner()
        self.setup_logging()

    def setup_logging(self):
        """Setup detailed logging configuration"""
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "sequential_generation.log")

        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Sequential logging initialized")

    def generate_all_data(self) -> bool:
        """
        Generate all training data sequentially

        Returns:
            True if successful, False if any errors occurred
        """
        start_time = time.time()

        try:
            # Load configuration
            self.logger.info("Loading configuration...")
            self.config_loader.load_config()

            # Generate parameter combinations
            self.logger.info("Generating parameter combinations...")
            combinations = self.config_loader.generate_parameter_combinations()
            total_simulations = len(combinations)
            self.logger.info(f"Generated {total_simulations} parameter combinations")

            # Process combinations sequentially
            successful_simulations = 0
            failed_simulations = 0
            completed_simulations = 0
            start_processing_time = time.time()

            self.logger.info(
                f"Starting sequential processing of {total_simulations} combinations"
            )

            for i, combination in enumerate(combinations):
                combination_start_time = time.time()

                print(f"\n🔄 Processing combination {i + 1}/{total_simulations}")
                print(
                    f"  Parameters: accuracy={combination['user_parameters']['accuracy_threshold']}, "
                    f"latency={combination['user_parameters']['latency_threshold_seconds']}"
                )
                print(
                    f"  Battery: {combination['battery_config']['battery_capacity_mwh']} mWh"
                )
                print(
                    f"  Date: {combination['date']}, Location: {combination['location']}"
                )

                self.logger.info(
                    f"Starting combination {i + 1}/{total_simulations} (ID: {combination['combination_id']})"
                )

                try:
                    # Run simulation
                    simulation_start = time.time()
                    result = self.oracle_runner.run_simulation(combination)
                    simulation_time = time.time() - simulation_start

                    self.logger.info(f"Simulation completed in {simulation_time:.2f}s")

                    if result["success"]:
                        # Save chunk
                        chunk_start = time.time()
                        chunk_path = self.oracle_runner.save_chunk([result], i)
                        chunk_time = time.time() - chunk_start

                        # Validate chunk
                        validation_start = time.time()
                        if not self.oracle_runner.validate_chunk_data(chunk_path):
                            raise Exception(f"Chunk {i} validation failed")
                        validation_time = time.time() - validation_start

                        total_combination_time = time.time() - combination_start_time

                        timesteps = result["metadata"]["total_timesteps"]
                        reward = result["metadata"]["total_reward"]
                        optimal_value = result["metadata"]["optimal_value"]

                        print(f"  ✓ Combination {i + 1} completed successfully")
                        print(f"    - Timesteps: {timesteps}")
                        print(f"    - Total reward: {reward:.3f}")
                        print(f"    - Optimal value: {optimal_value:.3f}")
                        print(f"    - Simulation time: {simulation_time:.2f}s")
                        print(f"    - Chunk save time: {chunk_time:.2f}s")
                        print(f"    - Validation time: {validation_time:.2f}s")
                        print(f"    - Total time: {total_combination_time:.2f}s")

                        self.logger.info(
                            f"Combination {i + 1} success - "
                            f"timesteps: {timesteps}, reward: {reward:.3f}, "
                            f"sim_time: {simulation_time:.2f}s, "
                            f"chunk_time: {chunk_time:.2f}s, "
                            "val_time: {validation_time:.2f}s, "
                            f"total_time: {total_combination_time:.2f}s"
                        )

                        successful_simulations += 1
                        completed_simulations += 1

                    else:
                        error_msg = (
                            f"{result['error']['type']}: {result['error']['message']}"
                        )
                        print(f"  ✗ Combination {i + 1} failed: {error_msg}")
                        self.logger.error(f"Combination {i + 1} failed: {error_msg}")
                        failed_simulations += 1

                        # Continue with other combinations instead of aborting
                        print("  Continuing with next combination...")
                        self.logger.info(
                            "Continuing with next combination after failure"
                        )

                except Exception as e:
                    print(f"  ✗ Combination {i + 1} failed with exception: {e}")
                    self.logger.error(f"Combination {i + 1} exception: {e}")
                    failed_simulations += 1

                    # Continue with other combinations
                    print("  Continuing with next combination...")
                    self.logger.info("Continuing with next combination after exception")

                # Progress update
                progress_percent = ((i + 1) / total_simulations) * 100
                elapsed = time.time() - start_processing_time
                avg_time_per_combination = elapsed / (i + 1)
                remaining_combinations = total_simulations - (i + 1)
                eta_seconds = avg_time_per_combination * remaining_combinations
                eta_hours = eta_seconds / 3600

                print(f"  Progress: {progress_percent:.1f}% | ETA: {eta_hours:.1f}h")

            # Generation completed
            total_time = time.time() - start_time

            print("\n🎉 Sequential processing completed!")
            print(f"Total time: {total_time / 3600:.1f} hours")
            print(
                f"Successful simulations: {successful_simulations}/{total_simulations}"
            )
            print(f"Failed simulations: {failed_simulations}/{total_simulations}")

            self.logger.info(
                f"All combinations completed in {total_time / 3600:.1f} hours"
            )
            self.logger.info(
                f"Successful: {successful_simulations}/{total_simulations}"
            )
            self.logger.info(f"Failed: {failed_simulations}/{total_simulations}")

            return failed_simulations == 0

        except Exception as e:
            print(f"\n✗ Data generation failed: {e}")
            self.logger.error(f"Data generation failed: {e}")
            self.logger.error(f"Traceback: {sys.exc_info()}")
            return False

    def cleanup_temp_files(self):
        """Clean up temporary files on failure"""
        temp_dir = "data/temp"
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.logger.info("Cleaned up temporary files")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp files: {e}")

    def get_generation_summary(self) -> Dict[str, Any]:
        """Get summary of generated data"""
        temp_dir = "data/temp"
        chunk_files = [
            f
            for f in os.listdir(temp_dir)
            if f.startswith("chunk_") and f.endswith(".npz")
        ]

        total_timesteps = 0
        successful_simulations = 0

        for chunk_file in chunk_files:
            try:
                chunk_path = os.path.join(temp_dir, chunk_file)
                data = np.load(chunk_path)
                metadata = json.loads(str(data["metadata"]))

                total_timesteps += metadata["total_timesteps"]
                successful_simulations += metadata["successful_simulations"]

            except Exception as e:
                self.logger.warning(f"Failed to read chunk {chunk_file}: {e}")

        return {
            "total_chunks": len(chunk_files),
            "successful_simulations": successful_simulations,
            "total_timesteps": total_timesteps,
            "temp_directory": temp_dir,
        }


def main():
    """Main entry point for sequential generation"""
    generator = SequentialGenerator()

    try:
        print("🚀 Starting sequential data generation...")
        success = generator.generate_all_data()

        if success:
            print("✅ Data generation completed successfully")

            # Get summary
            summary = generator.get_generation_summary()
            print(f"  - Total chunks: {summary['total_chunks']}")
            print(f"  - Successful simulations: {summary['successful_simulations']}")
            print(f"  - Total timesteps: {summary['total_timesteps']}")
            print(f"  - Temp directory: {summary['temp_directory']}")

            return 0
        else:
            print("✗ Data generation completed with some failures")
            generator.cleanup_temp_files()
            return 1

    except KeyboardInterrupt:
        print("\n✗ Data generation interrupted by user")
        generator.cleanup_temp_files()
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        generator.cleanup_temp_files()
        return 1


if __name__ == "__main__":
    exit(main())
