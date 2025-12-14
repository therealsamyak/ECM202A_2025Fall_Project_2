#!/usr/bin/env python3
"""
Parallel multiprocessing orchestrator for training data generation.
Handles chunked processing with detailed logging and error handling.
"""

import os
import sys
import json
import time
import logging
import multiprocessing as mp
import numpy as np
from typing import Dict, List, Any
import shutil

from generation_config import ConfigLoader
from oracle_runner import OracleRunner


class ParallelGenerator:
    """Orchestrates parallel data generation with chunked processing"""

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

        log_file = os.path.join(log_dir, "generation.log")

        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized")

    def generate_all_data(self) -> bool:
        """
        Generate all training data with parallel processing

        Returns:
            True if successful, False if any errors occurred
        """
        start_time = time.time()

        try:
            # Load configuration
            self.logger.info("Loading configuration...")
            config = self.config_loader.load_config()

            # Generate parameter combinations
            self.logger.info("Generating parameter combinations...")
            combinations = self.config_loader.generate_parameter_combinations()
            total_simulations = len(combinations)
            self.logger.info(f"Generated {total_simulations} parameter combinations")

            # Create chunks
            chunks = self.config_loader.calculate_chunks(combinations)
            total_chunks = len(chunks)
            self.logger.info(f"Created {total_chunks} chunks (1 combination per chunk)")

            # Initialize process pool
            max_workers = config.output["max_workers"]
            self.logger.info(f"Initializing process pool with {max_workers} workers")

            # Process chunks with worker recycling
            successful_chunks = 0
            failed_chunks = 0
            completed_chunks = 0
            start_processing_time = time.time()

            # Process chunks in batches to recycle workers
            batch_size = max_workers
            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]

                print(
                    f"\n🚀 Processing batch {batch_start // batch_size + 1}: chunks {batch_start + 1}-{batch_end}"
                )
                self.logger.info(
                    f"Processing batch {batch_start // batch_size + 1}: chunks {batch_start + 1}-{batch_end}"
                )

                with mp.Pool(processes=max_workers) as pool:
                    # Submit batch jobs
                    chunk_jobs = []
                    for i, chunk in enumerate(batch_chunks):
                        actual_chunk_id = batch_start + i
                        job = pool.apply_async(
                            self._process_chunk_worker,
                            args=(chunk, actual_chunk_id),
                            callback=self._chunk_success_callback,
                            error_callback=self._chunk_error_callback,
                        )
                        chunk_jobs.append((actual_chunk_id, job))

                    # Wait for batch completion with progress tracking
                    batch_completed = 0
                    for chunk_id, job in chunk_jobs:
                        try:
                            # Wait for job completion
                            job.get()  # This will raise exception if job failed
                            completed_chunks += 1
                            successful_chunks += 1
                            batch_completed += 1

                            # Calculate progress
                            progress_percent = (completed_chunks / total_chunks) * 100
                            elapsed = time.time() - start_processing_time
                            avg_time_per_chunk = elapsed / completed_chunks
                            remaining_chunks = total_chunks - completed_chunks
                            eta_seconds = avg_time_per_chunk * remaining_chunks
                            eta_hours = eta_seconds / 3600

                            # Console progress bar
                            progress_bar = "█" * int(progress_percent // 2) + "░" * (
                                50 - int(progress_percent // 2)
                            )

                            print(
                                f"\r[{progress_bar}] {progress_percent:.1f}% | "
                                f"Chunk {chunk_id + 1}/{total_chunks} | "
                                f"ETA: {eta_hours:.1f}h | "
                                f"Batch: {batch_completed}/{len(chunk_jobs)}",
                                end="",
                                flush=True,
                            )

                            self.logger.debug(
                                f"Chunk {chunk_id + 1}/{total_chunks} completed - "
                                f"ETA: {eta_hours:.1f} hours"
                            )

                        except Exception as e:
                            self.logger.error(f"Chunk {chunk_id + 1} failed: {e}")
                            failed_chunks += 1

                            # Zero-error tolerance - abort on first failure
                            print(f"\n✗ Chunk {chunk_id + 1} failed: {e}")
                            self.logger.error(
                                "Aborting generation due to zero-error tolerance policy"
                            )
                            pool.terminate()
                            pool.join()
                            return False

                    print(
                        f"\n✅ Batch {batch_start // batch_size + 1} completed successfully"
                    )
                    self.logger.info(
                        f"Batch {batch_start // batch_size + 1} completed successfully"
                    )

                # Workers automatically recycled when pool exits

            # Generation completed successfully
            total_time = time.time() - start_time

            # Final progress bar (100%)
            print(
                f"\r[{'█' * 50}] 100.0% | All chunks completed | "
                f"Time: {total_time / 3600:.1f}h | "
                f"Success: {successful_chunks}/{total_chunks}",
                flush=True,
            )
            print(f"\n🎉 All {successful_chunks} combinations completed successfully!")

            self.logger.info(
                f"All chunks completed successfully in {total_time / 3600:.1f} hours"
            )
            self.logger.info(f"Successful chunks: {successful_chunks}/{total_chunks}")

            return True

        except Exception as e:
            print(f"\n✗ Data generation failed: {e}")
            self.logger.error(f"Data generation failed: {e}")
            self.logger.error(f"Traceback: {sys.exc_info()}")
            return False

    def _process_chunk_worker(self, chunk: List[Dict[str, Any]], chunk_id: int) -> str:
        """
        Worker function to process a single chunk

        Args:
            chunk: List of parameter combinations (should be 1)
            chunk_id: Chunk identifier

        Returns:
            Path to saved chunk file
        """
        chunk_start_time = time.time()

        try:
            # Process each simulation in chunk (should be just 1)
            chunk_results = []

            for i, combination in enumerate(chunk):
                # Console progress for worker
                print(
                    f"  Worker {os.getpid()} processing combination {combination['combination_id']} "
                    f"({combination['user_parameters']['accuracy_threshold']}, "
                    f"{combination['reward_weights']['success_weight']})"
                )

                # Run simulation
                result = self.oracle_runner.run_simulation(combination)
                chunk_results.append(result)

                if result["success"]:
                    timesteps = result["metadata"]["total_timesteps"]
                    reward = result["metadata"]["total_reward"]
                    exec_time = result["metadata"]["execution_time_seconds"]
                    print(
                        f"  ✓ Combination {combination['combination_id']} completed - "
                        f"{timesteps} steps, reward: {reward:.2f}, time: {exec_time:.1f}s"
                    )
                else:
                    print(
                        f"  ✗ Combination {combination['combination_id']} failed - "
                        f"{result['error']['type']}: {result['error']['message']}"
                    )
                    # This will be caught by zero-error tolerance
                    raise Exception(
                        f"Simulation {combination['combination_id']} failed: {result['error']['message']}"
                    )

            # Save chunk
            chunk_path = self.oracle_runner.save_chunk(chunk_results, chunk_id)

            # Validate chunk
            if not self.oracle_runner.validate_chunk_data(chunk_path):
                raise Exception(f"Chunk {chunk_id} validation failed")

            chunk_time = time.time() - chunk_start_time
            self.logger.debug(
                f"Chunk {chunk_id + 1} completed in {chunk_time:.1f}s - "
                f"saved to {chunk_path}"
            )

            return chunk_path

        except Exception as e:
            print(f"  ✗ Chunk {chunk_id + 1} processing failed: {e}")
            self.logger.error(f"Chunk {chunk_id + 1} processing failed: {e}")
            raise

    def _chunk_success_callback(self, result: str):
        """Callback for successful chunk completion"""
        self.logger.debug(f"Chunk completed successfully: {result}")

    def _chunk_error_callback(self, error):
        """Callback for chunk completion error"""
        self.logger.error(f"Chunk failed with error: {error}")

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
    """Main entry point for parallel generation"""
    generator = ParallelGenerator()

    try:
        print("🚀 Starting parallel data generation...")
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
            print("✗ Data generation failed")
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
