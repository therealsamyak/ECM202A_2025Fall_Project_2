#!/usr/bin/env python3
"""
Main entry point for Phase 1 training data generation.
Simple script that handles everything automatically.
"""

import os
import time
from datetime import datetime
import traceback

from generation_config import ConfigLoader
from sequential_generator import SequentialGenerator
from recombine_data import DataRecombiner


def main():
    """Generate training data - simple one script approach"""
    print("=" * 60)
    print("PHASE 1: TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    try:
        # Step 1: Load configuration
        print("STEP 1: Loading configuration...")
        config_loader = ConfigLoader("data/data.config.json")
        config = config_loader.load_config()
        combinations = config_loader.generate_parameter_combinations()
        chunks = config_loader.calculate_chunks(combinations)

        print("✓ Configuration loaded")
        print(f"  - Total combinations: {len(combinations)}")
        print(f"  - Total chunks: {len(chunks)}")
        print(f"  - Max workers: {config.output['max_workers']}")
        print()

        # Step 2: Generate all chunks
        print("STEP 2: Generating training data...")
        generator = SequentialGenerator("data.config.json")

        if not generator.generate_all_data():
            print("✗ Data generation failed")
            return False

        print("✓ All chunks generated successfully")
        print()

        # Step 3: Recombine data
        print("STEP 3: Creating final dataset...")
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

        # Final summary
        total_time = time.time() - start_time
        print("=" * 60)
        print("🎉 TRAINING DATA GENERATION COMPLETED!")
        print("=" * 60)
        print(f"Total time: {total_time / 3600:.1f} hours")
        print("Output directory: data/training_data/")
        print()
        print("Files created:")

        output_files = [
            "combined_dataset.npz",
            "train.npy",
            "train_actions.npy",
            "val.npy",
            "val_actions.npy",
            "test.npy",
            "test_actions.npy",
            "metadata.json",
        ]

        for filename in output_files:
            filepath = os.path.join("data/training_data", filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✓ {filename} ({size_mb:.1f} MB)")

        print()
        print("Ready for Phase 2: Model Training")
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
