#!/usr/bin/env python3
"""MNIST Model Retraining Script.

This standalone script provides a convenient way to retrain the MNIST digit
recognition model from the command line. It uses the same training pipeline
as the main model module but provides more detailed progress output.

Usage:
    python retrain_cpu.py
"""
import sys
import time
from pathlib import Path

# Add the current directory to the Python path.
sys.path.append(str(Path(__file__).parent))

from model import retrain_model

def main() -> None:
    """Main function to execute model retraining with progress tracking."""
    print("=" * 60)
    print("🚀 MNIST Model Retraining Script")
    print("=" * 60)
    print()
    try:
        # Record start time for duration tracking.
        start_time = time.time()
        print("📋 Starting model retraining process...")
        print("⏱️  This will take 5-15 minutes depending on your hardware.")
        print("🔄 The training will save checkpoints and can be interrupted safely.")
        print()
        # Execute the retraining.
        retrain_model()
        # Calculate and display duration.
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print()
        print("=" * 60)
        print("✅ Model retraining completed successfully!")
        print(f"⏱️  Total time: {minutes}m {seconds}s")
        print("💾 New model saved to: mnist_cnn.h5")
        print("🎯 Ready for improved digit recognition!")
        print("=" * 60)
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("⚠️  Training interrupted by user")
        print("💡 You can resume training by running this script again")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Error during retraining: {e}")
        print("💡 Check the error message above for troubleshooting")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()