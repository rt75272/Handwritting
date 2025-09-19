#!/usr/bin/env python3
"""
Quick test script to verify GPU optimization is working correctly.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np

def test_gpu_setup():
    """Test GPU configuration and mixed precision."""
    print("🔍 Testing GPU Setup...")
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✅ GPU configured successfully!")
            print(f"Available GPUs: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i}: {details.get('device_name', 'Unknown')}")
            print(f"Mixed precision enabled: {policy.name}")
            return True
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("❌ No GPU found")
        return False

def test_model_on_gpu():
    """Test model creation and execution on GPU."""
    print("\n🔍 Testing Model on GPU...")
    # Create a simple test model
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(26, activation='softmax', dtype='float32')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Model parameters: {model.count_params():,}")
    # Test with dummy data
    batch_size = 128
    dummy_images = np.random.random((batch_size, 28, 28, 1)).astype(np.float32)
    dummy_labels = np.random.randint(0, 26, (batch_size,))
    dummy_labels_onehot = tf.one_hot(dummy_labels, 26)
    # Test forward pass
    with tf.device('/GPU:0'):
        predictions = model(dummy_images)
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {predictions.shape}")
        print(f"Device placement: {predictions.device}")
        # Test one training step
        loss = model.train_on_batch(dummy_images, dummy_labels_onehot)
        print(f"✅ Training step successful!")
        print(f"Loss: {loss[0]:.4f}, Accuracy: {loss[1]:.4f}")
    return True

def test_data_pipeline():
    """Test GPU-optimized data pipeline."""
    print("\n🔍 Testing Data Pipeline...")
    try:
        # Create dummy dataset
        dummy_images = np.random.random((1000, 28, 28, 1)).astype(np.float32)
        dummy_labels = np.random.randint(0, 26, (1000,))
        dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
        # Apply GPU-optimized pipeline
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = (dataset
                  .cache()
                  .shuffle(1000)
                  .batch(128)
                  .prefetch(AUTOTUNE))
        # Test iteration
        for batch_images, batch_labels in dataset.take(1):
            print(f"✅ Data pipeline successful!")
            print(f"Batch shape: {batch_images.shape}")
            print(f"Labels shape: {batch_labels.shape}")
            break
        return True
    except Exception as e:
        print(f"❌ Data pipeline error: {e}")
        return False

def main():
    """Run all GPU optimization tests."""
    print("🚀 GPU Optimization Test Suite")
    print("=" * 50)
    tests_passed = 0
    total_tests = 3
    # Test 1: GPU Setup
    if test_gpu_setup():
        tests_passed += 1
    # Test 2: Model on GPU
    if test_model_on_gpu():
        tests_passed += 1
    # Test 3: Data Pipeline
    if test_data_pipeline():
        tests_passed += 1
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    if tests_passed == total_tests:
        print("🎉 All GPU optimizations are working correctly!")
        print("💡 Your system is ready for GPU-accelerated training.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    return tests_passed == total_tests

if __name__ == "__main__":
    """The big red activation button."""
    success = main()
    exit(0 if success else 1)