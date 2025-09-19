#!/usr/bin/env python3
"""
Performance test script to verify multithreading optimizations.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import multiprocessing as mp

def test_threading_config():
    """Test threading configuration."""
    print("🧵 Testing Threading Configuration...")
    num_cores = mp.cpu_count()
    print(f"CPU cores available: {num_cores}")
    print(f"Inter-op parallelism: {tf.config.threading.get_inter_op_parallelism_threads()}")
    print(f"Intra-op parallelism: {tf.config.threading.get_intra_op_parallelism_threads()}")
    return True

def test_data_pipeline_performance():
    """Test data pipeline performance with different parallelization settings."""
    print("\n📊 Testing Data Pipeline Performance...")
    # Create dummy dataset
    batch_size = 512
    num_samples = 10000
    dummy_images = np.random.random((num_samples, 28, 28, 1)).astype(np.float32)
    dummy_labels = np.random.randint(0, 26, (num_samples,))
    dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
    # Test 1: Basic pipeline
    print("Testing basic pipeline...")
    basic_ds = (dataset
                .batch(batch_size)
                .prefetch(1))
    
    start_time = time.time()
    for _ in basic_ds:
        pass
    basic_time = time.time() - start_time
    print(f"Basic pipeline time: {basic_time:.3f} seconds")
    # Test 2: Optimized pipeline
    print("Testing optimized pipeline...")
    num_cores = mp.cpu_count()
    
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label
    optimized_ds = (dataset
                   .map(normalize, num_parallel_calls=num_cores * 2)
                   .cache()
                   .shuffle(1000)
                   .batch(batch_size, drop_remainder=True)
                   .prefetch(tf.data.AUTOTUNE))
    start_time = time.time()
    for _ in optimized_ds:
        pass
    optimized_time = time.time() - start_time
    print(f"Optimized pipeline time: {optimized_time:.3f} seconds")
    speedup = basic_time / optimized_time if optimized_time > 0 else 1
    print(f"Speedup: {speedup:.2f}x")
    return speedup > 1

def test_model_training_speed():
    """Test model training speed with optimizations."""
    print("\n🏗️ Testing Model Training Speed...")
    # Configure GPU and threading
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    # Set threading
    num_cores = mp.cpu_count()
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    # Create simple model
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(26, activation='softmax', dtype='float32')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Create dummy training data
    batch_size = 512
    num_batches = 50
    dummy_images = np.random.random((batch_size * num_batches, 28, 28, 1)).astype(np.float32)
    dummy_labels = tf.one_hot(np.random.randint(0, 26, (batch_size * num_batches,)), 26)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
    # Optimized dataset
    optimized_dataset = (dataset
                        .cache()
                        .shuffle(1000)
                        .batch(batch_size)
                        .prefetch(tf.data.AUTOTUNE))
    # Time training
    print("Training model with optimizations...")
    start_time = time.time()
    with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
        model.fit(optimized_dataset, epochs=2, verbose=0)
    training_time = time.time() - start_time
    print(f"Training time (2 epochs): {training_time:.3f} seconds")
    print(f"Average time per epoch: {training_time/2:.3f} seconds")
    return training_time < 30

def test_mixed_precision():
    """Test mixed precision training."""
    print("\n🎯 Testing Mixed Precision...")
    policy = tf.keras.mixed_precision.global_policy()
    print(f"Mixed precision policy: {policy.name}")
    if policy.name == 'mixed_float16':
        print("✅ Mixed precision is enabled")
        return True
    else:
        print("❌ Mixed precision not enabled")
        return False

def main():
    """Run all performance tests."""
    print("🚀 Multithreading Performance Test Suite")
    print("=" * 60)
    tests_passed = 0
    total_tests = 4
    # Test 1: Threading configuration
    if test_threading_config():
        tests_passed += 1
    # Test 2: Data pipeline performance
    if test_data_pipeline_performance():
        tests_passed += 1
    # Test 3: Model training speed
    if test_model_training_speed():
        tests_passed += 1
    # Test 4: Mixed precision
    if test_mixed_precision():
        tests_passed += 1
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    if tests_passed == total_tests:
        print("🎉 All multithreading optimizations are working!")
        print("💡 Your system is configured for maximum performance.")
    else:
        print("⚠️ Some optimizations may not be working optimally.")
    return tests_passed == total_tests

if __name__ == "__main__":
    """The big red activation button."""
    success = main()
    exit(0 if success else 1)