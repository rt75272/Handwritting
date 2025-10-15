#!/usr/bin/env python3
"""
Debug script to test model loading and basic functionality.
Run this to verify models work before deployment.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

def test_model_loading():
    """Test if models can be loaded successfully."""
    print("🔍 Testing model loading...")
    
    # Test digit model
    digit_model_path = "mnist_cnn.h5"
    if os.path.exists(digit_model_path):
        try:
            digit_model = tf.keras.models.load_model(digit_model_path)
            print(f"✅ Digit model loaded: {digit_model_path}")
            print(f"   Input shape: {digit_model.input_shape}")
            print(f"   Output shape: {digit_model.output_shape}")
            
            # Test prediction with dummy data
            dummy_input = np.random.random((1, 28, 28, 1))
            prediction = digit_model.predict(dummy_input, verbose=0)
            print(f"   Test prediction shape: {prediction.shape}")
            print(f"   Prediction sum: {np.sum(prediction):.4f}")
            
        except Exception as e:
            print(f"❌ Failed to load digit model: {e}")
    else:
        print(f"❌ Digit model not found: {digit_model_path}")
    
    # Test alphanumeric model
    alphanumeric_model_path = "alphanumeric_cnn.keras"
    labels_path = "alphanumeric_labels.json"
    
    if os.path.exists(alphanumeric_model_path):
        try:
            alphanumeric_model = tf.keras.models.load_model(alphanumeric_model_path)
            print(f"✅ Alphanumeric model loaded: {alphanumeric_model_path}")
            print(f"   Input shape: {alphanumeric_model.input_shape}")
            print(f"   Output shape: {alphanumeric_model.output_shape}")
            
            # Test prediction with dummy data
            dummy_input = np.random.random((1, 28, 28, 1))
            prediction = alphanumeric_model.predict(dummy_input, verbose=0)
            print(f"   Test prediction shape: {prediction.shape}")
            print(f"   Prediction sum: {np.sum(prediction):.4f}")
            
        except Exception as e:
            print(f"❌ Failed to load alphanumeric model: {e}")
    else:
        print(f"❌ Alphanumeric model not found: {alphanumeric_model_path}")
    
    # Test labels
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            print(f"✅ Labels loaded: {len(labels)} classes")
            print(f"   Sample labels: {dict(list(labels.items())[:5])}")
        except Exception as e:
            print(f"❌ Failed to load labels: {e}")
    else:
        print(f"❌ Labels file not found: {labels_path}")

def test_tensorflow():
    """Test TensorFlow installation and capabilities."""
    print("\n🔍 Testing TensorFlow...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU available: {tf.config.list_physical_devices('CPU')}")
    
    # Test basic TensorFlow operations
    try:
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"✅ TensorFlow basic operations working")
        print(f"   Matrix multiplication result shape: {c.shape}")
    except Exception as e:
        print(f"❌ TensorFlow basic operations failed: {e}")

def test_dependencies():
    """Test required dependencies."""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        'tensorflow',
        'numpy',
        'PIL',
        'scipy',
        'flask'
    ]
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package} (Pillow) version: {PIL.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package} version: {version}")
        except ImportError:
            print(f"❌ {package} not installed")

def main():
    """Run all tests."""
    print("🚀 Starting deployment readiness check...\n")
    
    test_dependencies()
    test_tensorflow()
    test_model_loading()
    
    print("\n✅ Deployment readiness check complete!")
    print("\nIf all tests passed, your app should work on Render.")
    print("If there are errors, fix them before deploying.")

if __name__ == "__main__":
    main()