#!/usr/bin/env python3
"""
Quick Demo Alphanumeric Model Training Script

This script creates a fast demo version of the alphanumeric model using 
a smaller subset of data for testing purposes. Use this to quickly test
the functionality before running the full training.
"""

import os
import time
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def create_demo_data():
    """Create demo data for quick testing."""
    print("📝 Creating demo alphanumeric dataset...")
    
    # Create synthetic data for demonstration
    # 1000 samples each for digits and letters (smaller for fast training)
    num_samples_per_class = 100
    
    # Generate synthetic digit data (classes 0-9)
    x_digits = np.random.rand(num_samples_per_class * 10, 28, 28, 1)
    y_digits = np.repeat(np.arange(10), num_samples_per_class)
    
    # Generate synthetic letter data (classes 10-35)
    x_letters = np.random.rand(num_samples_per_class * 26, 28, 28, 1)
    y_letters = np.repeat(np.arange(10, 36), num_samples_per_class)
    
    # Combine data
    x_data = np.concatenate([x_digits, x_letters])
    y_data = np.concatenate([y_digits, y_letters])
    
    # Shuffle
    indices = np.random.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train/test
    split = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]
    
    # Convert to categorical
    y_train = to_categorical(y_train, 36)
    y_test = to_categorical(y_test, 36)
    
    print(f"✅ Demo data created:")
    print(f"   Training samples: {len(x_train)}")
    print(f"   Test samples: {len(x_test)}")
    print(f"   Classes: 36 (0-9 digits + A-Z letters)")
    
    return (x_train, y_train), (x_test, y_test)

def create_simple_model():
    """Create a simple CNN model for demo purposes."""
    print("🏗️ Building demo model...")
    
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(36, activation='softmax')  # 36 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Demo model created!")
    return model

def create_label_mapping():
    """Create label mapping for the 36 classes."""
    label_map = {}
    
    # Digits 0-9
    for i in range(10):
        label_map[i] = str(i)
    
    # Letters A-Z
    for i in range(26):
        label_map[i + 10] = chr(ord('A') + i)
    
    return label_map

def main():
    """Main demo training pipeline."""
    print("🚀 Quick Demo Alphanumeric Model Training")
    print("=" * 50)
    print("⚡ This is a fast demo using synthetic data")
    print("📍 For real training, use train_alphanumeric.py")
    print("=" * 50)
    
    # Create demo data
    (x_train, y_train), (x_test, y_test) = create_demo_data()
    
    # Create model
    model = create_simple_model()
    
    # Train model (just a few epochs for demo)
    print("🚀 Training demo model (5 epochs)...")
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ Demo training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"📊 Demo Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Save demo model
    model.save('alphanumeric_cnn_demo.keras')
    print("✅ Demo model saved as 'alphanumeric_cnn_demo.keras'")
    
    # Create and save label mapping
    import json
    label_map = create_label_mapping()
    with open('alphanumeric_labels_demo.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    print("✅ Demo label mapping saved as 'alphanumeric_labels_demo.json'")
    
    print("\n🎉 Demo training completed successfully!")
    print("📝 To use this demo model:")
    print("   1. Rename 'alphanumeric_cnn_demo.keras' to 'alphanumeric_cnn.keras'")
    print("   2. Rename 'alphanumeric_labels_demo.json' to 'alphanumeric_labels.json'")
    print("   3. Restart the enhanced app: python app_enhanced.py")
    print("   4. The alphanumeric mode will be available!")
    print("\n⚠️ Note: This demo model uses synthetic data and won't recognize real handwriting.")
    print("   For real recognition, let the full training (train_alphanumeric.py) complete.")

if __name__ == "__main__":
    main()