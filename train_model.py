#!/usr/bin/env python3
"""
Pre-train the model and save it for deployment.
Run this locally to create a trained model that can be loaded in production.
"""
import numpy as np
import tensorflow as tf
import os
import json

def create_and_train_model():
    """Create, train, and save an optimized digit recognition model."""
    print("🚀 Starting model training...")
    
    # Load MNIST dataset
    print("📥 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess data
    print("🔄 Preprocessing data...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Create optimized model architecture
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block  
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🏋️ Training model...")
    
    # Use 50% of data for good accuracy
    subset_size = len(x_train) // 2
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Train the model
    history = model.fit(
        x_train_subset, y_train_subset,
        epochs=8,  # More epochs for better accuracy
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Test accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"🎯 Final test accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    model_path = 'trained_digit_model.keras'
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Save training metadata
    metadata = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'epochs_trained': 8,
        'training_samples': subset_size,
        'model_architecture': 'Enhanced CNN with BatchNorm and Dropout'
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Training complete! Accuracy: {test_accuracy:.4f}")
    return model

if __name__ == '__main__':
    create_and_train_model()