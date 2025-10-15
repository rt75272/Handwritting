#!/usr/bin/env python3
"""MNIST CNN Model Definition and Training Module.

This module contains the neural network architecture definition and training
functionality for the MNIST digit recognition model. It implements a 
convolutional neural network optimized for handwritten digit classification.
"""
import os
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants.
MODEL_PATH = "mnist_cnn.h5"
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

def create_cnn_model() -> tf.keras.Model:
    """Create a Convolutional Neural Network for MNIST digit classification.
    
    The model architecture consists of:
    - Two convolutional layers with ReLU activation and max pooling
    - Dropout layers for regularization  
    - Dense layers for final classification
    - Softmax output for probability distribution over 10 digit classes
    
    Returns:
        tf.keras.Model: Compiled CNN model ready for training.
    """
    model = Sequential([
        # First convolutional block.
        # 32 filters of size 3x3, ReLU activation, same padding.
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),  # Reduce spatial dimensions by half.
        # Second convolutional block.
        # 64 filters for more complex feature detection.
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        # Third convolutional block.
        # 64 filters to maintain feature richness.
        Conv2D(64, (3, 3), activation='relu'),
        # Flatten feature maps for dense layers.
        Flatten(),
        # Dense layers for classification.
        Dense(64, activation='relu'),
        Dropout(0.5),  # Prevent overfitting during training.
        Dense(NUM_CLASSES, activation='softmax')  # Output probabilities.
    ])
    # Compile model with appropriate loss function and optimizer.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the MNIST dataset.
    
    Performs the following preprocessing steps:
    - Loads MNIST data from Keras datasets
    - Normalizes pixel values to [0, 1] range
    - Reshapes images to include channel dimension
    - Converts labels to one-hot encoded format
    
    Returns:
        Tuple containing:
        - x_train: Training images (60000, 28, 28, 1)
        - y_train: Training labels (60000, 10)
        - x_test: Test images (10000, 28, 28, 1)  
        - y_test: Test labels (10000, 10)
    """
    print("Loading MNIST dataset...")
    # Load the MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize pixel values to [0, 1] range.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # Reshape to add channel dimension (grayscale = 1 channel).
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Convert labels to one-hot encoded format.
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    return x_train, y_train, x_test, y_test

def train_model(model: tf.keras.Model, 
                x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray,
                epochs: int = 10) -> tf.keras.Model:
    """Train the CNN model on MNIST data with callbacks for optimization.
    
    Args:
        model: The compiled Keras model to train.
        x_train: Training images.
        y_train: Training labels (one-hot encoded).
        x_test: Validation images.
        y_test: Validation labels (one-hot encoded).
        epochs: Maximum number of training epochs.
        
    Returns:
        tf.keras.Model: The trained model.
    """
    print("Starting model training...")
    # Define callbacks for training optimization.
    callbacks = [
        # Stop training early if validation accuracy stops improving.
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus.
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.0001,
            verbose=1
        )
    ]
    # Train the model.
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    # Evaluate final model performance.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_accuracy:.4f}")
    return model

def save_model(model: tf.keras.Model, filepath: str = MODEL_PATH) -> None:
    """Save the trained model to disk.
    
    Args:
        model: The trained Keras model to save.
        filepath: Path where to save the model file.
    """
    print(f"Saving model to {filepath}...")
    model.save(filepath)
    print("Model saved successfully!")

def load_model(filepath: str = MODEL_PATH) -> Optional[tf.keras.Model]:
    """Load a pre-trained model from disk.
    
    Args:
        filepath: Path to the saved model file.
        
    Returns:
        tf.keras.Model: The loaded model, or None if file doesn't exist.
    """
    if os.path.exists(filepath):
        print(f"Loading existing model from {filepath}...")
        return tf.keras.models.load_model(filepath)
    else:
        print(f"No existing model found at {filepath}")
        return None

def retrain_model() -> None:
    """Retrain the model from scratch and save the new version.
    
    This function:
    1. Loads and preprocesses the MNIST dataset
    2. Creates a new CNN model
    3. Trains the model
    4. Saves the trained model to disk
    
    Raises:
        Exception: If training fails or model cannot be saved.
    """
    try:
        print("=== Starting Model Retraining ===")
        # Load and preprocess data.
        x_train, y_train, x_test, y_test = load_and_preprocess_data()
        # Create new model.
        model = create_cnn_model()
        # Display model architecture.
        print("\nModel Architecture:")
        model.summary()
        # Train the model.
        model = train_model(model, x_train, y_train, x_test, y_test, epochs=15)
        # Save the trained model.
        save_model(model)
        print("=== Model Retraining Complete ===")
    except Exception as e:
        print(f"Error during model retraining: {e}")
        raise

def main() -> None:
    """Main function to train a new model if none exists.
    
    This function is called when the module is run directly.
    It checks for an existing model and trains a new one if needed.
    """
    # Check if model already exists.
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        # Load and test existing model.
        model = load_model()
        if model is not None:
            print("Model loaded successfully!")
            # Quick test on a small subset.
            (x_test, _), (_, _) = mnist.load_data()
            x_test = x_test[:5].astype('float32') / 255.0
            x_test = x_test.reshape(5, 28, 28, 1)
            predictions = model.predict(x_test, verbose=0)
            print(f"Sample predictions: {np.argmax(predictions, axis=1)}")
        else:
            print("Failed to load existing model, training new one...")
            retrain_model()
    else:
        print("No existing model found, training new model...")
        retrain_model()

if __name__ == "__main__":
    main()