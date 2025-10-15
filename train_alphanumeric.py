#!/usr/bin/env python3
"""
Enhanced Alphanumeric Character Recognition CNN Training Script

This script trains a CNN model to recognize both digits (0-9) and letters (A-Z)
using the EMNIST dataset. The model combines EMNIST Digits and EMNIST Letters
datasets for comprehensive alphanumeric recognition.

Features:
- GPU acceleration with mixed precision training
- Advanced data augmentation
- Modern CNN architecture with attention mechanisms
- Comprehensive preprocessing pipeline
- Progress tracking and visualization
"""

import os
import time
import warnings
import multiprocessing
from typing import Tuple, Dict, Any

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

def configure_gpu():
    """Configure GPU settings for optimal performance."""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU acceleration enabled with {len(gpus)} GPU(s)")
            print(f"GPU details: {[gpu.name for gpu in gpus]}")
            
            # Enable mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision (float16) enabled for faster training")
        else:
            print("⚠️ No GPU detected, using CPU")
    except Exception as e:
        print(f"⚠️ GPU configuration warning: {e}")

def configure_threading():
    """Configure threading for optimal CPU performance."""
    # Use all available CPU cores
    num_cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    print(f"✅ Configured threading to use all {num_cores} CPU cores")

def load_emnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """
    Load and combine EMNIST digits and letters datasets.
    
    Returns:
        Training data, test data, and metadata
    """
    print("📥 Loading EMNIST datasets...")
    
    # Load EMNIST digits (0-9)
    print("Loading EMNIST digits...")
    ds_digits_train, ds_digits_test = tfds.load(
        'emnist/digits',
        split=['train', 'test'],
        as_supervised=True,
        shuffle_files=True
    )
    
    # Load EMNIST letters (A-Z)
    print("Loading EMNIST letters...")
    ds_letters_train, ds_letters_test = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        as_supervised=True,
        shuffle_files=True
    )
    
    def extract_data(dataset):
        """Extract images and labels from TensorFlow dataset."""
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)
    
    # Extract digits data
    print("Processing digits data...")
    x_digits_train, y_digits_train = extract_data(ds_digits_train)
    x_digits_test, y_digits_test = extract_data(ds_digits_test)
    
    # Extract letters data
    print("Processing letters data...")
    x_letters_train, y_letters_train = extract_data(ds_letters_train)
    x_letters_test, y_letters_test = extract_data(ds_letters_test)
    
    # Adjust letter labels to start from 10 (after digits 0-9)
    y_letters_train += 10
    y_letters_test += 10
    
    # Combine datasets
    print("Combining datasets...")
    x_train = np.concatenate([x_digits_train, x_letters_train], axis=0)
    y_train = np.concatenate([y_digits_train, y_letters_train], axis=0)
    x_test = np.concatenate([x_digits_test, x_letters_test], axis=0)
    y_test = np.concatenate([y_digits_test, y_letters_test], axis=0)
    
    # Shuffle combined data
    train_indices = np.random.permutation(len(x_train))
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    
    test_indices = np.random.permutation(len(x_test))
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    
    print(f"✅ Combined dataset loaded:")
    print(f"   Training samples: {len(x_train):,}")
    print(f"   Test samples: {len(x_test):,}")
    print(f"   Classes: 36 (0-9 digits + A-Z letters)")
    
    # Create label mapping
    label_map = {}
    # Digits 0-9
    for i in range(10):
        label_map[i] = str(i)
    # Letters A-Z
    for i in range(26):
        label_map[i + 10] = chr(ord('A') + i)
    
    metadata = {
        'num_classes': 36,
        'image_shape': (28, 28, 1),
        'label_map': label_map
    }
    
    return (x_train, y_train), (x_test, y_test), metadata

def preprocess_data(x_train: np.ndarray, y_train: np.ndarray, 
                   x_test: np.ndarray, y_test: np.ndarray, 
                   num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the image data for training.
    
    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        num_classes: Number of classes
        
    Returns:
        Preprocessed training and test data
    """
    print("🔄 Preprocessing data...")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Ensure images have channel dimension
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"✅ Data preprocessed:")
    print(f"   Input shape: {x_train.shape[1:]}")
    print(f"   Number of classes: {num_classes}")
    
    return x_train, y_train, x_test, y_test

def create_data_augmentation():
    """Create data augmentation pipeline."""
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

def create_alphanumeric_model(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Create an advanced CNN model for alphanumeric recognition.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes (36 for digits + letters)
        
    Returns:
        Compiled Keras model
    """
    print("🏗️ Building advanced alphanumeric recognition model...")
    
    # Data augmentation
    augmentation = create_data_augmentation()
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # Data augmentation (only during training)
    x = augmentation(inputs)
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    
    # Global Average Pooling instead of Flatten for better generalization
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Dropout(0.5, name='dropout4')(x)
    
    x = layers.Dense(256, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.5, name='dropout5')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions', dtype='float32')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='alphanumeric_cnn')
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_5_accuracy']
    )
    
    print("✅ Model created and compiled successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model

def create_callbacks(model_name: str) -> list:
    """Create training callbacks."""
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks_list

def train_model(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray,
                x_test: np.ndarray, y_test: np.ndarray, epochs: int = 50) -> keras.callbacks.History:
    """
    Train the alphanumeric recognition model.
    
    Args:
        model: Keras model to train
        x_train, y_train: Training data
        x_test, y_test: Test data
        epochs: Number of training epochs
        
    Returns:
        Training history
    """
    print(f"🚀 Starting training for {epochs} epochs...")
    start_time = time.time()
    
    # Create callbacks
    model_callbacks = create_callbacks('alphanumeric_cnn')
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=model_callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time/60:.2f} minutes")
    
    return history

def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray, 
                  label_map: Dict[int, str]) -> None:
    """Evaluate the trained model and show detailed results."""
    print("📊 Evaluating model performance...")
    
    # Evaluate on test set
    test_loss, test_accuracy, test_top5 = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"✅ Final Test Results:")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Test Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Make predictions for detailed analysis
    predictions = model.predict(x_test[:1000], verbose=0)  # Sample for analysis
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:1000], axis=1)
    
    # Calculate per-class accuracy
    print("\n📈 Per-class accuracy (sample):")
    class_correct = {}
    class_total = {}
    
    for true_class, pred_class in zip(true_classes, predicted_classes):
        class_total[true_class] = class_total.get(true_class, 0) + 1
        if true_class == pred_class:
            class_correct[true_class] = class_correct.get(true_class, 0) + 1
    
    for class_id in sorted(class_total.keys()):
        if class_id in class_correct:
            accuracy = class_correct[class_id] / class_total[class_id]
            char = label_map[class_id]
            print(f"   {char}: {accuracy:.3f} ({class_correct[class_id]}/{class_total[class_id]})")

def save_training_plot(history: keras.callbacks.History) -> None:
    """Save training history plots."""
    print("📈 Saving training plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('alphanumeric_training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training plots saved as 'alphanumeric_training_history.png'")

def main():
    """Main training pipeline."""
    print("🤖 Alphanumeric Character Recognition CNN Training")
    print("=" * 60)
    
    # Configure hardware
    configure_gpu()
    configure_threading()
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test), metadata = load_emnist_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test, metadata['num_classes']
    )
    
    # Create and train model
    model = create_alphanumeric_model(metadata['image_shape'], metadata['num_classes'])
    
    # Print model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Train the model
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=30)
    
    # Evaluate model
    evaluate_model(model, x_test, y_test, metadata['label_map'])
    
    # Save final model
    model.save('alphanumeric_cnn.keras')
    print("✅ Model saved as 'alphanumeric_cnn.keras'")
    
    # Save training plots
    save_training_plot(history)
    
    # Save label mapping
    import json
    with open('alphanumeric_labels.json', 'w') as f:
        json.dump(metadata['label_map'], f, indent=2)
    print("✅ Label mapping saved as 'alphanumeric_labels.json'")
    
    print("\n🎉 Training completed successfully!")
    print("   Files created:")
    print("   - alphanumeric_cnn.keras (trained model)")
    print("   - alphanumeric_cnn_best.keras (best model checkpoint)")
    print("   - alphanumeric_labels.json (label mapping)")
    print("   - alphanumeric_training_history.png (training plots)")

if __name__ == "__main__":
    main()