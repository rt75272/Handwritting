#!/usr/bin/env python3
"""
Train a scikit-learn model for ultra-lightweight deployment
Much smaller than TensorFlow and works everywhere
"""
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def train_sklearn_model():
    """Train a scikit-learn model on MNIST data."""
    print("🔄 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    
    print(f"📊 Training data shape: {x_train.shape}")
    print(f"📊 Test data shape: {x_test.shape}")
    
    # Scale the data
    print("🔄 Scaling data...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Try different models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use subset for SVM (it's slow)
        if name == 'SVM':
            subset_size = 10000
            model.fit(x_train_scaled[:subset_size], y_train[:subset_size])
        else:
            model.fit(x_train_scaled, y_train)
        
        # Test accuracy
        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ {name} accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} with accuracy {best_accuracy:.4f}")
    
    # Save the best model and scaler
    print("💾 Saving model and scaler...")
    with open('sklearn_digit_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('sklearn_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'model_type': best_name,
        'test_accuracy': float(best_accuracy),
        'features': 784,
        'classes': 10
    }
    
    with open('sklearn_metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    # Check file sizes
    import os
    model_size = os.path.getsize('sklearn_digit_model.pkl') / 1024
    scaler_size = os.path.getsize('sklearn_scaler.pkl') / 1024
    
    print(f"📊 Model size: {model_size:.1f} KB")
    print(f"📊 Scaler size: {scaler_size:.1f} KB")
    print(f"📊 Total size: {(model_size + scaler_size):.1f} KB")
    
    # Compare with TensorFlow model if it exists
    if os.path.exists('trained_digit_model.keras'):
        tf_size = os.path.getsize('trained_digit_model.keras') / 1024 / 1024
        sklearn_size_mb = (model_size + scaler_size) / 1024
        print(f"📊 TensorFlow model: {tf_size:.1f} MB")
        print(f"📊 Scikit-learn model: {sklearn_size_mb:.1f} MB")
        print(f"💾 Size reduction: {((tf_size - sklearn_size_mb) / tf_size * 100):.1f}%")
    
    print("✅ Scikit-learn model training complete!")
    return best_model, scaler, best_accuracy

if __name__ == "__main__":
    train_sklearn_model()