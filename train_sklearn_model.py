#!/usr/bin/env python3
"""
Train an enhanced scikit-learn model for ultra-lightweight deployment
Optimized for maximum accuracy with multiple techniques
"""
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def enhance_data(x_train, y_train):
    """Enhanced data preprocessing and augmentation."""
    print("🔄 Enhancing training data...")
    
    # Original data
    enhanced_x = x_train.copy()
    enhanced_y = y_train.copy()
    
    # Add slight noise for robustness (data augmentation)
    noise_factor = 0.1
    noisy_x = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    noisy_x = np.clip(noisy_x, 0., 1.)
    
    # Combine original and augmented data
    enhanced_x = np.vstack([enhanced_x, noisy_x])
    enhanced_y = np.hstack([enhanced_y, y_train])
    
    print(f"📊 Enhanced dataset size: {enhanced_x.shape[0]} samples")
    return enhanced_x, enhanced_y

def train_enhanced_sklearn_model():
    """Train an enhanced scikit-learn model with multiple optimizations."""
    print("🔄 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    
    print(f"📊 Original training data shape: {x_train.shape}")
    print(f"📊 Test data shape: {x_test.shape}")
    
    # Enhance training data
    x_train_enhanced, y_train_enhanced = enhance_data(x_train, y_train)
    
    # Scale the data
    print("🔄 Scaling data...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_enhanced)
    x_test_scaled = scaler.transform(x_test)
    
    # Define enhanced models with better hyperparameters
    models = {
        'Enhanced Neural Network': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        ),
        'Optimized Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        'RBF SVM': SVC(
            C=10,
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    
    trained_models = {}
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Use subset for SVM (it's slow)
        if 'SVM' in name:
            subset_size = 15000
            model.fit(x_train_scaled[:subset_size], y_train_enhanced[:subset_size])
        else:
            model.fit(x_train_scaled, y_train_enhanced)
        
        # Test accuracy
        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        
        print(f"✅ {name} accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name
    
    # Create ensemble model for even better accuracy
    print(f"\n🔄 Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('nn', trained_models['Enhanced Neural Network']),
            ('rf', trained_models['Optimized Random Forest']),
        ],
        voting='soft'
    )
    
    # Train ensemble (use smaller subset to avoid memory issues)
    subset_size = 30000
    ensemble.fit(x_train_scaled[:subset_size], y_train_enhanced[:subset_size])
    
    # Test ensemble
    ensemble_pred = ensemble.predict(x_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    print(f"✅ Ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Choose the best performing model
    if ensemble_accuracy > best_accuracy:
        best_accuracy = ensemble_accuracy
        best_model = ensemble
        best_name = "Ensemble"
    
    print(f"\n🏆 Best model: {best_name} with accuracy {best_accuracy:.4f}")
    
    # Save the best model and scaler
    print("💾 Saving enhanced model and scaler...")
    with open('sklearn_digit_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('sklearn_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save enhanced metadata
    metadata = {
        'model_type': best_name,
        'test_accuracy': float(best_accuracy),
        'features': 784,
        'classes': 10,
        'enhanced_training': True,
        'data_augmentation': True,
        'training_samples': len(x_train_enhanced)
    }
    
    with open('sklearn_metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    # Detailed classification report
    if best_name != "Ensemble":
        y_pred_best = best_model.predict(x_test_scaled)
    else:
        y_pred_best = ensemble_pred
    
    print(f"\n📊 Detailed Classification Report for {best_name}:")
    print(classification_report(y_test, y_pred_best))
    
    # Check file sizes
    import os
    model_size = os.path.getsize('sklearn_digit_model.pkl') / 1024
    scaler_size = os.path.getsize('sklearn_scaler.pkl') / 1024
    
    print(f"\n📊 Enhanced model size: {model_size:.1f} KB")
    print(f"📊 Scaler size: {scaler_size:.1f} KB")
    print(f"📊 Total size: {(model_size + scaler_size):.1f} KB")
    
    # Per-digit accuracy analysis
    print(f"\n🎯 Per-digit accuracy analysis:")
    for digit in range(10):
        digit_mask = y_test == digit
        if np.sum(digit_mask) > 0:
            digit_accuracy = accuracy_score(y_test[digit_mask], y_pred_best[digit_mask])
            print(f"   Digit {digit}: {digit_accuracy:.3f} ({np.sum(digit_mask)} samples)")
    
    print("✅ Enhanced scikit-learn model training complete!")
    print(f"🎯 Final accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    return best_model, scaler, best_accuracy

if __name__ == "__main__":
    train_enhanced_sklearn_model()