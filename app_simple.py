#!/usr/bin/env python3
"""
Minimal Working App - Recreates model architecture instead of loading saved models
This avoids TensorFlow version compatibility issues in cloud deployment
"""
import base64
import io
import json
import os
import logging
import traceback
from typing import Tuple, Optional, Dict, List
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
digit_model = None
current_mode = "digits"

def create_digit_model():
    """Create a new digit recognition model with simple architecture."""
    try:
        logger.info("🔧 Creating digit recognition model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ Created fresh digit model")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to create digit model: {e}")
        raise

def simple_preprocess(image_data: str) -> np.ndarray:
    """Simple image preprocessing without scipy dependencies."""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize directly to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize
        img_array = np.array(image)
        img_array = 255 - img_array  # Invert colors
        img_array = img_array / 255.0  # Normalize
        
        # Reshape for model
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

@app.route('/')
def index():
    """Render the main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h1>Character Recognition App</h1><p>Templates not found. App is running but UI needs templates folder.</p><p>Use /predict endpoint directly for testing.</p>", 200

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'digit_model': digit_model is not None,
            'alphanumeric_model': False  # Only digit model for now
        },
        'current_mode': current_mode,
        'simplified': True
    })

@app.route('/debug')
def debug_info():
    """Debug information."""
    return jsonify({
        'status': 'running',
        'current_mode': current_mode,
        'models': {
            'digit_model': digit_model is not None,
            'alphanumeric_model': False
        },
        'environment': {
            'python_version': os.sys.version,
            'tensorflow_version': tf.__version__,
            'working_directory': os.getcwd(),
            'files_in_directory': sorted(os.listdir('.')),
            'tf_devices': [str(d) for d in tf.config.list_physical_devices()]
        },
        'simplified_version': True
    })

@app.route('/api/mode', methods=['GET', 'POST'])
def handle_mode():
    """Handle mode (only digits supported in minimal version)."""
    global current_mode
    
    if request.method == 'POST':
        return jsonify({
            'success': True,
            'mode': 'digits',
            'message': 'Only digits mode available in minimal version'
        })
    
    return jsonify({
        'current_mode': 'digits',
        'available_modes': {
            'digits': digit_model is not None,
            'alphanumeric': False
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        logger.info("🔍 Prediction request received")
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        if digit_model is None:
            return jsonify({'error': 'Digit model not available'}), 500
        
        # Preprocess image
        image_array = simple_preprocess(data['image'])
        
        if image_array is None or image_array.size == 0:
            return jsonify({'error': 'Image preprocessing failed'}), 400
        
        # Make prediction
        logger.info("🔄 Making prediction...")
        predictions = digit_model.predict(image_array, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            results.append({
                'character': str(idx),  # Digit 0-9
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            })
        
        response = {
            'success': True,
            'mode': 'digits',
            'prediction': results[0]['character'],
            'confidence': results[0]['confidence_percent'],
            'top_3': results
        }
        
        logger.info(f"✅ Prediction: {results[0]['character']} ({results[0]['confidence_percent']})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test')
def test_prediction():
    """Test endpoint with dummy prediction."""
    try:
        if digit_model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Create test input
        test_input = np.random.random((1, 28, 28, 1))
        predictions = digit_model.predict(test_input, verbose=0)
        
        return jsonify({
            'test_successful': True,
            'prediction_shape': str(predictions.shape),
            'sample_predictions': predictions[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_models():
    """Initialize the digit model."""
    global digit_model
    
    logger.info("🚀 Initializing minimal digit recognition...")
    
    try:
        # Create a fresh model (no loading from file)
        digit_model = create_digit_model()
        logger.info("✅ Digit model created successfully")
        
        # Test the model
        test_input = np.random.random((1, 28, 28, 1))
        test_pred = digit_model.predict(test_input, verbose=0)
        logger.info(f"🧪 Model test successful: {test_pred.shape}")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        digit_model = None
    
    logger.info("✅ Initialization complete")

if __name__ == '__main__':
    # Initialize models
    initialize_models()
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting minimal working app on port {port}")
    logger.info("📝 Note: This is a simplified version with fresh model (not pre-trained)")
    logger.info("🎯 Predictions will be random until model is trained")
    
    app.run(host='0.0.0.0', port=port, debug=False)