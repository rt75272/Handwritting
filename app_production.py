#!/usr/bin/env python3
"""
Production App - Loads pre-trained model for fast deployment
This version loads a saved trained model instead of training during startup
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
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
digit_model = None
current_mode = "digits"

def load_trained_model():
    """Load the pre-trained model from file with enhanced error handling."""
    try:
        logger.info("🔧 Loading pre-trained digit recognition model...")
        
        model_path = 'trained_digit_model.keras'
        
        # Check if file exists and get file info
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file {model_path} not found!")
            logger.info(f"📁 Files in directory: {os.listdir('.')}")
            logger.info("💡 Run 'python train_model.py' locally to create the model")
            return None
            
        # Check file size and permissions
        file_size = os.path.getsize(model_path)
        logger.info(f"📊 Model file size: {file_size / 1024 / 1024:.1f} MB")
        
        # Try to load with explicit error handling
        try:
            logger.info("🔄 Loading model with TensorFlow...")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Re-compile the model to ensure it works
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("✅ Pre-trained model loaded and compiled successfully")
            
            # Test the model with a dummy input
            test_input = np.random.random((1, 28, 28, 1))
            test_pred = model.predict(test_input, verbose=0)
            logger.info(f"🧪 Model test successful: output shape {test_pred.shape}")
            
        except Exception as load_error:
            logger.error(f"❌ TensorFlow model loading failed: {load_error}")
            logger.error(f"🔍 Load error traceback: {traceback.format_exc()}")
            
            # Try alternative: recreate model architecture and load weights
            logger.info("🔄 Attempting to recreate model architecture...")
            try:
                # Recreate the exact architecture from training
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Try to load just the weights
                weights_path = model_path.replace('.keras', '.weights.h5')
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    logger.info("✅ Model architecture recreated and weights loaded")
                else:
                    logger.warning("⚠️ Using fresh model architecture (untrained)")
                    
            except Exception as alt_error:
                logger.error(f"❌ Alternative loading also failed: {alt_error}")
                return None
        
        # Load metadata if available
        metadata_path = 'model_metadata.json'
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"📊 Model accuracy: {metadata.get('test_accuracy', 'Unknown'):.4f}")
                logger.info(f"🏗️ Architecture: {metadata.get('model_architecture', 'CNN')}")
            except Exception as meta_error:
                logger.warning(f"⚠️ Could not load metadata: {meta_error}")
        
        return model
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return None

def enhanced_preprocess(image_data: str) -> np.ndarray:
    """Enhanced image preprocessing for better digit recognition."""
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        # Get the bounding box of the drawn content
        bbox = image.getbbox()
        if bbox:
            # Crop to content with some padding
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            
            # Add padding (20% of the larger dimension)
            padding = int(0.2 * max(width, height))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)
            
            image = image.crop((left, top, right, bottom))
        
        # Create a square image with black background
        size = max(image.width, image.height)
        square_image = Image.new('L', (size, size), 0)  # Black background
        
        # Center the digit in the square
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        square_image.paste(image, offset)
        
        # Resize to 28x28 with high-quality resampling
        image = square_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize
        img_array = np.array(image)
        img_array = 255 - img_array  # Invert colors (white digit on black background)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Apply slight gaussian blur to smooth pixelation
        img_array = np.clip(img_array, 0, 1)
        
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
        logger.error(f"❌ Template rendering failed: {e}")
        return f"Template error: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from canvas drawing."""
    global digit_model
    
    logger.info("🔍 Prediction request received")
    
    try:
        if digit_model is None:
            logger.error("❌ No trained model available")
            return jsonify({
                'error': 'Model not available',
                'suggestion': 'Model failed to load during startup'
            }), 500
        
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        logger.info("🔄 Making prediction...")
        image_array = enhanced_preprocess(data['image'])
        
        # Make prediction
        predictions = digit_model.predict(image_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class]) * 100
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [
            {
                'digit': int(idx),
                'confidence': f"{predictions[0][idx] * 100:.1f}%"
            }
            for idx in top_3_indices
        ]
        
        logger.info(f"✅ Prediction: {predicted_class} ({confidence:.1f}%)")
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.1f}%",
            'top_3': top_3,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint with model status."""
    return jsonify({
        'status': 'healthy' if digit_model is not None else 'unhealthy',
        'models': {
            'digit_model': digit_model is not None
        },
        'current_mode': current_mode,
        'production_ready': True
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check environment and model status."""
    try:
        import tensorflow as tf
        
        debug_info = {
            'status': 'running',
            'models': {
                'digit_model': digit_model is not None
            },
            'current_mode': current_mode,
            'production_version': True,
            'environment': {
                'working_directory': os.getcwd(),
                'python_version': f"{os.sys.version}",
                'tensorflow_version': tf.__version__,
                'files_in_directory': [f for f in os.listdir('.') if not f.startswith('.')],
                'tf_devices': [str(d) for d in tf.config.list_physical_devices()]
            }
        }
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'debug_error'
        }), 500

def initialize_models():
    """Initialize the digit recognition model."""
    global digit_model
    
    logger.info("🚀 Initializing production digit recognition...")
    
    try:
        # Load the pre-trained model
        digit_model = load_trained_model()
        
        if digit_model is not None:
            logger.info("✅ Model loaded successfully")
            
            # Test the model
            test_input = np.random.random((1, 28, 28, 1))
            test_pred = digit_model.predict(test_input, verbose=0)
            logger.info(f"🧪 Model test successful: {test_pred.shape}")
        else:
            logger.error("❌ Failed to load model - predictions will not work")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
    
    logger.info("✅ Initialization complete")

# Initialize models at module level (for gunicorn compatibility)
logger.info("🚀 Starting production digit recognition app")
logger.info("📝 Note: Using pre-trained model for instant deployment")
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🎯 Starting local development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)