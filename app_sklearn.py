#!/usr/bin/env python3
"""
Ultra-Lightweight App - Uses scikit-learn instead of TensorFlow
Smallest possible deployment with good accuracy
"""
import base64
import io
import json
import os
import logging
import traceback
import pickle
from typing import Tuple, Optional, Dict, List
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
digit_model = None
scaler = None
current_mode = "digits"

def load_sklearn_model():
    """Load the scikit-learn model for ultra-lightweight deployment."""
    try:
        logger.info("🔧 Loading scikit-learn digit recognition model...")
        
        model_path = 'sklearn_digit_model.pkl'
        scaler_path = 'sklearn_scaler.pkl'
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Sklearn model file {model_path} not found!")
            logger.info(f"📁 Files in directory: {os.listdir('.')}")
            logger.info("💡 Run 'python train_sklearn_model.py' locally to create the sklearn model")
            return None, None
        
        # Load model and scaler
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        logger.info(f"✅ Scikit-learn model loaded successfully")
        logger.info(f"📊 Model type: {type(model).__name__}")
        
        # Test the model
        test_input = np.random.random((1, 784))
        if scaler:
            test_input = scaler.transform(test_input)
        test_pred = model.predict(test_input)
        test_proba = model.predict_proba(test_input)
        logger.info(f"🧪 Model test successful: prediction={test_pred[0]}, shape={test_proba.shape}")
        
        return model, scaler
            
    except Exception as e:
        logger.error(f"❌ Failed to load sklearn model: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return None, None

def enhanced_preprocess(image_data: str) -> np.ndarray:
    """Enhanced image preprocessing for digit recognition."""
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
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        
        # Invert if needed (MNIST expects white digits on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Flatten for sklearn (784 features)
        img_array = img_array.reshape(1, 784)
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from canvas drawing."""
    try:
        if digit_model is None:
            logger.error("❌ No trained model available")
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess the image
        processed_image = enhanced_preprocess(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Apply scaling if available
        if scaler is not None:
            processed_image = scaler.transform(processed_image)
        
        # Make prediction with sklearn
        predicted_digit = int(digit_model.predict(processed_image)[0])
        probabilities = digit_model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities))
        
        logger.info(f"🎯 Prediction: {predicted_digit} (confidence: {confidence:.3f})")
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': digit_model is not None,
        'mode': current_mode
    })

def initialize_models():
    """Initialize the digit recognition model."""
    global digit_model, scaler
    
    try:
        logger.info("🚀 Initializing ultra-lightweight digit recognition...")
        
        # Load the sklearn model
        digit_model, scaler = load_sklearn_model()
        
        if digit_model is not None:
            logger.info("✅ Scikit-learn model loaded successfully")
        else:
            logger.error("❌ Failed to load model - predictions will not work")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
    
    logger.info("✅ Initialization complete")

# Initialize models at module level (for gunicorn compatibility)
logger.info("🚀 Starting ultra-lightweight digit recognition app")
logger.info("📝 Note: Using scikit-learn for minimal deployment")
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🎯 Starting local development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)