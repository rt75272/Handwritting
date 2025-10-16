#!/usr/bin/env python3
"""Handwriting Digit Recognition Web Application.

Ultra-lightweight Flask web application using scikit-learn for digit recognition.
Provides a web interface for drawing digits and getting predictions with confidence scores.
Optimized for cloud deployment with minimal dependencies and fast startup times.
"""
import base64
import io
import os
import logging
import traceback
import pickle
from typing import Optional
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Configure application logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application.
app = Flask(__name__)

# Global model variables.
digit_model = None
scaler = None
current_mode = "digits"

def load_sklearn_model():
    """Load the pre-trained scikit-learn model and scaler from disk.
    
    Returns:
        tuple: (model, scaler) if successful, (None, None) if failed
        
    The function loads a pickled scikit-learn model and StandardScaler
    that were created during training. These files must exist for the
    application to provide predictions.
    """
    try:
        logger.info("🔧 Loading scikit-learn digit recognition model...")
        model_path = 'sklearn_digit_model.pkl'
        scaler_path = 'sklearn_scaler.pkl'
        # Check if model file exists.
        if not os.path.exists(model_path):
            logger.error(f"❌ Sklearn model file {model_path} not found!")
            logger.info(f"📁 Files in directory: {os.listdir('.')}")
            logger.info("💡 Run 'python train_sklearn_model.py' locally to create the sklearn model")
            return None, None
        # Load the trained model.
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Load the data scaler (optional but recommended).
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        logger.info(f"✅ Scikit-learn model loaded successfully")
        logger.info(f"📊 Model type: {type(model).__name__}")
        # Test the model with random data to verify it works.
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

def enhanced_preprocess(image_data: str) -> Optional[np.ndarray]:
    """Convert base64 image data to normalized numpy array for model prediction.
    
    Args:
        image_data: Base64 encoded image string from HTML5 canvas
        
    Returns:
        numpy.ndarray: Processed 784-element array ready for model input,
                      or None if processing fails
                      
    This function performs several preprocessing steps:
    1. Decodes base64 image data
    2. Converts to grayscale if needed
    3. Crops to content with padding
    4. Resizes to 28x28 pixels (MNIST standard)
    5. Inverts colors if needed (white digits on black background)
    6. Normalizes pixel values to [0,1] range
    7. Flattens to 784-element vector
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        # Find the bounding box of drawn content to crop efficiently
        bbox = image.getbbox()
        if bbox:
            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top
            
            # Add padding (20% of larger dimension) to preserve aspect ratio
            padding = int(0.2 * max(width, height))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)
            image = image.crop((left, top, right, bottom))
        # Resize to MNIST standard 28x28 pixels
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        # Convert to numpy array for mathematical operations
        img_array = np.array(image, dtype=np.float32)
        # Invert colors if needed (MNIST expects white digits on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        # Normalize pixel values to [0, 1] range
        img_array = img_array / 255.0
        # Flatten to 784-element vector for scikit-learn input
        img_array = img_array.reshape(1, 784)
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return None

@app.route('/')
def index():
    """Serve the main drawing interface page.
    
    Returns:
        str: Rendered HTML template for the digit drawing interface
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from canvas drawing using the trained model.
    
    Expected JSON input:
        {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        }
    
    Returns:
        JSON response with prediction results:
        {
            "prediction": 7,
            "confidence": "98.2%", 
            "probabilities": [0.001, 0.002, ..., 0.982, ...]
        }
    """
    try:
        # Verify model is loaded
        if digit_model is None:
            logger.error("❌ No trained model available")
            return jsonify({'error': 'Model not loaded'}), 500
        # Extract image data from request
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        # Preprocess the image for model input
        processed_image = enhanced_preprocess(image_data)
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        # Apply data scaling if scaler is available
        if scaler is not None:
            processed_image = scaler.transform(processed_image)
        # Make prediction using the trained model
        predicted_digit = int(digit_model.predict(processed_image)[0])
        probabilities = digit_model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities))
        logger.info(f"🎯 Prediction: {predicted_digit} (confidence: {confidence:.3f})")
        # Return prediction results as JSON.
        return jsonify({
            'prediction': predicted_digit,
            'confidence': f"{confidence * 100:.1f}%",
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/health')
def health():
    """Application health check endpoint for monitoring.
    
    Returns:
        JSON response with application status and model availability.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': digit_model is not None,
        'mode': current_mode
    })

def initialize_models():
    """Initialize the digit recognition model and scaler.
    
    Loads the pre-trained scikit-learn model and StandardScaler
    from pickle files. This function is called automatically
    when the module is imported (for gunicorn compatibility).
    """
    global digit_model, scaler
    try:
        logger.info("🚀 Initializing ultra-lightweight digit recognition...")
        # Load the sklearn model and scaler from disk
        digit_model, scaler = load_sklearn_model()
        if digit_model is not None:
            logger.info("✅ Scikit-learn model loaded successfully")
        else:
            logger.error("❌ Failed to load model - predictions will not work")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
    logger.info("✅ Initialization complete")

# Initialize models at module level for gunicorn compatibility
# (gunicorn doesn't execute the if __name__ == '__main__' block)
logger.info("🚀 Starting ultra-lightweight digit recognition app")
logger.info("📝 Note: Using scikit-learn for minimal deployment")
initialize_models()

if __name__ == '__main__':
    # Local development server configuration
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🎯 Starting local development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)