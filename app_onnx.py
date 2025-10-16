#!/usr/bin/env python3
"""
Lightweight App - Uses ONNX Runtime instead of TensorFlow
Much smaller deployment size and faster startup
"""
import base64
import io
import json
import os
import logging
import traceback
from typing import Tuple, Optional, Dict, List
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
digit_model = None
current_mode = "digits"

def load_onnx_model():
    """Load the ONNX model for lightweight deployment."""
    try:
        logger.info("🔧 Loading ONNX digit recognition model...")
        
        model_path = 'trained_digit_model.onnx'
        
        if not os.path.exists(model_path):
            logger.error(f"❌ ONNX model file {model_path} not found!")
            logger.info(f"📁 Files in directory: {os.listdir('.')}")
            logger.info("💡 Run 'python convert_to_onnx.py' locally to create the ONNX model")
            return None
        
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        
        # Get model info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        logger.info(f"✅ ONNX model loaded successfully")
        logger.info(f"📊 Input: {input_name}, Output: {output_name}")
        
        # Test the model
        test_input = np.random.random((1, 28, 28, 1)).astype(np.float32)
        test_pred = session.run([output_name], {input_name: test_input})
        logger.info(f"🧪 Model test successful: output shape {test_pred[0].shape}")
        
        return session
            
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX model: {e}")
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
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        
        # Invert if needed (MNIST expects white digits on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
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
        
        # Make prediction with ONNX
        input_name = digit_model.get_inputs()[0].name
        output_name = digit_model.get_outputs()[0].name
        
        predictions = digit_model.run([output_name], {input_name: processed_image})
        probabilities = predictions[0][0]
        
        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(probabilities))
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
    global digit_model
    
    try:
        logger.info("🚀 Initializing lightweight digit recognition...")
        
        # Load the ONNX model
        digit_model = load_onnx_model()
        
        if digit_model is not None:
            logger.info("✅ ONNX model loaded successfully")
        else:
            logger.error("❌ Failed to load model - predictions will not work")
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
    
    logger.info("✅ Initialization complete")

# Initialize models at module level (for gunicorn compatibility)
logger.info("🚀 Starting lightweight digit recognition app")
logger.info("📝 Note: Using ONNX Runtime for fast deployment")
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🎯 Starting local development server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)