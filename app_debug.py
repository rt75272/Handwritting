#!/usr/bin/env python3
"""
Debug Version of Production App
Enhanced logging to troubleshoot Render deployment issues
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

# Enhanced logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Global variables for models and configurations
digit_model = None
alphanumeric_model = None
alphanumeric_labels = None
current_mode = "digits"

def debug_environment():
    """Debug function to check the environment."""
    logger.info("🔍 Environment Debug Info:")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    # Check for model files
    model_files = ['mnist_cnn.h5', 'alphanumeric_cnn.keras', 'alphanumeric_labels.json']
    for file in model_files:
        exists = os.path.exists(file)
        if exists:
            size = os.path.getsize(file)
            logger.info(f"✅ {file}: exists ({size} bytes)")
        else:
            logger.error(f"❌ {file}: not found")
    
    # Check TensorFlow GPU/CPU
    logger.info(f"TensorFlow devices: {tf.config.list_physical_devices()}")

def load_digit_model() -> tf.keras.Model:
    """Load the digit-only MNIST CNN model with enhanced error handling."""
    model_path = "mnist_cnn.h5"
    
    try:
        logger.info(f"🔍 Attempting to load digit model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"📁 Model file exists, size: {os.path.getsize(model_path)} bytes")
        
        # Load with custom objects to handle any compilation issues
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"📊 Input shape: {model.input_shape}")
        logger.info(f"📊 Output shape: {model.output_shape}")
        
        # Test prediction to ensure model works
        test_input = np.random.random((1, 28, 28, 1))
        test_pred = model.predict(test_input, verbose=0)
        logger.info(f"🧪 Test prediction shape: {test_pred.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load digit model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_alphanumeric_model() -> Tuple[tf.keras.Model, Dict[str, str]]:
    """Load the alphanumeric CNN model and label mapping with enhanced error handling."""
    model_path = "alphanumeric_cnn.keras"
    labels_path = "alphanumeric_labels.json"
    
    try:
        logger.info(f"🔍 Attempting to load alphanumeric model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Alphanumeric model not found: {model_path}")
            return None, None
        
        if not os.path.exists(labels_path):
            logger.warning(f"⚠️ Label mapping not found: {labels_path}")
            return None, None
        
        logger.info(f"📁 Model file exists, size: {os.path.getsize(model_path)} bytes")
        logger.info(f"📁 Labels file exists, size: {os.path.getsize(labels_path)} bytes")
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info(f"✅ Alphanumeric model loaded successfully")
        logger.info(f"📊 Input shape: {model.input_shape}")
        logger.info(f"📊 Output shape: {model.output_shape}")
        
        # Load labels
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            labels = {int(k): v for k, v in labels.items()}
        
        logger.info(f"✅ Labels loaded: {len(labels)} classes")
        logger.info(f"📝 Sample labels: {dict(list(labels.items())[:5])}")
        
        # Test prediction
        test_input = np.random.random((1, 28, 28, 1))
        test_pred = model.predict(test_input, verbose=0)
        logger.info(f"🧪 Test prediction shape: {test_pred.shape}")
        
        return model, labels
        
    except Exception as e:
        logger.error(f"❌ Failed to load alphanumeric model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def gaussian_blur_numpy(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Apply Gaussian blur using PIL."""
    try:
        pil_image = Image.fromarray(image.astype(np.uint8))
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(blurred)
    except Exception as e:
        logger.warning(f"Gaussian blur failed: {e}, returning original image")
        return image

def center_of_mass_numpy(image: np.ndarray) -> Tuple[float, float]:
    """Calculate center of mass using pure numpy."""
    try:
        total = np.sum(image)
        if total == 0:
            return np.nan, np.nan
        
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        cy = np.sum(y_coords * image) / total
        cx = np.sum(x_coords * image) / total
        
        return cy, cx
    except Exception as e:
        logger.warning(f"Center of mass calculation failed: {e}")
        return np.nan, np.nan

def shift_image_numpy(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Shift image using numpy roll."""
    try:
        shift_y_int = int(round(shift_y))
        shift_x_int = int(round(shift_x))
        
        shifted = np.roll(image, shift_y_int, axis=0)
        shifted = np.roll(shifted, shift_x_int, axis=1)
        
        if shift_y_int > 0:
            shifted[:shift_y_int, :] = 0
        elif shift_y_int < 0:
            shifted[shift_y_int:, :] = 0
            
        if shift_x_int > 0:
            shifted[:, :shift_x_int] = 0
        elif shift_x_int < 0:
            shifted[:, shift_x_int:] = 0
        
        return shifted
    except Exception as e:
        logger.warning(f"Image shift failed: {e}, returning original image")
        return image

def preprocess_image(image_data: str) -> np.ndarray:
    """Preprocess canvas image data with detailed error logging."""
    try:
        logger.debug("🔍 Starting image preprocessing")
        
        if not image_data or not isinstance(image_data, str):
            raise ValueError("Invalid image data provided")
        
        # Decode base64 image data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        logger.debug(f"📊 Base64 data length: {len(image_data)}")
        
        image_bytes = base64.b64decode(image_data)
        logger.debug(f"📊 Decoded image bytes: {len(image_bytes)}")
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        logger.debug(f"📊 PIL image mode: {image.mode}, size: {image.size}")
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        logger.debug(f"📊 Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        if img_array.size == 0:
            raise ValueError("Empty image provided")
        
        # Invert colors
        img_array = 255 - img_array
        
        # Apply Gaussian blur
        img_array = gaussian_blur_numpy(img_array, sigma=0.8)
        
        # Find bounding box
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            padding = 10
            rmin = max(0, rmin - padding)
            rmax = min(img_array.shape[0], rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(img_array.shape[1], cmax + padding)
            
            cropped = img_array[rmin:rmax, cmin:cmax]
            logger.debug(f"📊 Cropped to: {cropped.shape}")
        else:
            h, w = img_array.shape
            margin = min(h, w) // 4
            cropped = img_array[margin:h-margin, margin:w-margin]
            logger.debug(f"📊 No content detected, using center: {cropped.shape}")
        
        if cropped.size == 0:
            cropped = img_array
        
        # Resize to 20x20
        h, w = cropped.shape
        if h > w:
            new_h, new_w = 20, max(1, int(20 * w / h))
        else:
            new_h, new_w = max(1, int(20 * h / w)), 20
        
        resized = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS))
        logger.debug(f"📊 Resized to: {resized.shape}")
        
        # Center in 28x28 canvas
        final_img = np.zeros((28, 28), dtype=np.float32)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Center of mass adjustment
        cy, cx = center_of_mass_numpy(final_img)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = np.clip(14 - cy, -7, 7)
            shift_x = np.clip(14 - cx, -7, 7)
            final_img = shift_image_numpy(final_img, shift_y, shift_x)
        
        # Normalize and reshape
        final_img = final_img / 255.0
        final_img = final_img.reshape(1, 28, 28, 1)
        
        logger.debug(f"✅ Final image shape: {final_img.shape}, range: [{final_img.min():.3f}, {final_img.max():.3f}]")
        return final_img
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

@app.route('/')
def index():
    """Render the main application page."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return f"Error loading page: {e}", 500

@app.route('/debug')
def debug_info():
    """Debug endpoint to check app status."""
    try:
        info = {
            'status': 'running',
            'models': {
                'digit_model': digit_model is not None,
                'alphanumeric_model': alphanumeric_model is not None
            },
            'current_mode': current_mode,
            'environment': {
                'python_version': os.sys.version,
                'tensorflow_version': tf.__version__,
                'working_directory': os.getcwd(),
                'files_in_directory': os.listdir('.'),
                'tf_devices': [str(d) for d in tf.config.list_physical_devices()]
            }
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Debug info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'digit_model': digit_model is not None,
            'alphanumeric_model': alphanumeric_model is not None
        },
        'current_mode': current_mode
    })

@app.route('/api/mode', methods=['GET', 'POST'])
def handle_mode():
    """Handle mode switching with detailed logging."""
    global current_mode
    
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'No data provided'}), 400
            
            mode = data.get('mode', 'digits')
            
            if mode in ['digits', 'alphanumeric']:
                current_mode = mode
                logger.info(f"🔄 Mode switched to: {mode}")
                return jsonify({
                    'success': True,
                    'mode': current_mode,
                    'message': f'Switched to {mode} recognition mode'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Invalid mode. Use "digits" or "alphanumeric"'
                }), 400
        
        return jsonify({
            'current_mode': current_mode,
            'available_modes': {
                'digits': digit_model is not None,
                'alphanumeric': alphanumeric_model is not None
            }
        })
    except Exception as e:
        logger.error(f"Mode handling error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Internal error: {e}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with extensive error logging."""
    try:
        logger.info("🔍 Prediction request received")
        logger.debug(f"Request headers: {dict(request.headers)}")
        logger.debug(f"Request content type: {request.content_type}")
        
        if not request.is_json:
            logger.error("❌ Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        logger.debug(f"Request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data or 'image' not in data:
            logger.error("❌ No image data in request")
            return jsonify({'error': 'No image data provided'}), 400
        
        logger.debug(f"Image data length: {len(data['image'])}")
        
        # Check models
        logger.info(f"🔍 Current mode: {current_mode}")
        logger.info(f"🔍 Digit model available: {digit_model is not None}")
        logger.info(f"🔍 Alphanumeric model available: {alphanumeric_model is not None}")
        
        if current_mode == 'alphanumeric' and alphanumeric_model is not None:
            model = alphanumeric_model
            labels = alphanumeric_labels
            logger.info("🎯 Using alphanumeric model")
        elif digit_model is not None:
            model = digit_model
            labels = None
            logger.info("🎯 Using digit model")
        else:
            logger.error("❌ No models available")
            return jsonify({'error': 'No models available'}), 500
        
        # Preprocess image
        logger.info("🔄 Preprocessing image...")
        image_array = preprocess_image(data['image'])
        
        if image_array is None or image_array.size == 0:
            logger.error("❌ Image preprocessing failed")
            return jsonify({'error': 'Image preprocessing failed'}), 400
        
        logger.info(f"✅ Image preprocessed: {image_array.shape}")
        
        # Make prediction
        logger.info("🔄 Making prediction...")
        predictions = model.predict(image_array, verbose=0)
        logger.info(f"✅ Prediction complete: {predictions.shape}")
        logger.debug(f"Prediction values: {predictions[0][:10]}...")  # First 10 values
        
        if predictions is None or len(predictions) == 0:
            logger.error("❌ Model prediction failed")
            return jsonify({'error': 'Model prediction failed'}), 500
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            
            if labels:
                character = labels.get(idx, f"Unknown_{idx}")
            else:
                character = str(idx)
            
            results.append({
                'character': character,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%"
            })
        
        response = {
            'success': True,
            'mode': current_mode,
            'prediction': results[0]['character'],
            'confidence': results[0]['confidence_percent'],
            'top_3': results
        }
        
        logger.info(f"✅ Prediction successful: {results[0]['character']} ({results[0]['confidence_percent']})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

def initialize_models():
    """Initialize models with enhanced error handling."""
    global digit_model, alphanumeric_model, alphanumeric_labels, current_mode
    
    logger.info("🚀 Initializing character recognition models...")
    debug_environment()
    
    # Load digit model
    try:
        digit_model = load_digit_model()
    except Exception as e:
        logger.error(f"❌ Failed to load digit model: {e}")
        digit_model = None
    
    # Load alphanumeric model
    try:
        alphanumeric_model, alphanumeric_labels = load_alphanumeric_model()
    except Exception as e:
        logger.error(f"❌ Failed to load alphanumeric model: {e}")
        alphanumeric_model = None
        alphanumeric_labels = None
    
    # Set default mode
    if alphanumeric_model is not None:
        current_mode = "alphanumeric"
        logger.info("🎯 Default mode: alphanumeric")
    elif digit_model is not None:
        current_mode = "digits"
        logger.info("🎯 Default mode: digits")
    else:
        logger.error("❌ No models available!")
        current_mode = "digits"
    
    logger.info("✅ Model initialization complete")

if __name__ == '__main__':
    initialize_models()
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting debug app on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)