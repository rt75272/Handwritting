#!/usr/bin/env python3
"""
Production-Ready Enhanced Alphanumeric Character Recognition Web Application
NO SCIPY VERSION - for faster Render deployment

This version replaces scipy functions with numpy/PIL equivalents to avoid
Fortran compiler requirements during deployment.
"""
import base64
import io
import json
import os
import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageFilter

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Global variables for models and configurations
digit_model = None
alphanumeric_model = None
alphanumeric_labels = None
current_mode = "digits"  # Default mode

def load_digit_model() -> tf.keras.Model:
    """Load the digit-only MNIST CNN model."""
    model_path = "mnist_cnn.h5"
    if not os.path.exists(model_path):
        logger.error(f"Digit model file not found: {model_path}")
        raise FileNotFoundError(f"Digit model file not found: {model_path}")
    
    logger.info("Loading digit recognition model...")
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Digit recognition model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load digit model: {e}")
        raise

def load_alphanumeric_model() -> Tuple[tf.keras.Model, Dict[str, str]]:
    """Load the alphanumeric CNN model and label mapping."""
    model_path = "alphanumeric_cnn.keras"
    labels_path = "alphanumeric_labels.json"
    
    if not os.path.exists(model_path):
        logger.warning(f"Alphanumeric model not found: {model_path}")
        return None, None
    
    if not os.path.exists(labels_path):
        logger.warning(f"Label mapping not found: {labels_path}")
        return None, None
    
    logger.info("Loading alphanumeric recognition model...")
    try:
        model = tf.keras.models.load_model(model_path)
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            # Convert string keys to integers
            labels = {int(k): v for k, v in labels.items()}
        
        logger.info("✅ Alphanumeric recognition model loaded successfully")
        return model, labels
    except Exception as e:
        logger.error(f"Failed to load alphanumeric model: {e}")
        return None, None

def gaussian_blur_numpy(image: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Apply Gaussian blur using PIL instead of scipy.
    Converts numpy array to PIL, applies blur, converts back.
    """
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Apply Gaussian blur
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Convert back to numpy
        return np.array(blurred)
    except Exception:
        # Fallback: return original image if blur fails
        return image

def center_of_mass_numpy(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculate center of mass using pure numpy instead of scipy.
    """
    try:
        # Get coordinates where image has non-zero values
        total = np.sum(image)
        if total == 0:
            return np.nan, np.nan
        
        # Calculate center of mass
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        cy = np.sum(y_coords * image) / total
        cx = np.sum(x_coords * image) / total
        
        return cy, cx
    except Exception:
        return np.nan, np.nan

def shift_image_numpy(image: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """
    Shift image using numpy roll instead of scipy.ndimage.shift.
    This is a simpler approximation but works for our use case.
    """
    try:
        # Convert shifts to integers
        shift_y_int = int(round(shift_y))
        shift_x_int = int(round(shift_x))
        
        # Use numpy roll for shifting
        shifted = np.roll(image, shift_y_int, axis=0)
        shifted = np.roll(shifted, shift_x_int, axis=1)
        
        # Zero out the wrapped-around edges
        if shift_y_int > 0:
            shifted[:shift_y_int, :] = 0
        elif shift_y_int < 0:
            shifted[shift_y_int:, :] = 0
            
        if shift_x_int > 0:
            shifted[:, :shift_x_int] = 0
        elif shift_x_int < 0:
            shifted[:, shift_x_int:] = 0
        
        return shifted
    except Exception:
        return image

def preprocess_image(image_data: str) -> np.ndarray:
    """
    Preprocess canvas image data for character recognition.
    NO SCIPY VERSION - uses PIL and numpy alternatives.
    """
    try:
        logger.debug("Starting image preprocessing")
        
        # Validate input
        if not image_data or not isinstance(image_data, str):
            raise ValueError("Invalid image data provided")
        
        # Decode base64 image data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Validate image dimensions
        if img_array.size == 0:
            raise ValueError("Empty image provided")
        
        # Invert colors
        img_array = 255 - img_array
        
        # Apply Gaussian blur using PIL
        img_array = gaussian_blur_numpy(img_array, sigma=0.8)
        
        # Find bounding box of the drawn content
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Extract with padding
            padding = 10
            rmin = max(0, rmin - padding)
            rmax = min(img_array.shape[0], rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(img_array.shape[1], cmax + padding)
            
            cropped = img_array[rmin:rmax, cmin:cmax]
        else:
            # Use center region if no content detected
            h, w = img_array.shape
            margin = min(h, w) // 4
            cropped = img_array[margin:h-margin, margin:w-margin]
        
        # Ensure cropped image is not empty
        if cropped.size == 0:
            cropped = img_array
        
        # Resize to 20x20 while maintaining aspect ratio
        h, w = cropped.shape
        if h > w:
            new_h, new_w = 20, max(1, int(20 * w / h))
        else:
            new_h, new_w = max(1, int(20 * h / w)), 20
        
        resized = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS))
        
        # Center the image in a 28x28 canvas
        final_img = np.zeros((28, 28), dtype=np.float32)
        
        # Calculate position to center
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Apply center of mass centering using numpy alternative
        cy, cx = center_of_mass_numpy(final_img)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = 14 - cy
            shift_x = 14 - cx
            
            # Limit shift
            shift_y = np.clip(shift_y, -7, 7)
            shift_x = np.clip(shift_x, -7, 7)
            
            final_img = shift_image_numpy(final_img, shift_y, shift_x)
        
        # Normalize to [0, 1] range
        final_img = final_img / 255.0
        
        # Reshape for model input
        final_img = final_img.reshape(1, 28, 28, 1)
        
        logger.debug("Image preprocessing completed successfully")
        return final_img
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

def get_top_predictions(predictions: np.ndarray, labels: Optional[Dict[int, str]] = None, 
                       top_k: int = 3) -> List[Dict[str, any]]:
    """Get top-k predictions with confidence scores."""
    try:
        if predictions is None or len(predictions) == 0:
            return []
        
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
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
        
        return results
    except Exception as e:
        logger.error(f"Error in get_top_predictions: {e}")
        return []

@app.route('/')
def index():
    """Render the main application page."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return f"Error loading page: {e}", 500

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'digit_model': digit_model is not None,
            'alphanumeric_model': alphanumeric_model is not None
        },
        'current_mode': current_mode,
        'scipy_free': True
    })

@app.route('/api/mode', methods=['GET', 'POST'])
def handle_mode():
    """Handle getting and setting the recognition mode."""
    global current_mode
    
    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'No data provided'}), 400
            
            mode = data.get('mode', 'digits')
            
            if mode in ['digits', 'alphanumeric']:
                current_mode = mode
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
        
        alphanumeric_available = alphanumeric_model is not None
        return jsonify({
            'current_mode': current_mode,
            'available_modes': {
                'digits': digit_model is not None,
                'alphanumeric': alphanumeric_available
            }
        })
    except Exception as e:
        logger.error(f"Error handling mode request: {e}")
        return jsonify({'success': False, 'message': f'Internal error: {e}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests for handwritten characters."""
    try:
        logger.info("Received prediction request")
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Select model based on current mode
        if current_mode == 'alphanumeric' and alphanumeric_model is not None:
            model = alphanumeric_model
            labels = alphanumeric_labels
        elif digit_model is not None:
            model = digit_model
            labels = None
        else:
            return jsonify({'error': 'No models available'}), 500
        
        # Preprocess the image
        image_array = preprocess_image(data['image'])
        
        if image_array is None or image_array.size == 0:
            return jsonify({'error': 'Image preprocessing failed'}), 400
        
        # Make prediction
        logger.info(f"Making prediction with {current_mode} model")
        predictions = model.predict(image_array, verbose=0)
        
        if predictions is None or len(predictions) == 0:
            return jsonify({'error': 'Model prediction failed'}), 500
        
        # Get top predictions
        top_predictions = get_top_predictions(predictions, labels, top_k=3)
        
        if not top_predictions:
            return jsonify({'error': 'Failed to process predictions'}), 500
        
        response = {
            'success': True,
            'mode': current_mode,
            'prediction': top_predictions[0]['character'],
            'confidence': top_predictions[0]['confidence_percent'],
            'top_3': top_predictions
        }
        
        logger.info(f"Prediction successful: {top_predictions[0]['character']} ({top_predictions[0]['confidence_percent']})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/models/status')
def model_status():
    """Get the status of available models."""
    try:
        digit_available = digit_model is not None
        alphanumeric_available = alphanumeric_model is not None
        
        return jsonify({
            'models': {
                'digits': {
                    'available': digit_available,
                    'file': 'mnist_cnn.h5',
                    'classes': 10,
                    'description': 'MNIST digit recognition (0-9)'
                },
                'alphanumeric': {
                    'available': alphanumeric_available,
                    'file': 'alphanumeric_cnn.keras',
                    'classes': 36,
                    'description': 'EMNIST alphanumeric recognition (0-9, A-Z)'
                }
            },
            'current_mode': current_mode,
            'scipy_free': True
        })
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({'error': f'Failed to get model status: {e}'}), 500

def initialize_models():
    """Initialize all available models on startup."""
    global digit_model, alphanumeric_model, alphanumeric_labels
    
    logger.info("🤖 Initializing character recognition models...")
    
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
    global current_mode
    if alphanumeric_model is not None:
        current_mode = "alphanumeric"
        logger.info("🎯 Default mode: alphanumeric (digits + letters)")
    elif digit_model is not None:
        current_mode = "digits"
        logger.info("🎯 Default mode: digits only")
    else:
        logger.error("❌ No models available!")
        current_mode = "digits"

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize models
    initialize_models()
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("\n🚀 Starting Enhanced Character Recognition Web App (No SciPy)")
    logger.info(f"Port: {port}")
    logger.info("Available endpoints:")
    logger.info("  - / : Main application")
    logger.info("  - /health : Health check")
    logger.info("  - /api/mode : Get/set recognition mode")
    logger.info("  - /predict : Character prediction")
    logger.info("  - /api/models/status : Model status")
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)