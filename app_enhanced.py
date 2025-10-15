#!/usr/bin/env python3
"""
Enhanced Alphanumeric Character Recognition Web Application

This Flask application provides a web interface for handwritten character recognition
supporting both digits (0-9) and letters (A-Z). Users can draw characters on an 
HTML5 canvas, and the application processes the image and returns predictions.

Features:
- Dual model support (digits-only and alphanumeric)
- Advanced image preprocessing
- Real-time predictions with confidence scores
- Modern web interface with visual feedback
"""
import base64
import io
import json
import os
from typing import Tuple, Optional, Dict, List
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
from scipy import ndimage

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
        raise FileNotFoundError(f"Digit model file not found: {model_path}")
    print("Loading digit recognition model...")
    return tf.keras.models.load_model(model_path)

def load_alphanumeric_model() -> Tuple[tf.keras.Model, Dict[str, str]]:
    """Load the alphanumeric CNN model and label mapping."""
    model_path = "alphanumeric_cnn.keras"
    labels_path = "alphanumeric_labels.json"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Alphanumeric model not found: {model_path}")
        return None, None
    
    if not os.path.exists(labels_path):
        print(f"⚠️ Label mapping not found: {labels_path}")
        return None, None
    
    print("Loading alphanumeric recognition model...")
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
        # Convert string keys to integers
        labels = {int(k): v for k, v in labels.items()}
    
    return model, labels

def preprocess_image(image_data: str) -> np.ndarray:
    """
    Preprocess canvas image data for character recognition.
    
    Converts base64 encoded canvas image to a format suitable for CNN models.
    Applies morphological operations, intelligent cropping, and proper normalization.
    
    Args:
        image_data (str): Base64 encoded image data from the canvas.
        
    Returns:
        np.ndarray: Preprocessed image array with shape (1, 28, 28, 1).
    """
    try:
        # Decode base64 image data
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert colors (make white background black, black drawing white)
        img_array = 255 - img_array
        
        # Apply Gaussian blur to smooth the image
        img_array = ndimage.gaussian_filter(img_array, sigma=0.8)
        
        # Find bounding box of the drawn content
        rows = np.any(img_array > 30, axis=1)  # Lowered threshold for better detection
        cols = np.any(img_array > 30, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Extract the drawn region with some padding
            padding = 10
            rmin = max(0, rmin - padding)
            rmax = min(img_array.shape[0], rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(img_array.shape[1], cmax + padding)
            
            cropped = img_array[rmin:rmax, cmin:cmax]
        else:
            # If no content detected, use center region
            h, w = img_array.shape
            margin = min(h, w) // 4
            cropped = img_array[margin:h-margin, margin:w-margin]
        
        # Resize to 20x20 while maintaining aspect ratio
        h, w = cropped.shape
        if h > w:
            new_h, new_w = 20, int(20 * w / h)
        else:
            new_h, new_w = int(20 * h / w), 20
        
        # Ensure minimum size
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        resized = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS))
        
        # Center the image in a 28x28 canvas
        final_img = np.zeros((28, 28), dtype=np.float32)
        
        # Calculate position to center the resized image
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Apply center of mass centering for better alignment
        cy, cx = ndimage.center_of_mass(final_img)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = 14 - cy  # Center of 28x28 image
            shift_x = 14 - cx
            
            # Limit shift to prevent image from going out of bounds
            shift_y = np.clip(shift_y, -7, 7)
            shift_x = np.clip(shift_x, -7, 7)
            
            final_img = ndimage.shift(final_img, [shift_y, shift_x], cval=0)
        
        # Normalize to [0, 1] range
        final_img = final_img / 255.0
        
        # Reshape for model input: (1, 28, 28, 1)
        final_img = final_img.reshape(1, 28, 28, 1)
        
        return final_img
        
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        # Return a blank image if preprocessing fails
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

def get_top_predictions(predictions: np.ndarray, labels: Optional[Dict[int, str]] = None, 
                       top_k: int = 3) -> List[Dict[str, any]]:
    """
    Get top-k predictions with confidence scores.
    
    Args:
        predictions: Model prediction probabilities
        labels: Label mapping (None for digit mode)
        top_k: Number of top predictions to return
        
    Returns:
        List of prediction dictionaries with character and confidence
    """
    # Get top k indices
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        confidence = float(predictions[0][idx])
        
        if labels:  # Alphanumeric mode
            character = labels.get(idx, f"Unknown_{idx}")
        else:  # Digit mode
            character = str(idx)
        
        results.append({
            'character': character,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.1f}%"
        })
    
    return results

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """Render the admin page for model management."""
    return render_template('admin.html')

@app.route('/debug')
def debug():
    """Render the debug page for testing."""
    return render_template('debug.html')

@app.route('/api/mode', methods=['GET', 'POST'])
def handle_mode():
    """Handle getting and setting the recognition mode."""
    global current_mode
    
    if request.method == 'POST':
        data = request.get_json()
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
    
    # GET request
    alphanumeric_available = alphanumeric_model is not None
    return jsonify({
        'current_mode': current_mode,
        'available_modes': {
            'digits': True,
            'alphanumeric': alphanumeric_available
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests for handwritten characters.
    
    Processes the uploaded image and returns the predicted character
    with confidence scores using the appropriate model.
    """
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess the image
        image_array = preprocess_image(data['image'])
        
        # Choose model based on current mode
        if current_mode == 'alphanumeric' and alphanumeric_model is not None:
            model = alphanumeric_model
            labels = alphanumeric_labels
        else:
            model = digit_model
            labels = None
        
        if model is None:
            return jsonify({'error': f'Model not available for {current_mode} mode'}), 500
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get top predictions
        top_predictions = get_top_predictions(predictions, labels, top_k=3)
        
        # Prepare response
        response = {
            'success': True,
            'mode': current_mode,
            'prediction': top_predictions[0]['character'],
            'confidence': top_predictions[0]['confidence_percent'],
            'top_3': top_predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/models/status')
def model_status():
    """Get the status of available models."""
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
        'current_mode': current_mode
    })

def initialize_models():
    """Initialize all available models on startup."""
    global digit_model, alphanumeric_model, alphanumeric_labels
    
    print("🤖 Initializing character recognition models...")
    
    # Load digit model
    try:
        digit_model = load_digit_model()
        print("✅ Digit recognition model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load digit model: {e}")
        digit_model = None
    
    # Load alphanumeric model
    try:
        alphanumeric_model, alphanumeric_labels = load_alphanumeric_model()
        if alphanumeric_model is not None:
            print("✅ Alphanumeric recognition model loaded successfully")
        else:
            print("⚠️ Alphanumeric model not available")
    except Exception as e:
        print(f"❌ Failed to load alphanumeric model: {e}")
        alphanumeric_model = None
        alphanumeric_labels = None
    
    # Set default mode based on available models
    if alphanumeric_model is not None:
        current_mode = "alphanumeric"
        print("🎯 Default mode: alphanumeric (digits + letters)")
    elif digit_model is not None:
        current_mode = "digits"
        print("🎯 Default mode: digits only")
    else:
        print("❌ No models available!")

if __name__ == '__main__':
    # Initialize models
    initialize_models()
    
    print("\n🚀 Starting Enhanced Character Recognition Web App")
    print("Available endpoints:")
    print("  - / : Main application")
    print("  - /admin : Model management")
    print("  - /debug : Debug interface")
    print("  - /api/mode : Get/set recognition mode")
    print("  - /predict : Character prediction")
    print("  - /api/models/status : Model status")
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)