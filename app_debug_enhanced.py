#!/usr/bin/env python3
"""
Enhanced Debug App with Robust Model Loading
Handles TensorFlow version compatibility issues in cloud environments
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

# Enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
digit_model = None
alphanumeric_model = None
alphanumeric_labels = None
current_mode = "digits"

def debug_environment():
    """Enhanced environment debugging."""
    logger.info("🔍 Environment Debug Info:")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check TensorFlow configuration
    logger.info(f"TF devices: {tf.config.list_physical_devices()}")
    logger.info(f"TF build info: {tf.sysconfig.get_build_info()}")
    
    # Check model files with detailed info
    model_files = ['mnist_cnn.h5', 'alphanumeric_cnn.keras', 'alphanumeric_labels.json']
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            logger.info(f"✅ {file}: exists ({size:,} bytes)")
            
            # Check file permissions
            import stat
            mode = os.stat(file).st_mode
            logger.info(f"   Permissions: {stat.filemode(mode)}")
        else:
            logger.error(f"❌ {file}: not found")

def load_digit_model_robust() -> tf.keras.Model:
    """Enhanced digit model loading with multiple fallback strategies."""
    model_path = "mnist_cnn.h5"
    
    try:
        logger.info(f"🔍 Loading digit model: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        size = os.path.getsize(model_path)
        logger.info(f"📁 Model file size: {size:,} bytes")
        
        # Strategy 1: Load with compile=False (most compatible)
        try:
            logger.info("🔄 Strategy 1: Loading with compile=False")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Manually compile with simple optimizer
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ Strategy 1 successful - model loaded and compiled")
            
        except Exception as e1:
            logger.warning(f"⚠️ Strategy 1 failed: {e1}")
            
            # Strategy 2: Load with custom objects
            try:
                logger.info("🔄 Strategy 2: Loading with custom objects")
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects={}
                )
                logger.info("✅ Strategy 2 successful")
                
            except Exception as e2:
                logger.warning(f"⚠️ Strategy 2 failed: {e2}")
                
                # Strategy 3: Load weights only (last resort)
                try:
                    logger.info("🔄 Strategy 3: Reconstructing model architecture")
                    
                    # Recreate the CNN architecture
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
                    
                    # Try to load weights
                    model.load_weights(model_path)
                    logger.info("✅ Strategy 3 successful - reconstructed model")
                    
                except Exception as e3:
                    logger.error(f"❌ All strategies failed: {e3}")
                    raise e3
        
        # Test the model
        logger.info("🧪 Testing model with dummy input")
        test_input = np.random.random((1, 28, 28, 1))
        test_pred = model.predict(test_input, verbose=0)
        logger.info(f"✅ Model test successful: {test_pred.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load digit model: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_alphanumeric_model_robust() -> Tuple[tf.keras.Model, Dict[str, str]]:
    """Enhanced alphanumeric model loading."""
    model_path = "alphanumeric_cnn.keras"
    labels_path = "alphanumeric_labels.json"
    
    try:
        logger.info(f"🔍 Loading alphanumeric model: {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Alphanumeric model not found: {model_path}")
            return None, None
        
        if not os.path.exists(labels_path):
            logger.warning(f"⚠️ Labels file not found: {labels_path}")
            return None, None
        
        # Load labels first
        logger.info(f"📁 Loading labels from: {labels_path}")
        with open(labels_path, 'r') as f:
            labels = json.load(f)
            labels = {int(k): v for k, v in labels.items()}
        logger.info(f"✅ Labels loaded: {len(labels)} classes")
        
        # Load model with same robust strategies
        size = os.path.getsize(model_path)
        logger.info(f"📁 Model file size: {size:,} bytes")
        
        try:
            logger.info("🔄 Loading alphanumeric model with compile=False")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Compile manually
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ Alphanumeric model loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Alphanumeric model loading failed: {e}")
            return None, None
        
        # Test the model
        test_input = np.random.random((1, 28, 28, 1))
        test_pred = model.predict(test_input, verbose=0)
        logger.info(f"✅ Alphanumeric model test successful: {test_pred.shape}")
        
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
        logger.warning(f"Gaussian blur failed: {e}")
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
    except Exception:
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
    except Exception:
        return image

def preprocess_image(image_data: str) -> np.ndarray:
    """Preprocess image with enhanced error handling."""
    try:
        logger.debug("🔍 Starting image preprocessing")
        
        if not image_data or not isinstance(image_data, str):
            raise ValueError("Invalid image data")
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        if img_array.size == 0:
            raise ValueError("Empty image")
        
        # Invert colors
        img_array = 255 - img_array
        
        # Apply blur
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
        else:
            h, w = img_array.shape
            margin = min(h, w) // 4
            cropped = img_array[margin:h-margin, margin:w-margin]
        
        if cropped.size == 0:
            cropped = img_array
        
        # Resize
        h, w = cropped.shape
        if h > w:
            new_h, new_w = 20, max(1, int(20 * w / h))
        else:
            new_h, new_w = max(1, int(20 * h / w)), 20
        
        resized = np.array(Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS))
        
        # Center in 28x28
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
        
        logger.debug(f"✅ Image preprocessed: {final_img.shape}")
        return final_img
        
    except Exception as e:
        logger.error(f"❌ Image preprocessing failed: {e}")
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug_info():
    """Enhanced debug endpoint."""
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
                'files_in_directory': sorted(os.listdir('.')),
                'tf_devices': [str(d) for d in tf.config.list_physical_devices()],
                'tf_build_info': dict(tf.sysconfig.get_build_info())
            },
            'model_files': {}
        }
        
        # Add detailed model file info
        model_files = ['mnist_cnn.h5', 'alphanumeric_cnn.keras', 'alphanumeric_labels.json']
        for file in model_files:
            if os.path.exists(file):
                info['model_files'][file] = {
                    'exists': True,
                    'size': os.path.getsize(file),
                    'readable': os.access(file, os.R_OK)
                }
            else:
                info['model_files'][file] = {'exists': False}
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Debug info error: {e}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/health')
def health_check():
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
                    'message': 'Invalid mode'
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
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction with detailed logging."""
    try:
        logger.info("🔍 Prediction request received")
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        logger.info(f"🔍 Current mode: {current_mode}")
        logger.info(f"🔍 Digit model available: {digit_model is not None}")
        logger.info(f"🔍 Alphanumeric model available: {alphanumeric_model is not None}")
        
        # Select model
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
        
        # Preprocess
        image_array = preprocess_image(data['image'])
        if image_array is None or image_array.size == 0:
            return jsonify({'error': 'Image preprocessing failed'}), 400
        
        # Predict
        logger.info("🔄 Making prediction...")
        predictions = model.predict(image_array, verbose=0)
        logger.info(f"✅ Prediction shape: {predictions.shape}")
        
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
        
        logger.info(f"✅ Prediction: {results[0]['character']} ({results[0]['confidence_percent']})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

def initialize_models():
    """Enhanced model initialization."""
    global digit_model, alphanumeric_model, alphanumeric_labels, current_mode
    
    logger.info("🚀 Initializing models with enhanced loading...")
    debug_environment()
    
    # Load digit model
    try:
        digit_model = load_digit_model_robust()
        logger.info("✅ Digit model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Digit model failed: {e}")
        digit_model = None
    
    # Load alphanumeric model
    try:
        alphanumeric_model, alphanumeric_labels = load_alphanumeric_model_robust()
        if alphanumeric_model is not None:
            logger.info("✅ Alphanumeric model loaded successfully")
        else:
            logger.warning("⚠️ Alphanumeric model not available")
    except Exception as e:
        logger.error(f"❌ Alphanumeric model failed: {e}")
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
    logger.info(f"🚀 Starting enhanced debug app on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)