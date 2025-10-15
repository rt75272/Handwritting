#!/usr/bin/env python3
"""MNIST Digit Recognition Web Application.

This Flask application provides a web interface for handwritten digit recognition
using a trained CNN model. Users can draw digits on an HTML5 canvas, and the
application processes the image and returns predictions.
"""
import base64
import io
import os
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
from scipy import ndimage

# Initialize Flask application.
app = Flask(__name__)

# Global model variable to store the loaded CNN model.
model = None

def load_model() -> tf.keras.Model:
    """Load the pre-trained MNIST CNN model from disk.
    
    Returns:
        tf.keras.Model: The loaded model ready for inference.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist.
        Exception: If model loading fails.
    """
    model_path = "mnist_cnn.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print("Loading saved model...")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_data: str) -> np.ndarray:
    """Preprocess canvas image data for MNIST digit recognition.
    
    Converts base64 encoded canvas image to a format suitable for the CNN model.
    Applies morphological operations, intelligent cropping, and proper normalization.
    
    Args:
        image_data (str): Base64 encoded image data from the canvas.
        
    Returns:
        np.ndarray: Preprocessed image array with shape (1, 28, 28, 1).
        
    Raises:
        ValueError: If image data is invalid or cannot be processed.
    """
    try:
        # Remove the data URL prefix and decode base64 image.
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        print(f"[DEBUG] Received image data length: {len(image_data)}")
        print(f"[DEBUG] Decoded bytes length: {len(image_bytes)}")
        # Open image with PIL and convert to grayscale.
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        print(f"[DEBUG] Original PIL image size: {img.size}, mode: {img.mode}")
        # Save original for debugging purposes.
        img.save("debug_original.png")
        print(f"[DEBUG] Saved debug_original.png")
        # Convert to numpy array for processing.
        img_array = np.array(img)
        # Apply morphological operations to clean up the digit.
        # This helps remove noise and improve digit boundaries.
        threshold_value = 128
        binary_img = img_array < threshold_value
        if binary_img.any():
            # Apply erosion followed by dilation (opening) to clean noise.
            kernel = ndimage.generate_binary_structure(2, 1)
            cleaned = ndimage.binary_opening(binary_img, structure=kernel, iterations=1)
            # Apply dilation followed by erosion (closing) to fill gaps.
            cleaned = ndimage.binary_closing(cleaned, structure=kernel, iterations=1)
            # Convert back to grayscale.
            img_array = (1 - cleaned.astype(np.uint8)) * 255
            # Find bounding box of the digit for intelligent cropping.
            coords = np.column_stack(np.where(cleaned))
            if len(coords) > 0:
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                # Add intelligent padding based on content size.
                height = max_y - min_y
                width = max_x - min_x
                padding = max(int(min(height, width) * 0.15), 8)  # Adaptive padding.
                # Ensure padding doesn't exceed image boundaries.
                min_y = max(0, min_y - padding)
                max_y = min(img_array.shape[0], max_y + padding)
                min_x = max(0, min_x - padding)
                max_x = min(img_array.shape[1], max_x + padding)
                # Crop to bounding box.
                cropped = img_array[min_y:max_y, min_x:max_x]
                # Create square image with smart centering.
                height, width = cropped.shape
                max_dim = max(height, width)
                # Add extra space for better digit recognition.
                final_dim = int(max_dim * 1.2)
                # Create square canvas with white background.
                square_img = np.full((final_dim, final_dim), 255, dtype=np.uint8)
                # Center the cropped image.
                y_offset = (final_dim - height) // 2
                x_offset = (final_dim - width) // 2
                square_img[y_offset:y_offset+height, x_offset:x_offset+width] = cropped
                img = Image.fromarray(square_img)
                print(f"[DEBUG] Applied advanced preprocessing with morphological operations")
            else:
                print(f"[DEBUG] No significant drawing detected after cleaning")
        else:
            print(f"[DEBUG] No drawing detected, using original image")
        # High-quality resize to 28x28 pixels using anti-aliasing.
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        # Convert to numpy array and normalize to [0, 1] range.
        img_array = np.array(img).astype(np.float32) / 255.0
        print(f"[DEBUG] Before inversion - shape: {img_array.shape}, "
              f"min: {img_array.min():.3f}, max: {img_array.max():.3f}, "
              f"mean: {img_array.mean():.3f}")
        # Invert colors to match MNIST format: white digits on black background.
        # MNIST has white digits (1.0) on black background (0.0).
        # Canvas draws black digits (0.0) on white background (1.0).
        img_array = 1.0 - img_array
        print(f"[DEBUG] After inversion - min: {img_array.min():.3f}, "
              f"max: {img_array.max():.3f}, mean: {img_array.mean():.3f}")
        # Save processed image for debugging purposes.
        debug_img = Image.fromarray((img_array * 255).astype(np.uint8))
        debug_img.save("debug_processed.png")
        print(f"[DEBUG] Saved debug_processed.png")
        # Reshape to match model input requirements: (batch_size, height, width, channels).
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route("/")
def index() -> str:
    """Render the main page with the digit drawing canvas.
    
    Returns:
        str: Rendered HTML template for the main page.
    """
    return render_template("index.html")

@app.route("/debug")
def debug() -> str:
    """Render the debug page for development and troubleshooting.
    
    Returns:
        str: Rendered HTML template for the debug page.
    """
    return render_template("debug.html")

@app.route("/admin")
def admin() -> str:
    """Render the admin page for model management.
    
    Returns:
        str: Rendered HTML template for the admin page.
    """
    return render_template("admin.html")

@app.route("/predict", methods=["POST"])
def predict() -> jsonify:
    """Process digit image and return prediction results.
    
    Expects JSON payload with base64 encoded image data.
    Returns prediction with confidence score and top 3 alternatives.
    
    Returns:
        jsonify: JSON response containing prediction results or error message.
    """
    try:
        # Extract image data from request.
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        image_data = data['image']
        # Preprocess the image for model inference.
        processed_img = preprocess_image(image_data)
        print(f"[DEBUG] Processed image shape: {processed_img.shape}")
        print(f"[DEBUG] Image stats - min: {processed_img.min():.3f}, "
              f"max: {processed_img.max():.3f}, mean: {processed_img.mean():.3f}")
        # Make prediction using the loaded model.
        predictions = model.predict(processed_img)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        # Get top 3 predictions for additional context.
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(int(idx), predictions[0][idx]) for idx in top_3_indices]
        print(f"[DEBUG] Prediction: digit={predicted_digit}, confidence={confidence:.3f}")
        print(f"[DEBUG] Top 3 predictions: {top_3_predictions}")
        return jsonify({
            "prediction": predicted_digit,
            "confidence": f"{confidence:.1%}",
            "top_3": [{"digit": int(digit), "confidence": f"{conf:.1%}"} 
                     for digit, conf in top_3_predictions]
        })
    except ValueError as e:
        print(f"[ERROR] Image processing error: {e}")
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/retrain", methods=["POST"])
def retrain() -> jsonify:
    """Retrain the model with fresh MNIST data.
    
    This endpoint triggers model retraining. Use with caution as it
    can take several minutes to complete.
    
    Returns:
        jsonify: JSON response indicating success or failure.
    """
    try:
        # Import here to avoid circular imports.
        from model import retrain_model
        print("[INFO] Starting model retraining...")
        retrain_model()
        print("[INFO] Model retraining completed!")
        # Reload the model after retraining.
        global model
        model = load_model()
        return jsonify({
            "status": "success", 
            "message": "Model retrained successfully!"
        })
    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Retraining failed: {str(e)}"
        }), 500

if __name__ == "__main__":
    try:
        # Load the model at startup.
        model = load_model()
        print("Model loaded successfully!")
        # Start the Flask development server.
        # Note: Use a production WSGI server (gunicorn, uwsgi) for deployment.
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"Failed to start application: {e}")
        exit(1)