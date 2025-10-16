#!/usr/bin/env python3
"""
Convert TensorFlow model to ONNX format for lighter deployment
"""
import tensorflow as tf
import tf2onnx
import onnx

def convert_model_to_onnx():
    """Convert the Keras model to ONNX format."""
    print("🔄 Loading TensorFlow model...")
    model = tf.keras.models.load_model('trained_digit_model.keras')
    
    print("🔄 Converting to ONNX...")
    # Convert to ONNX
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    output_path = "trained_digit_model.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(model_proto, output_path)
    
    print(f"✅ Model converted to ONNX: {output_path}")
    
    # Check file sizes
    import os
    keras_size = os.path.getsize('trained_digit_model.keras') / 1024 / 1024
    onnx_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"📊 Keras model: {keras_size:.1f} MB")
    print(f"📊 ONNX model: {onnx_size:.1f} MB")
    print(f"💾 Size reduction: {((keras_size - onnx_size) / keras_size * 100):.1f}%")

if __name__ == "__main__":
    convert_model_to_onnx()