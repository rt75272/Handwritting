# MNIST Digit Recognition Web Application

A modern, interactive web application for handwritten digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw digits on an HTML5 canvas and get real-time predictions with confidence scores.

## 🚀 Features

- **Interactive Drawing Canvas**: Draw digits with mouse or touch input
- **Real-time Predictions**: Get instant digit recognition with confidence scores
- **High Accuracy Model**: CNN trained on MNIST dataset with >99% accuracy
- **Advanced Image Processing**: Morphological operations for improved recognition
- **Mobile-Friendly**: Responsive design with touch support
- **Debug Interface**: Development tools for troubleshooting
- **Model Retraining**: Easy model retraining with fresh data

## 🏗️ Architecture

### Backend (Python/Flask)
- **Flask Web Framework**: Lightweight web server and API
- **TensorFlow/Keras**: Deep learning model implementation
- **Advanced Preprocessing**: Image cleaning and normalization
- **RESTful API**: JSON-based prediction endpoint

### Frontend (HTML/CSS/JavaScript)
- **HTML5 Canvas**: Interactive drawing interface
- **Responsive Design**: Works on desktop and mobile
- **Real-time Communication**: AJAX-based predictions
- **Touch Support**: Mobile-friendly drawing

### Machine Learning
- **CNN Architecture**: Convolutional layers with pooling and dropout
- **MNIST Dataset**: 60,000 training images, 10,000 test images
- **Data Augmentation**: Rotation, shifting, and noise for robustness
- **Regularization**: Dropout and early stopping to prevent overfitting

## 📋 Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone or download the project**:
   ```bash
   cd digit_recognizer
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv digits
   
   # On Linux/Mac:
   source digits/bin/activate
   
   # On Windows:
   digits\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if no model exists):
   ```bash
   python model.py
   ```

## 🚀 Usage

### Starting the Application

1. **Activate virtual environment** (if not already active):
   ```bash
   source digits/bin/activate  # Linux/Mac
   # or
   digits\Scripts\activate     # Windows
   ```

2. **Run the Flask application**:
   ```bash
   python app.py
   ```

3. **Open in browser**:
   - Navigate to `http://localhost:5002`
   - Draw a digit (0-9) on the canvas
   - Click "Predict" to get the result

### Available Endpoints

- `/` - Main drawing interface
- `/debug` - Debug page with additional information
- `/admin` - Model management interface
- `/predict` - API endpoint for predictions (POST)
- `/retrain` - Model retraining endpoint (POST)

### Model Retraining

To retrain the model with fresh data:

**Option 1: Command Line**
```bash
python retrain_cpu.py
```

**Option 2: Web Interface**
- Go to `/admin` in your browser
- Click "Retrain Model" button
- Wait for training to complete (~5-15 minutes)

## 📁 Project Structure

```
digit_recognizer/
├── app.py                 # Main Flask application
├── model.py               # CNN model definition and training
├── retrain_cpu.py         # Standalone retraining script
├── requirements.txt       # Python dependencies
├── mnist_cnn.h5          # Trained model file
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── static/
│   └── script.js         # Frontend JavaScript
├── templates/
│   ├── index.html        # Main drawing interface
│   ├── debug.html        # Debug page
│   └── admin.html        # Admin interface
└── digits/               # Virtual environment
```

## 🔧 Technical Details

### Model Architecture
```
Input (28x28x1) → Conv2D (32) → MaxPool → Conv2D (64) → MaxPool → 
Conv2D (64) → Flatten → Dense (64) → Dropout → Dense (10, softmax)
```

### Image Processing Pipeline
1. **Canvas Capture**: HTML5 canvas to base64 data URL
2. **Morphological Operations**: Noise removal and gap filling
3. **Intelligent Cropping**: Automatic bounding box detection
4. **Normalization**: Resize to 28x28, normalize to [0,1]
5. **Color Inversion**: Convert to MNIST format (white on black)

### Performance Metrics
- **Training Accuracy**: >99%
- **Validation Accuracy**: >99%
- **Inference Time**: <100ms per prediction
- **Model Size**: ~6.5MB

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
FileNotFoundError: Model file not found: mnist_cnn.h5
```
**Solution**: Run `python model.py` to train a new model.

**2. TensorFlow installation issues**
```
ImportError: No module named 'tensorflow'
```
**Solution**: Ensure virtual environment is activated and run `pip install -r requirements.txt`.

**3. Canvas not drawing**
**Solution**: Check browser console for JavaScript errors. Ensure all static files are loaded.

**4. Poor prediction accuracy**
**Solution**: 
- Draw digits clearly in the center of the canvas
- Use thick, connected strokes
- Retrain the model if needed

### Debug Mode

Enable debug logging by setting the debug flag in `app.py`:
```python
app.run(host="0.0.0.0", port=5002, debug=True)
```

Visit `/debug` for additional debugging information.

## 🚀 Deployment

### Development Server
The application runs on Flask's development server by default (port 5002).

### Production Deployment
For production use, deploy with a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 📊 Performance Optimization

### Memory Usage
- Model size: ~6.5MB
- RAM usage: ~500MB during inference
- CPU usage: Optimized for inference

### Speed Improvements
- Model uses CPU-optimized TensorFlow build
- Efficient image preprocessing
- Minimal memory allocation during prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add appropriate comments and documentation
5. Test thoroughly
6. Submit a pull request

## 📝 License

This project is created for educational purposes. Feel free to use and modify as needed.

## 🙏 Acknowledgments

- MNIST dataset creators and maintainers
- TensorFlow and Keras teams
- Flask development team
- Open source community

---

**Built with ❤️ for machine learning education and demonstration.**