# 🖊️ Handwriting Digit Recognition Web App

Ultra-lightweight web application for handwritten digit recognition using scikit-learn. Draw digits on an HTML5 canvas and get real-time predictions with confidence scores.

[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-blue?logo=render)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.13+-brightgreen?logo=python)](https://python.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.90%25-success)](https://scikit-learn.org)

## 🚀 Features

- **Interactive Drawing Canvas**: Draw digits with mouse or touch input (280×280 resolution)
- **Real-time Predictions**: Get instant digit recognition with confidence scores  
- **High Accuracy Model**: Enhanced Neural Network with **97.90% accuracy**
- **Ultra-Lightweight**: Uses scikit-learn (5.7MB) instead of TensorFlow (259MB+)
- **Fast Deployment**: Minimal dependencies (5 packages) for reliable cloud deployment
- **Data Augmentation**: Robust model trained on 120,000 augmented MNIST samples
- **Mobile-Friendly**: Responsive design with full touch support
- **Cloud Optimized**: Production-ready deployment on Render with gunicorn
- **Professional Code**: Clean, documented, and maintainable codebase

## 🏗️ Architecture

### Backend (Python/Flask)
- **Flask Web Framework**: Lightweight web server and RESTful API
- **Scikit-Learn MLPClassifier**: Enhanced Neural Network (256→128→64 architecture)
- **Advanced Preprocessing**: Multi-stage image processing pipeline
- **Error Handling**: Comprehensive error management and logging
- **Production Ready**: Gunicorn WSGI server with health checks

### Frontend (HTML/CSS/JavaScript)
- **HTML5 Canvas**: Interactive drawing interface with smooth rendering
- **Responsive Design**: Centered layout that works on all screen sizes
- **Real-time Communication**: Async fetch API with timeout handling
- **Touch Support**: Full mobile compatibility with gesture support
- **User Experience**: Clear feedback and error messages

### Machine Learning
- **Enhanced Neural Network**: MLPClassifier with optimized hyperparameters
- **MNIST Dataset**: 60,000 training + 60,000 augmented samples = 120,000 total
- **Data Augmentation**: Gaussian noise injection for improved robustness
- **Model Selection**: Ensemble voting between Neural Network and Random Forest
- **Preprocessing**: StandardScaler normalization and intelligent image cropping

## 📋 Requirements

- **Python 3.8+** (Tested on Python 3.13.4)
- **pip** (Python package manager)
- **Virtual environment** (highly recommended)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/rt75272/Handwritting.git
cd Handwritting
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv handwritting

# Activate on Linux/Mac:
source handwritting/bin/activate

# Activate on Windows:
handwritting\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies (Ultra-Minimal):**
- `Flask==3.1.0` - Web framework
- `numpy==2.2.1` - Numerical computing  
- `scikit-learn==1.6.0` - Machine learning
- `Pillow==11.1.0` - Image processing
- `gunicorn==23.0.0` - Production server

### 4. Train the Model (Optional)
```bash
# Train enhanced model with data augmentation
python train_sklearn_model.py
```

**Note**: Pre-trained model (`sklearn_digit_model.pkl`) is included for immediate use.

## 🚀 Usage

### Starting the Application

1. **Activate virtual environment** (if not already active):
   ```bash
   source handwritting/bin/activate  # Linux/Mac
   # or
   handwritting\Scripts\activate     # Windows
   ```

2. **Run the Flask application**:
   ```bash
   python app_sklearn.py
   ```

3. **Open in browser**:
   - Navigate to `http://localhost:5000`
   - Draw a digit (0-9) on the 280×280 canvas
   - Click "Predict" to get instant results with confidence scores
   - Use "Clear" to reset and try another digit

### 🌐 Live Demo

**Production App**: [https://handwritting-latest.onrender.com](https://handwritting-latest.onrender.com)

### API Endpoints

- **`GET /`** - Main drawing interface (HTML page)
- **`POST /predict`** - Digit prediction API (JSON response)
- **`GET /health`** - Application health check

#### API Usage Example
```bash
# POST to /predict with base64 image data
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}'

# Response
{
  "prediction": 7,
  "confidence": "98.2%",
  "probabilities": [0.001, 0.002, ..., 0.982, ...]
}
```

### Model Retraining

To retrain with enhanced data augmentation:

```bash
python train_sklearn_model.py
```

**Training Features:**
- Multiple model architectures (Neural Network, Random Forest, SVM)
- Ensemble voting for optimal performance
- Data augmentation with Gaussian noise
- Comprehensive evaluation and per-digit analysis
- Training time: ~5-10 minutes on modern CPU

## 📁 Project Structure

```
Handwritting/
├── app_sklearn.py                    # 🚀 Main Flask application (production)
├── train_sklearn_model.py            # 🧠 Enhanced model training script  
├── requirements.txt                  # 📦 Python dependencies (5 packages)
├── Procfile                         # ⚙️  Render deployment config
├── README.md                        # 📖 Project documentation
├── .gitignore                       # 🚫 Git ignore rules
├── sklearn_digit_model.pkl          # 🎯 Trained model (5.7MB)
├── sklearn_scaler.pkl               # 📏 Data scaler for preprocessing
├── sklearn_metadata.json            # 📊 Model performance metrics
├── static/
│   ├── css/
│   │   └── style.css               # 🎨 Responsive styling
│   └── js/
│       └── script.js               # ⚡ Canvas drawing & API calls
├── templates/
│   └── index.html                  # 🖼️  Drawing interface
└── handwritting/                   # 🐍 Virtual environment
```

**Key Files:**
- **`app_sklearn.py`**: Production Flask app with error handling
- **`train_sklearn_model.py`**: Complete training pipeline with augmentation
- **`static/js/script.js`**: Interactive canvas with touch support
- **`templates/index.html`**: Clean, responsive drawing interface

## 🔧 Technical Details

### Model Architecture
```
Enhanced Neural Network (MLPClassifier):
Input (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Output(10, Softmax)

Features:
- Adam optimizer with adaptive learning rate
- Early stopping (patience=20) to prevent overfitting  
- L2 regularization (alpha=0.0001)
- Batch normalization and dropout equivalent via alpha parameter
```

### Image Processing Pipeline
1. **Canvas Capture**: HTML5 canvas (280×280) to base64 PNG data URL
2. **Base64 Decoding**: Convert to PIL Image object
3. **Grayscale Conversion**: RGB to single channel if needed
4. **Intelligent Cropping**: Automatic bounding box detection with 20% padding
5. **MNIST Resize**: Resize to standard 28×28 pixels using Lanczos resampling
6. **Color Inversion**: Convert to MNIST format (white digits on black background)
7. **Normalization**: Pixel values normalized to [0,1] range
8. **Vectorization**: Flatten to 784-element vector for scikit-learn input
9. **Scaling**: StandardScaler normalization for optimal model performance

### Performance Metrics
- **Test Accuracy**: **97.90%** (Enhanced Neural Network)
- **Training Samples**: 120,000 (60k original + 60k augmented)
- **Model Size**: 5.7MB (sklearn) vs 259MB+ (TensorFlow equivalent)
- **Inference Time**: ~50ms per prediction
- **Memory Usage**: <100MB RAM during inference
- **Dependencies**: 5 packages vs 20+ for TensorFlow
- **Startup Time**: <2 seconds vs 10+ seconds for TensorFlow

### Per-Digit Accuracy Analysis
```
Digit 0: 0.990 (980 samples)    Digit 5: 0.978 (892 samples)
Digit 1: 0.991 (1135 samples)   Digit 6: 0.983 (958 samples)  
Digit 2: 0.976 (1032 samples)   Digit 7: 0.981 (1028 samples)
Digit 3: 0.980 (1010 samples)   Digit 8: 0.966 (974 samples)
Digit 4: 0.979 (982 samples)    Digit 9: 0.973 (1009 samples)
```

## 🐛 Troubleshooting

### Common Issues

**1. Model files not found**
```
❌ Sklearn model file sklearn_digit_model.pkl not found!
```
**Solution**: Run `python train_sklearn_model.py` to create the model files.

**2. Import errors**
```
ImportError: No module named 'sklearn'
```
**Solution**: Activate virtual environment and install dependencies:
```bash
source handwritting/bin/activate  # or handwritting\Scripts\activate on Windows
pip install -r requirements.txt
```

**3. Canvas not drawing**
**Solution**: 
- Check browser console (F12) for JavaScript errors
- Ensure all static files load correctly
- Try refreshing the page

**4. Poor prediction accuracy**
**Solution**: 
- Draw digits **clearly and centered** in the canvas
- Use **thick, connected strokes** (pen width is optimized at 20px)
- Avoid drawing too close to edges
- Make sure digits fill a reasonable portion of the canvas
- Clear and redraw if needed

**5. "Predicting..." stuck**
**Solution**:
- Check network connection
- Verify Flask app is running on correct port
- Check browser console for fetch errors

### Debug Mode

Enable debug logging in `app_sklearn.py`:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug=True
```

**Debugging Tips:**
- Check `/health` endpoint for model status
- Monitor console logs for detailed error information
- Use browser Network tab to inspect API requests

## 🚀 Deployment

### Local Development Server
```bash
python app_sklearn.py
# Serves on http://localhost:5000
```

### Production Deployment (Render)

The app is configured for **one-click deployment** on Render:

1. **Fork this repository** on GitHub
2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Create new Web Service from GitHub repo
   - Render auto-detects Python and uses `Procfile`
3. **Automatic build**: 
   ```bash
   # Render runs automatically:
   pip install -r requirements.txt
   gunicorn app_sklearn:app
   ```

**Production Configuration:**
- **Web Server**: Gunicorn WSGI server
- **Port**: Dynamic (set by Render via `$PORT` environment variable)  
- **Health Check**: `/health` endpoint for monitoring
- **Build Time**: ~2-3 minutes (vs 10+ minutes for TensorFlow apps)
- **Cold Start**: <2 seconds (vs 15+ seconds for TensorFlow apps)

### Alternative Deployment Options

**Heroku:**
```bash
# Procfile already configured
git push heroku main
```

**Docker:**
```dockerfile
FROM python:3.13-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD gunicorn app_sklearn:app --host 0.0.0.0 --port $PORT
```

**Local Production Server:**
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app_sklearn:app
```

## 📊 Performance Optimization

### Memory Efficiency
- **Model size**: 5.7MB (sklearn) vs 259MB+ (TensorFlow equivalent)
- **RAM usage**: <100MB during inference vs 500MB+ for TensorFlow
- **Startup memory**: Minimal vs large TensorFlow graph loading

### Speed Optimizations
- **CPU-only inference**: No GPU dependencies or CUDA overhead
- **Optimized preprocessing**: NumPy + PIL operations
- **Minimal dependencies**: 5 packages vs 20+ for deep learning frameworks
- **Fast model loading**: Pickle deserialization vs complex model graph loading
- **Efficient predictions**: Direct scikit-learn inference vs TensorFlow session management

### Deployment Benefits
- **Faster builds**: 2-3 minutes vs 10+ minutes
- **Reliable deployments**: Fewer dependency conflicts
- **Lower resource costs**: Smaller memory footprint
- **Better cold start performance**: <2s vs 15+ seconds

## � Development & Testing

### Code Quality
- **Professional Standards**: Clean, documented, and maintainable code
- **Error Handling**: Comprehensive exception management and user feedback
- **Logging**: Detailed application logs for debugging and monitoring
- **Type Hints**: Python type annotations for better code clarity
- **Documentation**: Extensive docstrings and inline comments

### Testing the Model
```bash
# Quick accuracy test
python -c "
from train_sklearn_model import train_enhanced_sklearn_model
model, scaler, accuracy = train_enhanced_sklearn_model()
print(f'Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
"
```

### Performance Monitoring
```bash
# Check model performance
curl http://localhost:5000/health

# Response
{
  "status": "healthy",
  "model_loaded": true,
  "mode": "digits"
}
```

## �🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/amazing-improvement`
3. **Make your changes** with proper documentation
4. **Add tests** if applicable
5. **Follow code style**: Clean formatting with proper comments
6. **Test thoroughly** on both local and production environments
7. **Submit a pull request** with detailed description

### Code Style Guidelines
- Use meaningful variable and function names
- Add docstrings to all functions
- Include type hints where appropriate
- End all comments with periods
- Remove extra spaces within functions
- Follow PEP 8 Python style guide

## 📝 License

This project is created for **educational and demonstration purposes**. 

**MIT License** - Feel free to use, modify, and distribute as needed.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **Scikit-Learn Team**: For the excellent machine learning library
- **Flask Team**: For the lightweight and powerful web framework
- **Render**: For providing reliable and affordable cloud hosting
- **Open Source Community**: For the foundational tools and libraries

## 📈 Project Evolution

This project evolved from a TensorFlow-based CNN to an ultra-lightweight scikit-learn solution:

**Original Approach:**
- TensorFlow/Keras CNN model (259MB+)
- Complex dependencies (20+ packages)
- Slow deployment and cold starts
- Memory intensive (500MB+ RAM)

**Current Approach:**
- Scikit-learn MLPClassifier (5.7MB)
- Minimal dependencies (5 packages)  
- Fast deployment and startup (<2s)
- Memory efficient (<100MB RAM)
- **Same 97.90% accuracy achieved!**

---

**🎯 Built with ❤️ for machine learning education, demonstration, and production deployment.**

---

**Live Demo**: [https://handwritting-latest.onrender.com](https://handwritting-latest.onrender.com)

*Last Updated: October 16, 2025*
