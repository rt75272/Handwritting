# 🔍 DEBUG DEPLOYMENT - Fix 500 Prediction Error

## Problem
Your app deployed successfully but predictions are failing with 500 errors.

## 🚀 IMMEDIATE DEBUG SOLUTION

### Step 1: Switch to Debug App
In your Render dashboard, change the **Start Command** to:
```bash
gunicorn app_debug:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --preload --access-logfile - --error-logfile -
```

### Step 2: Deploy and Test
1. Redeploy with the debug version
2. Check these endpoints:
   - `/debug` - Shows detailed environment info
   - `/health` - Shows model loading status

### Step 3: Check Logs
The debug version will show detailed logs including:
- ✅ Which model files exist
- ✅ Model loading success/failure
- ✅ Detailed prediction process
- ✅ Environment information

## 🔍 Common Issues & Solutions

### Issue 1: Model Files Missing
**Symptoms:** Debug shows model files not found
**Solution:** Ensure these files are in your repo:
- `mnist_cnn.h5`
- `alphanumeric_cnn.keras` 
- `alphanumeric_labels.json`

### Issue 2: TensorFlow Loading Issues
**Symptoms:** Models exist but fail to load
**Solution:** Models might be incompatible with cloud TensorFlow version

### Issue 3: Memory Issues
**Symptoms:** App starts but crashes on prediction
**Solution:** Models too large for free tier memory

## 🛠️ Quick Fixes to Try

### Fix 1: Use Only Digit Model
If alphanumeric model is causing issues, temporarily remove it and use only digits.

### Fix 2: Model Compatibility
The debug app loads models with `compile=False` to avoid compatibility issues.

### Fix 3: Memory Optimization
Using single worker and preload to minimize memory usage.

## 📊 Debug Endpoints

Once deployed with debug version:

```bash
# Check environment and model status
curl https://handwritting.onrender.com/debug

# Check health
curl https://handwritting.onrender.com/health

# Test prediction (will show detailed logs)
# Draw something and submit - check Render logs
```

## 🎯 Expected Debug Output

In Render logs, you should see:
```
🔍 Environment Debug Info:
✅ mnist_cnn.h5: exists (X bytes)
✅ Model loaded successfully
🔍 Prediction request received
🔄 Preprocessing image...
✅ Prediction successful: 5 (87.2%)
```

**Deploy the debug version and check the logs - this will tell us exactly what's failing!** 🚀