#!/usr/bin/env python3
"""
Simple test to verify the production app works without running a full server.
"""

import sys
sys.path.insert(0, '/home/bob/Handwritting')

# Import the production app
from app_production import app, initialize_models

def test_production_app():
    """Test the production app functionality."""
    print("🧪 Testing production app...")
    
    # Initialize models
    print("1. Initializing models...")
    initialize_models()
    
    # Test app context
    print("2. Testing Flask app context...")
    with app.test_client() as client:
        # Test health endpoint
        print("3. Testing health endpoint...")
        response = client.get('/health')
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"   Response: {data}")
        
        # Test model status endpoint
        print("4. Testing model status endpoint...")
        response = client.get('/api/models/status')
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"   Models available: {data.get('models', {})}")
        
        # Test mode endpoint
        print("5. Testing mode endpoint...")
        response = client.get('/api/mode')
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"   Current mode: {data.get('current_mode')}")
        
        # Test prediction endpoint with dummy data
        print("6. Testing prediction endpoint...")
        dummy_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        response = client.post('/predict', 
                              json={'image': dummy_image},
                              content_type='application/json')
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Confidence: {data.get('confidence')}")
        else:
            print(f"   Error: {response.get_data(as_text=True)}")
    
    print("✅ Production app test complete!")

if __name__ == "__main__":
    test_production_app()