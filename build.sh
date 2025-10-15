#!/bin/bash
# Build script for Render deployment
# This fixes setuptools/pip compatibility issues

echo "🔧 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing dependencies..."
pip install -r requirements_minimal.txt

echo "✅ Build complete!"