#!/bin/bash

# Netlify build script for Smart Farming AI
echo "🌱 Building Smart Farming AI..."

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p trained_models

# Set environment variables for production
export FLASK_ENV=production
export FLASK_DEBUG=False

echo "✅ Build completed successfully!"
