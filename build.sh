#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Checking for model files..."
if [ ! -f "model.pth" ]; then
    echo "Model file not found. Training will happen on first run."
fi

if [ ! -f "scaler.pkl" ]; then
    echo "Scaler file not found. Training will happen on first run."
fi

echo "Build completed successfully!" 