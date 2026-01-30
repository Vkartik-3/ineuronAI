#!/bin/bash
# Environment Recreation Script for ML Pipeline

set -e  # Exit on error

echo "=========================================="
echo "ML Pipeline Environment Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "train_pipeline.py" ]]; then
    echo "❌ Error: Must run from ml_pipeline/train directory"
    echo "Run: cd ml_pipeline/train && bash setup_env.sh"
    exit 1
fi

echo "✓ In correct directory: $(pwd)"
echo ""

# Navigate to project root
cd ../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo "⚠️  Warning: Python 3.10 recommended, found $PYTHON_VERSION"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Option to remove old environment
if [[ -d ".venv" ]]; then
    echo "Found existing .venv directory"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old environment..."
        rm -rf .venv
        echo "✓ Removed"
    fi
fi
echo ""

# Create virtual environment
if [[ ! -d ".venv" ]]; then
    echo "Creating new virtual environment..."

    # Check if uv is available
    if command -v uv &> /dev/null; then
        echo "Using uv..."
        uv venv --python 3.10
    else
        echo "Using standard venv..."
        python -m venv .venv
    fi

    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating environment..."
source .venv/Scripts/activate || source .venv/bin/activate
echo "✓ Environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet
echo "✓ Updated"
echo ""

# Install ML pipeline dependencies
echo "Installing ML pipeline dependencies..."
cd ml_pipeline/train

if [[ -f "requirements-fixed.txt" ]]; then
    echo "Using requirements-fixed.txt (recommended)..."
    pip install -r requirements-fixed.txt
else
    echo "Using requirements.txt..."
    pip install -r requirements.txt
fi

echo "✓ Dependencies installed"
echo ""

# Verify installations
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

echo -n "TensorFlow: "
python -c "import tensorflow as tf; print(tf.__version__)" 2>&1 || echo "❌ FAILED"

echo -n "scikit-learn: "
python -c "import sklearn; print(sklearn.__version__)" 2>&1 || echo "❌ FAILED"

echo -n "NumPy: "
python -c "import numpy; print(numpy.__version__)" 2>&1 || echo "❌ FAILED"

echo -n "Pandas: "
python -c "import pandas; print(pandas.__version__)" 2>&1 || echo "❌ FAILED"

echo -n "MLflow: "
python -c "import mlflow; print(mlflow.__version__)" 2>&1 || echo "❌ FAILED"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate this environment in the future:"
echo "  source $PROJECT_ROOT/.venv/Scripts/activate"
echo ""
echo "To run training:"
echo "  cd $PROJECT_ROOT/ml_pipeline/train"
echo "  python train_pipeline.py --config config/training_config.yaml"
echo ""
