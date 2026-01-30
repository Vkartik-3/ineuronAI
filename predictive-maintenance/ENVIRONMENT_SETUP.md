# Python Environment Setup Guide

## Issue: Corrupted Python Environment

Your Python environment has corrupted Cython modules, causing TensorFlow and scikit-learn imports to fail.

## Solution: Create Fresh Environment

### Option 1: Using `uv` (Recommended)

```bash
# Navigate to the project root
cd /d/data-science-projects/predictive-maintenance

# Remove existing virtual environment
rm -rf .venv

# Create fresh environment with Python 3.10
uv venv --python 3.10

# Activate the environment
source .venv/Scripts/activate  # Git Bash
# OR
.venv\Scripts\activate.bat      # CMD
# OR
.venv\Scripts\Activate.ps1      # PowerShell

# Install ML pipeline dependencies
cd ml_pipeline/train
uv pip install -r requirements-fixed.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

### Option 2: Using Standard `venv`

```bash
# Navigate to project root
cd /d/data-science-projects/predictive-maintenance

# Remove old environment
rm -rf venv

# Create new virtual environment
python -m venv venv

# Activate
source venv/Scripts/activate  # Git Bash

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
cd ml_pipeline/train
pip install -r requirements-fixed.txt

# Verify
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### Option 3: Using Conda (Most Reliable)

```bash
# Create conda environment
conda create -n pred-maint python=3.10 -y

# Activate
conda activate pred-maint

# Install dependencies
cd /d/data-science-projects/predictive-maintenance/ml_pipeline/train
pip install -r requirements-fixed.txt

# Verify
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

## After Setup

Test the training pipeline:

```bash
cd /d/data-science-projects/predictive-maintenance/ml_pipeline/train
python train_pipeline.py --config config/training_config.yaml
```

## Key Changes in `requirements-fixed.txt`

- **TensorFlow**: Downgraded from 2.15.0 → 2.13.0 (better Python 3.10 compatibility)
- **Keras**: Matched to TensorFlow version (2.13.1)
- All other dependencies remain the same

## Troubleshooting

If imports still fail after recreation:

1. **Check Python version**:

   ```bash
   python --version  # Should be 3.10.x
   ```

2. **Clear pip cache**:

   ```bash
   pip cache purge
   ```

3. **Install with no cache**:

   ```bash
   pip install --no-cache-dir -r requirements-fixed.txt
   ```

4. **Try tensorflow-cpu** (lighter, faster install):
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu==2.13.0
   ```

## Other Components

Each component has its own requirements. Install them separately:

```bash
# Data Generator
cd data_generator
pip install -r requirements.txt

# Data Loader
cd ../data_loader
pip install -r requirements.txt

# Feature Store
cd ../feature_store
pip install -r requirements.txt
```
