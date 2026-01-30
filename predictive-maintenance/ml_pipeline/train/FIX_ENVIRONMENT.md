# ML Training Pipeline - Environment Fix

## Problem Summary

Your Python 3.10.18 environment has **corrupted Cython extensions**, causing both TensorFlow and scikit-learn to fail during import. This is a common issue when packages are installed in an inconsistent state or with incompatible versions.

## Root Cause

1. **TensorFlow 2.15.0** has known compatibility issues with Python 3.10
2. Cython modules (`.pyx` files) failed to compile or load properly
3. Corrupted pip cache or partial installations

## Solution: 3-Step Fix

### Step 1: Test Current Environment (Optional)

```bash
cd /d/data-science-projects/predictive-maintenance/ml_pipeline/train
python test_environment.py
```

This will show which packages are failing.

### Step 2: Recreate Environment

**Option A: Automated (Recommended)**

```bash
cd /d/data-science-projects/predictive-maintenance/ml_pipeline/train
bash setup_env.sh
```

**Option B: Manual**

```bash
# Go to project root
cd /d/data-science-projects/predictive-maintenance

# Remove old environment
rm -rf .venv

# Create fresh environment
python -m venv .venv

# Activate
source .venv/Scripts/activate  # Git Bash

# Install with fixed requirements
pip install --upgrade pip setuptools wheel
cd ml_pipeline/train
pip install -r requirements-fixed.txt
```

### Step 3: Verify & Run

```bash
# Test environment
python test_environment.py

# If all tests pass, run training
python train_pipeline.py --config config/training_config.yaml
```

## Key Changes

### `requirements-fixed.txt` Changes:

- **TensorFlow**: 2.15.0 → 2.13.0 (stable with Python 3.10)
- **Keras**: 2.15.0 → 2.13.1 (matches TensorFlow)
- All other packages unchanged

## Files Created

1. **`requirements-fixed.txt`** - Compatible dependency versions
2. **`setup_env.sh`** - Automated environment setup script
3. **`test_environment.py`** - Environment verification script
4. **`ENVIRONMENT_SETUP.md`** - Detailed setup guide (project root)

## Troubleshooting

### If imports still fail:

```bash
# Clear pip cache
pip cache purge

# Reinstall with no cache
pip install --no-cache-dir -r requirements-fixed.txt

# Try tensorflow-cpu (lighter, faster)
pip uninstall tensorflow
pip install tensorflow-cpu==2.13.0
```

### If you prefer Conda:

```bash
conda create -n pred-maint python=3.10 -y
conda activate pred-maint
cd /d/data-science-projects/predictive-maintenance/ml_pipeline/train
pip install -r requirements-fixed.txt
```

## Expected Output After Fix

```
Testing ML Pipeline Environment
============================================================

Python Version   : ✓ PASS (v3.10.18)
TensorFlow       : ✓ PASS (v2.13.0)
Keras            : ✓ PASS (v2.13.1)
scikit-learn     : ✓ PASS (v1.3.2)
XGBoost          : ✓ PASS (v2.0.3)
NumPy            : ✓ PASS (v1.24.3)
Pandas           : ✓ PASS (v2.0.3)
SciPy            : ✓ PASS (v1.11.3)
MLflow           : ✓ PASS (v2.9.2)
Matplotlib       : ✓ PASS (v3.8.2)
Seaborn          : ✓ PASS (v0.13.0)
PyYAML           : ✓ PASS (v6.0.1)
Joblib           : ✓ PASS (v1.3.2)

============================================================
✓ All tests passed! Environment is ready.

You can now run:
  python train_pipeline.py --config config/training_config.yaml
============================================================
```

## Alternative: Use `uv` for Faster Setup

If you have `uv` installed:

```bash
cd /d/data-science-projects/predictive-maintenance
uv venv --python 3.10
source .venv/Scripts/activate
cd ml_pipeline/train
uv pip install -r requirements-fixed.txt
```

This is significantly faster than standard pip.

## Next Steps

Once environment is working:

1. **Run training**: `python train_pipeline.py --config config/training_config.yaml`
2. **Check data paths** in `config/training_config.yaml`
3. **Monitor training** with MLflow UI: `mlflow ui`

---

**Quick Command Sequence:**

```bash
cd /d/data-science-projects/predictive-maintenance/ml_pipeline/train
bash setup_env.sh
python test_environment.py
python train_pipeline.py --config config/training_config.yaml
```
