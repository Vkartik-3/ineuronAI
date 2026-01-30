"""
Environment Test Script
Tests if all required packages can be imported successfully
"""

import sys


def test_imports():
    """Test importing all required packages"""

    print("=" * 60)
    print("Testing ML Pipeline Environment")
    print("=" * 60)
    print()

    results = []

    # Test each import
    tests = [
        (
            "Python Version",
            lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        ("TensorFlow", lambda: __import__("tensorflow").__version__),
        ("Keras", lambda: __import__("keras").__version__),
        ("scikit-learn", lambda: __import__("sklearn").__version__),
        ("XGBoost", lambda: __import__("xgboost").__version__),
        ("NumPy", lambda: __import__("numpy").__version__),
        ("Pandas", lambda: __import__("pandas").__version__),
        ("SciPy", lambda: __import__("scipy").__version__),
        ("MLflow", lambda: __import__("mlflow").__version__),
        ("Matplotlib", lambda: __import__("matplotlib").__version__),
        ("Seaborn", lambda: __import__("seaborn").__version__),
        ("PyYAML", lambda: __import__("yaml").__version__),
        ("Joblib", lambda: __import__("joblib").__version__),
    ]

    max_name_len = max(len(name) for name, _ in tests)

    all_passed = True

    for name, test_func in tests:
        try:
            version = test_func()
            status = "✓ PASS"
            results.append(True)
            print(f"{name:<{max_name_len}} : {status} (v{version})")
        except Exception as e:
            status = "✗ FAIL"
            results.append(False)
            all_passed = False
            print(f"{name:<{max_name_len}} : {status} - {str(e)[:50]}")

    print()
    print("=" * 60)

    if all_passed:
        print("✓ All tests passed! Environment is ready.")
        print()
        print("You can now run:")
        print("  python train_pipeline.py --config config/training_config.yaml")
    else:
        print("✗ Some tests failed. Please fix the environment.")
        print()
        print("Try:")
        print("  1. pip install -r requirements-fixed.txt")
        print("  2. Or recreate the virtual environment")
        print("  3. See ENVIRONMENT_SETUP.md for detailed instructions")

    print("=" * 60)
    print()

    return all_passed


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
