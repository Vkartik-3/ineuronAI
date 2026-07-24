"""
Environment verification (Fix 25).

Fails loudly if the numpy/scipy ABI mismatch that broke the global Anaconda
environment is present, and reports the versions of the load-bearing packages.

Usage:
    .venv/bin/python verify_env.py
"""

import sys


def main() -> int:
    problems = []
    try:
        import numpy
        from scipy import signal, stats  # noqa: F401  (import is the test)
        import sklearn  # noqa: F401
        import torch  # noqa: F401
        import mlflow  # noqa: F401
        import pandas  # noqa: F401
    except Exception as exc:  # pragma: no cover - the failure path we guard
        print(f"FAIL: core import raised: {exc}")
        return 1

    if numpy.__version__.startswith("2."):
        problems.append(
            f"numpy {numpy.__version__} is 2.x; scipy 1.13.x is built for numpy "
            "1.x and will fail at import. Pin numpy==1.26.4."
        )

    print("Environment check")
    print(f"  python        {sys.version.split()[0]}")
    for name in ("numpy", "scipy", "sklearn", "torch", "mlflow", "pandas"):
        mod = sys.modules.get(name) or __import__(name)
        print(f"  {name:<13} {getattr(mod, '__version__', '?')}")

    # Prove the contract imports and derives correctly.
    sys.path.insert(0, ".")
    from shared.feature_contract import INPUT_DIM, FEATURE_SCHEMA_HASH
    print(f"  feature_dim   {INPUT_DIM}")
    print(f"  schema_hash   {FEATURE_SCHEMA_HASH[:16]}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK — environment is consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
