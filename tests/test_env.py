"""Phase 0: environment verification.

Confirms that the Python environment can run everything subsequent.
Run from the project root:

    python -m pytest tests/test_env.py -v

Pass condition: zero failures, zero errors.
"""
import sys
import importlib


REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 14

REQUIRED_PACKAGES = ["sympy", "mpmath", "gmpy2", "pytest"]
OPTIONAL_PACKAGES = ["cupy", "pycuda", "primesieve", "numba"]


def test_python_version():
    """Python is at the expected version (3.14.x)."""
    assert sys.version_info.major == REQUIRED_PYTHON_MAJOR, (
        f"Need Python {REQUIRED_PYTHON_MAJOR}.x, got {sys.version}"
    )
    assert sys.version_info.minor >= REQUIRED_PYTHON_MINOR, (
        f"Need Python {REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}+, "
        f"got {sys.version}"
    )


def test_required_packages_importable():
    """All packages listed as required can be imported."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    assert not missing, f"Required packages missing: {missing}"


def test_gmpy2_basic_functionality():
    """gmpy2.powmod works on a tiny test input."""
    import gmpy2
    # 2^10 mod 1000 = 1024 mod 1000 = 24
    assert int(gmpy2.powmod(2, 10, 1000)) == 24


def test_sympy_primerange_basic():
    """sympy.primerange returns the expected primes for a small range."""
    from sympy import primerange
    primes_below_30 = list(primerange(2, 30))
    expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert primes_below_30 == expected


def test_optional_packages_reported():
    """List optional packages and whether they are available. Never fails."""
    available = []
    unavailable = []
    for pkg in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg)
            available.append(pkg)
        except ImportError:
            unavailable.append(pkg)
    # This test always passes, but prints status for visibility
    print(f"\nOptional packages available: {available}")
    print(f"Optional packages unavailable: {unavailable}")


def test_modexp_module_importable():
    """The scripts.modexp module can be imported."""
    # Add the project root to sys.path so scripts.modexp resolves
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from scripts.modexp import modexp_python, modexp_gmpy2, is_wieferich
    # Smoke test: 2^10 mod 1000 = 24
    assert modexp_python(2, 10, 1000) == 24
    assert modexp_gmpy2(2, 10, 1000) == 24
    # is_wieferich on the canonical example
    assert is_wieferich(2, 1093) is True
