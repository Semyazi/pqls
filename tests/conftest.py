import pytest


@pytest.fixture(scope="session")
def tol_parity() -> float:
    """Maximum allowable absolute/relative difference vs QuTiP ground truth."""
    return 1e-11


@pytest.fixture(scope="session")
def tol_invariant() -> float:
    """Tolerance for fundamental physical invariants (Trace=1, Hermiticity)."""
    return 1e-14


@pytest.fixture(scope="session")
def tol_pos_def() -> float:
    """Noise floor for zero-eigenvalues in positive semi-definite verification."""
    return -1e-13