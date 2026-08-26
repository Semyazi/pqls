"""
Atoms management and caching for ARC physics parameters.
"""

from functools import lru_cache

from arc import Cesium, Rubidium85, Rubidium87

from .types import (
    ArrayOrScalar,
    AtomicLevel,
    AtomName,
    AtomType,
    QuantumState,
)
from .units import au_to_coulomb_metre, electric_field_to_rabi_freq

_ATOM_MAP = {
    "rb85": Rubidium85,
    "rb87": Rubidium87,
    "cs": Cesium,
}


@lru_cache(maxsize=3)
def get_atom(atom_name: AtomName) -> AtomType:
    """Returns a cached instance of the requested atom from the ARC package.

    Args:
        atom_name (AtomName): The common name of the atom (e.g., 'Rb85', 'Cs').

    Returns:
        The instantiated atom object from the ARC library.

    Raises:
        ValueError: If the provided atom_name is not in the supported list.
    """
    atom_cls = _ATOM_MAP.get(atom_name.lower())
    if atom_cls is None:
        raise ValueError(f"Atom {atom_name} not supported.")
    return atom_cls()


def get_ground_state(atom_name: AtomName) -> AtomicLevel:
    """Returns the ground state of the specified atom.

    Args:
        atom_name (AtomName): The common name of the atom.

    Returns:
        AtomicLevel: An object representing the n, l, and j quantum numbers
        of the ground state.
    """
    arc_atom = get_atom(atom_name)
    return AtomicLevel(n=arc_atom.groundStateN, l=0, j=0.5)


def calculate_rabi_frequency[T: ArrayOrScalar](
    atom: AtomName,
    lower: QuantumState,
    upper: QuantumState,
    e_field: T,
    q: int | None = None,
) -> T:
    """Calculates angular Rabi frequency (rad/s) between two atomic states for a given electric field.

    Parameters
    ----------
    atom : str, AtomName, or ARC Atom object
        The target alkali atom.
    lower : QuantumState
        Initial / lower energy state.
    upper : QuantumState
        Target / upper energy state.
    e_field : ArrayOrScalar
        Electric field amplitude in V/m (scalar or array).
    q : int, optional
        Laser polarization transition (-1 for sigma-, 0 for pi, +1 for sigma+).
        If None, inferred automatically from delta m_j.

    Returns
    -------
    ArrayOrScalar
        Angular Rabi frequency Omega in rad/s matching the shape of `e_field`.
    """
    arc_atom = get_atom(atom)

    if q is None:
        q = round(upper.mj - lower.mj)

    dipole_au = abs(
        arc_atom.getDipoleMatrixElement(
            *lower.to_arc_args(),
            *upper.to_arc_args(),
            q=q,
        )
    )
    dipole_cm = au_to_coulomb_metre(dipole_au)
    return electric_field_to_rabi_freq(e_field, dipole_cm)
