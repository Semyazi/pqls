"""
PQLS: High-performance JAX-accelerated Lindblad steady-state solvers for quantum atomic systems.
"""

from .atoms import calculate_rabi_frequency, get_atom, get_ground_state
from .lindblad import (
    solve_custom_network,
    solve_ladder_system,
    solve_quantum_ladder,
    solve_steady_state,
)
from .types import (
    ArrayLike,
    ArrayOrScalar,
    AtomicLevel,
    AtomName,
    Driver,
    QuantumLadder,
    QuantumState,
)
from .units import (
    GHz_to_eV,
    au_to_coulomb_metre,
    coulomb_metre_to_au,
    electric_field_to_rabi_freq,
    eV_to_GHz,
    hz_to_rad_s,
    rabi_freq_to_electric_field,
    rad_s_to_hz,
)

__version__ = "0.1.0"

__all__ = [
    # Core Solvers
    "solve_steady_state",
    "solve_custom_network",
    "solve_ladder_system",
    "solve_quantum_ladder",
    # Data Models & Types
    "ArrayLike",
    "ArrayOrScalar",
    "AtomicLevel",
    "QuantumState",
    "QuantumLadder",
    "Driver",
    "AtomName",
    # Atom Helpers
    "get_atom",
    "get_ground_state",
    "calculate_rabi_frequency",
    # Units & Conversions
    "au_to_coulomb_metre",
    "coulomb_metre_to_au",
    "eV_to_GHz",
    "GHz_to_eV",
    "electric_field_to_rabi_freq",
    "rabi_freq_to_electric_field",
    "hz_to_rad_s",
    "rad_s_to_hz",
]
