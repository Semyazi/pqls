"""
Type definitions.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from arc import Cesium, Rubidium85, Rubidium87

if TYPE_CHECKING:
    import jax

type ArrayLike = float | np.ndarray | jax.Array
type ArrayOrScalar = float | np.ndarray

type AtomType = Rubidium85 | Rubidium87 | Cesium
type AtomName = Literal["Rb85", "Rb87", "Cs", "rb85", "rb87", "cs"]


@dataclass(frozen=True)
class AtomicLevel:
    """
    Fine-structure atomic energy level.

    Attributes:
        n: Principal quantum number.
        l: Orbital angular momentum (0=S, 1=P, 2=D, 3=F, ...).
        j: Total angular momentum (l + s).
    """

    n: int
    l: int
    j: float

    def __str__(self) -> str:
        """Spectroscopic notation (e.g., 5S_{1/2}, 5P_{3/2}, 50D_{5/2})."""
        l_symbols = ["S", "P", "D", "F", "G", "H"]
        l_str = l_symbols[self.l] if self.l < len(l_symbols) else f"l={self.l}"

        j_fractions = {0.5: "1/2", 1.5: "3/2", 2.5: "5/2", 3.5: "7/2", 4.5: "9/2"}
        j_str = j_fractions.get(self.j, str(self.j))

        return f"{self.n}{l_str}_{{{j_str}}}"

    def to_arc_args(self) -> tuple[int, int, float]:
        """Returns the argument tuple required by ARC: (n, l, j)."""
        return (self.n, self.l, self.j)

    def to_arc_args_mj(self, mj: float) -> tuple[int, int, float, float]:
        """Returns the argument tuple required by ARC with magnetic sub-level: (n, l, j, mj)."""
        return (self.n, self.l, self.j, mj)


@dataclass(frozen=True)
class QuantumState:
    """
    Fully resolved atomic quantum state including magnetic quantum number.

    Attributes:
        n: Principal quantum number.
        l: Orbital angular momentum.
        j: Total angular momentum.
        mj: Magnetic quantum number.
    """

    n: int
    l: int
    j: float
    mj: float

    @property
    def level(self) -> AtomicLevel:
        """Underlying fine-structure level (n, l, j)."""
        return AtomicLevel(n=self.n, l=self.l, j=self.j)

    @classmethod
    def from_level(cls, level: AtomicLevel, mj: float) -> Self:
        """Constructs QuantumState from an AtomicLevel and mj value."""
        return cls(n=level.n, l=level.l, j=level.j, mj=mj)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, float, float]) -> Self:
        """Constructs QuantumState from an (n, l, j, mj) tuple."""
        return cls(*t)

    def to_arc_args(self) -> tuple[int, int, float, float]:
        """Returns the argument tuple required by ARC: (n, l, j, mj)."""
        return (self.n, self.l, self.j, self.mj)

    def __str__(self) -> str:
        """Spectroscopic notation with mj (e.g., 5S_{1/2}, mj=1/2)."""
        mj_fractions = {
            0.5: "1/2",
            -0.5: "-1/2",
            1.5: "3/2",
            -1.5: "-3/2",
            2.5: "5/2",
            -2.5: "-5/2",
        }
        mj_str = mj_fractions.get(self.mj, str(self.mj))
        return f"{self.level}, mj={mj_str}"


@dataclass(frozen=True)
class QuantumLadder:
    """
    Arbitrary N-level ladder system.

    Attributes:
        atom: Target alkali species
        states: Sequence of QuantumStates ordered from ground state upward.
    """

    atom: AtomName
    states: Sequence[QuantumState]

    def __post_init__(self) -> None:
        if not isinstance(self.states, tuple):
            object.__setattr__(self, "states", tuple(self.states))
        if len(self.states) < 2:
            raise ValueError("QuantumLadder requires at least 2 quantum states.")

    @property
    def num_levels(self) -> int:
        """Total number of levels in the ladder system."""
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        """Total number of adjacent driving transitions (N - 1)."""
        return len(self.states) - 1


@dataclass(frozen=True)
class Driver:
    """
    Laser or RF electromagnetic field driving an atomic transition.

    Attributes:
        amplitude: Electric field amplitude in V/m.
        frequency: Field frequency in Hz.
        account_for_detuning: If True, calculates detuning relative to atomic resonance.
                              If False, assumes resonant driving (detuning = 0.0).
    """

    amplitude: float
    frequency: float = 0.0
    account_for_detuning: bool = False
