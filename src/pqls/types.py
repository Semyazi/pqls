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
    r"""Fine-structure atomic energy level.

    Parameters
    ----------
    n : int
        Principal quantum number :math:`n`.
    l : int
        Orbital angular momentum quantum number :math:`l` (``0 = S``, ``1 = P``, ``2 = D``, ``3 = F``, ...).
    j : float
        Total angular momentum quantum number :math:`j = l + s`.
    """

    n: int
    l: int
    j: float

    def __str__(self) -> str:
        """Spectroscopic notation (e.g., ``5S_{1/2}``, ``5P_{3/2}``, ``50D_{5/2}``)."""
        l_symbols = ["S", "P", "D", "F", "G", "H"]
        l_str = l_symbols[self.l] if self.l < len(l_symbols) else f"l={self.l}"

        j_fractions = {0.5: "1/2", 1.5: "3/2", 2.5: "5/2", 3.5: "7/2", 4.5: "9/2"}
        j_str = j_fractions.get(self.j, str(self.j))

        return f"{self.n}{l_str}_{{{j_str}}}"

    def to_arc_args(self) -> tuple[int, int, float]:
        """Returns the argument tuple required by ARC.

        Returns
        -------
        tuple of (int, int, float)
            Tuple of quantum numbers ``(n, l, j)``.
        """
        return (self.n, self.l, self.j)

    def to_arc_args_mj(self, mj: float) -> tuple[int, int, float, float]:
        r"""Returns the argument tuple required by ARC with magnetic sub-level.

        Parameters
        ----------
        mj : float
            Magnetic quantum number :math:`m_j`.

        Returns
        -------
        tuple of (int, int, float, float)
            Tuple of quantum numbers ``(n, l, j, mj)``.
        """
        return (self.n, self.l, self.j, mj)


@dataclass(frozen=True)
class QuantumState:
    r"""Atomic quantum state including magnetic quantum number.

    Parameters
    ----------
    n : int
        Principal quantum number :math:`n`.
    l : int
        Orbital angular momentum quantum number :math:`l`.
    j : float
        Total angular momentum quantum number :math:`j`.
    mj : float
        Magnetic quantum number :math:`m_j`.
    """

    n: int
    l: int
    j: float
    mj: float

    @property
    def level(self) -> AtomicLevel:
        """Underlying fine-structure level ``(n, l, j)``.

        Returns
        -------
        AtomicLevel
            The base atomic level without magnetic sub-level.
        """
        return AtomicLevel(n=self.n, l=self.l, j=self.j)

    @classmethod
    def from_level(cls, level: AtomicLevel, mj: float) -> Self:
        r"""Constructs a QuantumState from an AtomicLevel and :math:`m_j` value.

        Parameters
        ----------
        level : AtomicLevel
            Underlying fine-structure level.
        mj : float
            Magnetic quantum number :math:`m_j`.

        Returns
        -------
        QuantumState
            Instantiated quantum state.
        """
        return cls(n=level.n, l=level.l, j=level.j, mj=mj)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, float, float]) -> Self:
        """Constructs a QuantumState from an ``(n, l, j, mj)`` tuple.

        Parameters
        ----------
        t : tuple of (int, int, float, float)
            Tuple containing ``(n, l, j, mj)``.

        Returns
        -------
        QuantumState
            Instantiated quantum state.
        """
        return cls(*t)

    def to_arc_args(self) -> tuple[int, int, float, float]:
        """Returns the argument tuple required by ARC.

        Returns
        -------
        tuple of (int, int, float, float)
            Tuple of quantum numbers ``(n, l, j, mj)``.
        """
        return (self.n, self.l, self.j, self.mj)

    def __str__(self) -> str:
        r"""Spectroscopic notation with :math:`m_j` (e.g., ``5S_{1/2}, mj=1/2``)."""
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
    r"""Arbitrary :math:`N`-level ladder system.

    Parameters
    ----------
    atom : AtomName
        Target alkali species.
    states : Sequence[QuantumState]
        Sequence of quantum states ordered from ground state upward.
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
        r"""Total number of levels :math:`N` in the ladder system."""
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        r"""Total number of adjacent driving transitions (:math:`N - 1`)."""
        return len(self.states) - 1


@dataclass(frozen=True)
class Driver:
    """Laser or RF electromagnetic field driving an atomic transition.

    Parameters
    ----------
    amplitude : float
        Electric field amplitude in ``V/m``.
    frequency : float, default=0.0
        Field frequency in ``Hz``.
    account_for_detuning : bool, default=False
        If ``True``, calculates detuning relative to atomic resonance.
        If ``False``, assumes resonant driving (detuning = ``0.0``).
    """

    amplitude: float
    frequency: float = 0.0
    account_for_detuning: bool = False
