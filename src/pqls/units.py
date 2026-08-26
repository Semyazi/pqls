"""
Physical constants and unit conversion utilities for quantum atomic calculations.
"""

import math

from scipy.constants import eV, h, hbar, physical_constants

from .types import ArrayLike

AU_DIPOLE_TO_SI: float = physical_constants["atomic unit of electric dipole moment"][0]
_EV_TO_GHZ: float = eV / (h * 1e9)
_GHZ_TO_EV: float = (1e9 * h) / eV


def au_to_coulomb_metre[T: ArrayLike](val: T) -> T:
    """
    Converts electric dipole moment from atomic units (a.u. / e*a0) to Coulomb-metres (C·m).
    """
    return val * AU_DIPOLE_TO_SI


def coulomb_metre_to_au[T: ArrayLike](val: T) -> T:
    """
    Converts electric dipole moment from Coulomb-metres (C·m) to atomic units (a.u.).
    """
    return val / AU_DIPOLE_TO_SI


def eV_to_GHz[T: ArrayLike](energy_ev: T) -> T:
    """
    Converts energy in electron-volts (eV) to linear frequency in gigahertz (GHz).
    """
    return energy_ev * _EV_TO_GHZ


def GHz_to_eV[T: ArrayLike](freq_ghz: T) -> T:
    """
    Converts linear frequency in gigahertz (GHz) to energy in electron-volts (eV).
    """
    return freq_ghz * _GHZ_TO_EV


def hz_to_rad_s[T: ArrayLike](freq_hz: T) -> T:
    """
    Converts linear frequency (Hz) to angular frequency (rad/s): omega = 2 * pi * f.
    """
    return freq_hz * math.tau


def rad_s_to_hz[T: ArrayLike](omega_rad_s: T) -> T:
    """
    Converts angular frequency (rad/s) to linear frequency (Hz): f = omega / (2 * pi).
    """
    return omega_rad_s / math.tau


def electric_field_to_rabi_freq[T: ArrayLike](
    e_field_v_m: T,
    dipole_c_m: float,
) -> T:
    """
    Converts electric field amplitude (V/m) to angular Rabi frequency (rad/s).

    Formula:
        Omega = (mu * E) / hbar
    """
    return (dipole_c_m * e_field_v_m) / hbar


def rabi_freq_to_electric_field[T: ArrayLike](
    omega_rad_s: T,
    dipole_c_m: float,
) -> T:
    """
    Converts angular Rabi frequency (rad/s) to electric field amplitude (V/m).

    Formula:
        E = (Omega * hbar) / mu
    """
    return (omega_rad_s * hbar) / dipole_c_m
