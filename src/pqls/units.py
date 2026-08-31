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
    r"""Converts electric dipole moment from atomic units (``a.u.``) to Coulomb-metres (``C·m``).

    Parameters
    ----------
    val : ArrayLike
        Electric dipole moment in ``a.u.``.

    Returns
    -------
    ArrayLike
        Electric dipole moment in ``C·m``.
    """
    return val * AU_DIPOLE_TO_SI


def coulomb_metre_to_au[T: ArrayLike](val: T) -> T:
    r"""Converts electric dipole moment from Coulomb-metres (``C·m``) to atomic units (``a.u.``).

    Parameters
    ----------
    val : ArrayLike
        Electric dipole moment in ``C·m``.

    Returns
    -------
    ArrayLike
        Electric dipole moment in ``a.u.``.
    """
    return val / AU_DIPOLE_TO_SI


def eV_to_GHz[T: ArrayLike](energy_ev: T) -> T:
    r"""Converts energy in electron-volts (``eV``) to linear frequency in gigahertz (``GHz``).

    Parameters
    ----------
    energy_ev : ArrayLike
        Energy in ``eV``.

    Returns
    -------
    ArrayLike
        Linear frequency in ``GHz``.
    """
    return energy_ev * _EV_TO_GHZ


def GHz_to_eV[T: ArrayLike](freq_ghz: T) -> T:
    r"""Converts linear frequency in gigahertz (``GHz``) to energy in electron-volts (``eV``).

    Parameters
    ----------
    freq_ghz : ArrayLike
        Linear frequency in ``GHz``.

    Returns
    -------
    ArrayLike
        Energy in ``eV``.
    """
    return freq_ghz * _GHZ_TO_EV


def hz_to_rad_s[T: ArrayLike](freq_hz: T) -> T:
    r"""Converts linear frequency (``Hz``) to angular frequency (``rad/s``).

    Parameters
    ----------
    freq_hz : ArrayLike
        Linear frequency :math:`f` in ``Hz``.

    Returns
    -------
    ArrayLike
        Angular frequency :math:`\omega` in ``rad/s``.
    """
    return freq_hz * math.tau


def rad_s_to_hz[T: ArrayLike](omega_rad_s: T) -> T:
    r"""Converts angular frequency (``rad/s``) to linear frequency (``Hz``).

    Parameters
    ----------
    omega_rad_s : ArrayLike
        Angular frequency :math:`\omega` in ``rad/s``.

    Returns
    -------
    ArrayLike
        Linear frequency :math:`f` in ``Hz``.
    """
    return omega_rad_s / math.tau


def electric_field_to_rabi_freq[T: ArrayLike](
    e_field_v_m: T,
    dipole_c_m: float,
) -> T:
    r"""Converts electric field amplitude (``V/m``) to angular Rabi frequency (``rad/s``).

    .. math::

        \Omega = \frac{\mu E}{\hbar}

    Parameters
    ----------
    e_field_v_m : ArrayLike
        Electric field amplitude :math:`E` in ``V/m``.
    dipole_c_m : float
        Electric dipole moment :math:`\mu` in ``C·m``.

    Returns
    -------
    ArrayLike
        Angular Rabi frequency :math:`\Omega` in ``rad/s``.
    """
    return (dipole_c_m * e_field_v_m) / hbar


def rabi_freq_to_electric_field[T: ArrayLike](
    omega_rad_s: T,
    dipole_c_m: float,
) -> T:
    r"""Converts angular Rabi frequency (``rad/s``) to electric field amplitude (``V/m``).

    .. math::

        E = \frac{\Omega \hbar}{\mu}

    Parameters
    ----------
    omega_rad_s : ArrayLike
        Angular Rabi frequency :math:`\Omega` in ``rad/s``.
    dipole_c_m : float
        Electric dipole moment :math:`\mu` in ``C·m``.

    Returns
    -------
    ArrayLike
        Electric field amplitude :math:`E` in ``V/m``.
    """
    return (omega_rad_s * hbar) / dipole_c_m
