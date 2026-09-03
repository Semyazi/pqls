"""
03_heatmap.py

5-level Rydberg Autler-Townes probe transmission heatmap simulation demonstrating vectorized 2D parameter sweeps
(Interference Field vs. Coupling Detuning) comparing the presence and absence of an RF Local Oscillator.

Adapted from simulation codes by Javane Rostampoor and Raviraj Adve for:
"Interference Resilient Quantum Receivers with Rydberg Atoms" (IEEE GLOBECOM Workshops 2025),
reproducing the 2D parameter space transmission heatmaps in Fig. 4(a) and Fig. 4(b).
All physical parameters, detunings, and decay rates directly mirror their implementation.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, hbar

from pqls.atoms import calculate_rabi_frequency, get_atom
from pqls.lindblad import solve_ladder_system
from pqls.types import QuantumState
from pqls.units import au_to_coulomb_metre

# Ladder: 5S_1/2 -> 5P_3/2 -> 53D_5/2 -> 54P_3/2 -> 52D_5/2
s0 = QuantumState(n=5, l=0, j=0.5, mj=0.5)
s1 = QuantumState(n=5, l=1, j=1.5, mj=1.5)
s2 = QuantumState(n=53, l=2, j=2.5, mj=2.5)
s3 = QuantumState(n=54, l=1, j=1.5, mj=1.5)
s4 = QuantumState(n=52, l=2, j=2.5, mj=2.5)

atom = "Rb85"
gammas = [
    0.0,
    2 * np.pi * 6e6,
    2 * np.pi * 0.003e6,
    2 * np.pi * 0.002e6,
    2 * np.pi * 0.002e6,
]

Omega_probe = calculate_rabi_frequency(atom, s0, s1, e_field=1.0)
Omega_coupling = calculate_rabi_frequency(atom, s1, s2, e_field=0.8e5)
Omega_rf_off = calculate_rabi_frequency(atom, s2, s3, e_field=0.0)
Omega_rf_on = calculate_rabi_frequency(atom, s2, s3, e_field=70e-5)
Omega_intf_baseline = calculate_rabi_frequency(atom, s3, s4, e_field=1.0)

# Normalization Constant (Beer-Lambert)
dipole_au = get_atom(atom).getDipoleMatrixElement(
    *s0.to_arc_args(), *s1.to_arc_args(), q=round(s1.mj - s0.mj)
)
mu_p = au_to_coulomb_metre(abs(dipole_au))

atom_density = 1e17
cell_length = 0.075
lambda_probe = 780.24e-9
normalization_constant = (4 * np.pi * atom_density * cell_length * mu_p**2) / (
    hbar * epsilon_0 * lambda_probe * Omega_probe
)

# Detunings
delta_probe = 0.0
delta_rf = 0.0
delta_interference = -2 * np.pi * 31e9

# Baseline Normalization
delta_c_baseline = np.linspace(-0.12e6, 0.12e6, 1001) * 2 * np.pi

coh_baseline = solve_ladder_system(
    detunings=[delta_probe, delta_c_baseline, delta_rf, delta_interference],
    rabis=[Omega_probe, Omega_coupling, Omega_rf_off, Omega_intf_baseline],
    gammas=gammas,
    coherence_index=0,
)
if hasattr(coh_baseline, "block_until_ready"):
    coh_baseline = coh_baseline.block_until_ready()
max_signal_off = float(np.exp(normalization_constant * coh_baseline).max())

# 2D Mesh Setup
n_intf, n_det = 50, 200
E_intf_vals = np.linspace(0, 2, n_intf)
delta_c_vals = np.linspace(-0.05e6, 0.05e6, n_det) * 2 * np.pi

E_intf_mesh, delta_c_mesh = np.meshgrid(E_intf_vals, delta_c_vals, indexing="ij")
E_intf_flat, delta_c_flat = E_intf_mesh.flatten(), delta_c_mesh.flatten()

Omega_intf_flat = calculate_rabi_frequency(atom, s3, s4, E_intf_flat)

print(f"Executing {n_intf * n_det} points per sweep...")

# Dry-run to compile the JAX kernel
_ = solve_ladder_system(
    detunings=[delta_probe, delta_c_flat, delta_rf, delta_interference],
    rabis=[Omega_probe, Omega_coupling, Omega_rf_off, Omega_intf_flat],
    gammas=gammas,
    coherence_index=0,
)
if hasattr(_, "block_until_ready"):
    _.block_until_ready()

# Case A - No RF
t0 = time.perf_counter()
coh_no_rf = solve_ladder_system(
    detunings=[delta_probe, delta_c_flat, delta_rf, delta_interference],
    rabis=[Omega_probe, Omega_coupling, Omega_rf_off, Omega_intf_flat],
    gammas=gammas,
    coherence_index=0,
)
if hasattr(coh_no_rf, "block_until_ready"):
    coh_no_rf = coh_no_rf.block_until_ready()
time_no_rf = time.perf_counter() - t0

# Case B - With RF
t1 = time.perf_counter()
coh_rf = solve_ladder_system(
    detunings=[delta_probe, delta_c_flat, delta_rf, delta_interference],
    rabis=[Omega_probe, Omega_coupling, Omega_rf_on, Omega_intf_flat],
    gammas=gammas,
    coherence_index=0,
)
if hasattr(coh_rf, "block_until_ready"):
    coh_rf = coh_rf.block_until_ready()
time_rf = time.perf_counter() - t1

print(
    f"|-- Case A (No RF): {time_no_rf * 1000:.1f} ms | Case B (With RF): {time_rf * 1000:.1f} ms"
)

# Plotting
heatmap_no_rf = (np.exp(normalization_constant * coh_no_rf) / max_signal_off).reshape(
    n_intf, n_det
)
heatmap_rf = (np.exp(normalization_constant * coh_rf) / max_signal_off).reshape(
    n_intf, n_det
)

delta_c_mhz = delta_c_vals / (2 * np.pi * 1e6)
extent = [delta_c_mhz.min(), delta_c_mhz.max(), E_intf_vals.min(), E_intf_vals.max()]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, layout="constrained")

fig.suptitle(
    r"$^{85}\mathrm{Rb}$ 5-Level Rydberg Electrometry: Interference Heatmap",
    fontsize=15,
    fontweight="bold",
)
fig.supxlabel(r"Coupling Detuning $\Delta_c$ (MHz)", fontsize=13)

# Panel A
im0 = axes[0].imshow(
    heatmap_no_rf, aspect="auto", extent=extent, origin="lower", cmap="viridis"
)
axes[0].set_title(r"(a) Absence of RF Field ($E_{\mathrm{LO}} = 0$)", fontsize=13)
axes[0].set_ylabel(r"Interference Field $E_{\mathrm{intf}}$ (V/m)", fontsize=12)
cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
cbar0.set_label(r"Normalized Probe Transmission $T$ (a.u.)", fontsize=11)

# Panel B
im1 = axes[1].imshow(
    heatmap_rf, aspect="auto", extent=extent, origin="lower", cmap="viridis"
)
axes[1].set_title(
    r"(b) Presence of RF Field ($E_{\mathrm{LO}} = 700\,\mu\mathrm{V/m}$)", fontsize=13
)
cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
cbar1.set_label(r"Normalized Probe Transmission $T$ (a.u.)", fontsize=11)

for ax in axes:
    ax.tick_params(axis="both", which="major", labelsize=10)

plt.show()
