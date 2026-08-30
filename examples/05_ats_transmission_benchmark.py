"""
05_ats_transmission_benchmark.py

Benchmarking central probe transmission (resonant driving, all detunings = 0 Hz)
as a function of the RF Electric Field Amplitude (E_rf) across a logarithmic scale.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from _qutip_reference import solve_steady_state_qt

from pqls.atoms import calculate_rabi_frequency, get_atom
from pqls.lindblad import solve_ladder_system
from pqls.types import QuantumState

# Ladder: 5S_1/2 -> 5P_3/2 -> 50D_5/2 <-> 51P_3/2
s0 = QuantumState(n=5, l=0, j=0.5, mj=0.5)
s1 = QuantumState(n=5, l=1, j=1.5, mj=1.5)
s2 = QuantumState(n=50, l=2, j=2.5, mj=2.5)
s3 = QuantumState(n=51, l=1, j=1.5, mj=1.5)

atom = "Rb87"
gammas = [0.0] + [
    1.0
    / get_atom(atom).getStateLifetime(
        *s.level.to_arc_args(), temperature=300.0, includeLevelsUpTo=80
    )
    for s in [s1, s2, s3]
]

Op = calculate_rabi_frequency(atom, s0, s1, e_field=1.0)
Oc = calculate_rabi_frequency(atom, s1, s2, e_field=80000.0)
normalization_constant = 250.0  # Beer-Lambert law constant

sweep_points = 2000
E_rf_range = np.logspace(-4, 0, sweep_points)
Orf_values = calculate_rabi_frequency(atom, s2, s3, e_field=E_rf_range)

# Resonant Driving (0 Hz detunings)
detunings_list = [0.0, 0.0, 0.0]
rabis_list = [Op, Oc, Orf_values]

# Pre-compile the exact batch kernel to exclude JIT overhead
_ = solve_ladder_system(detunings_list, rabis_list, gammas, coherence_index=0)
if hasattr(_, "block_until_ready"):
    _.block_until_ready()

# Execution & Benchmarking
print(f"Running ATS Transmission Benchmark ({sweep_points} points)...")

# Time PQLS
t0_jax = time.perf_counter()
coh_jax = solve_ladder_system(detunings_list, rabis_list, gammas, coherence_index=0)

# Block async JAX dispatch to get true wall-clock time
if hasattr(coh_jax, "block_until_ready"):
    coh_jax = coh_jax.block_until_ready()

jax_time = time.perf_counter() - t0_jax

# Time QuTiP
t0_qt = time.perf_counter()
coh_qt = solve_steady_state_qt(*detunings_list, *rabis_list, gammas)
qt_time = time.perf_counter() - t0_qt

# Convert density matrix coherence into transmission
trans_qt = np.exp(normalization_constant * coh_qt)
trans_jax = np.exp(normalization_constant * coh_jax)

# Verification Metrics
speedup = qt_time / jax_time if jax_time > 0 else 1.0
max_diff = float(np.max(np.abs(trans_qt - trans_jax)))

print(
    f"|-- QuTiP: {qt_time:.4f}s | PQLS: {jax_time * 1000:.2f}ms | Speedup: {speedup:5.0f}x | Max Diff: {max_diff:.2e}"
)

# Plotting
fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")

# Overlay plots
ax.plot(E_rf_range, trans_qt, color="#1f77b4", lw=4.5, label="QuTiP")
ax.plot(E_rf_range, trans_jax, color="#d62728", lw=2.0, label="PQLS", linestyle="--")

ax.set_xscale("log")
ax.set_title(
    r"ATS Central Transmission vs. RF Amplitude"
    "\n"
    r"($^{87}\mathrm{Rb}$ | $\Delta_p = \Delta_c = \Delta_{\mathrm{rf}} = 0\ \mathrm{Hz}$)",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax.set_xlabel(r"RF Field Amplitude $E_{\mathrm{RF}}$ (V/m)", fontsize=11)
ax.set_ylabel(r"Central Probe Transmission $T$ (a.u.)", fontsize=11)
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(loc="lower left", fontsize=9.5)

# Benchmark Metric Card
metrics_text = (
    f"QuTiP  : {qt_time:.2f} s\n"
    f"PQLS   : {jax_time * 1000:.1f} ms\n"
    f"Speedup: {speedup:.0f}x\n"
    f"Error  : {max_diff:.1e}"
)
props = dict(
    boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9
)
ax.text(
    0.03,
    0.55,
    metrics_text,
    transform=ax.transAxes,
    fontsize=8.5,
    fontfamily="monospace",
    verticalalignment="center",
    bbox=props,
)

plt.show()
