"""
03_benchmark_ats.py

Autler-Townes splitting benchmark across multiple RF field strengths.
Compares the batched PQLS Lindblad solver against a serial QuTiP baseline for speed and precision.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from _qutip_reference import solve_steady_state_qt

from pqls.atoms import calculate_rabi_frequency, get_atom
from pqls.lindblad import solve_ladder_system
from pqls.types import QuantumState
from pqls.units import hz_to_rad_s

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

rf_conditions = [
    {"E_rf": 0.0, "title": "No RF Field\n(Single Peak)"},
    {"E_rf": 2.0, "title": "Moderate RF (2.0 V/m)\n(Split Peak)"},
    {"E_rf": 4.0, "title": "Strong RF (4.0 V/m)\n(Wide Split)"},
]

sweep_points = 2000
detunings_hz = np.linspace(-80e6, 80e6, sweep_points)
detunings_list = [0.0, hz_to_rad_s(detunings_hz), 0.0]

# Pre-compile the exact batch kernel to exclude JIT overhead
_ = solve_ladder_system(
    detunings=[0.0, np.zeros(sweep_points), 0.0],
    rabis=[Op, Oc, 0.0],
    gammas=gammas,
    coherence_index=0,
)

fig, axes = plt.subplots(
    1, 3, figsize=(15, 5), sharex=True, sharey=True, layout="constrained"
)

print("Running ATS Multi-Field Benchmarks (QuTiP vs. PQLS)...")

for ax, cond in zip(axes, rf_conditions, strict=True):
    E_rf = cond["E_rf"]
    Orf = calculate_rabi_frequency(atom, s2, s3, e_field=E_rf)

    rabis_list = [Op, Oc, Orf]

    # Time PQLS
    t0_jax = time.perf_counter()
    coh_jax = solve_ladder_system(detunings_list, rabis_list, gammas, coherence_index=0)
    jax_time = time.perf_counter() - t0_jax

    # Time QuTiP
    t0_qt = time.perf_counter()
    coh_qt = solve_steady_state_qt(*detunings_list, *rabis_list, gammas)
    qt_time = time.perf_counter() - t0_qt

    # Verification Metrics
    speedup = qt_time / jax_time if jax_time > 0 else 1.0
    max_diff = float(np.max(np.abs(coh_qt - coh_jax)))

    # Convert density matrix coherence into transmission
    trans_qt = np.exp(normalization_constant * coh_qt)
    trans_jax = np.exp(normalization_constant * coh_jax)

    print(
        f"|-- E_rf = {E_rf:3.1f} V/m | QuTiP: {qt_time:.3f}s | PQLS: {jax_time * 1000:.1f}ms | Speedup: {speedup:5.0f}x | Max Error: {max_diff:.2e}"
    )

    ax.plot(detunings_hz / 1e6, trans_qt, color="#1f77b4", lw=4.5, label="QuTiP")
    ax.plot(
        detunings_hz / 1e6,
        trans_jax,
        color="#d62728",
        lw=2.0,
        label="PQLS",
        linestyle="--",
    )

    ax.set_title(cond["title"], fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Metric Annotation Card
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
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=props,
    )

fig.supxlabel(r"Coupling Detuning $\Delta_c$ (MHz)", fontsize=11)
fig.supylabel(r"Probe Transmission $T$ (a.u.)", fontsize=11)
axes[0].legend(loc="lower right", fontsize=9)

fig.suptitle(
    r"$^{87}\mathrm{Rb}$ 4-Level Rydberg Electrometry: Autler-Townes Splitting Progression",
    fontsize=13,
    fontweight="bold",
)
plt.show()
