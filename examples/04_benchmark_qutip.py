"""
04_benchmark_qutip.py

PQLS vs. QuTiP vs. QuTiP-JAX.
Averages over multiple trials for rigorous benchmarking.
Generates a 3-panel visual comparison plot and outputs the exact
LaTeX table formatting for the paper.
"""

import os
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from _qutip_reference import solve_steady_state_qt

from pqls.atoms import calculate_rabi_frequency, get_atom
from pqls.lindblad import solve_ladder_system
from pqls.types import QuantumState
from pqls.units import hz_to_rad_s

jax.config.update("jax_platforms", "cpu")

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
norm_c = 250.0

conditions = [
    {"E_rf": 0.0, "title": "No RF Field (Single Peak)"},
    {"E_rf": 2.0, "title": "Moderate RF (Split Peak)"},
    {"E_rf": 4.0, "title": "Strong RF (Wide Split)"},
]

sweep_points = 1000
det_hz = np.linspace(-80e6, 80e6, sweep_points)
det_list = [0.0, hz_to_rad_s(det_hz), 0.0]

n_trials = 10

# Pre-compile the exact batch kernel to exclude JIT overhead
_ = solve_ladder_system(det_list, [Op, Oc, 0.0], gammas)
if hasattr(_, "block_until_ready"):
    _.block_until_ready()

results_pqls, times_pqls = [], []
results_qt, times_qt = [], []
results_qtj, times_qtj = [], []

print(f"Running Ecosystem Benchmark (Averaging {n_trials} trials per condition)...")

# PQLS
print("\nRunning PQLS...")
for cond in conditions:
    Orf = calculate_rabi_frequency(atom, s2, s3, e_field=cond["E_rf"])

    trial_times = np.zeros(n_trials)
    for i in range(n_trials):
        t0 = time.perf_counter()
        coh = solve_ladder_system(det_list, [Op, Oc, Orf], gammas, coherence_index=0)
        if hasattr(coh, "block_until_ready"):
            coh.block_until_ready()
        trial_times[i] = time.perf_counter() - t0

    times_pqls.append(np.mean(trial_times))
    results_pqls.append(coh)
    print(f"  -> {cond['title']:<30} | {times_pqls[-1] * 1000:>6.2f} ms")

# QuTiP (Standard/Serial)
print("\nRunning QuTiP (Serial)... (This will take a minute)")
for cond in conditions:
    Orf = calculate_rabi_frequency(atom, s2, s3, e_field=cond["E_rf"])

    # Warmup for QuTiP
    _ = solve_steady_state_qt(*det_list, Op, Oc, Orf, gammas)

    trial_times = np.zeros(n_trials)
    for i in range(n_trials):
        t0 = time.perf_counter()
        coh = solve_steady_state_qt(*det_list, Op, Oc, Orf, gammas)
        trial_times[i] = time.perf_counter() - t0

    times_qt.append(np.mean(trial_times))
    results_qt.append(coh)
    print(f"  -> {cond['title']:<30} | {times_qt[-1]:>6.2f} s")

# QuTiP-JAX
print("\nRunning QuTiP-JAX... (This will also take a minute)")
import qutip_jax

qutip_jax.set_as_default()

for cond in conditions:
    Orf = calculate_rabi_frequency(atom, s2, s3, e_field=cond["E_rf"])

    # Warmup for QuTiP-JAX
    _ = solve_steady_state_qt(*det_list, Op, Oc, Orf, gammas)

    trial_times = np.zeros(n_trials)
    for i in range(n_trials):
        t0 = time.perf_counter()
        coh = solve_steady_state_qt(*det_list, Op, Oc, Orf, gammas)
        trial_times[i] = time.perf_counter() - t0

    times_qtj.append(np.mean(trial_times))
    results_qtj.append(coh)
    print(f"  -> {cond['title']:<30} | {times_qtj[-1]:>6.2f} s")


# Generate LaTeX table (averaging across the conditions)
print("\n" + "=" * 60)
print(f"LATEX CODE FOR THE TABLE ({n_trials} Trials Averaged)")
print("=" * 60)
t1, t2, t3 = map(np.mean, (times_qt, times_qtj, times_pqls))
print(f"QuTiP (Serial) & {t1:.3f} & {int(sweep_points / t1):,} & $1\\times$ \\\\")
print(
    f"QuTiP-JAX & {t2:.3f} & {int(sweep_points / t2):,} & ${t1 / t2:.2f}\\times$ \\\\"
)
print(
    f"\\textbf{{PQLS (Ours)}} & \\textbf{{{t3:.4f}}} & \\textbf{{{int(sweep_points / t3):,}}} & \\textbf{{$\\sim${int(t1 / t3):,}$\\times$}} \\\\"
)
print("=" * 60 + "\n")


# Plotting
fig, axes = plt.subplots(
    1, 3, figsize=(15, 5), sharex=True, sharey=True, layout="constrained"
)

for i, (ax, cond) in enumerate(zip(axes, conditions, strict=True)):
    max_err = float(np.max(np.abs(results_qt[i] - results_pqls[i])))

    ax.plot(
        det_hz / 1e6,
        np.exp(norm_c * results_qt[i]),
        color="#1f77b4",
        lw=5,
        label="QuTiP (Serial)",
    )
    ax.plot(
        det_hz / 1e6,
        np.exp(norm_c * results_qtj[i]),
        color="#ff7f0e",
        lw=3,
        label="QuTiP-JAX",
    )
    ax.plot(
        det_hz / 1e6,
        np.exp(norm_c * results_pqls[i]),
        color="#2ca02c",
        lw=2,
        linestyle="--",
        label="PQLS",
    )

    ax.set_title(cond["title"], fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    metrics = (
        f"QuTiP    : {times_qt[i]:.2f} s\n"
        f"QuTiP-JAX: {times_qtj[i]:.2f} s\n"
        f"PQLS     : {times_pqls[i] * 1000:.1f} ms\n"
        f"Max Error: {max_err:.1e}"
    )
    ax.text(
        0.05,
        0.95,
        metrics,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="top",
        bbox=dict(facecolor="#f5f5f5", alpha=0.9),
    )

axes[0].legend(loc="lower right")
fig.supxlabel(r"Coupling Detuning $\Delta_c$ (MHz)")
fig.supylabel(r"Probe Transmission $T$ (a.u.)")
fig.suptitle(
    r"$^{87}\mathrm{Rb}$ Rydberg Electrometry: PQLS vs QuTiP",
    fontsize=14,
    fontweight="bold",
)

os.makedirs("assets", exist_ok=True)
save_path = "assets/04_benchmark_qutip.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()
