"""
05_benchmark_scalability.py

Compares Rydiqule (NumPy) vs PQLS (JAX) on the CPU,
and then showcases PQLS executing on the GPU.
Generates the log-log scaling plot and the LaTeX table for the paper.
"""

import os
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import rydiqule as rq

from pqls.atoms import calculate_rabi_frequency, get_atom
from pqls.lindblad import solve_ladder_system
from pqls.types import QuantumState
from pqls.units import hz_to_rad_s

# Hardware check
devices = jax.devices()
has_gpu = devices[0].platform != "cpu"
cpu_device = jax.devices("cpu")[0]

print("=" * 70)
print(f"HARDWARE DETECTED: {devices[0].device_kind.upper()} ({len(devices)} devices)")
print("=" * 70)
if not has_gpu:
    print("WARNING: JAX is running on CPU. To utilize GPU hardware, you must")
    print("install the hardware-accelerated version of JAX.")
    print("For modern NVIDIA GPUs, this is typically:")
    print('    pip install -U "jax[cuda12]"')
    print(
        "See the official guide: https://jax.readthedocs.io/en/latest/installation.html"
    )
    print("=" * 70)

# Ladder: 5S_1/2 -> 5P_3/2 -> 50D_5/2 <-> 51P_3/2
s0 = QuantumState(5, 0, 0.5, 0.5)
s1 = QuantumState(5, 1, 1.5, 1.5)
s2 = QuantumState(50, 2, 2.5, 2.5)
s3 = QuantumState(51, 1, 1.5, 1.5)
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
Orf = calculate_rabi_frequency(atom, s2, s3, e_field=4.0)
rabis_list = [Op, Oc, Orf]

# Run all 6 points for a smooth graph
batch_sizes = [1_000, 5_000, 25_000, 100_000, 500_000, 1_000_000]
n_trials = 10

ryd_times, pqls_cpu_times, pqls_gpu_times = [], [], []

print(f"\nRunning Scalability Benchmarks (Averaging {n_trials} trials)...")
print("Note: The 1,000,000 batch will take a few minutes to complete on the CPU.\n")

print(
    f"{'Batch Size':>12} | {'Rydiqule (CPU)':>16} | {'PQLS (CPU)':>16} | {'PQLS (GPU)':>16}"
)
print("-" * 72)

for N in batch_sizes:
    detunings_hz = np.linspace(-80e6, 80e6, N)
    det_list = [0.0, hz_to_rad_s(detunings_hz), 0.0]

    # PQLS on CPU
    with jax.default_device(cpu_device):
        _ = solve_ladder_system(det_list, rabis_list, gammas)
        if hasattr(_, "block_until_ready"):
            _.block_until_ready()

        t_cpu_trials = np.zeros(n_trials)
        for i in range(n_trials):
            t0 = time.perf_counter()
            res = solve_ladder_system(det_list, rabis_list, gammas)
            if hasattr(res, "block_until_ready"):
                res.block_until_ready()
            t_cpu_trials[i] = time.perf_counter() - t0
        t_pqls_cpu = np.mean(t_cpu_trials)
        pqls_cpu_times.append(t_pqls_cpu)

    # PQLS on GPU
    t_pqls_gpu = np.nan
    if has_gpu:
        _ = solve_ladder_system(det_list, rabis_list, gammas)
        if hasattr(_, "block_until_ready"):
            _.block_until_ready()

        t_gpu_trials = np.zeros(n_trials)
        for i in range(n_trials):
            t0 = time.perf_counter()
            res = solve_ladder_system(det_list, rabis_list, gammas)
            if hasattr(res, "block_until_ready"):
                res.block_until_ready()
            t_gpu_trials[i] = time.perf_counter() - t0
        t_pqls_gpu = np.mean(t_gpu_trials)
        pqls_gpu_times.append(t_pqls_gpu)

    # Rydiqule on CPU
    sensor = rq.Sensor(4)
    sensor.add_decoherence((1, 0), gammas[1])
    sensor.add_decoherence((2, 1), gammas[2])
    sensor.add_decoherence((3, 2), gammas[3])
    sensor.add_coupling(states=(0, 1), rabi_frequency=Op, detuning=0.0)
    sensor.add_coupling(states=(1, 2), rabi_frequency=Oc, detuning=det_list[1])
    sensor.add_coupling(states=(2, 3), rabi_frequency=Orf, detuning=0.0)
    _ = rq.solve_steady_state(sensor)

    t_ryd_trials = np.zeros(n_trials)
    for i in range(n_trials):
        t0 = time.perf_counter()
        rq.solve_steady_state(sensor)
        t_ryd_trials[i] = time.perf_counter() - t0
    t_ryd = np.mean(t_ryd_trials)
    ryd_times.append(t_ryd)

    gpu_str = f"{t_pqls_gpu:14.4f} s" if has_gpu else "N/A"
    print(f"{N:12,d} | {t_ryd:14.4f} s | {t_pqls_cpu:14.4f} s | {gpu_str}")

# Throughput printout
ryd_peak_throughput = batch_sizes[-1] / ryd_times[-1]
pqls_cpu_peak_throughput = batch_sizes[-1] / pqls_cpu_times[-1]
print("\n" + "=" * 70)
print(f"Throughputs for the {batch_sizes[-1]:,} batch:")
print(f"Rydiqule (CPU) Peak Throughput: {ryd_peak_throughput:,.0f} systems / second")
print(
    f"PQLS (CPU) Peak Throughput    : {pqls_cpu_peak_throughput:,.0f} systems / second"
)
if has_gpu:
    pqls_gpu_peak_throughput = batch_sizes[-1] / pqls_gpu_times[-1]
    print(
        f"PQLS (GPU) Peak Throughput    : {pqls_gpu_peak_throughput:,.0f} systems / second"
    )
print("=" * 70)


# LaTeX table
print("\n" + "=" * 60)
print("LATEX CODE FOR TABLE")
print("=" * 60)
table_target_sizes = [1_000, 25_000, 100_000, 1_000_000]

for i, N in enumerate(batch_sizes):
    if N in table_target_sizes:
        t_r = ryd_times[i]
        t_cpu = pqls_cpu_times[i]
        if has_gpu:
            t_gpu = pqls_gpu_times[i]
            speedup = int(t_r / t_gpu)
            print(
                f"{N:,} & {t_r:.3f} & {t_cpu:.3f} & {t_gpu:.3f} & ${speedup:,}\\times$ \\\\"
            )
        else:
            speedup = int(t_r / t_cpu)
            print(f"{N:,} & {t_r:.3f} & {t_cpu:.3f} & ${speedup:,}\\times$ \\\\")
print("=" * 60 + "\n")


# Plotting
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

ax.plot(
    batch_sizes,
    ryd_times,
    marker="o",
    color="#1f77b4",
    lw=3,
    markersize=8,
    label="Rydiqule (CPU)",
)
ax.plot(
    batch_sizes,
    pqls_cpu_times,
    marker="^",
    color="#ff7f0e",
    lw=3,
    markersize=8,
    label="PQLS (CPU)",
)
if has_gpu:
    ax.plot(
        batch_sizes,
        pqls_gpu_times,
        marker="s",
        color="#d62728",
        lw=3,
        markersize=8,
        label=f"PQLS ({devices[0].device_kind.upper()})",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Simulated Systems (Batch Size)", fontsize=12)
ax.set_ylabel("Average Wall-Clock Execution Time (Seconds)", fontsize=12)
ax.set_title(
    "Lindblad Solver Scalability: PQLS vs. Rydiqule",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax.grid(True, which="both", linestyle=":", alpha=0.7)
ax.legend(fontsize=12, loc="upper left")

# Text box highlighting Throughputs
props = dict(
    boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9
)
throughput_text = (
    f"Peak Throughput (Systems/sec)\n"
    f"-----------------------------\n"
    f"Rydiqule (CPU): {ryd_peak_throughput / 1e3:>6.1f} K\n"
    f"PQLS (CPU)    : {pqls_cpu_peak_throughput / 1e3:>6.1f} K\n"
)
if has_gpu:
    throughput_text += f"PQLS (GPU)    : {pqls_gpu_peak_throughput / 1e6:>6.1f} M"

ax.text(
    0.95,
    0.05,
    throughput_text,
    transform=ax.transAxes,
    fontsize=11,
    fontfamily="monospace",
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=props,
)

os.makedirs("assets", exist_ok=True)
save_path = "assets/05_benchmark_scalability.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()
