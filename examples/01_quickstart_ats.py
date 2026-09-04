"""
01_quickstart_ats.py

Minimal 4-level Autler-Townes Splitting simulation using high-level ladder abstractions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pqls import Driver, QuantumLadder, QuantumState, solve_quantum_ladder

# Define 4-level atomic ladder: 5S_1/2 -> 5P_3/2 -> 50D_5/2 -> 51P_3/2
ladder = QuantumLadder(
    "Rb85",
    [
        QuantumState(5, 0, 0.5, 0.5),
        QuantumState(5, 1, 1.5, 1.5),
        QuantumState(50, 2, 2.5, 2.5),
        QuantumState(51, 1, 1.5, 1.5),
    ],
)

# Set field amplitudes (Probe, Coupling, RF in V/m) and detuning sweep
drivers = [Driver(1.0), Driver(8.0e4), Driver(2.0)]
delta_c = np.linspace(-60e6, 60e6, 400)

# Solve steady state across coupling detuning in a single batched call
# The detunings list is aligned with the driver order: [Probe, Coupling, RF]
coherences = solve_quantum_ladder(ladder, drivers, detunings=[0.0, delta_c, 0.0])

# Plot transmission
plt.plot(delta_c / 1e6, np.exp(250.0 * coherences))
plt.xlabel("Coupling Detuning (MHz)")
plt.ylabel("Probe Transmission (a.u.)")
plt.title(r"$^{85}\mathrm{Rb}$ Rydberg Electrometry: Autler-Townes Splitting")
plt.grid(True)

os.makedirs("assets", exist_ok=True)
save_path = "assets/01_quickstart_ats.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()
