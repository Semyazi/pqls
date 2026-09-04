"""
02_custom_network.py

Arbitrary transition network simulation demonstrating Coherent Population Trapping (CPT)
in a 3-level Lambda-system with custom branching decay pathways.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pqls import get_atom, hz_to_rad_s, solve_custom_network

atom = get_atom("Rb87")
gamma_5p12 = 1.0 / atom.getStateLifetime(n=5, l=1, j=0.5)

# Topology: Rb87 D1 Lambda-System (Coherent Population Trapping)
# Transitions: |0> (F=1) <-> |2> (5P_1/2) [Probe], |1> (F=2) <-> |2> (5P_1/2) [Coupling]
transitions = [
    (0, 2),  # Transition 0: Probe
    (1, 2),  # Transition 1: Coupling
]

# Spontaneous decay branches into both hyperfine ground states with equal probability
decays = [
    (2, 0, 0.5 * gamma_5p12),  # Decay |2> -> |0> (F=1)
    (2, 1, 0.5 * gamma_5p12),  # Decay |2> -> |1> (F=2)
]

# Sweep the two-photon detuning across resonance (+/- 4 MHz)
det_sweep_hz = np.linspace(-4e6, 4e6, 500)
detunings = [0.0, hz_to_rad_s(det_sweep_hz), 0.0]
rabis = [hz_to_rad_s(0.2e6), hz_to_rad_s(2.0e6)]  # Probe: 200 kHz, Coupling: 2 MHz

# Batched steady-state solve (coherence_index=0 returns Im(rho_20))
coh_probe = solve_custom_network(
    transitions=transitions,
    decays=decays,
    detunings=detunings,
    rabis=rabis,
    n_levels=3,
    coherence_index=0,
)

# Plot the 3-level interference spectrum
plt.plot(det_sweep_hz / 1e6, -coh_probe, color="#d62728", lw=2)
plt.xlabel(r"Two-Photon Detuning $\delta$ (MHz)")
plt.ylabel(r"Probe Absorption $[-\mathrm{Im}(\rho_{20})]$")
plt.title(r"$^{87}\mathrm{Rb}$ Coherent Population Trapping ($\Lambda$-System)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs("assets", exist_ok=True)
save_path = "assets/02_custom_network.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()
