"""
01b_quickstart_ats_low_level.py

Demonstrates the lowest-level API of PQLS (`solve_steady_state`).
This recreates the exact 4-level Autler-Townes Splitting simulation from Example A,
but requires the user to manually construct the batched Hamiltonian tensors
and collapse (jump) operators.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pqls import (
    QuantumState,
    calculate_rabi_frequency,
    get_atom,
    hz_to_rad_s,
    solve_steady_state,
)

# Define states and load atomic properties
atom_name = "Rb85"
s0 = QuantumState(n=5, l=0, j=0.5, mj=0.5)
s1 = QuantumState(n=5, l=1, j=1.5, mj=1.5)
s2 = QuantumState(n=50, l=2, j=2.5, mj=2.5)
s3 = QuantumState(n=51, l=1, j=1.5, mj=1.5)

arc_atom = get_atom(atom_name)

# Calculate Decay Rates (gammas) using ARC
# Gamma = 1 / lifetime (in rad/s)
gammas = [0.0] * 4
states = [s0, s1, s2, s3]
for i in range(1, 4):
    lifetime_s = arc_atom.getStateLifetime(
        *states[i].level.to_arc_args(), temperature=300.0, includeLevelsUpTo=80
    )
    gammas[i] = 1.0 / lifetime_s

# Calculate Rabi Frequencies from Electric Field Amplitudes (V/m)
Op = calculate_rabi_frequency(atom_name, s0, s1, e_field=1.0)
Oc = calculate_rabi_frequency(atom_name, s1, s2, e_field=8.0e4)
Orf = calculate_rabi_frequency(atom_name, s2, s3, e_field=2.0)

# Define the 1D parameter sweep (Coupling Detuning)
batch_size = 400
delta_c_hz = np.linspace(-60e6, 60e6, batch_size)
delta_c = hz_to_rad_s(delta_c_hz)

# Construct the Batched Hamiltonian (Shape: [Batch, N, N])
H = np.zeros((batch_size, 4, 4), dtype=complex)

H[:, 0, 0] = 0.0
H[:, 1, 1] = 0.0
H[:, 2, 2] = -delta_c
H[:, 3, 3] = -delta_c

H[:, 0, 1] = Op / 2.0
H[:, 1, 0] = Op / 2.0

H[:, 1, 2] = Oc / 2.0
H[:, 2, 1] = Oc / 2.0

H[:, 2, 3] = Orf / 2.0
H[:, 3, 2] = Orf / 2.0

# Construct Collapse Operators (Shape: [K, N, N])
# C_k = sqrt(gamma) * |lower><upper|
C_ops = np.zeros((3, 4, 4), dtype=complex)
for i in range(1, 4):
    C_ops[i - 1, i - 1, i] = np.sqrt(gammas[i])

# Solve Steady State using Low-Level API
# Passes the batched Hamiltonian and shared C_ops directly to the lowest-level API
rho_ss = solve_steady_state(hamiltonian=H, c_ops=C_ops)

# Extract the Coherence (Im[rho_{1, 0}])
# Since rho_ss has shape [Batch, 4, 4], coherence is at index [:, 1, 0]
coherences = np.imag(rho_ss[:, 1, 0])

# Plot transmission
plt.plot(delta_c_hz / 1e6, np.exp(250.0 * coherences))
plt.xlabel("Coupling Detuning (MHz)")
plt.ylabel("Probe Transmission (a.u.)")
plt.title(r"$^{85}\mathrm{Rb}$ Rydberg Electrometry: Autler-Townes Splitting")
plt.grid(True)

os.makedirs("assets", exist_ok=True)
save_path = "assets/01b_quickstart_ats_low_level"
plt.savefig(save_path + ".png", format="png", dpi=300, bbox_inches="tight")
plt.savefig(save_path + ".eps", format="eps", dpi=300, bbox_inches="tight")

plt.show()
