from collections.abc import Sequence

import numpy as np
import pytest
import qutip as qt

from pqls.atoms import get_atom
from pqls.lindblad import (
    solve_custom_network,
    solve_ladder_system,
    solve_quantum_ladder,
    solve_steady_state,
)
from pqls.types import Driver, QuantumLadder, QuantumState
from pqls.units import au_to_coulomb_metre, electric_field_to_rabi_freq, hz_to_rad_s


# Helper Generators
def generate_random_system(n_levels: int, num_c_ops: int, seed: int):
    """Generates a random Hermitian Hamiltonian and valid jump operators."""
    rng = np.random.default_rng(seed)
    
    # 1. Random Hermitian Hamiltonian: H = 0.5 * (A + A^dag)
    a = rng.normal(size=(n_levels, n_levels)) + 1j * rng.normal(size=(n_levels, n_levels))
    h = 0.5 * (a + a.conj().T)
    
    # Random Cascade Jump Operators
    c_ops = []
    for _ in range(num_c_ops):
        c = np.zeros((n_levels, n_levels), dtype=complex)
        i = rng.integers(1, n_levels)
        j = rng.integers(0, i)
        c[j, i] = rng.uniform(0.5, 5.0)  # Decay from level i -> level j
        c_ops.append(c)
        
    return h, np.stack(c_ops)

def build_ladder_hamiltonian(detunings: np.ndarray, rabis: np.ndarray) -> np.ndarray:
    """Constructs an N-level ladder Hamiltonian in the rotating wave frame."""
    diag = np.concatenate(([0.0], -np.cumsum(detunings)))
    
    h = np.diag(diag).astype(complex)
    h += np.diag(rabis / 2.0, k=1)
    h += np.diag(rabis / 2.0, k=-1)
    return h


def build_ladder_collapse_operators(gammas: Sequence[float] | np.ndarray) -> np.ndarray:
    """Constructs downward cascade jump operators for an N-level ladder."""
    n_levels = len(gammas)
    c_ops = []
    for i in range(1, n_levels):
        if gammas[i] > 0.0:
            c = np.zeros((n_levels, n_levels), dtype=complex)
            c[i - 1, i] = np.sqrt(gammas[i])  # Decay |i> -> |i-1>
            c_ops.append(c)
            
    if not c_ops:
        return np.empty((0, n_levels, n_levels), dtype=complex)
    return np.stack(c_ops)

def solve_qutip_steady_state(h: np.ndarray, c_ops: np.ndarray) -> np.ndarray:
    """Ground truth reference solver using QuTiP."""
    h_qt = qt.Qobj(h)
    c_ops_qt = [qt.Qobj(c) for c in c_ops]
    rho_ss = qt.steadystate(h_qt, c_ops_qt)
    return rho_ss.full()


# Test Suites
@pytest.mark.parametrize("n_levels", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [313, 42, 67, 2026])
def test_raw_solve_steady_state_vs_qutip(
    n_levels: int, seed: int, tol_parity: float, tol_invariant: float, tol_pos_def: float
):
    """Verifies solve_steady_state against QuTiP across arbitrary random systems."""
    h, c_ops = generate_random_system(n_levels=n_levels, num_c_ops=n_levels - 1, seed=seed)
    
    rho_qt = solve_qutip_steady_state(h, c_ops)
    rho_jax = solve_steady_state(h, c_ops)
    
    # Parity with QuTiP
    np.testing.assert_allclose(rho_jax, rho_qt, atol=tol_parity, rtol=tol_parity)
    
    # Physical Invariants: Unit Trace & Hermiticity
    assert np.isclose(np.trace(rho_jax), 1.0, atol=tol_invariant)
    np.testing.assert_allclose(rho_jax, rho_jax.conj().T, atol=tol_invariant)
    
    # Physical Invariant: Positive Semi-Definiteness (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(rho_jax)
    assert np.all(eigenvalues >= tol_pos_def)

    # Verify non-trivial off-diagonal coherences exist (not just classical populations)
    off_diagonals = rho_jax[~np.eye(n_levels, dtype=bool)]
    assert np.any(np.abs(off_diagonals) > 1e-6)

@pytest.mark.parametrize("n_levels", [2, 3, 4])
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("seed", [313, 42, 67, 2026])
def test_solve_steady_state_direct_batch_vs_qutip(
    n_levels: int,
    batch_size: int,
    seed: int,
    tol_parity: float,
    tol_invariant: float,
    tol_pos_def: float,
):
    """Verifies pre-batched 3D Hamiltonian tensors against iterated QuTiP ground truth."""
    rng = np.random.default_rng(seed)
    num_c_ops = n_levels - 1

    _, c_ops = generate_random_system(n_levels=n_levels, num_c_ops=num_c_ops, seed=seed)

    # Generate a batch of random Hermitian Hamiltonians
    a = rng.normal(size=(batch_size, n_levels, n_levels)) + 1j * rng.normal(size=(batch_size, n_levels, n_levels))
    h_batch = 0.5 * (a + np.swapaxes(a.conj(), -1, -2))  # Shape (batch_size, n_levels, n_levels)

    rho_pqls = solve_steady_state(h_batch, c_ops)
    assert rho_pqls.shape == (batch_size, n_levels, n_levels)

    rho_qt = np.stack([solve_qutip_steady_state(h_batch[k], c_ops) for k in range(batch_size)], axis=0)
    np.testing.assert_allclose(rho_pqls, rho_qt, atol=tol_parity, rtol=tol_parity)

    # Enforce physical invariants across every batch slice
    for k in range(batch_size):
        assert np.isclose(np.trace(rho_pqls[k]), 1.0, atol=tol_invariant)
        np.testing.assert_allclose(rho_pqls[k], rho_pqls[k].conj().T, atol=tol_invariant)
        eigenvalues = np.linalg.eigvalsh(rho_pqls[k])
        assert np.all(eigenvalues >= tol_pos_def)

@pytest.mark.parametrize("n_levels", [2, 3, 4, 5])
@pytest.mark.parametrize("batch_size", [1, 2, 64, 128])
@pytest.mark.parametrize("seed", [313])
def test_ladder_system_vs_qutip(n_levels: int, batch_size: int, seed: int, tol_parity: float):
    """Verifies batched solve_ladder_system against iterated QuTiP ground truth."""
    rng = np.random.default_rng(seed)
    num_transitions = n_levels - 1

    detunings = [rng.uniform(-20e6, 20e6, size=batch_size) if batch_size > 1 else float(rng.uniform(-20e6, 20e6)) 
                 for _ in range(num_transitions)]
    rabis = [rng.uniform(1e5, 5e6, size=batch_size) if batch_size > 1 else float(rng.uniform(1e5, 5e6)) 
             for _ in range(num_transitions)]
    gammas = [0.0] + [float(rng.uniform(1e4, 1e6)) for _ in range(num_transitions)]

    # Solve via PQLS (JAX Batch)
    pqls_coherences = solve_ladder_system(
        detunings=detunings,
        rabis=rabis,
        gammas=gammas,
        coherence_index=0,
    )

    # Reference Ground Truth via Helpers + QuTiP
    c_ops = build_ladder_collapse_operators(gammas)
    all_bcast = np.broadcast_arrays(*detunings, *rabis)
    n_points = all_bcast[0].size
    dets_mat = np.stack(all_bcast[:num_transitions], axis=-1).reshape(n_points, num_transitions)
    rabs_mat = np.stack(all_bcast[num_transitions:], axis=-1).reshape(n_points, num_transitions)

    qt_coherences = [
        np.imag(solve_qutip_steady_state(build_ladder_hamiltonian(dets_mat[k], rabs_mat[k]), c_ops)[1, 0])
        for k in range(n_points)
    ]

    qt_res = np.array(qt_coherences[0]) if batch_size == 1 else np.array(qt_coherences)
    np.testing.assert_allclose(pqls_coherences, qt_res, atol=tol_parity, rtol=tol_parity)

def test_coherence_index_out_of_bounds():
    """Ensures improper index requests raise standard Python exceptions."""
    detunings = [0.0, 0.0]
    rabis = [1.0, 1.0]
    gammas = [0.0, 1.0, 1.0]

    with pytest.raises(IndexError):
        solve_ladder_system(
            detunings=detunings,
            rabis=rabis,
            gammas=gammas,
            coherence_index=5,
        )

def test_solve_custom_network_vs_qutip(tol_parity: float):
    """Verifies a 3-level system with branching decays against QuTiP."""
    transitions = [(0, 2), (1, 2)]
    decays = [(2, 0, 4.0e6), (2, 1, 2.0e6)]
    detunings = [1.5e6, -0.5e6, 0.0]  # 3 levels: |0>, |1>, and |2>
    rabis = [3.0e6, 1.0e6]
    n_levels = 3

    rho_pqls = solve_custom_network(
        transitions=transitions,
        decays=decays,
        detunings=detunings,
        rabis=rabis,
        n_levels=n_levels,
        coherence_index=None,
    )

    h_qt = np.zeros((3, 3), dtype=complex)
    h_qt[0, 0] = detunings[0]
    h_qt[1, 1] = detunings[1]
    h_qt[2, 2] = detunings[2]
    h_qt[0, 2] = h_qt[2, 0] = rabis[0] / 2.0
    h_qt[1, 2] = h_qt[2, 1] = rabis[1] / 2.0

    c0 = np.zeros((3, 3), dtype=complex)
    c0[0, 2] = np.sqrt(4.0e6)
    c1 = np.zeros((3, 3), dtype=complex)
    c1[1, 2] = np.sqrt(2.0e6)

    rho_qt = solve_qutip_steady_state(h_qt, [c0, c1])
    np.testing.assert_allclose(rho_pqls, rho_qt, atol=tol_parity, rtol=tol_parity)

    # Verify coherence extraction for both transitions
    coh_pqls_0 = solve_custom_network(
        transitions=transitions,
        decays=decays,
        detunings=detunings,
        rabis=rabis,
        n_levels=n_levels,
        coherence_index=0,
    )
    assert np.isclose(coh_pqls_0, np.imag(rho_qt[2, 0]), atol=tol_parity)

    coh_pqls_1 = solve_custom_network(
        transitions=transitions,
        decays=decays,
        detunings=detunings,
        rabis=rabis,
        n_levels=n_levels,
        coherence_index=1,
    )
    assert np.isclose(coh_pqls_1, np.imag(rho_qt[2, 1]), atol=tol_parity)

def test_solve_quantum_ladder_validation_and_e2e(tol_parity: float):
    """Verifies 4-level Rydberg RF ladder against QuTiP and ARC physics."""

    # 4-Level Ladder: |5S_1/2> -> |5P_3/2> -> |50D_5/2> -> |51P_3/2>
    s0 = QuantumState(n=5, l=0, j=0.5, mj=0.5)
    s1 = QuantumState(n=5, l=1, j=1.5, mj=1.5)
    s2 = QuantumState(n=50, l=2, j=2.5, mj=2.5)
    s3 = QuantumState(n=51, l=1, j=1.5, mj=1.5)
    ladder = QuantumLadder(atom="Rb85", states=[s0, s1, s2, s3])

    d_probe = Driver(amplitude=5.0, frequency=384.23e12, account_for_detuning=False)
    d_couple = Driver(amplitude=15.0, frequency=625.0e12, account_for_detuning=False)
    d_rf = Driver(amplitude=1.0, frequency=15.0e9, account_for_detuning=False)

    with pytest.raises(ValueError, match=r"Expected 3 drivers"):
        solve_quantum_ladder(ladder, [d_probe, d_couple])

    det_sweep = np.linspace(-50e6, 50e6, 10)
    pqls_coherences = solve_quantum_ladder(
        ladder=ladder,
        drivers=[d_probe, d_couple, d_rf],
        detunings=[det_sweep, 0.0, 0.0],
        coherence_index=0,
    )

    atom = get_atom("Rb85")
    gammas = [
        0.0,
        1.0 / atom.getStateLifetime(5, 1, 1.5, temperature=300.0, includeLevelsUpTo=80),
        1.0 / atom.getStateLifetime(50, 2, 2.5, temperature=300.0, includeLevelsUpTo=80),
        1.0 / atom.getStateLifetime(51, 1, 1.5, temperature=300.0, includeLevelsUpTo=80),
    ]

    dipoles = [
        au_to_coulomb_metre(abs(atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 5, 1, 1.5, 1.5, 1))),
        au_to_coulomb_metre(abs(atom.getDipoleMatrixElement(5, 1, 1.5, 1.5, 50, 2, 2.5, 2.5, 1))),
        au_to_coulomb_metre(abs(atom.getDipoleMatrixElement(50, 2, 2.5, 2.5, 51, 1, 1.5, 1.5, -1))),
    ]
    rabis = np.array([
        electric_field_to_rabi_freq(5.0, dipoles[0]),
        electric_field_to_rabi_freq(15.0, dipoles[1]),
        electric_field_to_rabi_freq(1.0, dipoles[2]),
    ])

    c_ops = build_ladder_collapse_operators(gammas)
    qt_coherences = []
    for det in det_sweep:
        dets_rad_s = np.array([hz_to_rad_s(det), 0.0, 0.0])
        h_qt = build_ladder_hamiltonian(dets_rad_s, rabis)
        rho_ss = solve_qutip_steady_state(h_qt, c_ops)
        qt_coherences.append(np.imag(rho_ss[1, 0]))

    np.testing.assert_allclose(pqls_coherences, qt_coherences, atol=tol_parity, rtol=tol_parity)


def test_solve_custom_network_batched_vs_qutip(
    tol_parity: float, tol_invariant: float, tol_pos_def: float
):
    """Verifies batched parameter sweeps in custom networks against QuTiP and physical invariants."""
    transitions = [(0, 2), (1, 2)]
    decays = [(2, 0, 4.0e6), (2, 1, 2.0e6)]
    n_levels = 3
    
    n_points = 25
    det_sweep = np.linspace(-5e6, 5e6, n_points)
    detunings = [det_sweep, -0.5e6, 0.0]
    rabis = [3.0e6, 1.0e6]

    rho_batch = solve_custom_network(
        transitions=transitions,
        decays=decays,
        detunings=detunings,
        rabis=rabis,
        n_levels=n_levels,
        coherence_index=None,
    )
    assert rho_batch.shape == (n_points, n_levels, n_levels)

    coh_0_pqls = solve_custom_network(
        transitions=transitions,
        decays=decays,
        detunings=detunings,
        rabis=rabis,
        n_levels=n_levels,
        coherence_index=0,
    )
    coh_1_pqls = solve_custom_network(
        transitions=transitions,
        decays=decays,
        detunings=detunings,
        rabis=rabis,
        n_levels=n_levels,
        coherence_index=1,
    )

    # 3. Reference ground truth via QuTiP
    c0 = np.zeros((3, 3), dtype=complex)
    c0[0, 2] = np.sqrt(4.0e6)
    c1 = np.zeros((3, 3), dtype=complex)
    c1[1, 2] = np.sqrt(2.0e6)
    c_ops_qt = [c0, c1]

    qt_rho_list = []
    for k in range(n_points):
        h_qt = np.zeros((3, 3), dtype=complex)
        h_qt[0, 0] = det_sweep[k]
        h_qt[1, 1] = -0.5e6
        h_qt[2, 2] = 0.0
        h_qt[0, 2] = h_qt[2, 0] = rabis[0] / 2.0
        h_qt[1, 2] = h_qt[2, 1] = rabis[1] / 2.0
        
        rho_ss_qt = solve_qutip_steady_state(h_qt, c_ops_qt)
        qt_rho_list.append(rho_ss_qt)

    rho_qt_batch = np.stack(qt_rho_list, axis=0)

    np.testing.assert_allclose(rho_batch, rho_qt_batch, atol=tol_parity, rtol=tol_parity)
    np.testing.assert_allclose(coh_0_pqls, np.imag(rho_qt_batch[:, 2, 0]), atol=tol_parity, rtol=tol_parity)
    np.testing.assert_allclose(coh_1_pqls, np.imag(rho_qt_batch[:, 2, 1]), atol=tol_parity, rtol=tol_parity)

    # Verify physical invariants across the entire sweep
    for k in range(n_points):
        assert np.isclose(np.trace(rho_batch[k]), 1.0, atol=tol_invariant)
        np.testing.assert_allclose(rho_batch[k], rho_batch[k].conj().T, atol=tol_invariant)
        eigenvalues = np.linalg.eigvalsh(rho_batch[k])
        assert np.all(eigenvalues >= tol_pos_def)

def test_solve_quantum_ladder_parameter_sweeps_and_overrides_vs_qutip(tol_parity: float):
    """Verifies solve_quantum_ladder parameter overrides and sweeps directly against QuTiP."""
    # 2-Level System: Rb85 |5S_1/2> -> |5P_3/2>
    s0 = QuantumState(n=5, l=0, j=0.5, mj=0.5)
    s1 = QuantumState(n=5, l=1, j=1.5, mj=1.5)
    ladder = QuantumLadder(atom="Rb85", states=[s0, s1])

    driver_base = Driver(amplitude=1.0, frequency=0.0, account_for_detuning=False)

    atom = get_atom("Rb85")
    gamma_1 = 1.0 / atom.getStateLifetime(5, 1, 1.5, temperature=300.0, includeLevelsUpTo=80)
    dipole_au = atom.getDipoleMatrixElement(5, 0, 0.5, 0.5, 5, 1, 1.5, 1.5, 1)
    dipole_c_m = au_to_coulomb_metre(abs(dipole_au))
    rabi_base = electric_field_to_rabi_freq(1.0, dipole_c_m)
    c_ops = build_ladder_collapse_operators([0.0, gamma_1])

    coh_default = solve_quantum_ladder(ladder, [driver_base])
    h_qt_res = build_ladder_hamiltonian(np.array([0.0]), np.array([rabi_base]))
    rho_qt_res = solve_qutip_steady_state(h_qt_res, c_ops)
    np.testing.assert_allclose(coh_default, np.imag(rho_qt_res[1, 0]), atol=tol_parity, rtol=tol_parity)

    n_points_det = 15
    det_sweep = np.linspace(-20e6, 20e6, n_points_det)
    coh_det = solve_quantum_ladder(
        ladder=ladder,
        drivers=[driver_base],
        detunings=[det_sweep],
    )

    qt_det_coherences = []
    for det in det_sweep:
        h_qt = build_ladder_hamiltonian(np.array([hz_to_rad_s(det)]), np.array([rabi_base]))
        rho_ss = solve_qutip_steady_state(h_qt, c_ops)
        qt_det_coherences.append(np.imag(rho_ss[1, 0]))

    assert coh_det.shape == (n_points_det,)
    np.testing.assert_allclose(coh_det, qt_det_coherences, atol=tol_parity, rtol=tol_parity)

    n_points_amp = 10
    amp_sweep = np.linspace(0.5, 10.0, n_points_amp)
    coh_amp = solve_quantum_ladder(
        ladder=ladder,
        drivers=[driver_base],
        detunings=[0.0],
        amplitudes=[amp_sweep],
    )

    qt_amp_coherences = []
    for amp in amp_sweep:
        rabi_val = electric_field_to_rabi_freq(amp, dipole_c_m)
        h_qt = build_ladder_hamiltonian(np.array([0.0]), np.array([rabi_val]))
        rho_ss = solve_qutip_steady_state(h_qt, c_ops)
        qt_amp_coherences.append(np.imag(rho_ss[1, 0]))

    assert coh_amp.shape == (n_points_amp,)
    np.testing.assert_allclose(coh_amp, qt_amp_coherences, atol=tol_parity, rtol=tol_parity)