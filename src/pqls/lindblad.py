"""
High-performance JAX-accelerated solvers for the Lindblad master equation steady-state.
Provides both raw matrix-level APIs and high-level ladder system functions.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .atoms import get_atom
from .types import ArrayOrScalar, Driver, QuantumLadder
from .units import au_to_coulomb_metre, electric_field_to_rabi_freq, hz_to_rad_s

jax.config.update("jax_enable_x64", True)


# --- Private JAX Kernels ---


def _build_hamiltonian(
    detunings: jnp.ndarray,
    rabis: jnp.ndarray,
    from_idx: jnp.ndarray,
    to_idx: jnp.ndarray,
    n_levels: int,
) -> jnp.ndarray:
    """Constructs the N x N Hamiltonian matrix."""
    H = jnp.zeros((n_levels, n_levels), dtype=jnp.complex128)

    H = H.at[jnp.diag_indices(n_levels)].set(detunings)

    H = H.at[(from_idx, to_idx)].set(rabis / 2.0)
    H = H.at[(to_idx, from_idx)].set(jnp.conj(rabis) / 2.0)

    return H


def _build_collapse_operators(
    decay_from: jnp.ndarray, decay_to: jnp.ndarray, gammas: jnp.ndarray, n_levels: int
) -> jnp.ndarray:
    """Constructs a stack of K sparse decay/jump operators."""
    K = gammas.shape[0]
    C_ops = jnp.zeros((K, n_levels, n_levels), dtype=jnp.complex128)

    # C_k = sqrt(gamma) * |to><from|
    sqrt_g = jnp.sqrt(jnp.maximum(gammas, 0.0))
    C_ops = C_ops.at[jnp.arange(K), decay_to, decay_from].set(sqrt_g)

    return C_ops


def _extract_output(
    rho_ss: jnp.ndarray,
    transitions: list[tuple[int, int]],
    coherence_index: int | None,
    is_scalar: bool,
) -> ArrayOrScalar:
    """Extracts either target coherence or full density matrix and formats dimensions."""
    if coherence_index is not None:
        try:
            from_s, to_s = transitions[coherence_index]
        except IndexError as err:
            raise IndexError(
                f"coherence_index {coherence_index} out of range for {len(transitions)} transitions."
            ) from err
        res = np.asarray(jnp.imag(rho_ss[:, to_s, from_s]))
        return float(res[0]) if is_scalar else res

    res = np.asarray(rho_ss)
    return res[0] if is_scalar else res


def _solve_steady_state_single(H: jnp.ndarray, C_ops: jnp.ndarray) -> jnp.ndarray:
    """Solves the steady-state Lindblad master equation for an unbatched system.

    Parameters
    ----------
    H : jnp.ndarray
        Hamiltonian matrix of shape (N, N).
    C_ops : jnp.ndarray
        Stack of collapse/jump operators of shape (K, N, N).

    Returns
    -------
    jnp.ndarray
        Steady-state density matrix rho_ss of shape (N, N) with unit trace.
    """

    N = H.shape[0]
    Ident = jnp.eye(N, dtype=jnp.complex128)

    L = -1j * (jnp.kron(H, Ident) - jnp.kron(Ident, H.T))

    def make_decay_superop(C):
        CTC = C.T.conj() @ C
        return (
            jnp.kron(C, C.conj())
            - 0.5 * jnp.kron(CTC, Ident)
            - 0.5 * jnp.kron(Ident, CTC.T)
        )

    if C_ops.shape[0] > 0:
        L_decay = jnp.sum(jax.vmap(make_decay_superop)(C_ops), axis=0)
        L = L + L_decay

    L = L.at[0, :].set(0.0)
    diag_indices = jnp.arange(0, N**2, N + 1)
    L = L.at[0, diag_indices].set(1.0)

    b = jnp.zeros(N**2, dtype=jnp.complex128).at[0].set(1.0)
    rho_vec = jnp.linalg.solve(L, b)

    return rho_vec.reshape((N, N))


def _single_point_network(
    d_arr, r_arr, from_idx, to_idx, C_ops, n_levels
) -> jnp.ndarray:
    """Maps parameters to a Hamiltonian and solves the steady state."""
    H = _build_hamiltonian(d_arr, r_arr, from_idx, to_idx, n_levels)
    return _solve_steady_state_single(H, C_ops)


def _single_point_ladder(det_vec, rab_vec, gammas_vec, n_levels) -> jnp.ndarray:
    """N-level ladder point solver composing shared JAX primitives."""
    cum_dets = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(det_vec)])
    diagonals = -1.0 * cum_dets

    from_idx = jnp.arange(n_levels - 1)
    to_idx = jnp.arange(1, n_levels)

    H = _build_hamiltonian(diagonals, rab_vec, from_idx, to_idx, n_levels)
    C_ops = _build_collapse_operators(to_idx, from_idx, gammas_vec[1:], n_levels)

    return _solve_steady_state_single(H, C_ops)


_compiled_solve_single = jax.jit(_solve_steady_state_single)
_compiled_solve_batch = jax.jit(jax.vmap(_solve_steady_state_single, in_axes=(0, None)))
_compiled_batched_network = jax.jit(
    jax.vmap(_single_point_network, in_axes=(0, 0, None, None, None, None)),
    static_argnums=(5,),  # n_levels must be a static integer for matrix allocation
)
_compiled_batched_ladder = jax.jit(
    jax.vmap(_single_point_ladder, in_axes=(0, 0, None, None)), static_argnums=(3,)
)

# --- Public APIs ---


def solve_steady_state(hamiltonian: np.ndarray, c_ops: np.ndarray) -> np.ndarray:
    """
    Solve for the steady state given the Hamiltonian & collapse operators.
    """
    if hamiltonian.ndim == 2:
        # Single unbatched matrix solve
        res = _compiled_solve_single(jnp.asarray(hamiltonian), jnp.asarray(c_ops))
    else:
        # Batched across axis 0
        res = _compiled_solve_batch(jnp.asarray(hamiltonian), jnp.asarray(c_ops))

    return np.asarray(res)


def solve_custom_network(
    transitions: list[tuple[int, int]],
    decays: list[tuple[int, int, float]],
    detunings: list[ArrayOrScalar],
    rabis: list[ArrayOrScalar],
    n_levels: int,
    coherence_index: int | None = None,
) -> ArrayOrScalar:
    """Solve steady state for an arbitrary transition network.

    Parameters
    ----------
    coherence_index : int, optional
        If specified (e.g. `0`, `1`), returns `Im(rho)` for `transitions[coherence_index]`.
        If `None` (default), returns the full density matrix batch of shape `(Batch, N, N)`.
    """

    num_dets = len(detunings)
    all_bcast = np.broadcast_arrays(*detunings, *rabis)
    is_scalar = all_bcast[0].ndim == 0

    if is_scalar:
        all_bcast = [np.expand_dims(x, 0) for x in all_bcast]

    dets_stacked = np.stack(all_bcast[:num_dets], axis=-1)
    rabs_stacked = np.stack(all_bcast[num_dets:], axis=-1)

    from_idx = jnp.array([t[0] for t in transitions])
    to_idx = jnp.array([t[1] for t in transitions])

    decay_from = jnp.array([d[0] for d in decays], dtype=jnp.int32)
    decay_to = jnp.array([d[1] for d in decays], dtype=jnp.int32)
    gamma_vals = jnp.array([d[2] for d in decays], dtype=jnp.float64)

    # Pre-build collapse operators (these do not change during parameter sweeps)
    C_ops = _build_collapse_operators(decay_from, decay_to, gamma_vals, n_levels)

    rho_ss = _compiled_batched_network(
        jnp.asarray(dets_stacked),
        jnp.asarray(rabs_stacked),
        from_idx,
        to_idx,
        C_ops,
        n_levels,
    )

    return _extract_output(rho_ss, transitions, coherence_index, is_scalar)


def solve_ladder_system(
    detunings: list[ArrayOrScalar],
    rabis: list[ArrayOrScalar],
    gammas: list[float],
    coherence_index: int | None = 0,
) -> ArrayOrScalar:
    """Solve steady state for an N-level ladder system at native XLA speed."""
    N = len(rabis) + 1
    num_dets = len(detunings)
    all_bcast = np.broadcast_arrays(*detunings, *rabis)
    is_scalar = all_bcast[0].ndim == 0

    if is_scalar:
        all_bcast = [np.expand_dims(x, 0) for x in all_bcast]

    dets_stacked = jnp.asarray(
        np.stack(all_bcast[:num_dets], axis=-1), dtype=jnp.float64
    )
    rabs_stacked = jnp.asarray(
        np.stack(all_bcast[num_dets:], axis=-1), dtype=jnp.complex128
    )
    gammas_vec = jnp.asarray(gammas, dtype=jnp.float64)

    rho_ss = _compiled_batched_ladder(dets_stacked, rabs_stacked, gammas_vec, N)
    transitions = [(i, i + 1) for i in range(N - 1)]

    return _extract_output(rho_ss, transitions, coherence_index, is_scalar)


def solve_quantum_ladder(
    ladder: QuantumLadder,
    drivers: Sequence[Driver],
    detunings: Sequence[ArrayOrScalar | None] | None = None,
    frequencies: Sequence[ArrayOrScalar | None] | None = None,
    amplitudes: Sequence[ArrayOrScalar | None] | None = None,
    temperature: float = 300.0,
    max_n: int = 80,
    coherence_index: int | None = 0,
) -> ArrayOrScalar:
    """Solves the steady-state for a physical atomic ladder system using ARC and JAX."""
    n_transitions = ladder.num_transitions
    n_levels = ladder.num_levels

    if len(drivers) != n_transitions:
        raise ValueError(
            f"Expected {n_transitions} drivers for a {n_levels}-level ladder, got {len(drivers)}."
        )

    atom = get_atom(ladder.atom)

    # Compute gammas (decay rates in rad/s) from ARC
    gammas = [0.0] * n_levels
    for i in range(1, n_levels):
        lifetime_s = atom.getStateLifetime(
            *ladder.states[i].level.to_arc_args(),
            temperature=temperature,
            includeLevelsUpTo=max_n,
        )
        gammas[i] = 1.0 / lifetime_s if lifetime_s > 0 else 0.0

    # Resolve Frequencies, Detunings, and Rabi Frequencies
    detunings_list: list[ArrayOrScalar] = []
    rabis_list: list[ArrayOrScalar] = []

    for i in range(n_transitions):
        lower = ladder.states[i]
        upper = ladder.states[i + 1]
        driver = drivers[i]

        if detunings is not None and i < len(detunings) and detunings[i] is not None:
            detuning_rad_s = hz_to_rad_s(detunings[i])

        elif (
            frequencies is not None
            and i < len(frequencies)
            and frequencies[i] is not None
        ):
            freq_val = frequencies[i]
            if driver.account_for_detuning:
                f_res_hz = atom.getTransitionFrequency(
                    *lower.level.to_arc_args(),
                    *upper.level.to_arc_args(),
                )
                detuning_rad_s = hz_to_rad_s(freq_val - f_res_hz)
            else:
                detuning_rad_s = (
                    np.zeros_like(freq_val) if not np.isscalar(freq_val) else 0.0
                )

        else:
            if driver.account_for_detuning:
                f_res_hz = atom.getTransitionFrequency(
                    *lower.level.to_arc_args(),
                    *upper.level.to_arc_args(),
                )
                detuning_rad_s = hz_to_rad_s(driver.frequency - f_res_hz)
            else:
                detuning_rad_s = 0.0

        detunings_list.append(detuning_rad_s)

        if amplitudes is not None and i < len(amplitudes) and amplitudes[i] is not None:
            amp_v_m = amplitudes[i]
        else:
            amp_v_m = driver.amplitude

        # Calculate transition dipole moment via ARC (a.u. -> C*m)
        dipole_au = atom.getDipoleMatrixElement(
            *lower.to_arc_args(),
            *upper.to_arc_args(),
            q=round(upper.mj - lower.mj),
        )
        dipole_c_m = au_to_coulomb_metre(abs(dipole_au))

        # Convert electric field to angular Rabi frequency (rad/s)
        rabi_rad_s = electric_field_to_rabi_freq(amp_v_m, dipole_c_m)
        rabis_list.append(rabi_rad_s)

    return solve_ladder_system(
        detunings=detunings_list,
        rabis=rabis_list,
        gammas=gammas,
        coherence_index=coherence_index,
    )
