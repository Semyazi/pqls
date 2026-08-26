"""
Reference implementation using QuTiP for baseline comparison and verification.
"""

import numpy as np
import qutip as qt


def solve_steady_state_qt(
    delta_p, delta_c, delta_rf, Omega_p, Omega_c, Omega_rf, gamma_list
) -> np.ndarray:
    """Legacy QuTiP 4-level ladder steady-state solver iterating point-by-point."""
    dp, dc, drf, Op, Oc, Orf = np.broadcast_arrays(
        delta_p, delta_c, delta_rf, Omega_p, Omega_c, Omega_rf
    )
    n_points = dc.size

    c_ops = [
        np.sqrt(gamma) * qt.basis(4, i - 1) * qt.basis(4, i).dag()
        for i, gamma in enumerate(gamma_list)
        if i > 0 and gamma > 0
    ]

    results = np.zeros(n_points)
    for i in range(n_points):
        mat = np.zeros((4, 4), dtype=complex)
        mat[0, 1] = mat[1, 0] = Op[i]
        mat[1, 2] = mat[2, 1] = Oc[i]
        mat[2, 3] = mat[3, 2] = Orf[i]

        mat[1, 1] = -2.0 * dp[i]
        mat[2, 2] = -2.0 * (dp[i] + dc[i])
        mat[3, 3] = -2.0 * (dp[i] + dc[i] + drf[i])

        H = qt.Qobj(0.5 * mat)
        rho_ss = qt.steadystate(H, c_ops)
        results[i] = np.imag(rho_ss.full()[1, 0])

    return results
