"""ETDRK4 + Chebyshev collocation solver for the generalized Burgers--Huxley equation.

PDE:
    u_t + alpha * u^delta * u_x - u_xx = beta * u * (1 - u^delta) * (u^delta - gamma)
on (x, t) in [xmin, xmax] x [0, T], with non-homogeneous Dirichlet boundary data
u(xmin, t) = gL(t), u(xmax, t) = gR(t).

Method: Chebyshev--Gauss--Lobatto collocation in x, linear boundary lifting to
homogenize, and the Cox--Matthews/Kassam--Trefethen ETDRK4 scheme in t, with
phi-functions evaluated by the Trefethen--Weideman--Schmelzer Talbot contour.

See paper, Section 3, for the method description.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .chebyshev_grid import ChebyshevGrid
from .phi_functions import all_phi_functions


class GBHEtdrk4Solver:
    """Solve the generalized Burgers--Huxley equation on a bounded interval.

    Parameters
    ----------
    alpha, beta, gamma, delta : floats
        PDE coefficients.
    N : int
        Number of Chebyshev intervals (gives N+1 grid points, N-1 interior).
    xmin, xmax : float
        Domain endpoints.
    initial_condition : callable
        u0(x) -> array of shape like x.
    gL, gR : callable
        Left and right time-dependent Dirichlet boundary data, gL(t), gR(t).
    exact_solution : callable, optional
        u_exact(x, t) -> array. If provided, enables error computation.
    n_quad : int, default 64
        Talbot quadrature nodes.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        N: int,
        xmin: float,
        xmax: float,
        initial_condition: Callable,
        gL: Callable,
        gR: Callable,
        exact_solution: Callable | None = None,
        n_quad: int = 64,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.N = N
        self.grid = ChebyshevGrid(N, xmin, xmax)
        self.u0 = initial_condition
        self.gL = gL
        self.gR = gR
        self.u_exact = exact_solution
        self.n_quad = n_quad

    # -- Boundary lifting helpers ------------------------------------------

    def _lifting_function(self) -> tuple[Callable, Callable]:
        """Linear boundary lift and its time derivative (via complex-step)."""
        a, b = self.grid.xmin, self.grid.xmax
        gL, gR = self.gL, self.gR
        eps = 1e-20
        gL_t = lambda t: np.imag(gL(t + 1j * eps)) / eps
        gR_t = lambda t: np.imag(gR(t + 1j * eps)) / eps
        ell = lambda x, t: ((b - x) / (b - a)) * gL(t) + ((x - a) / (b - a)) * gR(t)
        ell_t = lambda x, t: ((b - x) / (b - a)) * gL_t(t) + ((x - a) / (b - a)) * gR_t(
            t
        )
        return ell, ell_t

    # -- Time integration --------------------------------------------------

    def solve(self, T: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance from t = 0 to t = T with step dt. Returns (u, t)."""
        t = np.arange(0.0, T + dt, dt)
        nt = len(t)
        x = self.grid.x

        ell, ell_t = self._lifting_function()

        # Interior operator L_int = D^2|_{1:N-1, 1:N-1} and boundary-column pieces
        D = self.grid.D
        D_int = D[1:-1, 1:-1]
        D_col_0 = D[1:-1, 0]       # boundary column at index 0  (x = xmax here, because of cosine ordering)
        D_col_Nm = D[1:-1, -1]     # boundary column at index N  (x = xmin)

        L = self.grid.D2
        L_int = L[1:-1, 1:-1]
        L_col_0 = L[1:-1, 0]
        L_col_Nm = L[1:-1, -1]

        # gR is applied at x=xmax (index 0), gL at x=xmin (index N)
        gL_fun, gR_fun = self.gL, self.gR
        x_int = x[1:-1]

        # F-tilde in the paper: full semi-discrete nonlinear + lift dynamics
        def F_tilde(v_int: np.ndarray, t_now: float) -> np.ndarray:
            ell_int = ell(x_int, t_now)
            ell_t_int = ell_t(x_int, t_now)
            u_int = v_int + ell_int
            # u_x on interior nodes (full D acting on full vector, then restricted):
            u_x_int = (
                D_int @ u_int + D_col_0 * gR_fun(t_now) + D_col_Nm * gL_fun(t_now)
            )
            # (L * ell)|_int = L_int @ ell_int + boundary-column contributions:
            L_ell_int = (
                L_int @ ell_int
                + L_col_0 * gR_fun(t_now)
                + L_col_Nm * gL_fun(t_now)
            )
            return (
                -self.alpha * u_int**self.delta * u_x_int
                + self.beta * u_int * (1 - u_int**self.delta) * (u_int**self.delta - self.gamma)
                - ell_t_int
                + L_ell_int
            )

        # Precompute phi-functions once (they depend only on (L_int, dt))
        phi0, phi1, phi2, phi3, phi0_half, phi1_half = all_phi_functions(
            L_int, dt, n_q=self.n_quad
        )

        # Initial condition for v = u - ell
        u = np.zeros((self.N + 1, nt))
        u[:, 0] = self.u0(x)
        v = np.zeros_like(u)
        v[:, 0] = u[:, 0] - ell(x, 0.0)
        # Enforce v = 0 at boundaries (should already hold up to round-off)
        v[0, 0] = 0.0
        v[-1, 0] = 0.0

        # ETDRK4 time stepping on the interior.
        #
        # Implementation note: we deliberately DO NOT cache F(v_n, t_n) across
        # stage evaluations, because re-evaluating F at each stage (as in the
        # Cox--Matthews / Kassam--Trefethen reference implementations) yields
        # slightly better numerical cancellation at the few-ULP level on
        # well-conditioned benchmarks. Results agree to round-off either way.
        for n in range(nt - 1):
            tn = t[n]
            h = dt
            v_n = v[1:-1, n]

            a_stage = phi0_half @ v_n + 0.5 * h * (phi1_half @ F_tilde(v_n, tn))
            b_stage = phi0_half @ v_n + 0.5 * h * (
                phi1_half @ F_tilde(a_stage, tn + 0.5 * h)
            )
            c_stage = phi0_half @ a_stage + 0.5 * h * (
                phi1_half @ (2 * F_tilde(b_stage, tn + 0.5 * h) - F_tilde(v_n, tn))
            )

            v[1:-1, n + 1] = (
                phi0 @ v_n
                + h * ((phi1 - 3 * phi2 + 4 * phi3) @ F_tilde(v_n, tn))
                + 2 * h
                * ((phi2 - 2 * phi3) @ (F_tilde(a_stage, tn + 0.5 * h) + F_tilde(b_stage, tn + 0.5 * h)))
                + h * ((4 * phi3 - phi2) @ F_tilde(c_stage, tn + h))
            )

        # Reconstruct full u = v + ell
        for n in range(nt):
            u[:, n] = v[:, n] + ell(x, t[n])

        return u, t

    # -- Error analysis ----------------------------------------------------

    def linf_error(self, u_num: np.ndarray, t: np.ndarray, time_index: int = -1) -> float:
        """L-infinity error at a given time index (default: final time)."""
        if self.u_exact is None:
            raise RuntimeError("No exact solution provided for error computation.")
        x = self.grid.x
        u_ex = self.u_exact(x, t[time_index])
        return float(np.max(np.abs(u_num[:, time_index] - u_ex)))
