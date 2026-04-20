"""Exact travelling-wave solutions for the generalized Burgers--Huxley equation.

Two wave-speed formulas are provided:

- A2_correct (Deng 2008; verified by Appadu & Tijani 2022)
- A2_wang    (Wang, Zhu, Lu 1990; contains a sign error, does NOT satisfy the PDE)

See paper, Section 2, equations (4) and (5).
"""
from __future__ import annotations

import numpy as np


def A1_coefficient(alpha: float, beta: float, gamma: float, delta: float) -> float:
    """Amplitude A1 of the travelling-wave ansatz, eq. (3) of the paper."""
    disc = np.sqrt(alpha**2 + 4.0 * beta * (1.0 + delta))
    return gamma * delta * (-alpha + disc) / (4.0 * (1.0 + delta))


def A2_correct(alpha: float, beta: float, gamma: float, delta: float) -> float:
    """Corrected wave speed of Deng (2008), eq. (5) of the paper.

    This is the UNIQUE value making the travelling-wave ansatz a solution.
    """
    disc = np.sqrt(alpha**2 + 4.0 * beta * (1.0 + delta))
    return alpha * gamma / (1.0 + delta) + (1.0 + delta - gamma) * (
        alpha + disc
    ) / (2.0 * (1.0 + delta))


def A2_wang(alpha: float, beta: float, gamma: float, delta: float) -> float:
    """Wang--Zhu--Lu (1990) wave speed, eq. (4) of the paper.

    WARNING: This formula contains a sign error and does NOT satisfy the PDE.
    Provided only so that users can reproduce the Section 4 diagnostic
    (the plateau at |A2_correct - A2_wang| gamma A_1 / 2).
    """
    disc = np.sqrt(alpha**2 + 4.0 * beta * (1.0 + delta))
    return alpha * gamma / (1.0 + delta) - (1.0 + delta - gamma) * (
        -alpha + disc
    ) / (2.0 * (1.0 + delta))


def travelling_wave(
    x: np.ndarray,
    t: np.ndarray | float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    use_wang: bool = False,
) -> np.ndarray:
    """Evaluate the travelling-wave exact solution, eq. (2) of the paper.

        u(x, t) = [ gamma/2 + gamma/2 * tanh( A1 (x - A2 t) ) ]^(1/delta)

    Parameters
    ----------
    use_wang : if True, use A2_wang (WRONG); if False, use A2_correct.
    """
    A1 = A1_coefficient(alpha, beta, gamma, delta)
    A2 = A2_wang(alpha, beta, gamma, delta) if use_wang else A2_correct(
        alpha, beta, gamma, delta
    )
    return (gamma / 2.0 + (gamma / 2.0) * np.tanh(A1 * (x - A2 * t))) ** (1.0 / delta)


def make_benchmark(
    alpha: float, beta: float, gamma: float, delta: float, use_wang: bool = False
):
    """Package the IC, BCs, and exact solution for a given parameter set.

    Returns
    -------
    (u_exact, u0, gL, gR)
        u_exact(x, t) : exact solution (vectorized)
        u0(x)         : initial condition at t = 0
        gL(t)         : left Dirichlet data (at x = xmin used by caller)
        gR(t)         : right Dirichlet data (at x = xmax)

    Caller supplies the domain [xmin, xmax] to gL, gR as implicit fixed points.
    """
    A1 = A1_coefficient(alpha, beta, gamma, delta)
    A2 = A2_wang(alpha, beta, gamma, delta) if use_wang else A2_correct(
        alpha, beta, gamma, delta
    )

    def u_exact(x, t):
        return (gamma / 2.0 + (gamma / 2.0) * np.tanh(A1 * (x - A2 * t))) ** (
            1.0 / delta
        )

    def u0(x):
        return u_exact(x, 0.0)

    return u_exact, u0, A1, A2
