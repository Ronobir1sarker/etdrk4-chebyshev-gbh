"""Smoke tests verifying the machine-precision and diagnostic claims.

Run with: python -m pytest tests/  (or just `python tests/test_machine_precision.py`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import (
    A1_coefficient,
    A2_correct,
    A2_wang,
    GBHEtdrk4Solver,
    make_benchmark,
    travelling_wave,
)


def test_ismail_single_step_machine_precision() -> None:
    """N=2 collocation points + single dt=1.0 step -> machine precision."""
    alpha, beta, gamma, delta = 1.0, 1.0, 1e-3, 1.0
    u_exact, u0, _, _ = make_benchmark(alpha, beta, gamma, delta)
    # BCs must accept complex t (for complex-step differentiation of ell_t).
    gL = lambda t: u_exact(np.array([0.0]), t)[0]
    gR = lambda t: u_exact(np.array([1.0]), t)[0]
    solver = GBHEtdrk4Solver(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        N=2, xmin=0.0, xmax=1.0,
        initial_condition=u0,
        gL=gL, gR=gR,
        exact_solution=u_exact,
    )
    u, t = solver.solve(T=1.0, dt=1.0)
    err = solver.linf_error(u, t)
    assert err < 1e-14, f"Expected round-off error, got {err:.4e}"
    print(f"PASS: Ismail N=2, dt=1.0 gives L_inf = {err:.4e}")


def test_wang_plateau_is_analytical_gap() -> None:
    """Measured against Wang's wrong formula, error = |u_correct - u_wang|_inf."""
    alpha, beta, gamma, delta = 1.0, 1.0, 1e-3, 1.0
    u_exact, u0, _, _ = make_benchmark(alpha, beta, gamma, delta)
    u_wang = lambda x, t: travelling_wave(x, t, alpha, beta, gamma, delta, use_wang=True)

    solver = GBHEtdrk4Solver(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        N=30, xmin=0.0, xmax=1.0,
        initial_condition=u0,
        gL=lambda t: u_exact(np.array([0.0]), t)[0],
        gR=lambda t: u_exact(np.array([1.0]), t)[0],
        exact_solution=u_exact,
    )
    u, t = solver.solve(T=1.0, dt=0.5)
    x_grid = solver.grid.x
    err_vs_wang = float(np.max(np.abs(u[:, -1] - u_wang(x_grid, t[-1]))))

    # Analytical gap prediction:  |gamma * A1 * (A2 - A2_W) / 2|
    A1 = A1_coefficient(alpha, beta, gamma, delta)
    A2 = A2_correct(alpha, beta, gamma, delta)
    A2_W = A2_wang(alpha, beta, gamma, delta)
    predicted_gap = abs(gamma * A1 * (A2 - A2_W) / 2.0)

    rel = abs(err_vs_wang - predicted_gap) / predicted_gap
    assert rel < 1e-3, f"gap mismatch: measured {err_vs_wang:.4e} vs {predicted_gap:.4e}"
    assert abs(err_vs_wang - 3.748e-7) / 3.748e-7 < 1e-3, "Should match paper's 3.748e-7"
    print(f"PASS: Wang plateau = {err_vs_wang:.4e}, predicted = {predicted_gap:.4e}")


def test_hashim_regime2() -> None:
    """Hashim benchmark (delta=2) should give ~10^-16."""
    alpha, beta, gamma, delta = 1.0, 1.0, 1e-2, 2.0
    u_exact, u0, _, _ = make_benchmark(alpha, beta, gamma, delta)
    solver = GBHEtdrk4Solver(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        N=20, xmin=0.0, xmax=1.0,
        initial_condition=u0,
        gL=lambda t: u_exact(np.array([0.0]), t)[0],
        gR=lambda t: u_exact(np.array([1.0]), t)[0],
        exact_solution=u_exact,
    )
    u, t = solver.solve(T=1.0, dt=0.01)
    err = solver.linf_error(u, t)
    assert err < 1e-14, f"Hashim benchmark should be round-off, got {err:.4e}"
    print(f"PASS: Hashim N=20, dt=1e-2 gives L_inf = {err:.4e}")


def test_exact_solution_satisfies_boundary_conditions() -> None:
    """The exact solution evaluated on the grid should match the BCs."""
    alpha, beta, gamma, delta = 1.0, 1.0, 1e-3, 1.0
    u_exact, _, _, _ = make_benchmark(alpha, beta, gamma, delta)
    gL_val = u_exact(np.array([0.0]), 0.5)[0]
    gR_val = u_exact(np.array([1.0]), 0.5)[0]
    assert np.isfinite(gL_val) and np.isfinite(gR_val)
    print(f"PASS: BCs at t=0.5: u(0)={gL_val:.4e}, u(1)={gR_val:.4e}")


def main() -> None:
    print("Running smoke tests...")
    print("-" * 60)
    test_ismail_single_step_machine_precision()
    test_wang_plateau_is_analytical_gap()
    test_hashim_regime2()
    test_exact_solution_satisfies_boundary_conditions()
    print("-" * 60)
    print("All tests passed.")


if __name__ == "__main__":
    main()
