"""Reproduce the ETDRK4-Chebyshev row(s) of Table 1 in the paper.

Regime 1 (R1): Ismail benchmark, alpha=beta=delta=1, gamma=1e-3
  - at t=1:  ~10^-19 (round-off)
  - at t=10: ~10^-19 (no temporal accumulation)

Regime 2 (R2): Hashim benchmark, alpha=beta=1, delta=2, gamma=1e-2
  - at t=1:  ~10^-16

Other published methods' numbers (ADM, B-spline, Haar, NSFD, BI-SQLM) are
tabulated in the paper from the original sources; they cannot be regenerated here.

Usage:  python scripts/reproduce_table1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import GBHEtdrk4Solver, make_benchmark


def run_case(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    N: int,
    dt: float,
    T: float,
    label: str,
) -> float:
    """Solve and return L_inf error at final time."""
    xmin, xmax = 0.0, 1.0
    u_exact, u0, _, _ = make_benchmark(alpha, beta, gamma, delta)
    solver = GBHEtdrk4Solver(
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        N=N, xmin=xmin, xmax=xmax,
        initial_condition=u0,
        gL=lambda t: u_exact(np.array([xmin]), t)[0],
        gR=lambda t: u_exact(np.array([xmax]), t)[0],
        exact_solution=u_exact,
    )
    u, t = solver.solve(T=T, dt=dt)
    err = solver.linf_error(u, t)
    print(f"  {label:<35s}  L_inf = {err:.4e}")
    return err


def main() -> None:
    print("=" * 70)
    print("Reproducing ETDRK4-Chebyshev row of Table 1")
    print("=" * 70)

    print("\nRegime 1 (R1): Ismail benchmark (alpha=beta=delta=1, gamma=1e-3)")
    run_case(1.0, 1.0, 1e-3, 1.0, N=20, dt=1e-2, T=1.0,  label="at t=1  (N=20, dt=1e-2)")
    run_case(1.0, 1.0, 1e-3, 1.0, N=20, dt=1e-2, T=10.0, label="at t=10 (N=20, dt=1e-2)")

    print("\nRegime 2 (R2): Hashim benchmark (alpha=beta=1, delta=2, gamma=1e-2)")
    run_case(1.0, 1.0, 1e-2, 2.0, N=20, dt=1e-2, T=1.0,  label="at t=1  (N=20, dt=1e-2)")

    print("\nPaper claims (Table 1, ETDRK4-Chebyshev row):")
    print("  R1, t=1  : 6.5e-19")
    print("  R1, t=10 : 5.4e-19")
    print("  R2, t=1  : 1.1e-16")


if __name__ == "__main__":
    main()
