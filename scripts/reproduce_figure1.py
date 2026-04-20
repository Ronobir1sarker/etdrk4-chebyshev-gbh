"""Reproduce Figure 1 of the paper.

Panel (a): N-refinement at t=1 for the Ismail benchmark (alpha=beta=delta=1,
gamma=1e-3) with a SINGLE ETDRK4 step of size dt=1.0, covering [0,1].
Two curves: vs the corrected exact solution (blue), and vs Wang's (red).

Panel (b): dt-refinement at t=1 in the strongly nonlinear regime gamma=0.5,
delta=1, alpha=beta=1, with N=30. Reveals the Hochbruck-Ostermann order
reduction (observed order ~ 2.5 vs theoretical O(dt^4)).

Usage:  python scripts/reproduce_figure1.py [--output PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import GBHEtdrk4Solver, make_benchmark, travelling_wave


def panel_a_ismail_N_refinement() -> tuple[list[int], list[float], list[float]]:
    """N-refinement at the Ismail benchmark. Single ETDRK4 step, dt=1.0."""
    alpha, beta, gamma, delta = 1.0, 1.0, 1e-3, 1.0
    xmin, xmax, T, dt = 0.0, 1.0, 1.0, 1.0

    u_exact, u0, _, _ = make_benchmark(alpha, beta, gamma, delta)
    u_wang = lambda x, t: travelling_wave(x, t, alpha, beta, gamma, delta, use_wang=True)

    N_values = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30]
    errors_vs_correct = []
    errors_vs_wang = []

    for N in N_values:
        solver = GBHEtdrk4Solver(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            N=N, xmin=xmin, xmax=xmax,
            initial_condition=u0,
            gL=lambda t: u_exact(np.array([xmin]), t)[0],
            gR=lambda t: u_exact(np.array([xmax]), t)[0],
            exact_solution=u_exact,
        )
        u, t = solver.solve(T=T, dt=dt)
        x_grid = solver.grid.x

        err_correct = np.max(np.abs(u[:, -1] - u_exact(x_grid, t[-1])))
        err_wang = np.max(np.abs(u[:, -1] - u_wang(x_grid, t[-1])))

        # Floor at machine epsilon for log plotting
        errors_vs_correct.append(max(err_correct, 1e-20))
        errors_vs_wang.append(err_wang)

    return N_values, errors_vs_correct, errors_vs_wang


def panel_b_strongly_nonlinear_dt_refinement() -> tuple[list[float], list[float]]:
    """dt-refinement at gamma=0.5 with N=30 fixed. Shows order reduction."""
    alpha, beta, gamma, delta = 1.0, 1.0, 0.5, 1.0
    xmin, xmax, T, N = 0.0, 1.0, 1.0, 30

    u_exact, u0, _, _ = make_benchmark(alpha, beta, gamma, delta)

    dt_values = [0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]
    errors = []
    for dt in dt_values:
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
        errors.append(err)

    return dt_values, errors


def plot_figure(output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel (a)
    N_values, err_correct, err_wang = panel_a_ismail_N_refinement()
    ax = axes[0]
    ax.semilogy(N_values, err_correct, "o-", color="tab:blue",
                label=r"vs.\ corrected exact $A_2^+$")
    ax.semilogy(N_values, err_wang, "s-", color="tab:red",
                label=r"vs.\ Wang's $A_2^-$")
    ax.axhline(np.finfo(float).eps, linestyle=":", color="gray", label="machine epsilon")
    ax.set_xlabel(r"$N$ (collocation points)")
    ax.set_ylabel(r"$\|u_{\rm num}-u_{\rm ref}\|_\infty$ at $t=1$")
    ax.set_title(r"(a) Ismail benchmark $\gamma=10^{-3}$; $\Delta t = 1.0$ (single step)")
    ax.legend(loc="center right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-21, 1e-5)

    # Panel (b)
    dt_values, errs = panel_b_strongly_nonlinear_dt_refinement()
    ax = axes[1]
    ax.loglog(dt_values, errs, "o-", color="tab:green", label="present scheme")

    # Reference O(dt^4) line anchored at the smallest dt
    ref_constant = errs[-1] / dt_values[-1] ** 4
    dt_ref = np.array(dt_values)
    ax.loglog(dt_ref, ref_constant * dt_ref**4, "--", color="gray",
              label=r"$\mathcal{O}(\Delta t^4)$ reference")
    ax.axhline(np.finfo(float).eps, linestyle=":", color="gray", label="machine epsilon")

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\|u_{\rm num}-u_{\rm exact}\|_\infty$ at $t=1$")
    ax.set_title(r"(b) Strongly nonlinear regime $\gamma=0.5$; $N=30$")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/convergence.pdf"),
                        help="Output path for the figure (pdf/png).")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_figure(args.output)


if __name__ == "__main__":
    main()
