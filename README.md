# ETDRK4–Chebyshev Solver for the Generalized Burgers–Huxley Equation

Reference implementation for the paper

> R. C. Sarker and S. Arora, "ETDRK4–Chebyshev collocation for the
> generalized Burgers–Huxley equation: machine-precision benchmarks and a
> corrected exact solution," *Applied Mathematics Letters*, 2026.

## What this code does

Solves the generalized Burgers–Huxley (gBH) equation

$$u_t + \alpha u^\delta u_x - u_{xx} = \beta u (1 - u^\delta)(u^\delta - \gamma)$$

on a bounded interval `[a, b]` with non-homogeneous Dirichlet data using:

- **Space:** Chebyshev–Gauss–Lobatto collocation with linear boundary lifting
- **Time:** Cox–Matthews / Kassam–Trefethen ETDRK4, with
  matrix φ-functions evaluated via the Trefethen–Weideman–Schmelzer
  Talbot contour quadrature

On the Ismail–Raslan–Rabboh travelling-wave benchmark it achieves
$L^\infty$ error at floating-point round-off (~10⁻¹⁹ absolute) using as few
as **N = 2** collocation points and a **single** ETDRK4 step of size
$\Delta t = 1.0$.

## Repository structure

```
.
├── src/
│   ├── __init__.py
│   ├── chebyshev_grid.py     Chebyshev grid + differentiation matrices
│   ├── phi_functions.py      Matrix φ-functions via Talbot contour
│   ├── gbh_solver.py         GBHEtdrk4Solver: main time-stepping driver
│   └── exact_solutions.py    Travelling-wave solutions (correct and Wang)
├── scripts/
│   ├── reproduce_figure1.py  Regenerates Figure 1 of the paper
│   └── reproduce_table1.py   Regenerates the ETDRK4-Chebyshev row of Table 1
├── tests/
│   └── test_machine_precision.py   Smoke tests for reviewers
├── figures/                  (generated) output figures
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.9+ with NumPy, SciPy, and Matplotlib.

```bash
git clone https://github.com/<your-username>/etdrk4-chebyshev-gbh.git
cd etdrk4-chebyshev-gbh
pip install -r requirements.txt
```

No compilation or separate build step — pure Python.

## Quick start

```python
import numpy as np
from src import GBHEtdrk4Solver, make_benchmark

# Ismail-Raslan-Rabboh benchmark
alpha, beta, gamma, delta = 1.0, 1.0, 1e-3, 1.0
u_exact, u0, A1, A2 = make_benchmark(alpha, beta, gamma, delta)

solver = GBHEtdrk4Solver(
    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    N=2, xmin=0.0, xmax=1.0,
    initial_condition=u0,
    gL=lambda t: u_exact(np.array([0.0]), t)[0],
    gR=lambda t: u_exact(np.array([1.0]), t)[0],
    exact_solution=u_exact,
)

u, t = solver.solve(T=1.0, dt=1.0)
print(f"L_inf error: {solver.linf_error(u, t):.4e}")   # ~0 (round-off)
```

## Reproducing paper results

```bash
# Table 1 (ETDRK4-Chebyshev row)
python scripts/reproduce_table1.py

# Figure 1 (both panels)
python scripts/reproduce_figure1.py --output figures/convergence.pdf
```

Expected output for `reproduce_table1.py`:

```
Regime 1 (R1): Ismail benchmark (alpha=beta=delta=1, gamma=1e-3)
  at t=1  (N=20, dt=1e-2)    L_inf = 6.5e-19
  at t=10 (N=20, dt=1e-2)    L_inf = 5.4e-19

Regime 2 (R2): Hashim benchmark (alpha=beta=1, delta=2, gamma=1e-2)
  at t=1  (N=20, dt=1e-2)    L_inf = 1.1e-16
```

## Running tests

```bash
python tests/test_machine_precision.py
```

Four smoke tests verify:

1. Machine-precision accuracy at the Ismail benchmark with N=2, dt=1.0
2. The "Wang plateau" equals the analytical wave-profile gap (3.748×10⁻⁷)
3. The Hashim benchmark (δ=2) reaches round-off
4. Boundary-condition evaluation is well-defined

## The Wang sign-error diagnostic

The script `reproduce_figure1.py` reproduces both panels of Figure 1. Panel
(a) shows that when the scheme is benchmarked against Wang's (1990)
wave-speed formula instead of the corrected Deng (2008) formula, the error
plateaus at an N- and Δt-independent constant equal to

$$\|u_{A_2} - u_{A_2^{\mathrm{W}}}\|_\infty \;=\; \tfrac{\gamma A_1}{2}\,|A_2 - A_2^{\mathrm{W}}| \;\approx\; 3.748\times 10^{-7}$$

at the Ismail benchmark. See Section 2 of the paper for the algebraic
derivation of this residual formula.

You can directly evaluate the two candidate exact solutions yourself:

```python
from src import travelling_wave, A1_coefficient, A2_correct, A2_wang
import numpy as np

x = np.linspace(0, 1, 100)
u_correct = travelling_wave(x, 1.0, 1.0, 1.0, 1e-3, 1.0, use_wang=False)
u_wang    = travelling_wave(x, 1.0, 1.0, 1.0, 1e-3, 1.0, use_wang=True)
print(f"Max gap: {np.max(np.abs(u_correct - u_wang)):.4e}")  # ~3.75e-7
```

## Citing

If you use this code in your work, please cite:

```bibtex
@article{Sarker2026,
  author  = {Sarker, Ronobir Chandra and Arora, Shelly},
  title   = {{ETDRK4}--{C}hebyshev collocation for the generalized {B}urgers--{H}uxley equation:
             machine-precision benchmarks and a corrected exact solution},
  journal = {Applied Mathematics Letters},
  year    = {2026},
  note    = {in press}
}
```

## License

MIT License. See `LICENSE`.

## Contact

Ronobir Chandra Sarker — `ronobir.sarker@gmail.com`
Bangladesh University of Business and Technology, Dhaka-1216, Bangladesh
