# Notebooks


gbh_demo.ipynb](gbh_demo.ipynb) — a ready-to-run companion notebook for the
ETDRK4–Chebyshev solver. Open it and run Kernel → Restart & Run All.


It is self-contained (the solver is embedded) and reproduces the paper's
headline results:


- machine-precision accuracy on the Ismail benchmark at N = 2, single step;
- the constant 3.748e-7 Wang gap, matching its closed-form value;
- convergence to round-off against the corrected (Deng) solution while the
  error against the erroneous (Wang) solution flatlines at the gap;
- an editable cell to sweep your own (alpha, beta, gamma, delta).


Requires numpy and matplotlib.
