"""ETDRK4--Chebyshev solver for the generalized Burgers--Huxley equation."""

from .chebyshev_grid import ChebyshevGrid
from .exact_solutions import (
    A1_coefficient,
    A2_correct,
    A2_wang,
    make_benchmark,
    travelling_wave,
)
from .gbh_solver import GBHEtdrk4Solver
from .phi_functions import all_phi_functions, phi_k_matrix

__version__ = "1.0.0"

__all__ = [
    "ChebyshevGrid",
    "GBHEtdrk4Solver",
    "phi_k_matrix",
    "all_phi_functions",
    "A1_coefficient",
    "A2_correct",
    "A2_wang",
    "travelling_wave",
    "make_benchmark",
]
