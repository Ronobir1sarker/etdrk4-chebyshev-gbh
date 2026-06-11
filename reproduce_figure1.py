#!/usr/bin/env python3
"""
reproduce_figure1.py
====================
Regenerates the two paper figures with publication polish:

  figures/wave_profile_gap.pdf   the constant 3.748e-7 gap between the corrected
                                 (Deng 2008) and erroneous (Wang 1990) profiles
  figures/convergence.pdf        L-infinity error vs N and vs dt, showing the
                                 scheme reaching round-off against the corrected
                                 solution while plateauing at the Wang gap

Polish applied (reviewer R2-3): subtle dashed gridlines, enlarged axis labels,
colorblind-safe palette (Okabe-Ito: blue = corrected/Deng, vermillion = Wang).

Usage
-----
    python scripts/reproduce_figure1.py
    python scripts/reproduce_figure1.py --outdir figures

The solver is embedded below so the script is fully standalone; the production
solver used for the paper's tighter benchmarks lives in ``src/``.
"""
import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Chebyshev-Gauss-Lobatto grid + differentiation matrix
# ----------------------------------------------------------------------
def cheb(N, xmin=0.0, xmax=1.0):
    if N == 0:
        return np.array([xmax]), np.zeros((1, 1))
    k = np.arange(N + 1)
    xc = np.cos(np.pi * k / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1) ** k
    X = np.tile(xc, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    x = xmin + (xmax - xmin) * (xc + 1) / 2.0
    return x, D * (2.0 / (xmax - xmin))


# ----------------------------------------------------------------------
# ETDRK4 coefficient functions (Taylor-guarded scalar evaluation)
# ----------------------------------------------------------------------
def _etd_coeffs(z, h):
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    ez = np.exp(zs)
    Q = h * (np.exp(zs / 2) - 1) / zs
    f1 = h * (-4 - zs + ez * (4 - 3 * zs + zs ** 2)) / zs ** 3
    f2 = h * (2 + zs + ez * (-2 + zs)) / zs ** 3
    f3 = h * (-4 - 3 * zs - zs ** 2 + ez * (4 - zs)) / zs ** 3
    Qs = h * (1 / 2 + z / 8 + z ** 2 / 48 + z ** 3 / 384)
    f1s = h * (1 / 6 + z / 6 + 3 * z ** 2 / 40 + z ** 3 / 45)
    f23s = h * (1 / 6 + z / 12 + z ** 2 / 40 + z ** 3 / 180)
    Q = np.where(small, Qs, Q)
    f1 = np.where(small, f1s, f1)
    f2 = np.where(small, f23s, f2)
    f3 = np.where(small, f23s, f3)
    return Q, f1, f2, f3


def _matfun(L, h):
    A = h * L
    w, V = np.linalg.eig(A)
    Vi = np.linalg.inv(V)
    E = (V * np.exp(w)) @ Vi
    E2 = (V * np.exp(w / 2)) @ Vi
    Q, f1, f2, f3 = _etd_coeffs(w, h)
    out = [E, E2, (V * Q) @ Vi, (V * f1) @ Vi, (V * f2) @ Vi, (V * f3) @ Vi]
    return [M.real if np.allclose(M.imag, 0, atol=1e-10) else M for M in out]


class GBHEtdrk4:
    """ETDRK4 + Chebyshev collocation solver with linear boundary lifting."""

    def __init__(self, alpha, beta, gamma, delta, N, xmin, xmax,
                 initial_condition, gL, gR, exact_solution=None, eps=1.0):
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.eps = eps
        self.gL, self.gR, self.exact = gL, gR, exact_solution
        self.x, self.D = cheb(N, xmin, xmax)
        self.D2 = self.D @ self.D
        self.N = N
        self.u0 = initial_condition(self.x)
        self.iI = np.arange(1, N)
        self.L = eps * self.D2[np.ix_(self.iI, self.iI)]

    def _N(self, uI, t):
        a, b, g, d, eps = self.alpha, self.beta, self.gamma, self.delta, self.eps
        u = np.empty(self.N + 1)
        u[self.iI] = uI
        u[0], u[-1] = self.gR(t), self.gL(t)
        ux = self.D @ u
        uxx = self.D2 @ u
        rhs = (-a * u ** d * ux + b * u * (1 - u ** d) * (u ** d - g) + eps * uxx)
        return rhs[self.iI] - self.L @ uI

    def solve(self, T, dt):
        h = dt
        E, E2, Q, f1, f2, f3 = _matfun(self.L, h)
        uI = self.u0[self.iI].copy()
        t = 0.0
        for _ in range(int(round(T / dt))):
            Nu = self._N(uI, t)
            a = E2 @ uI + Q @ Nu
            Na = self._N(a, t + h / 2)
            b = E2 @ uI + Q @ Na
            Nb = self._N(b, t + h / 2)
            c = E2 @ a + Q @ (2 * Nb - Nu)
            Nc = self._N(c, t + h)
            uI = E @ uI + f1 @ Nu + 2 * f2 @ (Na + Nb) + f3 @ Nc
            t += h
        u = np.empty(self.N + 1)
        u[self.iI] = uI
        u[0], u[-1] = self.gR(t), self.gL(t)
        return u, t


def make_benchmark(alpha, beta, gamma, delta):
    """Returns (u_exact, u0, A1, A2_corrected, A2_wang)."""
    s = np.sqrt(alpha ** 2 + 4 * beta * (1 + delta))
    A1 = ((-delta * alpha + delta * s) * gamma) / (4 * (1 + delta))
    A2 = (alpha * gamma) / (1 + delta) + ((1 + delta - gamma) * (alpha + s)) / (2 * (1 + delta))
    A2W = (alpha * gamma) / (1 + delta) - ((1 + delta - gamma) * (-alpha + s)) / (2 * (1 + delta))
    u_exact = lambda x, t: (gamma / 2 + (gamma / 2) * np.tanh(A1 * (x - A2 * t))) ** (1 / delta)
    u0 = lambda x: u_exact(x, 0.0)
    return u_exact, u0, A1, A2, A2W


# ----------------------------------------------------------------------
# figure styling
# ----------------------------------------------------------------------
DENG = "#0072B2"   # blue       -> corrected (Deng 2008)
WANG = "#D55E00"   # vermillion -> erroneous (Wang-Zhu-Lu 1990)


def _style():
    mpl.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
        "axes.linewidth": 1.0, "lines.linewidth": 2.0, "figure.dpi": 120,
    })


def figure_wave_profile_gap(outdir):
    a, b, g, d = 1.0, 1.0, 1e-3, 1.0
    u_exact, _, A1, A2, A2W = make_benchmark(a, b, g, d)
    uW = lambda x, t: (g / 2 + g / 2 * np.tanh(A1 * (x - A2W * t))) ** (1 / d)
    x = np.linspace(0, 1, 1000)
    t = 1.0

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(x, u_exact(x, t), color=DENG, label="Corrected (Deng 2008)")
    ax[0].plot(x, uW(x, t), color=WANG, ls="--", label="Erroneous (Wang 1990)")
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$u(x,\,t=1)$")
    ax[0].set_title("Travelling-wave profiles")
    ax[0].legend(frameon=False)

    gap = np.abs(u_exact(x, t) - uW(x, t))
    ax[1].plot(x, gap, color="#444444")
    ax[1].axhline(3.748e-7, color=WANG, ls=":", lw=1.5)
    ax[1].annotate(r"$\dfrac{\gamma A_1}{2}\,|A_2-A_2^{W}| \approx 3.748\times10^{-7}$",
                   xy=(0.5, 3.748e-7), xytext=(0.12, 2.4e-7),
                   arrowprops=dict(arrowstyle="->", color=WANG), color=WANG)
    ax[1].set_ylim(0, 4.3e-7)
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$|u_{\mathrm{Deng}}-u_{\mathrm{Wang}}|$")
    ax[1].set_title("Wave-profile gap at $t=1$")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.90, wspace=0.30)
    path = os.path.join(outdir, "wave_profile_gap.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}   (max gap = {gap.max():.4e})")


def figure_convergence(outdir):
    a, b, g, d = 1.0, 1.0, 1e-3, 1.0
    u_exact, u0, A1, A2, A2W = make_benchmark(a, b, g, d)
    uW = lambda x, t: (g / 2 + g / 2 * np.tanh(A1 * (x - A2W * t))) ** (1 / d)
    gL = lambda t: u_exact(np.array([0.0]), t)[0]
    gR = lambda t: u_exact(np.array([1.0]), t)[0]

    Ns = [2, 4, 6, 8, 10, 12, 16, 20]
    eD_N, eW_N = [], []
    for N in Ns:
        s = GBHEtdrk4(a, b, g, d, N, 0.0, 1.0, u0, gL, gR, u_exact)
        u, tt = s.solve(1.0, 1e-2)
        eD_N.append(np.max(np.abs(u - u_exact(s.x, tt))))
        eW_N.append(np.max(np.abs(u - uW(s.x, tt))))

    dts = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    eD_dt, eW_dt = [], []
    for dt in dts:
        s = GBHEtdrk4(a, b, g, d, 20, 0.0, 1.0, u0, gL, gR, u_exact)
        u, tt = s.solve(1.0, dt)
        eD_dt.append(np.max(np.abs(u - u_exact(s.x, tt))))
        eW_dt.append(np.max(np.abs(u - uW(s.x, tt))))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(Ns, eD_N, "o-", color=DENG, label="vs corrected (Deng)")
    ax[0].semilogy(Ns, eW_N, "s--", color=WANG, label="vs erroneous (Wang)")
    ax[0].axhline(3.748e-7, color=WANG, lw=1, alpha=0.5)
    ax[0].set_xlabel(r"$N$ (collocation points)")
    ax[0].set_ylabel(r"$L^\infty$ error at $t=1$")
    ax[0].set_title(r"Spatial refinement ($\Delta t=10^{-2}$)")
    ax[0].legend(frameon=False)

    ax[1].loglog(dts, eD_dt, "o-", color=DENG, label="vs corrected (Deng)")
    ax[1].loglog(dts, eW_dt, "s--", color=WANG, label="vs erroneous (Wang)")
    ax[1].axhline(3.748e-7, color=WANG, lw=1, alpha=0.5)
    ax[1].set_xlabel(r"$\Delta t$")
    ax[1].set_ylabel(r"$L^\infty$ error at $t=1$")
    ax[1].set_title(r"Temporal refinement ($N=20$)")
    ax[1].legend(frameon=False)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.90, wspace=0.30)
    path = os.path.join(outdir, "convergence.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}   (min Deng error = {min(eD_N):.3e}, Wang plateau = {np.median(eW_N):.3e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    _style()
    print("Generating polished figures...")
    figure_wave_profile_gap(args.outdir)
    figure_convergence(args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
