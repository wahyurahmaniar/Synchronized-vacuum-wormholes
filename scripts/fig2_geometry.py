"""Fig. 2: exact beyond-Horndeski holding geometry (M = 1, alpha = 1/10,
lambda = 1/2), including the even scalar profile and the curvature invariants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from svw import geometry as geom
from svw.figstyle import apply, BLU, VER, GRN, PNK, label, ticks
from svw.paths import FIGURES

apply()
R0 = geom.R0
ell = np.linspace(0.0, 40 * R0, 400001)
r = np.sqrt(ell**2 + R0**2)
H = geom.h(r)
F = geom.F_of_ell(ell)
rho = np.concatenate(([0.0], cumulative_trapezoid(1 / np.sqrt(F), ell)))
drdrho = np.sqrt(F) * ell / r
# scalar: phi(ell) - phi(0) = int_{r0}^{r} phi'(r') dr', even in ell
php = geom.phi_prime_r(r)
phi = np.concatenate(([0.0], cumulative_trapezoid(php * ell / r, ell)))
phi1 = geom.phi_prime_r(R0)
# curvature invariants in the proper-distance chart
sp_r, sp_P = CubicSpline(rho, r), CubicSpline(rho, 0.5 * np.log(H))
rp, rpp = sp_r(rho, 1), sp_r(rho, 2)
Pp, Ppp = sp_P(rho, 1), sp_P(rho, 2)
A1 = Ppp + Pp**2                  # R_trtr
A2 = Pp * rp / r                  # R_t th t th
A3 = -rpp / r                     # R_r th r th
A4 = (1 - rp**2) / r**2           # R_th ph th ph
K = 4 * A1**2 + 8 * A2**2 + 8 * A3**2 + 4 * A4**2
Ric = -2 * A1 - 4 * A2 + 4 * A3 + 2 * A4      # R = -2 R_trtr ... (sign checked below)
G_rho = 2 * rpp / r + (rp**2 - 1) / r**2
G_pr = 2 * Pp * rp / r + (rp**2 - 1) / r**2
G_pt = Ppp + Pp**2 + Pp * rp / r + rpp / r
Ric2 = -G_rho - G_pr - 2 * G_pt                # R = -trace(G) with rho = -G^t_t
print("throat: R r0^2 = %.3f (from G: %.3f), K r0^4 = %.3f, phi1 = %.7f" %
      (Ric[0] * R0**2, Ric2[0] * R0**2, K[0] * R0**4, phi1))
i8 = np.argmin(abs(rho / R0 - 8))
print("at rho = 8 r0: K r0^4 = %.1e, R r0^2 = %.1e ; phi(inf)-phi(0) = %.4f" %
      (K[i8] * R0**4, Ric2[i8] * R0**2, phi[-1]))

x = ell / R0
X = np.concatenate((-x[:0:-1], x))
ev = lambda a: np.concatenate((a[:0:-1], a))
od = lambda a: np.concatenate((-a[:0:-1], a))
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
a = ax[0, 0]
a.plot(X, ev(H), color=BLU, lw=1.4, label=r"$H(\ell)$")
a.plot(X, ev(F), color=VER, lw=1.4, ls=(0, (4, 2)), label=r"$F(\ell)$")
a.plot([0], [H[0]], "o", color=BLU, ms=4)
a.plot([0], [F[0]], "s", color=VER, ms=4)
a.set_xlim(-8, 8); a.set_ylim(0, 1.05)
a.set_xlabel(r"$\ell/r_0$"); a.set_ylabel("metric functions")
a.legend(loc="lower right", frameon=False)
label(a, "(a)")
a = ax[0, 1]
a.axhline(1, color="0.75", lw=0.7); a.axhline(0, color="0.75", lw=0.7)
a.plot(X, ev(r) / R0, color=BLU, lw=1.4, label=r"$r/r_0$")
a.plot(X, od(drdrho), color=GRN, lw=1.4, ls=(0, (4, 2)), label=r"$dr/d\rho$")
a.set_xlim(-8, 8); a.set_ylim(-1.5, 8.5)
a.set_xlabel(r"$\ell/r_0$"); a.set_ylabel("areal radius, flare-out")
a.legend(loc="upper center", frameon=False)
label(a, "(b)")
a = ax[1, 0]
a.axhline(0, color="0.75", lw=0.7); a.axvline(0, color="0.75", lw=0.7)
a.plot(X, ev(phi), color=PNK, lw=1.4, label=r"$\varphi(\ell)-\varphi(0)$")
xq = np.linspace(-0.9, 0.9, 200)
a.plot(xq, phi1 * R0 * xq**2 / 2, color="k", lw=1.0, ls=":",
       label=r"$\varphi_1\ell^{2}/(2r_0)$")
a.set_xlim(-8, 8); a.set_ylim(-0.72, 0.12)
a.set_xlabel(r"$\ell/r_0$"); a.set_ylabel(r"$\varphi(\ell)-\varphi(0)$")
a.legend(loc="upper right", frameon=False, borderaxespad=0.3)
label(a, "(c)")
a = ax[1, 1]
a.axhline(0, color="0.75", lw=0.7)
a.plot(rho / R0, Ric2 * R0**2, color=BLU, lw=1.4, label=r"$R\,r_0^{2}$")
a.plot(rho / R0, K * R0**4, color=GRN, lw=1.4, ls=(0, (4, 2)),
       label=r"$\mathcal{K}\,r_0^{4}$")
a.set_xlim(0, 8); a.set_ylim(-0.6, 6.2)
a.set_xlabel(r"$\rho/r_0$"); a.set_ylabel("curvature invariants")
a.legend(loc="upper right", frameon=False)
label(a, "(d)")
for a in ax.ravel():
    ticks(a)
fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.12, wspace=0.28,
                    hspace=0.52)
fig.savefig(FIGURES / "svw_geometry.png", dpi=400)
print("wrote svw_geometry.png")
