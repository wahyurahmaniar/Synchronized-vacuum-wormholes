"""Figure 1: four panels on one line.

(a) candidate-mouth compatibility, (b) synchronized activation drawn as a
physical schematic, (c) retention on the reduced open branch, (d) reduced
normal-form activation model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLU, VER, GRN, PNK, ORA, SKY = ("#0072B2", "#D55E00", "#009E73",
                                "#CC79A7", "#E69F00", "#56B4E9")
GRY = "0.55"
from svw.paths import FIGURES
OUT = FIGURES / "svw_framework.png"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
})

fig = plt.figure(figsize=(6.6, 2.95))
AX = [fig.add_axes(r) for r in ([0.005, 0.185, 0.228, 0.775],
                                [0.245, 0.185, 0.228, 0.775],
                                [0.485, 0.185, 0.228, 0.775],
                                [0.800, 0.300, 0.192, 0.635])]
for a in AX[:3]:
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis("off")
CX = [0.119, 0.359, 0.599, 0.896]
for cx, sub, lab in zip(CX,
                        ["candidate-mouth compatibility", "synchronized activation",
                         "retention on the open branch", "reduced normal-form model"],
                        ["(a)", "(b)", "(c)", "(d)"]):
    fig.text(cx, 0.105, lab, ha="center", va="top", fontsize=9)


def wavelet(ax, x, y, w, h, color, n=9, lw=0.9):
    t = np.linspace(-1, 1, 200)
    ax.plot(x + w * t, y + h * np.exp(-3.4 * t**2) * np.cos(n * t),
            color=color, lw=lw, zorder=4)


def throat(ax, x0, y0, hw, sy, a, color, ls="-", lw=1.5, fill=None, z=3):
    l = np.linspace(-1, 1, 260)
    R = np.sqrt(l**2 + a**2)
    xs, up, dn = x0 + hw * l, y0 + sy * R, y0 - sy * R
    if fill:
        ax.fill_between(xs, dn, up, color=fill, alpha=0.18, lw=0, zorder=1)
    ax.plot(xs, up, color=color, ls=ls, lw=lw, zorder=z)
    ax.plot(xs, dn, color=color, ls=ls, lw=lw, zorder=z)


# ------------------------------------------------------------------- (a) ----
ax = AX[0]
ax.add_patch(FancyBboxPatch((0.03, 0.385), 0.28, 0.23,
                            boxstyle="round,pad=0.01,rounding_size=0.04",
                            fc="white", ec=BLU, lw=1.3, zorder=3))
wavelet(ax, 0.17, 0.555, 0.10, 0.035, BLU)
ax.text(0.17, 0.437, r"$R_A$", ha="center", va="center", color=BLU, fontsize=9)
for yy, lab, col, dl, ok in [(0.845, r"$R_{B_1}$", GRY, r"$D_1>0$", False),
                             (0.500, r"$R_{B_2}$", GRN, r"$D_2=0$", True),
                             (0.155, r"$R_{B_3}$", GRY, r"$D_3>0$", False)]:
    ax.add_patch(FancyBboxPatch((0.66, yy - 0.115), 0.30, 0.23,
                                boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc="white", ec=col, lw=1.4 if ok else 0.9,
                                ls="-" if ok else (0, (2.4, 1.6)), zorder=3))
    wavelet(ax, 0.81, yy + 0.055, 0.10, 0.033, col, lw=0.8)
    ax.text(0.81, yy - 0.062, lab, ha="center", va="center", color=col, fontsize=9)
    ax.add_patch(FancyArrowPatch((0.325, 0.50), (0.645, yy), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.5 if ok else 0.9,
                                 color=col, linestyle="-" if ok else (0, (2.4, 1.6)),
                                 zorder=2))
    xm, ym = 0.482, 0.50 + (yy - 0.50) * 0.49
    ax.plot([xm], [ym], marker="o" if ok else "x", ms=4.2, mew=1.5,
            color=col, zorder=5)
    if ok:
        ax.text(xm, ym + 0.055, dl, ha="center", va="bottom", color=col,
                fontsize=7.2)
    else:
        up = yy > 0.5
        ax.text(xm - 0.045, ym + (0.03 if up else -0.03), dl, ha="right",
                va="bottom" if up else "top", color=col, fontsize=7.2)

# ------------------------------------------------------------------- (b) ----
ax = AX[1]
wavelet(ax, 0.15, 0.905, 0.125, 0.052, BLU, lw=1.0)
wavelet(ax, 0.85, 0.905, 0.125, 0.052, PNK, lw=1.0)
ax.text(0.15, 0.995, r"$\psi_A$", ha="center", va="bottom", color=BLU, fontsize=9)
ax.text(0.85, 0.995, r"$\psi_B$", ha="center", va="bottom", color=PNK, fontsize=9)
ax.add_patch(FancyArrowPatch((0.30, 0.905), (0.70, 0.905), arrowstyle="<|-|>",
                             mutation_scale=6.5, lw=1.0, color=ORA,
                             linestyle=(0, (3, 1.8))))
ax.text(0.50, 0.845, r"$M$", ha="center", va="top", color=ORA, fontsize=9)
ax.text(0.50, 0.755, r"$\Omega_A+\Omega_B=\omega_Q$", ha="center", va="top",
        color=GRN, fontsize=8.6)
throat(ax, 0.50, 0.275, 0.47, 0.155, 0.28, GRY, ls=(0, (2.4, 1.6)), lw=1.1, z=3)
throat(ax, 0.50, 0.275, 0.47, 0.155, 0.62, BLU, lw=1.6, z=4)
for xs, xe in ((0.17, 0.40), (0.83, 0.60)):
    ax.add_patch(FancyArrowPatch((xs, 0.635), (xe, 0.455), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.2, color=GRN,
                                 connectionstyle="arc3,rad=%.2f" % (0.28 if xs < 0.5 else -0.28),
                                 zorder=5))
ax.text(0.50, 0.560, r"$J_{\rm corr}(t)$", ha="center", va="center",
        color=GRN, fontsize=8.6)
ax.text(0.50, 0.045, r"$Q(t)>Q_{\rm b}$", ha="center", va="bottom",
        color=VER, fontsize=8.8)

# ------------------------------------------------------------------- (c) ----
ax = AX[2]
SY, AQ = 0.160, 0.55
throat(ax, 0.50, 0.600, 0.47, SY, AQ, BLU, lw=1.6, fill=SKY, z=3)
# field lines threading the sector, squeezed through the narrow waist
_l = np.linspace(-1, 1, 300)
_R = np.sqrt(_l**2 + AQ**2)
for _f in (0.42, 0.72):
    for _sg in (+1, -1):
        ax.plot(0.50 + 0.47 * _l, 0.600 + _sg * _f * SY * _R,
                color=BLU, lw=0.6, alpha=0.30, zorder=2)
# support contours of the vacuum sector, parallel to the throat surface and
# pinching at the waist
lq = np.linspace(-1, 1, 300)
Rq = np.sqrt(lq**2 + AQ**2)
xq = 0.50 + 0.47 * lq
for d, al in ((0.040, 0.85), (0.082, 0.58), (0.124, 0.34)):
    for sgn in (1, -1):
        ax.plot(xq, 0.600 + sgn * (SY * Rq + d), color=GRN, lw=1.0,
                alpha=al, zorder=2)
for xx in (0.30, 0.70):
    yt = 0.600 + SY * np.sqrt(((xx - 0.5) / 0.47)**2 + AQ**2)
    ax.add_patch(FancyArrowPatch((xx, yt + 0.168), (xx, yt + 0.028),
                                 arrowstyle="-|>", mutation_scale=7, lw=1.2,
                                 color=GRN, zorder=6))
ax.text(0.50, 0.995, "beyond-Horndeski", ha="center", va="bottom",
        color=BLU, fontsize=7.6)
ax.text(0.50, 0.945, "vacuum sector", ha="center", va="bottom",
        color=BLU, fontsize=7.6)
ax.text(0.50, 0.888, r"$\tilde\rho+\tilde p_r<0$", ha="center", va="center",
        color=GRN, fontsize=8.4)
ax.add_patch(FancyArrowPatch((0.07, 0.600), (0.30, 0.600), arrowstyle="-|>",
                             mutation_scale=7, lw=1.3, color=ORA,
                             linestyle=(0, (3, 1.8)), zorder=5))
ax.text(0.335, 0.600, r"transient $J_{\rm corr}$", ha="left", va="center",
        color=ORA, fontsize=7.0, zorder=6)
ax.text(0.50, 0.205, r"$Q(t_{\rm off})>Q_{\rm b}$", ha="center", va="center",
        fontsize=8.4)
ax.text(0.50, 0.085, r"$E_Q(t_{\rm off})<U_{\rm eff}(Q_{\rm b})$", ha="center",
        va="center", fontsize=8.4)

# ------------------------------------------------------------------- (d) ----
ax = AX[3]
mu = 0.35
Q = np.linspace(-0.16, 1.28, 700)
U = (Q**4 / 4 - (1 + mu) * Q**3 / 3 + mu * Q**2 / 2) / mu
Ub = (mu**4 / 4 - (1 + mu) * mu**3 / 3 + mu * mu**2 / 2) / mu
U1 = (0.25 - (1 + mu) / 3 + mu / 2) / mu
ax.axhline(0.0, color="0.85", lw=0.7, zorder=1)
ax.plot(Q, U, color=BLU, lw=1.6, zorder=3)
ax.plot([mu, 1.28], [Ub, Ub], ls=(0, (2, 1.8)), color="0.45", lw=0.8, zorder=2)
for qq, uu, cc in ((0, 0.0, GRN), (mu, Ub, VER), (1, U1, BLU)):
    ax.plot([qq], [uu], "o", color=cc, ms=4.6, zorder=4)
ax.annotate("", xy=(0.63, Ub), xytext=(0.63, 0.0),
            arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="0.3",
                            mutation_scale=5))
ax.text(0.68, Ub / 2, r"$\Delta\mathcal{U}_{\rm eff}$", fontsize=7.6, va="center")
ax.text(0.04, -0.013, r"$Q=0$", ha="center", va="top", color=GRN, fontsize=7.4)
ax.text(mu + 0.03, Ub + 0.006, r"$Q_{\rm b}=\mu$", ha="left", va="bottom",
        color=VER, fontsize=7.4)
ax.text(1.0, U1 - 0.007, r"$Q=1$", ha="center", va="top", color=BLU, fontsize=7.4)
ax.set_xlabel(r"$Q$", labelpad=1.5)
ax.set_ylabel(r"$\mathcal{U}_{\rm eff}/\omega_Q^{2}$", labelpad=1.5)
ax.set_xlim(-0.16, 1.28)
ax.set_ylim(-0.095, 0.048)
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([-0.075, -0.05, -0.025, 0.0, 0.025])
ax.tick_params(top=True, right=True, pad=1.8)

fig.savefig(OUT, dpi=400)
print("svw_framework.png written")
print(f"  panel (d): Delta U/w^2 = {Ub:.4f} (0.0168), U(1)/w^2 = {U1:.4f} (-0.0714)")
