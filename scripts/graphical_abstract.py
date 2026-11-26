"""Graphical abstract of the paper in a wide 2:1 format. Panels 2 and 3 are
computed from results/sources.npz and results/table_z0.02.json, so the image
shows the actual drive spectra and reduced trajectories of the paper.

Writes figures/graphical_abstract.png (2560 x 1280 px), the image shown at the
top of the README.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from svw import pipeline as P
from svw.figstyle import BLU, VER, GRN, PNK, ORA, SKY
from svw.paths import RESULTS, FIGURES, load_sources

INK, SUB, GRY = "#1f1f1f", "#4a4a4a", "0.62"
MU, WQ, SIGMA, XM = 0.35, 0.700, 0.0175, 12.0
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"],
    "mathtext.fontset": "stix", "font.size": 15, "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15,
})

# ------------------------------------------------------------------ data ----
d = load_sources()
tg0 = d["tg"]
sel = (tg0 >= XM - 2.7 / SIGMA - 1e-9) & (tg0 <= XM + 2.7 / SIGMA + 1e-9)
tg = tg0[sel]
row = {round(r["mu"], 4): r for r in json.load(open(RESULTS / "table_z0.02.json"))}[MU]
j = {k: 2 * np.real(np.exp(1j * row[k + "_chi"]) * d[k]["anom"][sel]) for k in "ACS"}
wb = np.linspace(0.45, 1.15, 1401)
spec = {k: np.abs(P.spectrum(j[k], tg, wb)) for k in "ACS"}
m0 = 1.01 * row["A"]
tt = np.linspace(tg[0], 620.0, 6000)
traj = {k: P.evolve(m0, j[k], tg, MU, full=True).sol(tt)[0] for k in "ACS"}

fig = plt.figure(figsize=(12.8, 6.4))
fig.patch.set_facecolor("white")

# ------------------------------------------------------------ title band ----
fig.text(0.5, 0.925, "Synchronized-vacuum wormholes", ha="center", va="center",
         fontsize=31, color=INK, fontweight="bold")
fig.text(0.5, 0.855, "state matching and resonant activation of reduced throat mode "
         "on exact beyond-Horndeski background", ha="center", va="center",
         fontsize=17, color=SUB)

# ---------------------------------------------------------- panel headers ---
HY = 0.765
heads = [(0.178, "1", "State matching", GRN), (0.525, "2", "Carrier-sum resonance", ORA),
         (0.855, "3", "Activation", VER)]
renderer = fig.canvas.get_renderer()
spans = []
for x, n, t, c in heads:
    txt = fig.text(x, HY, t, ha="center", va="center", fontsize=19, color=INK,
                   fontweight="bold")
    bb = txt.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    shift = 0.0165                       # make room for the badge, keep the group centred
    txt.set_x(x + shift)
    bx = bb.x0 + shift - 0.024
    fig.text(bx, HY, n, ha="center", va="center", fontsize=14, color="white",
             fontweight="bold", bbox=dict(boxstyle="circle,pad=0.28", fc=c, ec="none"))
    spans.append((bx - 0.016, bb.x1 + shift))
for (a0, a1), (b0, b1) in zip(spans[:-1], spans[1:]):
    xm = 0.5 * (a1 + b0)
    fig.patches.append(FancyArrowPatch((xm - 0.022, HY), (xm + 0.022, HY),
                                       transform=fig.transFigure, figure=fig,
                                       arrowstyle="-|>", mutation_scale=17, lw=1.6,
                                       color="0.65"))

# ------------------------------------------------- panel 1: state matching --
ax = fig.add_axes([0.020, 0.175, 0.315, 0.540])
ax.set_xlim(-3.25, 3.25)
ax.set_ylim(-2.78, 2.78)
ax.axis("off")
S, A0, XT = 0.30, 1.25, 2.75


def prof(x):
    return S * np.sqrt(x**2 + A0**2)


xs = np.linspace(-XT, XT, 500)
ax.fill_between(xs, -prof(xs), prof(xs), color=SKY, alpha=0.20, lw=0, zorder=1)
ax.plot(xs, prof(xs), color=BLU, lw=2.2, zorder=4)
ax.plot(xs, -prof(xs), color=BLU, lw=2.2, zorder=4)
for xr in np.linspace(-XT, XT, 11)[1:-1]:
    ax.add_patch(Ellipse((xr, 0), 0.22, 2 * prof(xr), fill=False, ec=BLU, lw=0.8,
                         alpha=0.28, zorder=2))
for xr in (-XT, XT):
    ax.add_patch(Ellipse((xr, 0), 0.34, 2 * prof(xr), fc="white", ec=BLU, lw=2.0, zorder=5))
t = np.linspace(-1, 1, 300)
for xc, col, sgn in ((-1.55, BLU, 1), (1.55, PNK, -1)):
    ax.plot(xc + 0.62 * t, 0.26 * np.exp(-3.0 * t**2) * np.cos(11 * sgn * t),
            color=col, lw=2.0, zorder=6)
ax.text(-1.55, -0.78, r"$\psi_A\;\rightarrow$", ha="center", va="top", color=BLU,
        fontsize=19, zorder=7)
ax.text(1.55, -0.78, r"$\leftarrow\;\psi_B$", ha="center", va="top", color=PNK,
        fontsize=19, zorder=7)
ax.plot([0, 0], [-0.62, 0.62], color=GRN, lw=1.4, ls=(0, (3, 2.2)), zorder=3)
ax.add_patch(FancyArrowPatch((-1.55, 1.12), (1.55, 1.12), arrowstyle="<|-|>",
                             mutation_scale=15, lw=1.6, color=GRN,
                             connectionstyle="arc3,rad=-0.32", zorder=6))
ax.text(0, 1.86, r"$\psi_B=\psi_A\circ\mathcal{I}$", ha="center", va="bottom",
        color=GRN, fontsize=19)
ax.text(-XT, -1.20, r"$R_A$", ha="center", va="top", color=BLU, fontsize=18)
ax.text(XT, -1.20, r"$R_B$", ha="center", va="top", color=BLU, fontsize=18)
ax.text(0, -1.62, r"$D=0$", ha="center", va="top", color=GRN, fontsize=19)

# ------------------------------------------- panel 2: carrier-sum spectra ---
ax = fig.add_axes([0.405, 0.255, 0.240, 0.445])
ax.axvline(WQ, color="0.55", lw=1.1, ls=(0, (2, 2)), zorder=1)
ax.plot(wb, spec["A"], color=BLU, lw=2.4, zorder=4)
ax.plot(wb, spec["C"], color=VER, lw=2.0, ls=(0, (5, 2.2)), zorder=3)
ax.plot(wb, spec["S"], color=GRN, lw=2.0, ls=(0, (1.3, 1.6)), zorder=3)
ax.set_xlim(0.45, 1.15)
ax.set_ylim(0, 8.4)
ax.set_xticks([0.5, 0.7, 0.9, 1.1])
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_xlabel(r"$\omega\ \ (c^{3}/GM)$", labelpad=2)
ax.set_ylabel(r"$|J(\omega)|$", labelpad=3)
ax.tick_params(top=True, right=True, length=4)
iA = np.argmax(spec["A"])
ax.text(0.655, 4.9, "A", ha="right", va="center", color=BLU, fontsize=17)
ax.text(WQ - 0.015, 7.85, r"$\omega_Q$", ha="right", va="top", color="0.35", fontsize=17)
iS = np.argmax(spec["S"])
ax.text(wb[iS] + 0.045, spec["S"][iS] - 0.2, r"$C_\Delta^{\rm sym}$", ha="left", va="center",
        color=GRN, fontsize=16)
iC = np.argmax(spec["C"])
ax.text(wb[iC] + 0.060, spec["C"][iC] - 1.2, r"$C_\Delta$", ha="left", va="center",
        color=VER, fontsize=16)

# ------------------------------------------------ panel 3: Q(t) trajectories
ax = fig.add_axes([0.735, 0.255, 0.240, 0.445])
ax.axhspan(MU, 1.55, color=VER, alpha=0.06, lw=0, zorder=0)
ax.axhline(MU, color="0.45", lw=1.1, ls=(0, (2, 2)), zorder=1)
ax.plot(tt, traj["C"], color=VER, lw=1.8, ls=(0, (5, 2.2)), zorder=3)
ax.plot(tt, traj["S"], color=GRN, lw=1.8, ls=(0, (1.3, 1.6)), zorder=3)
ax.plot(tt, traj["A"], color=BLU, lw=1.2, zorder=4)
ax.set_xlim(tt[0], tt[-1])
ax.set_ylim(-0.32, 1.55)
ax.set_xticks([0, 200, 400, 600])
ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.set_xlabel(r"$t\ \ (GM/c^{3})$", labelpad=2)
ax.set_ylabel(r"$Q$", labelpad=3)
ax.tick_params(top=True, right=True, length=4)
ax.text(605, 1.30, "A: held opening", ha="right", va="center", color=BLU, fontsize=15)
ax.text(605, -0.20, r"$C_\Delta$, $C_\Delta^{\rm sym}$: seed branch", ha="right",
        va="center", color=SUB, fontsize=14)
ax.text(605, MU + 0.04, r"barrier $Q=\mu$", ha="right", va="bottom", color="0.35",
        fontsize=14)

# --------------------------------------------------------------- captions ---
CY = 0.095
for x, s in ((0.178, "isometry-transported state matching\nselects the admissible mouth pairing"),
             (0.525, "throat stress couples counter-propagating\npackets at the sum of their carriers"),
             (0.855, "activation threshold 3.4 to 4.5 times\nlower than for the detuned partner")):
    fig.text(x, CY, s, ha="center", va="center", fontsize=14.5, color=SUB, linespacing=1.25)

out = FIGURES / "graphical_abstract.png"
fig.savefig(out, dpi=200, facecolor="white")
im = plt.imread(out)
print("wrote %s (%d x %d px); A crosses: %s, C max Q %.3f, S max Q %.3f" % (
    out.name, im.shape[1], im.shape[0], traj["A"][-1] > MU, traj["C"].max(), traj["S"].max()))
