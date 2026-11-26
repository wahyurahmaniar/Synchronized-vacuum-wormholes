"""Figure 1 (three-stage framework) and Figure 3 (holding-sector source).

House style: 400 dpi, Okabe-Ito, STIXGeneral, no plot titles, panel labels
bottom-centre non-bold, legends outside the data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from svw import geometry as geom
from svw.paths import FIGURES

BLU, VER, GRN, PNK, ORA, SKY = ("#0072B2", "#D55E00", "#009E73",
                                "#CC79A7", "#E69F00", "#56B4E9")
GRY = "0.55"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
})


def panel_label(ax, txt, dy=-0.155):
    ax.annotate(txt, xy=(0.5, dy), xycoords="axes fraction",
                ha="center", va="top", fontweight="normal", fontsize=11.5)


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def region(ax, x, y, w, h, label, color, wave=True, lw=1.3, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc="white", ec=color, lw=lw, ls=ls, zorder=3))
    if wave:
        t = np.linspace(-1, 1, 220)
        env = np.exp(-4.0 * t**2)
        ax.plot(x + 0.40 * w * t, y + 0.24 * h * env * np.cos(9 * t),
                color=color, lw=1.1, zorder=4)
    ax.text(x, y - h / 2 - 0.052, label, ha="center", va="top",
            color=color, fontsize=11.5, zorder=4)


# ================================================ Figure 3: holding source ===
ell = np.linspace(1e-6, 40 * geom.R0, 60000)
rho, rr, pr, pt = geom.einstein_orthonormal(ell)
x = rho / geom.R0
m = x <= 6.0

fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.85))
ax[0].axhline(0, color="0.8", lw=0.8)
ax[0].plot(x[m], rr[m], color=BLU, lw=1.9, label=r"$\tilde\rho$")
ax[0].plot(x[m], pr[m], color=VER, lw=1.9, ls="--", label=r"$\tilde p_r$")
ax[0].plot(x[m], pt[m], color=GRN, lw=1.9, ls="-.", label=r"$\tilde p_t$")
ax[0].set_ylabel("effective source")
ax[1].axhline(0, color="0.8", lw=0.8)
ax[1].fill_between(x[m], rr[m] + pr[m], 0, where=(rr + pr)[m] < 0,
                   color=SKY, alpha=0.28, lw=0)
ax[1].plot(x[m], (rr + pr)[m], color=BLU, lw=1.9, label=r"$\tilde\rho+\tilde p_r$")
ax[1].plot(x[m], (rr + pt)[m], color=PNK, lw=1.9, ls="--",
           label=r"$\tilde\rho+\tilde p_t$")
ax[1].set_ylabel("null combinations")
for a, lab in zip(ax, ["(a)", "(b)"]):
    a.set_xlabel(r"$\rho/r_0$")
    a.set_xlim(0, 6)
    a.tick_params(top=True, right=True)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3,
             frameon=False, handlelength=1.7, columnspacing=1.2)
    a.annotate(lab, xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
               va="top", fontweight="normal", fontsize=11.5)
fig.subplots_adjust(left=0.095, right=0.985, top=0.845, bottom=0.245, wspace=0.28)
fig.savefig(FIGURES / "svw_holding.png", dpi=400)
print("wrote svw_holding.png")
