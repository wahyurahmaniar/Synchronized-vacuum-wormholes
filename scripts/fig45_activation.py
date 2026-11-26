"""Figures 4 and 5 of the revised manuscript, from the revised pipeline.

Fig. 4  correlation-isolated drive and reduced trajectories at mu = 0.35
Fig. 5  (a) held-opening threshold against mu, (b) against carrier detuning,
        (c) maximal excursion against reduced amplitude near the window closure
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from svw import pipeline as P
from svw.paths import RESULTS, FIGURES, load_sources

BLU, VER, GRN, PNK, ORA, SKY = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                "#E69F00", "#56B4E9")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
})


def label(ax, s):
    ax.annotate(s, xy=(0.5, -0.27), xycoords="axes fraction", ha="center",
                va="top", fontsize=11.5)


def ticks(ax):
    ax.tick_params(which="both", top=True, right=True)


d = load_sources()
tg0 = d["tg"]
sel = (tg0 >= 12 - 2.7 / 0.0175 - 1e-9) & (tg0 <= 12 + 2.7 / 0.0175 + 1e-9)
tg = tg0[sel]
tab = {r["mu"]: r for r in json.load(open(RESULTS / "table_z0.02.json"))}
row = tab[0.35]
jA = 2 * np.real(np.exp(1j * row["A_chi"]) * d["A"]["anom"][sel])
jC = 2 * np.real(np.exp(1j * row["C_chi"]) * d["C"]["anom"][sel])
occA = d["A"]["occ"][sel]

# ------------------------------------------------------------------ Fig. 4
fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.9))
a = ax[0, 0]
s = np.abs(jA).max()
a.plot(tg, jA / s, color=BLU, lw=0.9, label="A")
a.plot(tg, jC / s, color=VER, lw=0.9, ls=(0, (3.5, 1.6)), label=r"$C_\Delta$")
a.set_xlim(-60, 90)
a.set_ylim(-1.15, 1.15)
a.set_xlabel(r"$\tau=ct/(GM/c^{2})$")
a.set_ylabel(r"$j(\tau)/\max|j_A|$")
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
label(a, "(a)")

a = ax[0, 1]
wb = np.linspace(0.001, 1.6, 1600)
SA = np.abs(P.spectrum(jA, tg, wb))
SC = np.abs(P.spectrum(jC, tg, wb))
SO = np.abs(P.spectrum(occA, tg, wb))
a.semilogy(wb, SA, color=BLU, lw=1.2, label="A")
a.semilogy(wb, SC, color=VER, lw=1.2, ls=(0, (3.5, 1.6)), label=r"$C_\Delta$")
a.semilogy(wb, SO, color="0.45", lw=1.0, ls=":", label=r"occupation, per unit $N$")
a.axvline(0.7, color="0.35", lw=0.8, ls=(0, (1, 2)))
a.axvline(0.9, color=VER, lw=0.8, ls=(0, (1, 2)))
a.text(0.685, 1.5e-5, r"$\omega_Q$", ha="right", va="bottom", fontsize=10, color="0.3")
a.text(1.10, 1.5e-5, r"$\Omega_A+\Omega_B^{C_\Delta}$", ha="left", va="bottom",
       fontsize=10, color=VER)
a.set_ylim(1e-6, 30)
a.set_xlim(0, 1.6)
a.set_xlabel(r"$\omega\,GM/c^{3}$")
a.set_ylabel(r"$|\hat j(\omega)|$")
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False,
         handlelength=1.6, columnspacing=0.9, fontsize=8.8)
label(a, "(b)")

mu = 0.35
tt = np.linspace(tg[0], 900, 60000)


def traj(j, m):
    sol = P.evolve(m, j, tg, mu, full=True)
    return sol.sol(tt)[0]


a = ax[1, 0]
m0 = 1.01 * row["A"]
QA, QC = traj(jA, m0), traj(jC, m0)
a.plot(tt, QA, color=BLU, lw=0.7, label="A")
a.plot(tt, QC, color=VER, lw=0.7, ls=(0, (3.5, 1.6)), label=r"$C_\Delta$")
a.plot(tt, 0 * tt, color="k", lw=1.0, ls=":", label="B, D, E")
a.axhline(mu, color="0.35", lw=0.8, ls="--")
a.text(700, mu + 0.05, r"$Q_{\rm b}=\mu$", fontsize=10, color="0.3")
a.set_xlim(tt[0], 900)
a.set_ylim(-0.3, 1.4)
a.set_xlabel(r"$\tau$")
a.set_ylabel(r"$Q(\tau)$")
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
label(a, "(c)")

a = ax[1, 1]
mc = row["A"]
for m, col, ls in [(0.90 * mc, GRN, "-."), (0.99 * mc, ORA, "--"),
                   (1.01 * mc, BLU, "-"), (1.55 * mc, PNK, ":")]:
    q = traj(jA, m)
    print("Fig4(d) m=%.4f final Q=%.4f max Q=%.3f" % (m, q[-1], q.max()))
    a.plot(tt, q, color=col, lw=0.7, ls=ls,
           label=r"$m=%.3f$" % m)
a.axhline(mu, color="0.35", lw=0.8, ls="--")
a.set_xlim(tt[0], 900)
a.set_ylim(-0.3, 1.4)
a.set_xlabel(r"$\tau$")
a.set_ylabel(r"$Q(\tau)$")
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.27), ncol=2, frameon=False,
         fontsize=9, columnspacing=1.0)
label(a, "(d)")
for a in ax.ravel():
    ticks(a)
fig.subplots_adjust(left=0.095, right=0.985, top=0.925, bottom=0.125,
                    wspace=0.28, hspace=0.66)
fig.savefig(FIGURES / "svw_opening.png", dpi=400)
plt.close(fig)
print("Fig. 4 written; trajectories at m0 =", m0, "A final", QA[-1], "C max", QC.max())

# ------------------------------------------------------------------ Fig. 5
T = json.load(open(RESULTS / "table_z0.02.json"))
MU = np.array([r["mu"] for r in T])
MA = np.array([r["A"] for r in T])
MC = np.array([r["C"] for r in T])
MS = np.array([r["S"] for r in T])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.75))
a = ax[0]
a.fill_between(MU, MA, MS, color="0.72", alpha=0.30, lw=0)
a.semilogy(MU, MC, "-s", color=VER, ms=3.6, lw=1.2, label=r"$C_\Delta$")
a.semilogy(MU, MS, "-^", color=GRN, ms=3.8, lw=1.2, label=r"$C_\Delta^{\rm sym}$")
a.semilogy(MU, MA, "-o", color=BLU, ms=3.6, lw=1.2, label="A")
a.set_xlabel(r"$\mu$")
a.set_ylabel(r"$m_{\rm crit}$")
a.set_xlim(0.02, 0.43)
a.set_ylim(1.2e-2, 1.2)
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False,
         handlelength=1.5, columnspacing=0.8, fontsize=9)
label(a, "(a)")

a = ax[1]
dp = json.load(open(RESULTS / "detuning_partner.json"))
ds = json.load(open(RESULTS / "detuning_sym.json"))
x1 = np.array([r["dw"] for r in dp]); y1 = np.array([r["m_phasemin"] for r in dp])
x2 = np.array([r["dw"] for r in ds]); y2 = np.array([r["m_phasemin"] for r in ds])
a.axvline(0.0, color="0.45", lw=0.8, ls=(0, (4, 2.5)))
a.axvline(0.6236 - 0.7, color="0.45", lw=0.8, ls=":")
a.semilogy(x1, y1, "-", color=VER, lw=1.2, label="partner shift")
a.semilogy(x2, y2, color=GRN, lw=1.2, ls=(0, (3.5, 1.6)), label="symmetric shift")
a.set_xlabel(r"$\Delta\omega\,GM/c^{3}$")
a.set_ylabel(r"$m_{\rm crit}$")
a.set_xlim(-0.205, 0.205)
a.set_ylim(5e-2, 1.0)
a.text(0.010, 0.80, "linear\nresonance", fontsize=8.2, color="0.35", va="top")
a.text(-0.081, 0.80, r"$\omega(\Delta U/2)$", fontsize=8.2, color="0.35",
       ha="right", va="top")
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False,
         handlelength=1.6, columnspacing=0.8, fontsize=9)
label(a, "(b)")

a = ax[2]
chiA = P.phase_for(d["A"]["anom"][sel], tg, 0.7)
jov = 2 * np.real(np.exp(1j * chiA) * d["A"]["anom"][sel])
amps = np.linspace(0.10, 0.22, 481)
for mu_, col, ls in [(0.390, BLU, "-"), (0.400, VER, (0, (3.5, 1.6)))]:
    e = P.evolve_many(amps, jov, tg, mu_)
    a.plot(amps, e["qmax"], color=col, lw=1.1, ls=ls, label=r"$\mu=%.2f$" % mu_)
    a.axhline(mu_, color=col, lw=0.7, ls=":")
a.set_xlabel(r"$m$")
a.set_ylabel(r"$\max_\tau Q$")
a.set_xlim(0.10, 0.22)
a.set_ylim(0.22, 0.42)
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False,
         handlelength=1.6, fontsize=9)
label(a, "(c)")
for a in ax:
    ticks(a)
fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.25, wspace=0.42)
fig.savefig(FIGURES / "svw_phase.png", dpi=400)
plt.close(fig)
print("Fig. 5 written")
