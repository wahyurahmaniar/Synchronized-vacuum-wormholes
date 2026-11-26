"""Shared figure style: Times New Roman text (Times-metric fallbacks where it is
not installed), STIX math, Okabe-Ito colours, panel labels centred below."""
import matplotlib.pyplot as plt

BLU, VER, GRN, PNK, ORA, SKY = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                "#E69F00", "#56B4E9")


def apply(base=10.5):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": base, "axes.labelsize": base + 1,
        "legend.fontsize": base - 1, "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
    })


def label(ax, s, y=-0.27):
    ax.annotate(s, xy=(0.5, y), xycoords="axes fraction", ha="center",
                va="top", fontsize=plt.rcParams["font.size"] + 1)


def ticks(ax):
    ax.tick_params(which="both", top=True, right=True)
