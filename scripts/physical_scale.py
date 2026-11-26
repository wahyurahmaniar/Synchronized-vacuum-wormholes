"""Physical scale of the drive (Section 6.1).

Restores hbar, G, and c in the reduced equation. A deformation of scale da on a
throat of radius r0 has kinetic norm M_Q = kappa_Q c^2 r0 da^2 / G, the barrier
is Delta U = M_Q omega_Q^2 mu^2 (2 - mu)/12, the projected drive is
J_corr = (hbar c da / L^2) j(tau) |M| with L = GM/c^2, and the thresholds of
Table 1 are values of m = eps_d |M| with eps_d = l_Pl^2 / (kappa_Q r0 da).

Reads results/table_z0.02.json and results/sources.npz; writes
results/physical_scale.json for r0 = 1 m and r0 = 1 km.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from svw import pipeline as P
from svw.paths import RESULTS, load_sources

G, C_LIGHT, HBAR = 6.67430e-11, 299792458.0, 1.054571817e-34      # CODATA 2018
WBAR = 0.700          # omega_Q in units c^3/GM
DA_OVER_R0 = 0.1      # deformation scale
KAPPA_Q = 1.0         # kinetic-norm coefficient (drops out of |M|_crit / n_coll)
SIGMA, XM = 0.0175, 12.0

table = {round(r["mu"], 4): r for r in json.load(open(RESULTS / "table_z0.02.json"))}


def barrier_factor(mu):
    return mu**2 * (2.0 - mu) / 12.0


def prefactor(mu):
    """|M|_crit / (Delta U / hbar omega_Q), independent of kappa_Q."""
    da_over_L = DA_OVER_R0 * P.R0
    return table[mu]["A"] / (WBAR * da_over_L * barrier_factor(mu))


# peak of the correlation drive of A at the phase of Table 1
d = load_sources()
tg = d["tg"]
sel = (tg >= XM - 2.7 / SIGMA - 1e-9) & (tg <= XM + 2.7 / SIGMA + 1e-9)
row = table[0.35]
jA = 2 * np.real(np.exp(1j * row["A_chi"]) * d["A"]["anom"][sel])
jmax = float(np.abs(jA).max())

out = {"prefactor_mu0.35": prefactor(0.35), "prefactor_mu0.05": prefactor(0.05),
       "jmax_A": jmax}
mu = 0.35
for r0 in (1.0, 1000.0):
    L = r0 / P.R0                              # metres per unit GM/c^2
    M = L * C_LIGHT**2 / G
    da = DA_OVER_R0 * r0
    lp2 = HBAR * G / C_LIGHT**3
    eps_d = lp2 / (KAPPA_Q * r0 * da)
    M_Q = KAPPA_Q * C_LIGHT**2 * r0 * da**2 / G
    omega_Q = WBAR * C_LIGHT / L
    dU = M_Q * omega_Q**2 * barrier_factor(mu)
    n_coll = dU / (HBAR * omega_Q)
    Mcrit = {k: row[k] / eps_d for k in ("A", "C", "S")}
    carrier_sum = {"A": 0.70, "C": 0.90, "S": 0.90}           # c^3/GM
    E_in = {k: Mcrit[k] * HBAR * carrier_sum[k] * C_LIGHT / L for k in Mcrit}
    J_peak = HBAR * C_LIGHT * da / L**2 * jmax * Mcrit["A"]
    res = dict(M_kg=M, Mc2_J=M * C_LIGHT**2, eps_d=eps_d, M_Q=M_Q, omega_Q=omega_Q,
               DeltaU_J=dU, ncoll=n_coll, Mcrit_A=Mcrit["A"], Ein_A=E_in["A"],
               Ein_C=E_in["C"], Ein_S=E_in["S"], Ein_over_DU=E_in["A"] / dU,
               Ein_over_Mc2=E_in["A"] / (M * C_LIGHT**2), J_peak=J_peak)
    out[str(r0)] = res
    print("r0 = %g m: M = %.3g kg, eps_d = %.2g, M_Q = %.3g kg m^2, omega_Q = %.2g /s, "
          "Delta U = %.2g J = %.2g quanta" % (r0, M, eps_d, M_Q, omega_Q, dU, n_coll))
    print("          |M|_crit(A) = %.2g, E_in = %.2g J = %.1f Delta U = %.3f Mc^2, "
          "peak J_corr = %.2g J" % (Mcrit["A"], E_in["A"], E_in["A"] / dU,
                                    E_in["A"] / (M * C_LIGHT**2), J_peak))
print("prefactor of Delta U/(hbar omega_Q) in |M|_crit: %.1f at mu = 0.35, %.0f at mu = 0.05"
      % (out["prefactor_mu0.35"], out["prefactor_mu0.05"]))
json.dump(out, open(RESULTS / "physical_scale.json", "w"), indent=1)
