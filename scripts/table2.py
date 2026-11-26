"""Table 2, sensitivity rows: thresholds at mu = 0.35 with zeta, sigma_Omega, and
ell_w halved and doubled, and the occupation-inclusive and coherent-product
controls (writes results/table2_sensitivity.json).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, json, time
from svw import pipeline as P
from svw.paths import RESULTS
t0 = time.time()
bg = P.build_background()
mu = 0.35; wQ = 0.7
out = {}
def tgrid(sig):
    h = 2.7 / sig
    return np.arange(12 - h, 12 + h + 1e-9, 0.25)
def thr(j, tg, zeta=0.02, a1=8.0):
    return P.thresholds(j, tg, mu, 0.01, a1, n=900, refine=80, zeta=zeta)[1]
def pair(sig, lw=6.0):
    tg = tgrid(sig)
    PA = P.Packet(bg, 0.35, sigma=sig); PB = P.Packet(bg, 0.35, sigma=sig, mirror=True)
    PC = P.Packet(bg, 0.55, sigma=sig, mirror=True)
    PS1 = P.Packet(bg, 0.45, sigma=sig); PS2 = P.Packet(bg, 0.45, sigma=sig, mirror=True)
    sA = P.sources(bg, tg, PA, PB, lw=lw, coherent=True)
    sC = P.sources(bg, tg, PA, PC, lw=lw, coherent=True)
    sS = P.sources(bg, tg, PS1, PS2, lw=lw, coherent=True)
    return tg, sA, sC, sS
cache = {}
for label, sig, lw, zeta in [("base", 0.0175, 6.0, 0.02), ("zeta_half", 0.0175, 6.0, 0.01), ("zeta_double", 0.0175, 6.0, 0.04),
                             ("sigma_half", 0.00875, 6.0, 0.02), ("sigma_double", 0.035, 6.0, 0.02),
                             ("lw_half", 0.0175, 3.0, 0.02), ("lw_double", 0.0175, 12.0, 0.02)]:
    key = (sig, lw)
    if key not in cache:
        cache[key] = pair(sig, lw)
    tg, sA, sC, sS = cache[key]
    jA = P.drive(sA, P.phase_for(sA["anom"], tg, wQ))
    jC = P.drive(sC, P.phase_for(sC["anom"], tg, wQ))
    jS = P.drive(sS, P.phase_for(sS["anom"], tg, wQ))
    mA, mC, mS = thr(jA, tg, zeta), thr(jC, tg, zeta), thr(jS, tg, zeta)
    # phase-minimized held thresholds (24 phases)
    pm = {}
    for nm, s in [("A", sA), ("C", sC), ("S", sS)]:
        chis, th = P.held_threshold_over_phase(s["anom"], tg, mu, 0.01, 8.0, n=700, nchi=24, zeta=zeta)
        pm[nm] = float(np.nanmin(th))
    out[label] = dict(sigma=sig, lw=lw, zeta=zeta, A=mA, C=mC, S=mS, RC=mC/mA, RS=mS/mA,
                      A_pm=pm["A"], C_pm=pm["C"], S_pm=pm["S"], RC_pm=pm["C"]/pm["A"], RS_pm=pm["S"]/pm["A"])
    print("%-12s A %.4f C %.4f S %.4f  R_C %.2f R_S %.2f | phase-min: A %.4f C %.4f S %.4f R_C %.2f R_S %.2f  [%.0fs]" % (
        label, mA, mC, mS, mC/mA, mS/mA, pm["A"], pm["C"], pm["S"], pm["C"]/pm["A"], pm["S"]/pm["A"], time.time()-t0), flush=True)
# coherent product state and occupation-inclusive drives at the base point
tg, sA, sC, sS = cache[(0.0175, 6.0)]
res = {}
for nm, s in [("A", sA), ("C", sC), ("S", sS)]:
    chi = P.phase_for(s["anom"], tg, wQ)
    j_corr = P.drive(s, chi)
    j_tot = j_corr + s["occ"]                                    # TMSS at |M|>>1, or separable mixture with |M| = N
    j_coh = j_corr + s["occ"] + np.real(np.exp(1j*chi)*(s["selfA"] + s["selfB"])) + 2*np.real(s["xdiff"])
    r = {}
    for k, j in [("corr", j_corr), ("total", j_tot), ("coherent", j_coh)]:
        r[k] = thr(j, tg)
    # phase-minimized coherent threshold (scan chi explicitly)
    best = np.inf
    for c in np.linspace(0, 2*np.pi, 12, endpoint=False):
        jj = 2*np.real(np.exp(1j*c)*s["anom"]) + s["occ"] + np.real(np.exp(1j*c)*(s["selfA"]+s["selfB"])) + 2*np.real(s["xdiff"])
        best = min(best, thr(jj, tg))
    r["coherent_phasemin"] = best
    r["max_corr"] = float(abs(j_corr).max()); r["max_occ"] = float(abs(s["occ"]).max())
    r["max_self"] = float(abs(s["selfA"] + s["selfB"]).max()); r["max_xdiff"] = float(abs(2*np.real(s["xdiff"])).max())
    res[nm] = r
    print(nm, {k: round(v, 5) for k, v in r.items()}, "[%.0fs]" % (time.time()-t0), flush=True)
out["controls"] = res
json.dump(out, open(RESULTS / "table2_sensitivity.json", "w"), indent=1)
