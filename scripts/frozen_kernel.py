"""Table 2, frozen-kernel rows, and the convergence checks: modes and projected
source recomputed on deformed members of the family, and thresholds with the
spatial step, domain, frequency nodes, and time step varied
(writes results/frozen_kernel_convergence.json).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, json, time
from svw import pipeline as P
from svw.paths import RESULTS
t0 = time.time(); mu = 0.35; wQ = 0.7
def tgrid(sig=0.0175, dt=0.25):
    h = 2.7 / sig
    return np.arange(12 - h, 12 + h + 1e-9, dt)
def pm(s, tg, dt=0.05):
    chis, th = P.held_threshold_over_phase(s["anom"], tg, mu, 0.01, 8.0, n=700, nchi=24)
    return float(np.nanmin(th))
out = {"frozen": {}, "conv": {}}
# ---------------- frozen kernel: modes and source on deformed members of the family
ref = None
for da_r0, Q in [(0.0, 1.0), (0.1, 0.0), (0.1, mu), (0.2, 0.0), (0.2, mu)]:
    bg = P.build_background(Q=Q, da=da_r0 * P.R0)
    tg = tgrid()
    PA = P.Packet(bg, 0.35); PB = P.Packet(bg, 0.35, mirror=True)
    PC = P.Packet(bg, 0.55, mirror=True)
    sA = P.sources(bg, tg, PA, PB); sC = P.sources(bg, tg, PA, PC)
    jA = P.drive(sA, P.phase_for(sA["anom"], tg, wQ))
    wA = abs(P.spectrum(jA, tg, wQ)[0])
    mA, mC = pm(sA, tg), pm(sC, tg)
    if ref is None:
        ref = dict(jA=jA, wA=wA, mA=mA, mC=mC)
    dJ = np.sqrt(np.trapezoid((jA - ref["jA"])**2, tg) / np.trapezoid(ref["jA"]**2, tg))
    out["frozen"][f"da{da_r0}_Q{Q}"] = dict(a_throat_over_r0=float(bg["a_throat"] / P.R0), Vmax=float(bg["Vmax"]),
        weight_wQ=float(wA), rel_weight=float(wA / ref["wA"] - 1), rel_L2_source=float(dJ),
        mA=mA, mC=mC, rel_mA=mA / ref["mA"] - 1, rel_mC=mC / ref["mC"] - 1, R=mC / mA)
    print("da=%.1f r0 Q=%.2f a/r0=%.3f Vmax=%.4f |J(wQ)| %.3f (%+.2f%%) L2 %.3f%%  mA %.4f (%+.2f%%) mC %.4f (%+.2f%%) R %.2f [%.0fs]" % (
        da_r0, Q, bg["a_throat"]/P.R0, bg["Vmax"], wA, 100*(wA/ref["wA"]-1), 100*dJ, mA, 100*(mA/ref["mA"]-1), mC, 100*(mC/ref["mC"]-1), mC/mA, time.time()-t0), flush=True)
# ---------------- convergence at mu = 0.35
for label, kw in [("base", {}), ("dx_half", dict(dx=0.03)), ("L_90", dict(L=90.0)), ("nfreq_161", dict(nfreq=161)), ("dt_half", dict(dt=0.125))]:
    bgk = {k: v for k, v in kw.items() if k in ("dx", "L")}
    bg = P.build_background(**bgk)
    pk = {"nfreq": kw["nfreq"]} if "nfreq" in kw else {}
    tg = tgrid(dt=kw.get("dt", 0.25))
    PA = P.Packet(bg, 0.35, **pk); PB = P.Packet(bg, 0.35, mirror=True, **pk); PC = P.Packet(bg, 0.55, mirror=True, **pk)
    sA = P.sources(bg, tg, PA, PB); sC = P.sources(bg, tg, PA, PC)
    jA = P.drive(sA, P.phase_for(sA["anom"], tg, wQ))
    mAo = P.thresholds(jA, tg, mu, 0.01, 8.0, n=900, refine=80)[1]
    mA, mC = pm(sA, tg), pm(sC, tg)
    out["conv"][label] = dict(mA_overlap=float(mAo), mA=mA, mC=mC, R=mC / mA, wA=float(abs(P.spectrum(jA, tg, wQ)[0])))
    print("conv %-9s mA(overlap) %.5f  mA %.5f  mC %.5f  R %.3f  |J(wQ)| %.4f [%.0fs]" % (label, mAo, mA, mC, mC/mA, out["conv"][label]["wA"], time.time()-t0), flush=True)
json.dump(out, open(RESULTS / "frozen_kernel_convergence.json", "w"), indent=1)
