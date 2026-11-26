"""Fig. 5(b): threshold at mu = 0.35 against the carrier-sum detuning, with the
partner carrier shifted alone or both carriers shifted together.
Usage: python scripts/detuning.py partner|sym   (writes results/detuning_MODE.json)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, json, time
from svw import pipeline as P
from svw.paths import RESULTS
mode = sys.argv[1]          # "partner" or "sym"
bg = P.build_background()
tg = np.arange(-150, 175.001, 0.25)
dws = np.round(np.arange(-0.20, 0.2001, 0.01), 3)
t0 = time.time()
PA = P.Packet(bg, 0.35)
res = []
for dw in dws:
    if mode == "partner":
        pa, pb = PA, P.Packet(bg, 0.35 + dw, mirror=True)
    else:
        Om = 0.35 + dw / 2
        pa, pb = P.Packet(bg, Om), P.Packet(bg, Om, mirror=True)
    s = P.sources(bg, tg, pa, pb)
    chi = P.phase_for(s["anom"], tg, 0.7)
    j = P.drive(s, chi)
    mc, mh = P.thresholds(j, tg, 0.35, 0.02, 8.0, n=900, refine=60)
    # truncated window of the original submission, same source
    sel = (tg >= -10) & (tg <= 150)
    mtr = P.thresholds(j[sel], tg[sel], 0.35, 0.005, 8.0, n=900, refine=60)[1]
    # held threshold minimized over 12 phases (dense grid, no refinement)
    chis, thp = P.held_threshold_over_phase(s["anom"], tg, 0.35, 0.02, 8.0, n=500, nchi=12)
    w = abs(P.spectrum(j, tg, 0.7)[0])
    res.append(dict(dw=float(dw), m_cross=float(mc), m_held=float(mh), m_trunc=float(mtr),
                    m_phasemin=float(np.nanmin(thp)), weight_wQ=float(w),
                    jmax=float(abs(j).max())))
    print("%s dw=%+.3f  cross %.4f held %.4f  phase-min %.4f  truncated-window %.4f  |J(wQ)| %.3g  [%.0fs]" % (
        mode, dw, mc, mh, np.nanmin(thp), mtr, w, time.time()-t0), flush=True)
json.dump(res, open(RESULTS / f"detuning_{mode}.json", "w"), indent=1)
