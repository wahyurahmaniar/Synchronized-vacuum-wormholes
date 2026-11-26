"""Held-opening threshold at mu = 0.35 against the squeezing phase for each
preparation (writes results/phase_scan_mu0.35.json).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, json, time
from svw import pipeline as P
from svw.paths import RESULTS, load_sources
d = load_sources()
tg = d["tg"]; t0 = time.time()
out = {}
for name in ["A", "C", "S"]:
    chis, th = P.held_threshold_over_phase(d[name]["anom"], tg, 0.35, 0.02, 3.0, n=700, nchi=24)
    k = np.nanargmin(th)
    chi_ov = P.phase_for(d[name]["anom"], tg, 0.7) % (2*np.pi)
    out[name] = dict(chis=chis.tolist(), th=th.tolist(), best_chi=float(chis[k]), best=float(th[k]), chi_overlap=float(chi_ov))
    print(name, "min over phase %.4f at chi=%.2f (overlap chi=%.2f) ; max %.4f ; [%.0fs]" % (th[k], chis[k], chi_ov, np.nanmax(th), time.time()-t0), flush=True)
json.dump(out, open(RESULTS / "phase_scan_mu0.35.json", "w"), indent=1)
