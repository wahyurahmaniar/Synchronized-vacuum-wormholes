"""Comment 11 of the review: thresholds located with a coarse bracketing scan of
the kind used in the original search reproduce a spurious jump of the A
threshold between mu = 0.300 and 0.325 (writes results/coarse_scan.json).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, time, json
from svw import pipeline as P
from svw.paths import RESULTS, load_sources
d = load_sources()
tg = d["tg"]; wQ = 0.7
chiA = P.phase_for(d["A"]["anom"], tg, wQ)
J = {"A": P.drive(d["A"], chiA),
     "C_Aphase": P.drive(d["C"], chiA),
     "C_own": P.drive(d["C"], P.phase_for(d["C"]["anom"], tg, wQ)),
     "S_own": P.drive(d["S"], P.phase_for(d["S"]["anom"], tg, wQ))}
mus = [0.050, 0.150, 0.250, 0.300, 0.325, 0.350, 0.375, 0.400]
res = {k: [] for k in J}
t0 = time.time()
for mu in mus:
    row = []
    for k, j in J.items():
        m = P.threshold(j, tg, mu, a0=1e-3, a1=1e3, npts=70)
        res[k].append(m); row.append(m)
    print("mu=%.3f  " % mu + "  ".join("%s=%.5g" % (k, v) for k, v in zip(J, row)) +
          "  R(C_own)=%.2f R(S)=%.2f  [%.0fs]" % (row[2]/row[0], row[3]/row[0], time.time()-t0), flush=True)
json.dump(dict(mu=mus, **{k: list(map(float, v)) for k, v in res.items()}), open(RESULTS / "coarse_scan.json", "w"), indent=1)
