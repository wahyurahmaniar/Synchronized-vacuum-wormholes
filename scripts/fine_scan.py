"""Fine scan in mu with the phase optimized (16 phases): continuity across
0.29 <= mu <= 0.34 and the 11 percent jump of the A threshold between
mu = 0.3925 and 0.395 (Section 7.2).
Usage: python scripts/fine_scan.py ZETA MU1,MU2,... LABEL
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, json, time
from svw import pipeline as P
from svw.paths import RESULTS, load_sources
d = load_sources()
tg0 = d["tg"]
# use the sigma-based window used in parts 2-3 (12 +- 2.7/sigma)
sel = (tg0 >= 12 - 2.7/0.0175 - 1e-9) & (tg0 <= 12 + 2.7/0.0175 + 1e-9)
tg = tg0[sel]
S = {k: {kk: (vv[sel] if isinstance(vv, np.ndarray) else vv) for kk, vv in d[k].items()} for k in ["A", "C", "S"]}
zeta = float(sys.argv[1]); mus = [float(x) for x in sys.argv[2].split(",")]
t0 = time.time(); res = []
for mu in mus:
    row = {"mu": mu}
    for k in ["A", "C", "S"]:
        chis, th = P.held_threshold_over_phase(S[k]["anom"], tg, mu, 0.005, 8.0, n=650, nchi=16, zeta=zeta)
        i = int(np.nanargmin(th)); row[k] = float(th[i]); row[k + "_chi"] = float(chis[i])
    row["RC"] = row["C"] / row["A"]; row["RS"] = row["S"] / row["A"]
    res.append(row)
    print("zeta=%.2f mu=%.4f  A %.5f  C %.5f  S %.5f  R_C %.2f  R_S %.2f  [%.0fs]" % (zeta, mu, row["A"], row["C"], row["S"], row["RC"], row["RS"], time.time()-t0), flush=True)
json.dump(res, open(RESULTS / f"fine_scan_z{zeta}_{sys.argv[3]}.json", "w"), indent=1)
