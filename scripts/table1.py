"""Table 1: held-opening thresholds m_crit of A, C_Delta, and C_Delta^sym, with the
squeezing phase optimized for each preparation (32 phases, then +-5.6 degrees).
Usage: python scripts/table1.py ZETA MU1,MU2,...   (writes results/table_zZETA.json)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, json, time
from svw import pipeline as P
from svw.paths import RESULTS, load_sources
d = load_sources()
tg0 = d["tg"]
sel = (tg0 >= 12 - 2.7/0.0175 - 1e-9) & (tg0 <= 12 + 2.7/0.0175 + 1e-9)
tg = tg0[sel]
S = {k: d[k]["anom"][sel] for k in ["A", "C", "S"]}
zeta = float(sys.argv[1]); mus = [float(x) for x in sys.argv[2].split(",")]
t0 = time.time(); res = []
for mu in mus:
    row = {"mu": mu}
    for k in ["A", "C", "S"]:
        chis, th = P.held_threshold_over_phase(S[k], tg, mu, 0.005, 8.0, n=520, nchi=32, zeta=zeta)
        i = int(np.nanargmin(th)); chi = chis[i]
        best = np.inf
        for dc in [-0.0982, 0.0, 0.0982]:          # +- 5.6 deg around the best coarse phase
            j = 2*np.real(np.exp(1j*(chi+dc))*S[k])
            m = P.thresholds(j, tg, mu, th[i]*0.85, th[i]*1.15, n=300, refine=100, zeta=zeta)[1]
            if np.isfinite(m) and m < best:
                best, bchi = m, chi + dc
        row[k] = float(best); row[k+"_chi"] = float(bchi % (2*np.pi))
    row["RC"] = row["C"]/row["A"]; row["RS"] = row["S"]/row["A"]
    res.append(row)
    print("zeta=%.2f mu=%.3f A %.5f C %.5f S %.5f R_C %.3f R_S %.3f [%.0fs]" % (zeta, mu, row["A"], row["C"], row["S"], row["RC"], row["RS"], time.time()-t0), flush=True)
json.dump(res, open(RESULTS / f"table_z{zeta}.json", "w"), indent=1)
