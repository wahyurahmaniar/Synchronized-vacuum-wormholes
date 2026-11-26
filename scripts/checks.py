"""Smaller numbers quoted in the text, at mu = 0.35 and zeta = 0.02.

* fractions of held, transient, seed-branch, and inconclusive outcomes of A on
  the amplitude interval [m_crit, 3 m_crit] (Section 7.2);
* threshold of C_Delta at the squeezing phase optimal for A, against its own
  optimal phase (Table 1);
* size of the single-packet second-harmonic terms relative to the cross-term
  drive (Section 4).

Reads results/sources.npz and results/table_z0.02.json; writes results/checks.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from svw import pipeline as P
from svw.paths import RESULTS, load_sources

MU, SIGMA, XM, WQ = 0.35, 0.0175, 12.0, 0.700
d = load_sources()
tg0 = d["tg"]
sel = (tg0 >= XM - 2.7 / SIGMA - 1e-9) & (tg0 <= XM + 2.7 / SIGMA + 1e-9)
tg = tg0[sel]
row = {round(r["mu"], 4): r for r in json.load(open(RESULTS / "table_z0.02.json"))}[MU]
out = {}

# outcome fractions above threshold
jA = 2 * np.real(np.exp(1j * row["A_chi"]) * d["A"]["anom"][sel])
amps = np.linspace(row["A"], 3 * row["A"], 801)
cls, _ = P.outcomes(amps, jA, tg, MU)
frac = {name: float(np.mean(cls == c)) for name, c in
        (("held", "H"), ("transient", "t"), ("seed", "."), ("inconclusive", "W"))}
out["outcome_fractions_A"] = frac
print("A on [m_crit, 3 m_crit]: " + ", ".join("%s %.2f" % kv for kv in frac.items()))

# C_Delta at the phase of A
jC = 2 * np.real(np.exp(1j * row["A_chi"]) * d["C"]["anom"][sel])
mC_Aphase = P.thresholds(jC, tg, MU, 0.3, 1.2, n=600, refine=100)[1]
out["C_at_A_phase"] = float(mC_Aphase)
out["C_own_phase"] = row["C"]
print("C_Delta threshold: %.4f at the phase of A, %.4f at its own optimal phase"
      % (mC_Aphase, row["C"]))

# single-packet terms against the cross-term drive
ratios = {}
for k in ("A", "C", "S"):
    an = d[k]["anom"][sel]
    j = P.drive({"anom": an, "occ": 0 * an.real}, P.phase_for(an, tg, WQ))
    ratios[k] = float(np.abs(d[k]["selfA"][sel] + d[k]["selfB"][sel]).max()
                      / np.abs(j).max())
out["single_packet_over_cross"] = ratios
print("single-packet second-harmonic terms / cross-term drive: "
      + ", ".join("%s %.3f" % kv for kv in ratios.items()))
json.dump(out, open(RESULTS / "checks.json", "w"), indent=1)
