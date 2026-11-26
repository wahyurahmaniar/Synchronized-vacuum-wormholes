"""Addressing functional for the isometry-related candidate regions (Section 6).

The mirror region displaced by a Killing time T carries a transported mode whose
Klein-Gordon overlap with the partner packet is exp(-sigma^2 T^2), so the
mismatch D(T) is bounded below by the trace distance between single-mode
thermal states of occupations N exp(-sigma^2 T^2) and N. A carrier-shifted
partner gives D >= N/(N+1).

Writes results/addressing.json with D(T) for the reference squeezing r_s = 1
(N = sinh^2 1) and for N >> 1 (N = 1e66, the regime of Section 6.1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from svw import pipeline as P
from svw.paths import RESULTS

SIGMA = 0.0175                      # packet bandwidth, c^3/GM
T = np.arange(0.0, 150.0 + 1e-9, 1.0)
N_RS1 = np.sinh(1.0) ** 2
N_LARGE = 1e66


def D(t, n):
    return P.thermal_trace_distance(n * np.exp(-SIGMA**2 * t**2), n)


out = {"D_rs1": [(float(t), D(t, N_RS1)) for t in T],
       "D_large": [(float(t), D(t, N_LARGE)) for t in T]}
for key, n in (("D_rs1", N_RS1), ("D_large", N_LARGE)):
    vals = dict(out[key])
    print("%-8s N = %.4g: " % (key, n)
          + "  ".join("D(%g) = %.3f" % (t, vals[t]) for t in (20.0, 40.0, 60.0, 100.0))
          + "   carrier-shifted partner: D >= %.6f" % (n / (n + 1)))
json.dump(out, open(RESULTS / "addressing.json", "w"), indent=1)
