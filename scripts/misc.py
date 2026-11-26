"""Anharmonic frequency of the seed well, spectral weights of the drives in one
normalization, and sample values of the addressing distance D(T)
(writes results/misc.json).
"""
import numpy as np, json
from scipy.integrate import quad
from scipy.optimize import brentq
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svw import pipeline as P
from svw.paths import RESULTS, load_sources
out = {}
# ---- (b) amplitude-dependent frequency of the seed well at mu = 0.35, omega_Q = 0.7
mu, w = 0.35, 0.7
U = lambda q: (w**2/mu) * (q**4/4 - (1+mu)*q**3/3 + mu*q**2/2)
dU = U(mu)
print("Delta U / w^2 = %.5f  U(1)/w^2 = %.5f" % (dU/w**2, U(1.0)/w**2))
freqs = {}
for frac in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    E = frac * dU
    qr = brentq(lambda q: U(q) - E, 1e-9, mu)          # right turning point below barrier
    ql = brentq(lambda q: U(q) - E, -1.0, -1e-9)
    T = 2 * quad(lambda q: 1/np.sqrt(2*(E - U(q))), ql, qr, limit=200)[0]
    freqs[frac] = (2*np.pi/T, ql, qr)
    print("E = %.2f DU : omega(E) = %.4f (%.3f omega_Q), turning points %.3f %.3f" % (frac, 2*np.pi/T, 2*np.pi/T/w, ql, qr))
out["omega_E"] = {str(k): v for k, v in freqs.items()}
# ---- (c) spectral weights in one normalization (per unit reduced amplitude m, or per unit N)
d = load_sources()
tg0 = d["tg"]; sel = (tg0 >= 12 - 2.7/0.0175 - 1e-9) & (tg0 <= 12 + 2.7/0.0175 + 1e-9); tg = tg0[sel]
wb = np.linspace(0.001, 2.0, 2000)
sw = {}
for k, ssum in [("A", 0.7), ("C", 0.9), ("S", 0.9)]:
    an = d[k]["anom"][sel]; occ = d[k]["occ"][sel]
    chi = P.phase_for(an, tg, 0.7)
    j = 2*np.real(np.exp(1j*chi)*an)
    Sj = np.abs(P.spectrum(j, tg, wb)); So = np.abs(P.spectrum(occ, tg, wb))
    sw[k] = dict(peak=float(wb[Sj.argmax()]), peak_val=float(Sj.max()),
                 at_wQ=float(abs(P.spectrum(j, tg, 0.7)[0])), at_sum=float(abs(P.spectrum(j, tg, ssum)[0])),
                 occ_peak=float(wb[So.argmax()]), occ_at_wQ=float(abs(P.spectrum(occ, tg, 0.7)[0])),
                 occ_at_0=float(abs(P.spectrum(occ, tg, 0.001)[0])),
                 jmax=float(abs(j).max()), occmax=float(abs(occ).max()),
                 self_max=float(abs(d[k]["selfA"][sel] + d[k]["selfB"][sel]).max()),
                 transit=float(abs(np.trapezoid(j, tg))/abs(j).max()))
    print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in sw[k].items()})
out["spectral"] = sw
# FWHM of the A spectrum around its peak
SA = np.abs(P.spectrum(2*np.real(np.exp(1j*P.phase_for(d["A"]["anom"][sel], tg, 0.7))*d["A"]["anom"][sel]), tg, wb))
half = SA >= SA.max()/2; idx = np.flatnonzero(half)
print("A spectral FWHM: %.4f (from %.4f to %.4f)" % (wb[idx[-1]]-wb[idx[0]], wb[idx[0]], wb[idx[-1]]))
out["A_fwhm"] = float(wb[idx[-1]]-wb[idx[0]])
# ---- (d) addressing functional for isometry-related candidates
sig = 0.0175
def D_shift(T, N):
    return P.thermal_trace_distance(N*np.exp(-sig**2*T**2), N)
for N in [np.sinh(1.0)**2, 1e3, 1e66]:
    vals = [(T, D_shift(T, N)) for T in [0, 10, 20, 40, 60, 100]]
    print("N=%.3g  D(T): " % N + " ".join("T=%g:%.4f" % (T, v) for T, v in vals), " ; carrier-shifted D = N/(N+1) = %.6f" % (N/(N+1)))
out["D_shift_rs1"] = [(T, D_shift(T, np.sinh(1.0)**2)) for T in np.linspace(0, 120, 121)]
out["D_shift_large"] = [(T, D_shift(T, 1e66)) for T in np.linspace(0, 120, 121)]
# large-N limit: trace distance between geometric distributions -> exponentials with ratio q = n1/n2
q = np.exp(-sig**2*40**2)
print("large-N analytic check at T=40: q=%.4f  D = q^{q/(1-q)} - q^{1/(1-q)} = %.4f" % (q, q**(q/(1-q)) - q**(1/(1-q))))
json.dump(out, open(RESULTS / "misc.json", "w"), indent=1, default=float)
