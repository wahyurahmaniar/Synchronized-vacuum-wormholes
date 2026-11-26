"""Projected sources of the three preparations on the exact background (mu-free):
A (carriers 0.35 + 0.35), C_Delta (0.35 + 0.55), and C_Delta^sym (0.45 + 0.45),
in units c^3/GM. Prints the barrier, Klein-Gordon norms, transmission at the
carriers, and the spectral content of each drive; writes results/sources.npz,
which every threshold script reads.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, time
from svw import pipeline as P
from svw.paths import save_sources
t0 = time.time()
bg = P.build_background()
print("Vmax=%.5f at x=%.3f  V(0)=%.5f  r0=%.5f" % (bg["Vmax"], bg["xVmax"], bg["V0"], P.R0), flush=True)
tg = np.arange(-150, 175.001, 0.25)
PA = P.Packet(bg, 0.35)
PB = P.Packet(bg, 0.35, mirror=True)
PC = P.Packet(bg, 0.55, mirror=True)
PS = P.Packet(bg, 0.45); PS2 = P.Packet(bg, 0.45, mirror=True)
print("modes %.0fs; |T|^2 at carrier: 0.35 %.4f 0.45 %.4f 0.55 %.4f" % (time.time()-t0,
      np.interp(0.35, PA.w, PA.T2), np.interp(0.45, PS.w, PS.T2), np.interp(0.55, PC.w, PC.T2)), flush=True)
for t in [-100.0, 0.0, 12.0, 60.0]:
    # unit-norm packets; the printed norm is the part inside the domain |x| < 60
    print("t=%6.1f  KG norm inside the domain: A %.4f  B %.4f   |(A,B)_KG| = %.1e" % (
          t, P.kg_inner(bg, PA, PA, t).real, P.kg_inner(bg, PB, PB, t).real,
          abs(P.kg_inner(bg, PA, PB, t))), flush=True)
srcA = P.sources(bg, tg, PA, PB, coherent=True)
srcC = P.sources(bg, tg, PA, PC, coherent=True)
srcS = P.sources(bg, tg, PS, PS2, coherent=True)
print("sources %.0fs" % (time.time()-t0), flush=True)
save_sources(tg, dict(A=srcA, C=srcC, S=srcS), Vmax=bg["Vmax"])
wQ = 0.7
wb = np.linspace(0.001, 2, 700)
for name, s, wsum in [("A", srcA, 0.7), ("C", srcC, 0.9), ("S", srcS, 0.9)]:
    chi = P.phase_for(s["anom"], tg, wQ)
    j = P.drive(s, chi)
    S = np.abs(P.spectrum(j, tg, wb))
    occS = np.abs(P.spectrum(s["occ"], tg, wb))
    print(f"{name}: max|j|={abs(j).max():.4g}  max|occ|={abs(s['occ']).max():.4g}  "
          f"peak j at {wb[S.argmax()]:.3f} (|J|={S.max():.4g})  |J(wQ)|={abs(P.spectrum(j,tg,wQ)[0]):.4g}  "
          f"|J(sum)|={abs(P.spectrum(j,tg,wsum)[0]):.4g}  occ peak {wb[occS.argmax()]:.3f} |occ(wQ)|={abs(P.spectrum(s['occ'],tg,wQ)[0]):.3g}  "
          f"int j dt / max|j| = {np.trapezoid(j,tg)/abs(j).max():.2e}", flush=True)
    print("   self/xdiff max: %.3g %.3g %.3g" % (abs(s["selfA"]).max(), abs(s["selfB"]).max(), abs(s["xdiff"]).max()))
print("j(t) at window edges relative to max: A %.1e %.1e" % (abs(P.drive(srcA,0)[0])/abs(srcA['anom']).max(), abs(P.drive(srcA,0)[-1])/abs(srcA['anom']).max()))
