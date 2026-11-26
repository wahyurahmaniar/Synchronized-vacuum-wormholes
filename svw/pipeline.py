"""Activation numerics of the revised manuscript.

Conventions
-----------
* Units G = c = hbar = 1 with M = 1 (lengths in GM/c^2, frequencies in
  c^3/GM).  The throat radius is r0 = 2.6382386 in these units.
* Packets carry unit Klein-Gordon norm: the delta-normalized scattering modes
  satisfy (psi_w, psi_w')_KG = 2 w delta(w - w'), so the spectral amplitude is
  g(w)/sqrt(2w) with int |g|^2 dw = 1.
* Deformation family r(l;Q)^2 = l^2 + r0^2 - 2 r0 da (1-Q) K(x(l)),
  K(x) = exp(-x^2 / 2 lw^2) in the tortoise coordinate, so r r_Q = r0 da K(x)
  exactly and for every Q.  The lapse H and the radial function F are those of
  the exact solution, so the tortoise coordinate does not depend on Q.
* The reduced drive per unit reduced amplitude m = eps_d |M| is
  j(t) = 8 pi int drho (r0 K) Delta p_t / |M|, with Delta p_t from
  equation (Tstate): anomalous part 4 Re[M T^theta_theta[psi_A, psi_B]].
"""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from . import geometry as geom

R0 = geom.R0                 # 2.6382386 in units of GM/c^2


# ---------------------------------------------------------------- background
def _ell_grid(ell_max=400.0, n=60000):
    s = np.linspace(0.0, 1.0, n)
    a = 7.0
    return ell_max * R0 * np.sinh(a * s) / np.sinh(a)


def build_background(Q=1.0, da=0.0, lw=6.0, L=60.0, dx=0.06, n=60000):
    """Tortoise grid, monopole potential and the fields needed by the packets
    on the member Q of the deformation family (Q = 1 or da = 0: exact
    solution)."""
    ell = _ell_grid(n=n)
    r_ex = np.sqrt(ell**2 + R0**2)
    H = geom.h(r_ex)
    F = geom.F_of_ell(ell)
    rho = np.concatenate(([0.0], cumulative_trapezoid(1.0 / np.sqrt(F), ell)))
    x = np.concatenate(([0.0], cumulative_trapezoid(1.0 / np.sqrt(H), rho)))
    K = np.exp(-x**2 / (2 * lw**2))
    r2 = ell**2 + R0**2 - 2 * R0 * da * (1.0 - Q) * K
    r = np.sqrt(r2)
    # derivatives along ell: dK/dl = K_x / (sqrt(F) sqrt(H))
    dxdl = 1.0 / (np.sqrt(F) * np.sqrt(H))
    Kx = -x / lw**2 * K
    Kxx = (x**2 / lw**4 - 1.0 / lw**2) * K
    Kl = Kx * dxdl
    dxdl_l = np.gradient(dxdl, ell, edge_order=2)
    Kll = Kxx * dxdl**2 + Kx * dxdl_l
    c = 2 * R0 * da * (1.0 - Q)
    r_l = (ell - 0.5 * c * Kl) / r
    r_ll = (1.0 - 0.5 * c * Kll - r_l**2) / r
    H_l = np.gradient(H, ell, edge_order=2)
    F_l = np.gradient(F, ell, edge_order=2)
    # V = H [ (F r_ll + F_l r_l / 2) / r + F H_l r_l / (2 H r) ]
    V = H * ((F * r_ll + 0.5 * F_l * r_l) / r + F * H_l * r_l / (2 * H * r))
    drdx = r_l / dxdl
    xu = np.arange(-L, L + 1e-9, dx)
    ax = np.abs(xu)
    out = {}
    for k, v in (("V", V), ("r", r), ("H", H), ("drdx", drdx)):
        out[k] = CubicSpline(x, v)(ax)
    out["drdx"] *= np.sign(xu)
    out["x"] = xu
    out["Vmax"] = V.max()
    out["xVmax"] = x[np.argmax(V)]
    out["V0"] = V[0]
    out["a_throat"] = r[0]
    out["lw"] = lw
    return out


# --------------------------------------------------------------- mode solver
def scattering_modes(bg, omegas):
    xu, Vsp = bg["x"], CubicSpline(bg["x"], bg["V"])
    xL, xR = xu[0], xu[-1]
    U = np.empty((len(omegas), len(xu)), complex)
    dU = np.empty_like(U)
    T2 = np.empty(len(omegas))
    for i, w in enumerate(omegas):
        def rhs(x, y):
            return [y[1], (Vsp(x) - w**2) * y[0]]
        y0 = [np.exp(1j * w * xR), 1j * w * np.exp(1j * w * xR)]
        sol = solve_ivp(rhs, (xR, xL), y0, t_eval=xu[::-1],
                        rtol=1e-10, atol=1e-12, method="DOP853")
        u, du = sol.y[0][::-1], sol.y[1][::-1]
        A = np.exp(-1j * w * xL) * (u[0] + du[0] / (1j * w)) / 2.0
        T2[i] = 1.0 / abs(A) ** 2
        U[i] = u / (A * np.sqrt(2 * np.pi))
        dU[i] = du / (A * np.sqrt(2 * np.pi))
    return U, dU, T2


class Packet:
    """Unit-KG-norm monopole packet; left-incident (mirror=False) or its
    throat-reflection image (mirror=True)."""

    def __init__(self, bg, Omega, sigma=0.0175, xm=12.0, nfreq=81, nsig=5.0,
                 mirror=False):
        w = np.linspace(Omega - nsig * sigma, Omega + nsig * sigma, nfreq)
        dw = w[1] - w[0]
        U, dU, self.T2 = scattering_modes(bg, w)
        g = (2 * np.pi * sigma**2) ** -0.25 \
            * np.exp(-(w - Omega) ** 2 / (4 * sigma**2)) \
            * np.exp(1j * w * xm) / np.sqrt(2 * w)
        r, H, drdx = bg["r"], bg["H"], bg["drdx"]
        if mirror:
            U, dU, drdx = U[:, ::-1], -dU[:, ::-1], -drdx[::-1]
            r, H = r[::-1], H[::-1]
        self.f = U / (np.sqrt(4 * np.pi) * r)
        dfdx = (dU / r - U * drdx / r**2) / np.sqrt(4 * np.pi)
        self.df = dfdx / np.sqrt(H)            # d/drho
        self.c = g * dw                          # quadrature weights
        self.w, self.Omega, self.sigma = w, Omega, sigma
        self.r = r

    def field(self, t):
        e = self.c * np.exp(-1j * self.w * t)
        return e @ self.f, e @ self.df, (-1j * self.w * e) @ self.f


def kg_inner(bg, P1, P2, t=0.0):
    """(psi_1, psi_2)_KG = i int dx r^2 (psi1^* d_t psi2 - psi2 d_t psi1^*) * 4pi
    in the tortoise chart (sqrt(-g) g^tt = -r^2 sin(theta))."""
    p1, _, t1 = P1.field(t)
    p2, _, t2 = P2.field(t)
    r = bg["r"]
    integrand = 1j * (np.conj(p1) * t2 - p2 * np.conj(t1)) * r**2 * 4 * np.pi
    return np.trapezoid(integrand, bg["x"])


# ------------------------------------------------------------------ sources
def sources(bg, tgrid, PA, PB, lw=None, coherent=False):
    """Projected drives on the time grid.

    Returns a dict with
      'anom' : complex, J_code(t) = 8 pi int drho rrQ P_AB, P_AB = 2 T^th_th[psi_A,psi_B]
               (the anomalous drive per unit |M| is 2 Re[e^{i chi} J_code])
      'occ'  : real, O_A + O_B with O_I = 8 pi int drho rrQ 2 T^th_th[psi_I, psi_I^*]
               (the occupation drive is N (O_A + O_B))
    and, if coherent=True, the extra coherent-state terms
      'selfA','selfB' : complex, 8 pi int drho rrQ 2 T^th_th[psi_I, psi_I]
      'xdiff'         : complex, 8 pi int drho rrQ 2 T^th_th[psi_A, psi_B^*]
    """
    lw = bg["lw"] if lw is None else lw
    x, H = bg["x"], bg["H"]
    wgt = 8 * np.pi * np.sqrt(H) * R0 * np.exp(-x**2 / (2 * lw**2))
    nt = len(tgrid)
    out = {"anom": np.empty(nt, complex), "occ": np.empty(nt)}
    if coherent:
        for k in ("selfA", "selfB", "xdiff"):
            out[k] = np.empty(nt, complex)
    for k, t in enumerate(tgrid):
        pA, dA, tA = PA.field(t)
        pB, dB, tB = PB.field(t)
        out["anom"][k] = np.trapezoid(wgt * (tA * tB / H - dA * dB), x)
        occ = (np.abs(tA)**2 + np.abs(tB)**2) / H - np.abs(dA)**2 - np.abs(dB)**2
        out["occ"][k] = np.trapezoid(wgt * occ, x)
        if coherent:
            out["selfA"][k] = np.trapezoid(wgt * (tA * tA / H - dA * dA), x)
            out["selfB"][k] = np.trapezoid(wgt * (tB * tB / H - dB * dB), x)
            out["xdiff"][k] = np.trapezoid(
                wgt * (tA * np.conj(tB) / H - dA * np.conj(dB)), x)
    return out


def drive(src, chi):
    """Anomalous drive per unit |M| for squeezing phase chi."""
    return 2.0 * np.real(np.exp(1j * chi) * src["anom"])


def phase_for(Janom, tgrid, wQ):
    return -np.angle(np.trapezoid(Janom * np.exp(1j * wQ * tgrid), tgrid))


def spectrum(J, tgrid, wbar):
    wbar = np.atleast_1d(wbar)
    return np.array([np.trapezoid(J * np.exp(1j * w * tgrid), tgrid)
                     for w in wbar])


# ---------------------------------------------------------- reduced dynamics
def evolve(amp, jt, tgrid, mu, wQ=0.700, zeta=0.02, t_end=900.0, full=False):
    """Q'' + 2 zeta wQ Q' + (wQ^2/mu) Q (Q-mu)(Q-1) = amp * j(t), from rest
    on the seed branch at the first grid time; the drive is the tabulated
    source and vanishes after the last grid time t_off."""
    js = CubicSpline(tgrid, jt)
    t_off = tgrid[-1]

    def rhs(t, y):
        d = amp * js(t) if t <= t_off else 0.0
        return [y[1], -2 * zeta * wQ * y[1]
                - (wQ**2 / mu) * y[0] * (y[0] - mu) * (y[0] - 1.0) + d]
    sol = solve_ivp(rhs, (tgrid[0], t_end), [0.0, 0.0], rtol=1e-8,
                    atol=1e-10, max_step=0.5, dense_output=full)
    if full:
        return sol
    Q, t = sol.y[0], sol.t
    ok = bool((Q > -0.35).all() and (Q < 1.8).all())
    after = t >= t_off
    return dict(qmax=Q.max(), qend=Q[-1], qmin_after=Q[after].min(),
                ok=ok)


def held(amp, jt, tgrid, mu, zeta=0.02, **kw):
    e = evolve(amp, jt, tgrid, mu, zeta=zeta, **kw)
    if not e["ok"]:
        return False
    if zeta > 0:
        return e["qend"] > mu
    return e["qmin_after"] > mu          # conservative case: never recrosses


def classify(amp, jt, tgrid, mu, zeta=0.02, **kw):
    e = evolve(amp, jt, tgrid, mu, zeta=zeta, **kw)
    if not e["ok"]:
        return "window"
    h = (e["qend"] > mu) if zeta > 0 else (e["qmin_after"] > mu)
    if h:
        return "held"
    return "transient" if e["qmax"] > mu else "none"


def threshold(jt, tgrid, mu, a0=1e-3, a1=1e2, npts=90, tol=1e-4, **kw):
    """Smallest amplitude giving held opening: geometric scan, then
    bisection of the first bracket."""
    grid = np.geomspace(a0, a1, npts)
    idx = None
    for i, a in enumerate(grid):
        if held(a, jt, tgrid, mu, **kw):
            idx = i
            break
    if idx in (None, 0):
        return np.nan
    lo, hi = grid[idx - 1], grid[idx]
    while (hi - lo) / hi > tol:
        mid = np.sqrt(lo * hi)
        if held(mid, jt, tgrid, mu, **kw):
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def threshold_seeded(jt, tgrid, mu, seed, span=6.0, npts=60, tol=1e-4, **kw):
    """Threshold searched in [seed/span, seed*span] first, then globally."""
    m = threshold(jt, tgrid, mu, a0=seed / span, a1=seed * span, npts=npts,
                  tol=tol, **kw)
    if not np.isfinite(m) or m <= seed / span * 1.0001:
        m = threshold(jt, tgrid, mu, tol=tol, **kw)
    return m


# ------------------------------------------------------ addressing functional
def thermal_trace_distance(n1, n2):
    """Trace distance between single-mode thermal states with mean occupations
    n1 and n2 (both diagonal in the number basis)."""
    if n1 == n2:
        return 0.0
    if n1 > n2:
        n1, n2 = n2, n1
    if n2 > 1e8:
        # continuum limit: geometric distributions -> exponentials, q = n1/n2
        q = n1 / n2
        if q <= 0.0:
            return 1.0
        return q ** (q / (1 - q)) - q ** (1 / (1 - q))
    l1, l2 = n1 / (n1 + 1.0), n2 / (n2 + 1.0)
    if l1 == 0.0:
        return l2
    kstar = np.log((1 - l1) / (1 - l2)) / np.log(l2 / l1)
    K = int(np.ceil(kstar - 1e-12))
    return l2**K - l1**K


# ------------------------------------------- vectorized fixed-step integrator
def evolve_many(amps, jt, tgrid, mu, wQ=0.700, zeta=0.02, t_end=900.0,
                dt=0.05):
    """Classical RK4 for an array of drive amplitudes at once.  Returns a dict
    of arrays: qmax (over the whole run), qend, qmin_after (t >= t_off), ok."""
    amps = np.asarray(amps, float)
    t0, t_off = tgrid[0], tgrid[-1]
    nsteps = int(round((t_end - t0) / dt))
    th = t0 + 0.5 * dt * np.arange(2 * nsteps + 1)
    js = CubicSpline(tgrid, jt)
    jh = np.where(th <= t_off, js(np.clip(th, t0, t_off)), 0.0)
    k2 = wQ**2 / mu
    g = 2 * zeta * wQ

    def acc(q, v, f):
        return -g * v - k2 * q * (q - mu) * (q - 1.0) + f

    q = np.zeros_like(amps)
    v = np.zeros_like(amps)
    qmax = np.zeros_like(amps)
    qmin_after = np.full_like(amps, np.inf)
    lo = np.zeros(amps.shape, bool)
    hi = np.zeros(amps.shape, bool)
    for n in range(nsteps):
        f0 = amps * jh[2 * n]
        f1 = amps * jh[2 * n + 1]
        f2 = amps * jh[2 * n + 2]
        a1 = acc(q, v, f0)
        q2, v2 = q + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = acc(q2, v2, f1)
        q3, v3 = q + 0.5 * dt * v2, v + 0.5 * dt * a2
        a3 = acc(q3, v3, f1)
        q4, v4 = q + dt * v3, v + dt * a3
        a4 = acc(q4, v4, f2)
        q = q + dt / 6.0 * (v + 2 * v2 + 2 * v3 + v4)
        v = v + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        np.maximum(qmax, q, out=qmax)
        lo |= q < -0.35
        hi |= q > 1.8
        if t0 + (n + 1) * dt >= t_off:
            np.minimum(qmin_after, q, out=qmin_after)
    return dict(qmax=qmax, qend=q, qmin_after=qmin_after, ok=~(lo | hi))


def outcomes(amps, jt, tgrid, mu, zeta=0.02, **kw):
    e = evolve_many(amps, jt, tgrid, mu, zeta=zeta, **kw)
    heldf = (e["qend"] > mu) if zeta > 0 else (e["qmin_after"] > mu)
    cls = np.where(~e["ok"], "W", np.where(heldf, "H",
                   np.where(e["qmax"] > mu, "t", ".")))
    return cls, e


def thresholds(jt, tgrid, mu, a0, a1, n=600, refine=120, zeta=0.02, **kw):
    """Crossing threshold (first amplitude with max Q > mu) and held
    threshold (first amplitude with held opening) on a dense geometric grid
    of n points in [a0, a1], each refined on a linear grid of `refine` points
    inside the bracketing cell.  Returns (m_cross, m_held)."""
    amps = np.geomspace(a0, a1, n)
    cls, e = outcomes(amps, jt, tgrid, mu, zeta=zeta, **kw)
    res = []
    for want in ("cross", "held"):
        sel = (e["qmax"] > mu) if want == "cross" else (cls == "H")
        idx = np.flatnonzero(sel)
        if len(idx) == 0 or idx[0] == 0:
            res.append(np.nan)
            continue
        i = idx[0]
        fine = np.linspace(amps[i - 1], amps[i], refine)
        c2, e2 = outcomes(fine, jt, tgrid, mu, zeta=zeta, **kw)
        s2 = (e2["qmax"] > mu) if want == "cross" else (c2 == "H")
        j = np.flatnonzero(s2)
        res.append(fine[j[0]] if len(j) else amps[i])
    return tuple(res)


def evolve_phases(amps, chis, Janom, tgrid, mu, wQ=0.700, zeta=0.02,
                  t_end=900.0, dt=0.05, extra=None):
    """RK4 over all (chi, amp) pairs for the anomalous drive
    2 Re[e^{i chi} Janom(t)] (+ optional real extra(t) per unit amplitude,
    e.g. the occupation drive).  Returns arrays of shape (len(chis), len(amps))."""
    A, C = np.meshgrid(np.asarray(amps, float), np.asarray(chis, float))
    ca, cb = (2 * A * np.cos(C)).ravel(), (-2 * A * np.sin(C)).ravel()
    ce = A.ravel()
    t0, t_off = tgrid[0], tgrid[-1]
    nsteps = int(round((t_end - t0) / dt))
    th = t0 + 0.5 * dt * np.arange(2 * nsteps + 1)
    inside = th <= t_off
    tc = np.clip(th, t0, t_off)
    ja = np.where(inside, CubicSpline(tgrid, Janom.real)(tc), 0.0)
    jb = np.where(inside, CubicSpline(tgrid, Janom.imag)(tc), 0.0)
    je = np.where(inside, CubicSpline(tgrid, extra)(tc), 0.0) if extra is not None else None
    k2, g = wQ**2 / mu, 2 * zeta * wQ

    def F(i):
        f = ca * ja[i] + cb * jb[i]
        if je is not None:
            f = f + ce * je[i]
        return f

    def acc(q, v, f):
        return -g * v - k2 * q * (q - mu) * (q - 1.0) + f
    q = np.zeros_like(ca); v = np.zeros_like(ca)
    qmax = np.zeros_like(ca); qmin_after = np.full_like(ca, np.inf)
    bad = np.zeros(ca.shape, bool)
    for n in range(nsteps):
        f0, f1, f2 = F(2 * n), F(2 * n + 1), F(2 * n + 2)
        a1 = acc(q, v, f0)
        q2, v2 = q + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = acc(q2, v2, f1)
        q3, v3 = q + 0.5 * dt * v2, v + 0.5 * dt * a2
        a3 = acc(q3, v3, f1)
        q4, v4 = q + dt * v3, v + dt * a3
        a4 = acc(q4, v4, f2)
        q = q + dt / 6.0 * (v + 2 * v2 + 2 * v3 + v4)
        v = v + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        np.maximum(qmax, q, out=qmax)
        bad |= (q < -0.35) | (q > 1.8)
        if t0 + (n + 1) * dt >= t_off:
            np.minimum(qmin_after, q, out=qmin_after)
    sh = A.shape
    return dict(qmax=qmax.reshape(sh), qend=q.reshape(sh),
                qmin_after=qmin_after.reshape(sh), ok=(~bad).reshape(sh))


def held_threshold_over_phase(Janom, tgrid, mu, a0, a1, n=500, nchi=12,
                              zeta=0.02, extra=None):
    """Held-opening threshold for each of nchi phases on a dense geometric
    amplitude grid (resolution (a1/a0)^(1/n)); returns (chis, thresholds)."""
    amps = np.geomspace(a0, a1, n)
    chis = np.linspace(0, 2 * np.pi, nchi, endpoint=False)
    e = evolve_phases(amps, chis, Janom, tgrid, mu, zeta=zeta, extra=extra)
    if zeta > 0:
        H = e["ok"] & (e["qend"] > mu)
    else:
        H = e["ok"] & (e["qmin_after"] > mu)
    th = np.full(nchi, np.nan)
    for k in range(nchi):
        idx = np.flatnonzero(H[k])
        if len(idx) and idx[0] > 0:
            th[k] = np.sqrt(amps[idx[0]] * amps[idx[0] - 1])
    return chis, th
