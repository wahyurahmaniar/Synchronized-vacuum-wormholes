"""Exact beyond-Horndeski holding geometry (Bakopoulos-Charmousis-Kanti branch).

Units: G = c = 1, lengths in units of r0 only where stated explicitly.
Operating point of the manuscript: M = 1, alpha = 1/10, lambda = 1/2.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

MASS = 1.0
ALPHA = 0.1
LAM = 0.5


def r0_of(M=MASS, alpha=ALPHA, lam=LAM):
    disc = M**2 - alpha * lam**3 * (2 - lam) ** 3
    return (M + np.sqrt(disc)) / (lam * (2 - lam))


R0 = r0_of()


def h(r, M=MASS, alpha=ALPHA):
    return 1.0 + r**2 / (2 * alpha) * (1.0 - np.sqrt(1.0 + 8 * alpha * M / r**3))


def dh(r, M=MASS, alpha=ALPHA):
    s = np.sqrt(1.0 + 8 * alpha * M / r**3)
    return r / alpha * (1.0 - s) + r**2 / (2 * alpha) * (12 * alpha * M / (r**4 * s))


def Winv(r, M=MASS, alpha=ALPHA, lam=LAM, r0=None):
    r0 = R0 if r0 is None else r0
    return 1.0 - r0 / (lam * r) * (1.0 - np.sqrt(h(r, M, alpha)))


def f(r, M=MASS, alpha=ALPHA, lam=LAM, r0=None):
    return h(r, M, alpha) * Winv(r, M, alpha, lam, r0)


def df(r, eps=1e-6):
    return (f(r + eps) - f(r - eps)) / (2 * eps)


# ---------------------------------------------------------------- ell chart
def r_of_ell(ell, r0=R0):
    return np.sqrt(ell**2 + r0**2)


def H_of_ell(ell, r0=R0):
    return h(r_of_ell(ell, r0))


def F_of_ell(ell, r0=R0):
    """F = f(r) r^2 / ell^2, with the removable value F(0) = r0 f'(r0)/2."""
    ell = np.asarray(ell, dtype=float)
    r = r_of_ell(ell, r0)
    out = np.empty_like(ell)
    small = np.abs(ell) < 1e-4 * r0
    out[~small] = f(r[~small]) * r[~small] ** 2 / ell[~small] ** 2
    out[small] = r0 * df(r0) / 2.0
    return out


def phi_prime_r(r):
    """phi'(r) = (sqrt(h)-1)/(r sqrt(h)) on the areal chart."""
    sh = np.sqrt(h(r))
    return (sh - 1.0) / (r * sh)


class Background:
    """Tortoise chart, potential and interpolants built once on a fixed grid."""

    def __init__(self, ell_max=400.0, n=40000, r0=R0):
        self.r0 = r0
        # geometric ell grid, dense near the throat
        u = np.linspace(0.0, 1.0, n)
        self.ell = ell_max * r0 * u**3
        self.r = r_of_ell(self.ell, r0)
        self.H = H_of_ell(self.ell, r0)
        self.F = F_of_ell(self.ell, r0)
        # proper distance and tortoise coordinate
        self.rho = np.concatenate(
            ([0.0], cumulative_trapezoid(1.0 / np.sqrt(self.F), self.ell))
        )
        self.x = np.concatenate(
            ([0.0], cumulative_trapezoid(1.0 / np.sqrt(self.H), self.rho))
        )
        # derivatives with respect to proper distance
        drdrho = np.sqrt(self.F) * self.ell / self.r
        Phi = 0.5 * np.log(self.H)
        sp_r = CubicSpline(self.rho, self.r)
        sp_P = CubicSpline(self.rho, Phi)
        self.drdrho = drdrho
        self.d2rdrho2 = sp_r(self.rho, 2)
        self.dPhidrho = sp_P(self.rho, 1)
        self.Phi = Phi
        # monopole potential in the tortoise chart
        self.V0 = self.H * (self.d2rdrho2 / self.r + self.dPhidrho * drdrho / self.r)
        # even/odd extension to the full line
        self.X = np.concatenate((-self.x[:0:-1], self.x))
        self.V0X = np.concatenate((self.V0[:0:-1], self.V0))
        self.rX = np.concatenate((self.r[:0:-1], self.r))
        self.HX = np.concatenate((self.H[:0:-1], self.H))
        self.rhoX = np.concatenate((-self.rho[:0:-1], self.rho))
        self.ellX = np.concatenate((-self.ell[:0:-1], self.ell))
        self.FX = np.concatenate((self.F[:0:-1], self.F))
        self.V_int = CubicSpline(self.X, self.V0X)
        self.r_int = CubicSpline(self.X, self.rX)
        self.H_int = CubicSpline(self.X, self.HX)
        self.rho_int = CubicSpline(self.X, self.rhoX)


def einstein_orthonormal(ell, r0=R0):
    """Orthonormal G_{ab} components in units c^4/(8 pi G r0^2) for ds^2 chart
    -H dt^2 + drho^2 + r^2 dOmega^2, built from the ell chart by splines."""
    r = r_of_ell(ell, r0)
    H = H_of_ell(ell, r0)
    F = F_of_ell(ell, r0)
    rho = np.concatenate(([0.0], cumulative_trapezoid(1.0 / np.sqrt(F), ell)))
    Phi = 0.5 * np.log(H)
    sp_r = CubicSpline(rho, r)
    sp_P = CubicSpline(rho, Phi)
    rp, rpp = sp_r(rho, 1), sp_r(rho, 2)
    Pp, Ppp = sp_P(rho, 1), sp_P(rho, 2)
    rho_ = 2 * rpp / r + (rp**2 - 1.0) / r**2      # -G^t_t
    p_r = 2 * Pp * rp / r + (rp**2 - 1.0) / r**2   # G^r_r
    p_t = Ppp + Pp**2 + Pp * rp / r + rpp / r      # G^th_th
    return rho, -r0**2 * rho_, r0**2 * p_r, r0**2 * p_t


if __name__ == "__main__":
    r0 = R0
    print(f"r0                      = {r0:.7f}   (paper 2.6382386)")
    print(f"h(r0)                   = {h(r0):.7f}   (paper 0.2500000)")
    print(f"Winv(r0)                = {Winv(r0):.3e}")
    print(f"f(r0)                   = {f(r0):.3e}")
    print(f"f'(r0)                  = {df(r0):.7f}   (paper 0.2324026)")
    print(f"F(0)                    = {r0*df(r0)/2:.7f}   (paper 0.3065667)")
    print(f"phi'(0) [ell]           = {LAM/(r0*(LAM-1)):.7f}   (paper -0.3790408)")
    print(f"b1 = r0/lambda          = {r0/LAM:.7f}   (paper 5.2764772)")
    bg = Background()
    print(f"V0max r0^2              = {bg.V0.max()*r0**2:.5f}   (paper 0.0186)")
    ell = np.linspace(1e-6, 40 * r0, 60000)
    rho, rr, pr, pt = einstein_orthonormal(ell)
    print(f"rho_th,p_r,p_t          = {rr[0]:+.3f} {pr[0]:+.3f} {pt[0]:+.3f}"
          f"   (paper +0.387 -1.000 +0.752)")
    print(f"rho+p_r at throat       = {rr[0]+pr[0]:+.3f}   (paper -0.613)")
    print(f"rho+p_t at throat       = {rr[0]+pt[0]:+.3f}   (paper +1.139)")
