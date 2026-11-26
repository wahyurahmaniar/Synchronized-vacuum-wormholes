"""Derivation of the transformed beyond-Horndeski functions from the disformal
rules of Bakopoulos-Charmousis-Kanti (Eqs. 2.15-2.21) with D of Eq. (A.1).

Compares each derived function with the printed form on the physical branch
(ratio 1 means agreement), and evaluates the degeneracy relation
X G5X F4 = 3 F5 (G4 - 2 X G4X) with the y^-3 and the y^-5 forms of F5: the
ratio is 1 for y^-3 and y^2 for y^-5 (Appendix A)."""
import sympy as sp
y, b, a = sp.symbols('y b_1 alpha', positive=True)
Xb = sp.Symbol('Xbar')
s2 = sp.sqrt(2)
D_of_Xb = -b/(2*b*Xb + sp.sqrt(-2*Xb))
# seed
G2b = 8*a*Xb**2; G3b = -8*a*Xb; G4b = 1 + 4*a*Xb; G5b = -4*a*sp.log(-Xb)
sub = {Xb: -y**2}
D   = sp.simplify(D_of_Xb.subs(sub))
DX  = sp.simplify((sp.diff(D_of_Xb, Xb)).subs(sub))
W   = sp.simplify(1 + 2*(-y**2)*D)            # 1 + 2 Xbar D
den = sp.simplify(1 - 2*(y**4)*DX)            # 1 - 2 Xbar^2 D_Xbar
print("D(y)            =", sp.factor(D))
print("1+2XbD          =", sp.factor(W))
print("1-2Xb^2 D_Xb    =", sp.factor(den))
X = sp.factor(-y**2 / W)
print("X = Xb/(1+2XbD) =", X)
G4b_X = sp.diff(G4b, Xb).subs(sub); G5b_X = sp.diff(G5b, Xb).subs(sub)
G3b_X = sp.diff(G3b, Xb).subs(sub)
G2  = sp.simplify(G2b.subs(sub) / sp.sqrt(W))
G4  = sp.simplify(G4b.subs(sub) / sp.sqrt(W))
G5X = sp.simplify(G5b_X * W**sp.Rational(5, 2) / den)
G3X = sp.simplify(G3b_X * W**sp.Rational(5, 2) / den)
F4  = sp.simplify((G4b.subs(sub) - 2*(-y**2)*G4b_X) * DX * W**sp.Rational(5, 2) / (2*den))
F5  = sp.simplify((-y**2) * G5b_X * DX * W**sp.Rational(7, 2) / (6*den))
u = 1 - s2*b*y
paper = {
 "G2":  8*a*y**4*sp.sqrt(u),
 "G3X": 16*a/((3*s2*b*y-2)*sp.sqrt(u)),
 "G4":  (1-4*a*y**2)*sp.sqrt(u),
 "G5X": -8*a/(y**2*(3*s2*b*y-2)*sp.sqrt(u)),
 "F4":  b*(1-2*s2*b*y)*(4*a*y**2+1)/(2**sp.Rational(3,2)*y**3*(3*s2*b*y-2)*u**sp.Rational(5,2)),
 "F5_y3": -s2*a*b*(1-2*s2*b*y)/(3*y**3*(3*s2*b*y-2)*u**sp.Rational(7,2)),
 "F5_y5": -s2*a*b*(1-2*s2*b*y)/(3*y**5*(3*s2*b*y-2)*u**sp.Rational(7,2)),
}
derived = {"G2": G2, "G3X": G3X, "G4": G4, "G5X": G5X, "F4": F4, "F5_y3": F5, "F5_y5": F5}
# compare numerically on the physical branch 0 < y < 1/(sqrt2 b1), b1 = 5.2764772, alpha = 0.1
import random
vals = {b: sp.Float('5.2764772'), a: sp.Rational(1, 10)}
for k in paper:
    ratios = []
    for yy in [0.02, 0.05, 0.08, 0.1, 0.12, 0.13]:
        v = {**vals, y: yy}
        ratios.append(float((derived[k] / paper[k]).subs(v).evalf(30)))
    print(f"{k:6s} derived/printed at y=0.02..0.13:", ["%.12g" % r for r in ratios])
print("F5 derived, factored:", sp.factor(sp.simplify(F5)))
# degeneracy X G5X F4 - 3 F5 (G4 - 2 X G4X), G4X via d/dX = (dX/dy)^-1 d/dy
dXdy = sp.diff(X, y)
G4X = sp.diff(G4, y) / dXdy
for name, F5c in [("y3", paper["F5_y3"]), ("y5", paper["F5_y5"])]:
    lhs = X*G5X*F4; rhs = 3*F5c*(G4 - 2*X*G4X)
    ratios = [float((lhs / rhs).subs({**vals, y: yy}).evalf(30))
              for yy in [0.02, 0.05, 0.08, 0.1, 0.12, 0.13]]
    print("degeneracy X G5X F4 / [3 F5 (G4 - 2X G4X)] with F5_%s:" % name,
          ["%.12g" % r for r in ratios])
