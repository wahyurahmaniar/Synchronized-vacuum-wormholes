#!/usr/bin/env bash
# Reproduce every number, table, and figure of the paper, in dependency order.
# Results go to results/, figures to figures/. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

MU_TABLE=0.05,0.1,0.15,0.2,0.25,0.3,0.325,0.35,0.375,0.4
MU_FINE=0.29,0.295,0.3,0.305,0.31,0.315,0.32,0.325,0.33,0.335,0.34,0.385,0.3875,0.39,0.3925,0.395,0.3975,0.4

$PY -m svw.geometry                       # throat values of the exact solution
$PY scripts/f5_derivation.py              # Appendix A, symbolic
$PY scripts/sources.py                    # projected sources -> results/sources.npz
$PY scripts/table1.py 0.02 $MU_TABLE      # Table 1, damped
$PY scripts/table1.py 0.0  $MU_TABLE      # Table 1, conservative
$PY scripts/table2.py                     # Table 2, sensitivity and controls
$PY scripts/frozen_kernel.py              # Table 2, frozen kernel; convergence
$PY scripts/detuning.py partner           # Fig. 5(b)
$PY scripts/detuning.py sym               # Fig. 5(b)
$PY scripts/fine_scan.py 0.02 $MU_FINE fine
$PY scripts/coarse_scan.py
$PY scripts/phase_scan.py
$PY scripts/misc.py
$PY scripts/addressing.py
$PY scripts/physical_scale.py
$PY scripts/checks.py

$PY scripts/fig1_framework.py
$PY scripts/fig2_geometry.py
$PY scripts/fig3_holding.py
$PY scripts/fig45_activation.py
$PY scripts/graphical_abstract.py
