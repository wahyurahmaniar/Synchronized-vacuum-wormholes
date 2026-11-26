"""Repository paths and storage of the projected sources."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
SOURCES = RESULTS / "sources.npz"
FIELDS = ("anom", "occ", "selfA", "selfB", "xdiff")


def save_sources(tg, preps, path=SOURCES, **scalars):
    """Store the time grid and, for each preparation, the projected source
    components returned by pipeline.sources(..., coherent=True)."""
    arrays = {"tg": tg}
    for name, src in preps.items():
        for f in FIELDS:
            arrays[f"{name}_{f}"] = src[f]
    arrays.update({k: np.asarray(v) for k, v in scalars.items()})
    np.savez_compressed(path, **arrays)


def load_sources(path=SOURCES):
    """Inverse of save_sources: {'tg', 'Vmax', 'A', 'C', 'S'}, each
    preparation a dict of the source components."""
    z = np.load(path)
    out = {"tg": z["tg"], "Vmax": float(z["Vmax"])}
    for name in ("A", "C", "S"):
        out[name] = {f: z[f"{name}_{f}"] for f in FIELDS}
    return out
