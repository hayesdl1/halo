"""The six capital-intensity metrics (spec section 4). Higher means more capital intensive.

All inputs are Compustat annual items (dollars in millions, ``emp`` in thousands of
employees, so ``clr`` is millions of dollars per thousand employees). Units do not
matter for percentile ranks, so nothing is rescaled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from halo import METRICS


def compute_metrics(f: pd.DataFrame, ppe: str = "ppent") -> pd.DataFrame:
    """Raw metric values plus two diagnostic flags, one row per input row.

    Invalid-input rules from spec section 4:
    - tai: sale <= 0 or at <= 0 gives NaN; missing intan counts as 0; missing che gives NaN.
    - fai, cxi: sale <= 0 gives NaN. cxi with missing capx is NaN, never 0.
    - fas: at <= 0 gives NaN.
    - clr: emp <= 0 or missing gives NaN.
    - cxl: ebitda <= 0 gives NaN here. ``ebitda_nonpos`` marks those rows so the
      ``nonpositive_ebitda: max_rank`` option can rank them at the top instead.
    ``ebitda`` falls back to ``oibdp`` when missing.
    """
    sale = f["sale"].where(f["sale"] > 0)
    at = f["at"].where(f["at"] > 0)
    intan_missing = f["intan"].isna()
    tangible = at - f["che"] - f["intan"].fillna(0.0)
    plant = f[ppe]
    emp = f["emp"].where(f["emp"] > 0)
    ebitda = f["ebitda"].fillna(f["oibdp"]) if "oibdp" in f else f["ebitda"]
    capx = f["capx"]
    out = pd.DataFrame({
        "tai": tangible / sale,
        "fai": plant / sale,
        "fas": plant / at,
        "clr": tangible / emp,
        "cxi": capx / sale,
        "cxl": capx / ebitda.where(ebitda > 0),
    }, index=f.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    out["intan_filled"] = intan_missing & f["at"].notna()
    out["ebitda_nonpos"] = (ebitda <= 0) & capx.notna()
    return out[[*METRICS, "intan_filled", "ebitda_nonpos"]]
