"""Percentile ranks and the capital intensity score (spec section 4).

Ranks are computed each month within the eligible universe only:
``pct = rank(method="average") / n_non_missing``, so values lie in (0, 1] and ties
share a rank. The score is the weighted mean of the available ranks, and a stock with
fewer than ``min_metrics`` ranks gets no score that month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from halo import METRICS


def pct_ranks(values: pd.DataFrame, groups: list[pd.Series] | pd.Series) -> pd.DataFrame:
    """Percentile rank of each column within each group (for example each month)."""
    out = {}
    for col in values.columns:
        g = values[col].groupby(groups)
        out[col] = g.rank(method="average") / g.transform("count")
    return pd.DataFrame(out, index=values.index)


def rank_inputs(metrics: pd.DataFrame, nonpositive_ebitda: str = "nan") -> pd.DataFrame:
    """Metric values ready to rank. With ``max_rank``, firms with capex but EBITDA <= 0
    get +inf for ``cxl``, so they share the top rank instead of dropping out."""
    vals = metrics[list(METRICS)].copy()
    if nonpositive_ebitda == "max_rank":
        vals.loc[metrics["ebitda_nonpos"].astype(bool), "cxl"] = np.inf
    return vals


def sector_groups(sic: pd.Series, month: pd.Series, min_group: int) -> pd.Series:
    """2-digit SIC group per row; groups with fewer than ``min_group`` members that
    month merge into their 1-digit parent."""
    s = pd.to_numeric(sic, errors="coerce")
    two = (s // 100).astype("Int64").astype(str)
    one = (s // 1000).astype("Int64").astype(str)
    size = two.groupby([month, two]).transform("size")
    return pd.Series(np.where(size >= min_group, "2d_" + two, "1d_" + one), index=sic.index)


def weighted_score(pct: pd.DataFrame, weights, min_metrics: int) -> tuple[pd.Series, pd.Series]:
    """Score and number of available metrics. ``weights`` follows METRICS order."""
    w = np.asarray(weights, dtype="float64")
    p = pct[list(METRICS)].to_numpy(dtype="float64")
    avail = ~np.isnan(p)
    n = avail.sum(axis=1)
    num = np.where(avail, p * w, 0.0).sum(axis=1)
    den = np.where(avail, w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where((n >= min_metrics) & (den > 0), num / den, np.nan)
    return pd.Series(score, index=pct.index), pd.Series(n, index=pct.index)


def equal_weights() -> np.ndarray:
    return np.full(len(METRICS), 1 / len(METRICS))
