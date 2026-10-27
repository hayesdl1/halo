"""Static charts for REPORT.md (spec 7.9). Every chart reads a results CSV.

Design rules (from the project's data-viz guidance): one y-axis per chart; thin lines;
quintiles on a single-hue ordered ramp (light = Q1, dark = Q5); two-series charts use
categorical slots 1 and 2 with direct labels as well as a legend; the six weights are
small multiples rather than six colours; recessive grid and axes; text in ink colours,
never in series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal blue, Q1 to Q5
HIST = "#9ec5f4"


def _style(ax, title: str, ylabel: str = ""):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK_2)


def _dates(idx) -> pd.DatetimeIndex:
    return pd.PeriodIndex(idx, freq="M").to_timestamp(how="end")


def quintile_growth(q: pd.DataFrame, path: Path, title: str):
    """q: columns q1..q5 of monthly returns indexed by holding month."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = _dates(q.index)
    for i, col in enumerate(["q1", "q2", "q3", "q4", "q5"]):
        w = (1 + q[col].fillna(0)).cumprod()
        ax.plot(x, w, color=RAMP[i], linewidth=1.6, label=col.upper())
        _end_label(ax, x[-1], w.iloc[-1], col.upper())
    ax.set_yscale("log")
    _style(ax, title, "growth of $1 (log scale)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def two_line_growth(a: pd.Series, b: pd.Series, labels: tuple[str, str], path: Path,
                    title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = _dates(a.index)
    for s, color, lab in [(a, SERIES_1, labels[0]), (b, SERIES_2, labels[1])]:
        w = (1 + s.fillna(0)).cumprod()
        ax.plot(x, w, color=color, linewidth=1.6, label=lab)
        _end_label(ax, x[-1], w.iloc[-1], lab)
    ax.axhline(1, color=INK_2, linewidth=0.8)
    _style(ax, title, "growth of $1")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def weights_path(wp: pd.DataFrame, metrics, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.8), dpi=150, sharex=True, sharey=True)
    for ax, m in zip(axes.flat, metrics, strict=True):
        ax.bar(wp["year"], wp[f"w_{m}"], color=SERIES_1, width=0.7)
        _style(ax, m)
        ax.set_ylim(0, 1)
    fig.suptitle("Tuned weight on each metric, by out-of-sample year", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def random_vs_tuned(rand: pd.Series, tuned: float, replica: float, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.hist(rand.dropna(), bins=40, color=HIST, edgecolor=SURFACE, linewidth=0.8)
    top = ax.get_ylim()[1]
    # Labels at different heights; the smaller value's label sits left of its line.
    marks = sorted([(tuned, SERIES_2, "-", "tuned"), (replica, INK_2, "--", "equal weight")])
    for (v, color, ls, lab), (h, side) in zip(marks, [(0.80, "left"), (0.94, "right")],
                                               strict=True):
        ax.axvline(v, color=color, linewidth=1.6, linestyle=ls)
        ax.annotate(f"{lab} {v:.2f}", (v, top * h), xytext=(-4 if side == "left" else 4, 0),
                    textcoords="offset points", fontsize=8, color=INK_2,
                    ha="right" if side == "left" else "left")
    _style(ax, f"Out-of-sample net Sharpe: {len(rand.dropna()):,} random weightings vs tuned",
           "count")
    ax.set_xlabel("annualised net Sharpe ratio", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def annual_bars(a: pd.Series, b: pd.Series | None, labels: tuple[str, str], path: Path,
                title: str):
    """Calendar-year compounded returns; b (optional) shares years with a."""
    def yearly(s):
        return s.groupby(s.index.year).apply(lambda r: (1 + r.dropna()).prod() - 1)
    ya = yearly(a)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    if b is None:
        ax.bar(ya.index, 100 * ya, color=SERIES_1, width=0.7, label=labels[0])
    else:
        yb = yearly(b).reindex(ya.index)
        x = np.arange(len(ya))
        ax.bar(x - 0.2, 100 * ya, width=0.38, color=SERIES_1, label=labels[0])
        ax.bar(x + 0.2, 100 * yb, width=0.38, color=SERIES_2, label=labels[1])
        ax.set_xticks(x, ya.index, rotation=90)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    _style(ax, title, "return, % per year")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
