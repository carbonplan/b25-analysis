"""The mineralogy mixture: turning two model runs into a published-comparable number.

The model runs once per mineralogy cluster, and the published results (`N2`) are
a weighted mixture of the mineralogy clusters, where

    N2(t) = w * run_1(t) + (1 - w) * run_2(t)

and `w` is a property of the source state: the fraction of that state's rock
analyses falling in cluster 1. We confirmed this holds against five independently
measured weights (to <3.1e-5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_COUNTS = Path(__file__).resolve().parents[2] / "data" / "mineralogy_cluster_counts.csv"


def cluster_counts():
    """The extdat Fig. 3a sample counts. Hand-transcribed -- see the header of the CSV."""
    d = pd.read_csv(_COUNTS, comment="#")
    assert (d.n_1 + d.n_2 == d.n_total).all(), "transcription check failed"
    return d


def weight(state):
    """Mixing weight w for a source state, e.g. 'Virginia' or 'North_Carolina'."""
    d = cluster_counts()
    s = state.replace("_", " ")
    row = d[d.state == s]
    if row.empty:
        raise KeyError(f"no cluster counts for '{state}'; have {list(d.state)}")
    r = row.iloc[0]
    return r.n_1 / (r.n_1 + r.n_2)


def mix(a, b, state):
    """Combine cluster-1 and cluster-2 series into the published-N2 equivalent."""
    w = weight(state)
    return w * np.asarray(a, dtype=float) + (1 - w) * np.asarray(b, dtype=float)


def recover_weight(a, b, published_n2):
    """Solve for w at each time from a run pair and the published N2.

    A constant w is the validation, not the fit: the a/b ratio moves 10-30% across
    a deployment window, so one weight matching every year constrains the shape and
    scale of both trajectories independently.
    """
    a, b, n2 = (np.asarray(x, dtype=float) for x in (a, b, published_n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (n2 - b) / (a - b)
