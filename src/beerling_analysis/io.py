"""Loading: model runs from S3, published outputs, and the CLM5 drivers.

Three sources:

  RUNS       our own re-runs of the model, on S3. Carry `totals.diag`, which the
             published outputs have no equivalent of.
  PUBLISHED  Zenodo 14755423, the paper's per-cell outputs. 15 flux files (we didn't 
             find calcite in any of them).
  DRIVERS    the 437-cluster `run_strc_6.mat`. NOTE there are two files with that
             name in the code deposit and only this one reproduces the published
             outputs.
"""
from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import scipy.io as sio

S3_PREFIX = ("s3://carbonplan-carbon-removal/ew-workflows-data/"
             "beerling-2025-model-extension/results/e0-sites/")

_REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("BEERLING_CACHE", _REPO / "cache"))
DATA = _REPO / "data"

# Overridable because these live outside the repo (they are large and not ours).
PUBLISHED_DIR = Path(os.environ.get(
    "BEERLING_PUBLISHED", Path.home() / "Downloads" / "USA_ERW" / "data"))
DRIVERS = Path(os.environ.get(
    "BEERLING_DRIVERS",
    Path.home() / "Downloads" / "weathering_model" / "run_strc_6.mat"))

#: The six representative driver clusters, and the deployment each was run for.
#: `drv` is 1-BASED, matching mylist{drv} and run_site's argument. Mixing this up
#: (learned the hard way it's easy to mix up with python's 0-based indexing).
SITES = {
    6:   dict(state="Virginia",       start=2043, years=28, crop="Maize",        where="Missouri",  stratum="acid + wet + hot"),
    390: dict(state="North_Carolina", start=2043, years=28, crop="Maize",        where="Indiana",   stratum="high efficiency"),
    214: dict(state="Minnesota",      start=2035, years=36, crop="Maize",        where="Iowa",      stratum="very wet"),
    276: dict(state="Oregon",         start=2056, years=15, crop="Spring wheat", where="Montana",   stratum="dry + alkaline"),
    180: dict(state="Minnesota",      start=2026, years=45, crop="Spring wheat", where="Minnesota", stratum="cold"),
    162: dict(state="Wisconsin",      start=2046, years=25, crop="Soy",          where="Illinois",  stratum="median"),
}

G_PER_M2_TO_T_PER_HA = 0.01   # 1 g/m2 = 0.01 t/ha


def run_filename(drv, cluster, experiment="E0"):
    s = SITES[drv]
    return f"{experiment}_drv{drv}_{s['state']}_{cluster}_{s['years']}yr.mat"


def fetch(name, quiet=True):
    """Download one result from S3 into the cache if not already there."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists():
        return dest
    cmd = ["aws", "s3", "cp", S3_PREFIX + name, str(dest)]
    if quiet:
        cmd.append("--quiet")
    subprocess.run(cmd, check=True)
    return dest


def fetch_all(experiment="E0", quiet=True):
    """Populate the cache with every run for the six sites (~250 MB for E0)."""
    out = []
    for drv in SITES:
        for cluster in (1, 2):
            out.append(fetch(run_filename(drv, cluster, experiment), quiet=quiet))
    return out


def load_run(drv, cluster, experiment="E0"):
    """Load one model run. Returns the raw mat_struct namespace.

    Time series are on fd15's regular grid: 100 steps per simulated year, so
    index 100*k is the end of year k. Depth is 31 nodes over 0-0.15 m.
    """
    path = fetch(run_filename(drv, cluster, experiment))
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def load_published(flux):
    """Load one published flux pickle, e.g. 'CO2', 'rock', 'pH', 'P_mass'."""
    p = PUBLISHED_DIR / f"compiled_run_{flux}_a_b_v6_varz.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Set BEERLING_PUBLISHED to the USA_ERW/data directory "
            "(Zenodo 14755423)."
        )
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _unwrap(e):
    while isinstance(e, np.ndarray) and e.dtype == object and e.size >= 1:
        e = e.ravel()[0]
    return e


def load_drivers():
    """The 437 driver clusters. Returns (list_of_clusters, {(crop, lind): drv}).

    The returned index is 1-BASED. See the note on SITES.
    """
    if not DRIVERS.exists():
        raise FileNotFoundError(
            f"{DRIVERS} not found. Set BEERLING_DRIVERS to the 437-cluster "
            "run_strc_6.mat (the one at the TOP of weathering_model/, not the "
            "525-cluster copy under particle_weathering_data/)."
        )
    ml = sio.loadmat(DRIVERS, squeeze_me=False, struct_as_record=False)["mylist"][0]
    clusters = [_unwrap(e) for e in ml]
    if len(clusters) != 437:
        raise ValueError(
            f"{DRIVERS} has {len(clusters)} clusters, not 437 -- this is the wrong "
            "run_strc_6.mat. See beerling-2025-model-extension/notes.md D6."
        )
    index = {}
    for i, e in enumerate(clusters):
        crop = str(np.atleast_1d(e.Crop).ravel()[0])
        for lind in np.atleast_1d(e.lind).ravel():
            index[(crop, int(lind))] = i + 1
    return clusters, index


def published_series(flux, drv, state, start, index=None, column="N2"):
    """The published series for one cluster + source state + deployment year.

    Returns (years, per-hectare values). Points in a cluster sharing a source
    state are identical per hectare to ~1e-16, so any member will do; this
    returns the first and does not average.
    """
    if index is None:
        _, index = load_drivers()
    db = load_published(flux)
    key = f"{flux}_a_b"
    years = np.array([int(str(x)[:4]) for x in db[next(iter(db))][key].index])
    want = state.replace("_", " ")
    for v in db.values():
        crop = str(np.atleast_1d(v["nam"]).ravel()[0])
        if index.get((crop, int(v["lind"]))) != drv:
            continue
        for S in (1, 2):
            if str(np.atleast_1d(v["bas"]).ravel()[S - 1]) != want:
                continue
            if int(np.atleast_1d(v["yy"]).ravel()[S - 1]) != start:
                continue
            n = {"N1": 1, "N2": 2, "N3": 3}[column]
            vals = v[key][f"S{S}_U1_N{n}_p100"].values
            per_ha = vals / float(v["ar"]) if flux not in ("pH",) else vals
            return years, per_ha
    raise LookupError(f"No published cell for drv={drv}, {state}, start={start}.")
