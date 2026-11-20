# b25-analysis

Analysis of the **Beerling et al. (2025)** enhanced-weathering study
([doi:10.1038/s41586-024-08429-2](https://doi.org/10.1038/s41586-024-08429-2)) and
of our own re-runs of its model. 

**E0** is the model re-run as published. **E1** and **E2** are post-processes on
E0's saved state, not new runs: E1 applies the thermodynamic calcite ceiling with
dissolution frozen at E0, and E2 holds the modeled pH and soil CO₂ and solves for
carbonate alkalinity. 

## Notebooks

| notebook | purpose |
|---|---|
| `01_e0_carbonate` | E0's carbonate system closure check |
| `02_e2_alkalinity_ceiling` | How much CDR is available at the modeled pH and soil CO2 (E2)? |
| `03_conus_1gt_figures`, `04_conus_2gt_figures` | CONUS-scale figures for the 1 Gt and 2 Gt batches. |
| `postprocess-narrow` | Turns a batch tag into `full.zarr`, `point.zarr` and `state.zarr` under `postproc/<tag>/` |
| `postprocess-dissolution` | Writes `dissolution.zarr`: cation charge applied vs. still locked in unweathered rock |
| `postprocess-all` | Unreduced archive, one Zarr per run |
| `validation` | Checks `point.zarr` against the published deposit, point by point and year by year |



## Layout

```
src/beerling_analysis/
  carbonate.py     the model's carbonate system, its thermodynamic counterpart, and their inversions
  io.py            model runs (S3), published outputs, CLM5 drivers
  mixing.py        per-state mineralogy weight w
  experiments.py   Run view over a saved run, plus the E1 / E2 recomputes
data/              values we transcribed ourselves (see each file's header)
notebooks/         analyses and post-processing
figures/           paper figures
cache/             model runs pulled from S3 (git-ignored, ~240 MB)
```

## Getting started

```bash
uv venv && uv pip install -e .
python -c "from beerling_analysis import io; io.fetch_all()"   # ~240 MB from S3
jupyter lab notebooks/01_e0_carbonate.ipynb
```

You can override paths outside the repo with these environment variables:

- `BEERLING_PUBLISHED`: the published `USA_ERW/data` pickles (Zenodo 14755423)
- `BEERLING_DRIVERS`: the **437-cluster** `run_strc_6.mat`, not the 525-cluster file of the same name
- `BEERLING_POSTPROC`: where `validation` reads the Zarr stores from
- `BEERLING_CACHE`: where fetched runs are stored

## Data

The runs come from
[b25-model-extension](https://github.com/carbonplan/b25-model-extension), which vendors the untouched Zenodo model and documents every change made. 


## License
This work is licensed under the MIT license.

## About us
CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/b25-analysis/issues/new) or [sending us an email](mailto:hello@carbonplan.org).