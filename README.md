# WRF Tropical Cyclone Analysis

Analysis notebooks and helpers for a WRF tropical-cyclone case study.
Compare two WRF runs (`era5_default` vs `slab_ocean`) against satellite
brightness-temperature observations, and extract the storm track from the
minimum sea-level pressure.

I also changed this.
ciao

## Setup

On the Euler cluster, run the setup script. It creates `.venv` in this
repo folder and registers a Jupyter kernel named "Our Venv":

```bash
./setup_venv.sh
```

Activate the environment with:

```bash
source .venv/bin/activate
```

Copy `example.env` to `.env` and set your username:

```bash
cp example.env .env
# edit .env and set USERNAME=<your_username>
```

## Files

- `project.ipynb` — main notebook. Interactive widget lets you pick a WRF
  surface variable; the plot cell shows an animation with three panels:
  observed Tb, `era5_default`, `slab_ocean`.
- `wrf_plot.py` — helpers: `load_wrf`, `load_obs`, `get_latlon`,
  `extract_var`, `olr_to_tb`, `plot_field`, `extract_and_plot`,
  `plot_time` (animated multi-panel), `plot_track_time`, `plot_compare`.
- `storm_track.py` — `storm_track(wrfout_pattern)` returns an
  `xr.Dataset` with `lat`, `lon`, `slp_min` per time step, using
  `wrf-python`'s `getvar(nc, "slp")`.
- `requirements.txt` — Python dependencies.
- `netcdf_env.sh` — legacy shell setup (kept for reference).