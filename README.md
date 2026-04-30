# WRF Tropical Cyclone Analysis

source ../../venv-wcm/bin/activate

Analysis notebooks and helpers for a WRF tropical-cyclone case study.
Compare two WRF runs (`era5_default` vs `slab_ocean`) against satellite
brightness-temperature observations, and extract the storm track from the
minimum sea-level pressure.

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

## Setup

On the Euler cluster, load Python 3.12 and create a local venv:

```bash
module purge
# Adjust the stack/version to match what your cluster offers for 3.9
module load stack/2024-06 gcc/12.2.0 python/3.9.18 

# Create the new venv
python -m venv /cluster/home/$USER/wcm/wrf-model/notebook/.venv
source /cluster/home/$USER/wcm/wrf-model/notebook/.venv/bin/activate

# Upgrade pip and install
pip install --upgrade pip
pip install -r requirements.txt
```

Register the venv as a Jupyter kernel (optional, if you want to pick it
from inside JupyterLab):

```bash
python -m ipykernel install --user --name wrf-tc --display-name "Python (wrf-tc)"
```
