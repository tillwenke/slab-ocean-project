# WRF Tropical Cyclone Analysis

Analysis notebooks and helpers for a WRF tropical-cyclone case study.
Compare two WRF runs (`era5_default` vs `slab_ocean`) against satellite
brightness-temperature observations, and extract the storm track from the
minimum sea-level pressure.

I also changed this.
ciao

## Setup

On the Euler cluster, load Python 3.9 and create a local venv:

```bash
module purge
module load stack/2024-06 gcc/12.2.0 python/3.12

python -m venv /cluster/home/$USER/wcm/wrf-model/notebook/.venv
source /cluster/home/$USER/wcm/wrf-model/notebook/.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Copy `example.env` to `.env` and set your username:

```bash
cp example.env .env
# edit .env and set USERNAME=<your_username>
```

Register the venv as a Jupyter kernel (optional, if you want to pick it
from inside JupyterLab):

```bash
python -m ipykernel install --user --name wrf-tc --display-name "Python (wrf-tc)"
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



## Usage

Activate the venv, then launch Jupyter and open `project.ipynb`:

```bash
source /cluster/home/$USER/wcm/wrf-model/notebook/.venv/bin/activate
jupyter lab
```

Update `user` and the three path variables in the settings cell to point
at your WRF output and observation files. Run the cells top to bottom:
the widget cell exposes a variable dropdown, and the plot cell renders
the interactive animation for that variable.
