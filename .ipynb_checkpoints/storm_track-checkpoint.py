"""Extract tropical-cyclone track (min-SLP location) from WRF output."""

from __future__ import annotations

import glob

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from wrf import ALL_TIMES, getvar


def storm_track(wrfout_pattern: str) -> xr.Dataset:
    """Return the coordinates of the lowest SLP at each WRF time step.

    For every time step in every wrfout file matched by the pattern,
    locate the minimum sea-level pressure using ``wrf-python``'s
    ``getvar(..., "slp")`` and record its latitude, longitude, and
    pressure value.

    Parameters
    ----------
    wrfout_pattern : str
        Glob pattern matching one or more wrfout files
        (e.g. ``".../wrfout_d01_*"``).

    Returns
    -------
    xr.Dataset
        Indexed by ``time`` with variables ``lat`` (deg), ``lon`` (deg)
        and ``slp_min`` (hPa). Carries ``pattern`` and
        ``units_slp_min`` attributes.

    Raises
    ------
    FileNotFoundError
        If the pattern matches no files.
    """
    files = sorted(glob.glob(wrfout_pattern))
    if not files:
        raise FileNotFoundError(f"No WRF files found for pattern: {wrfout_pattern}")

    times: list = []
    lats: list = []
    lons: list = []
    slp_min: list = []
    for path in files:
        nc = Dataset(path)
        try:
            slp = getvar(nc, "slp", timeidx=ALL_TIMES)
            if "Time" not in slp.dims:
                slp = slp.expand_dims("Time")

            lat2d = np.asarray(slp.coords["XLAT"])
            lon2d = np.asarray(slp.coords["XLONG"])
            slp_vals = np.asarray(slp)
            time_vals = np.asarray(slp.coords["Time"])

            for t in range(slp_vals.shape[0]):
                frame = slp_vals[t]
                j, i = np.unravel_index(np.argmin(frame), frame.shape)
                times.append(time_vals[t])
                lats.append(lat2d[j, i])
                lons.append(lon2d[j, i])
                slp_min.append(frame[j, i])
        finally:
            nc.close()

    return xr.Dataset(
        {
            "lat":     ("time", np.asarray(lats)),
            "lon":     ("time", np.asarray(lons)),
            "slp_min": ("time", np.asarray(slp_min)),
        },
        coords={"time": np.asarray(times)},
        attrs={"pattern": wrfout_pattern, "units_slp_min": "hPa"},
    )


if __name__ == "__main__":
    import os
    user = os.environ.get("USER", "twenke")
    pattern = f"/cluster/home/{user}/wcm/wrf-model/wrf-output/era5_default/wrfout_d01_*"
    print(storm_track(pattern))
