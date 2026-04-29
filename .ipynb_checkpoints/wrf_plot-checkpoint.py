"""Helpers for loading WRF output, extracting a variable, and plotting it."""

import glob

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def load_wrf(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No WRF files found for pattern: {pattern}")
    return xr.open_mfdataset(files, combine="nested", concat_dim="Time")


def get_latlon(ds):
    return ds["XLAT"].isel(Time=0), ds["XLONG"].isel(Time=0)


def extract_var(ds, var, t_index):
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    return ds[var].isel(Time=t_index)


def olr_to_tb(olr):
    # Stefan–Boltzmann: Tf = (OLR/sigma)^(1/4); then empirical IR-window conversion.
    sigma = 5.6693e-8
    Tf = (olr / sigma) ** 0.25
    a, b, c = -0.000917, 1.13333, 10.50007
    tb = (-b + np.sqrt(b ** 2 - 4 * a * (c - Tf))) / (2 * a)
    tb.name = "Tb"
    tb.attrs["units"] = "K"
    return tb


def load_obs(obs_file, time_dim="Time", var_name="Tb"):
    ds = xr.open_dataset(obs_file)
    if "time" in ds.dims and time_dim not in ds.dims:
        ds = ds.rename({"time": time_dim})
    if var_name not in ds.data_vars:
        first = list(ds.data_vars)[0]
        ds = ds.rename_vars({first: var_name})
    return ds


def plot_field(data, lat, lon, ax=None, title="", cmap="gray_r",
               vmin=None, vmax=None):
    if ax is None:
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                       vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return im


def extract_and_plot(ds, var, t_index, ax=None, title=None,
                     cmap="gray_r", vmin=None, vmax=None):
    """Extract `var` at `t_index` from `ds` and plot it on a PlateCarree map."""
    lat, lon = get_latlon(ds)
    data = extract_var(ds, var, t_index)
    return plot_field(data, lat, lon, ax=ax, title=title or var,
                      cmap=cmap, vmin=vmin, vmax=vmax), data


def plot_compare(obs, model, lat, lon, var_name="Tb", unit="K",
                 vmin=None, vmax=None, cmap="gray_r"):
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    plot_field(obs, lat, lon, ax=ax1, title=f"Observed {var_name}",
               cmap=cmap, vmin=vmin, vmax=vmax)
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    im = plot_field(model, lat, lon, ax=ax2, title=f"WRF {var_name}",
                    cmap=cmap, vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.20, -0.05, 0.6, 0.03])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(f"{var_name} ({unit})")
    plt.tight_layout()
    return fig
