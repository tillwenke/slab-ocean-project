"""Helpers for loading WRF output, extracting a variable, and plotting it."""

from __future__ import annotations

import glob
import math
from typing import Any

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import HTML
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from netCDF4 import Dataset
from wrf import ALL_TIMES, getvar


def load_wrf(pattern: str) -> xr.Dataset:
    """Load one or more WRF ``wrfout`` files into a single dataset.

    Parameters
    ----------
    pattern : str
        Glob pattern matching one or more wrfout files
        (e.g. ``".../wrfout_d01_*"``).

    Returns
    -------
    xr.Dataset
        Datasets concatenated along the ``Time`` dimension.

    Raises
    ------
    FileNotFoundError
        If the pattern matches no files.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No WRF files found for pattern: {pattern}")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="Time")

    slp_frames = []
    for path in files:
        nc = Dataset(path)
        try:
            slp = getvar(nc, "slp", timeidx=ALL_TIMES)
            if "Time" not in slp.dims:
                slp = slp.expand_dims("Time")
            slp_frames.append(np.asarray(slp))
        finally:
            nc.close()
    slp_arr = np.concatenate(slp_frames, axis=0)
    ds["slp"] = (("Time", "south_north", "west_east"), slp_arr,
                 {"units": "hPa", "description": "sea level pressure"})
    return ds


def get_latlon(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return the 2D ``XLAT`` and ``XLONG`` arrays at the first time step.

    Parameters
    ----------
    ds : xr.Dataset
        WRF dataset containing the ``XLAT`` and ``XLONG`` coordinates.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(lat, lon)``, each shaped ``(south_north, west_east)``.
    """
    return ds["XLAT"].isel(Time=0), ds["XLONG"].isel(Time=0)


def extract_var(ds: xr.Dataset, var: str, t_index: int) -> xr.DataArray:
    """Select a single time step of ``var`` from ``ds``.

    Parameters
    ----------
    ds : xr.Dataset
        WRF dataset.
    var : str
        Variable name to extract.
    t_index : int
        Index along the ``Time`` dimension.

    Returns
    -------
    xr.DataArray
        2D field at ``t_index``.

    Raises
    ------
    KeyError
        If ``var`` is not in ``ds``.
    """
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    return ds[var].isel(Time=t_index)


def olr_to_tb(olr: xr.DataArray) -> xr.DataArray:
    """Convert outgoing longwave radiation to brightness temperature.

    Uses the Stefan–Boltzmann law to obtain an effective temperature,
    then applies an empirical IR-window conversion.

    Parameters
    ----------
    olr : xr.DataArray
        Outgoing longwave radiation in W m^-2.

    Returns
    -------
    xr.DataArray
        Brightness temperature in K (``name="Tb"``, ``units="K"``).
    """
    sigma = 5.6693e-8
    Tf = (olr / sigma) ** 0.25
    a, b, c = -0.000917, 1.13333, 10.50007
    tb = (-b + np.sqrt(b ** 2 - 4 * a * (c - Tf))) / (2 * a)
    tb.name = "Tb"
    tb.attrs["units"] = "K"
    return tb


def load_obs(obs_file: str, time_dim: str = "Time",
             var_name: str = "Tb") -> xr.Dataset:
    """Open an observation file and align its dim/variable names to WRF.

    If the file uses ``"time"`` as the time dimension it is renamed to
    ``time_dim``. If ``var_name`` is missing, the first data variable
    is renamed to ``var_name``.

    Parameters
    ----------
    obs_file : str
        Path to the netCDF observation file.
    time_dim : str, default "Time"
        Target name of the time dimension.
    var_name : str, default "Tb"
        Target name of the observation variable.

    Returns
    -------
    xr.Dataset
        Dataset with ``var_name`` accessible via ``ds[var_name]``.
    """
    ds = xr.open_dataset(obs_file)
    if "time" in ds.dims and time_dim not in ds.dims:
        ds = ds.rename({"time": time_dim})
    if var_name not in ds.data_vars:
        first = list(ds.data_vars)[0]
        ds = ds.rename_vars({first: var_name})
    return ds


def plot_field(data: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray,
               ax: Axes | None = None, title: str = "",
               cmap: str = "gray_r", vmin: float | None = None,
               vmax: float | None = None) -> QuadMesh:
    """Plot a 2D field on a PlateCarree map with coastlines and gridlines.

    Parameters
    ----------
    data : xr.DataArray
        2D field to plot.
    lat, lon : xr.DataArray
        2D latitude/longitude grids matching ``data``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new ``PlateCarree`` axis is
        created.
    title : str, default ""
        Axes title.
    cmap : str, default "gray_r"
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color-scale limits.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mappable returned by ``ax.pcolormesh`` (suitable for
        ``plt.colorbar``).
    """
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


def extract_and_plot(ds: xr.Dataset, var: str, t_index: int,
                     ax: Axes | None = None, title: str | None = None,
                     cmap: str = "gray_r", vmin: float | None = None,
                     vmax: float | None = None
                     ) -> tuple[QuadMesh, xr.DataArray]:
    """Extract ``var`` at ``t_index`` and plot it on a PlateCarree map.

    Parameters
    ----------
    ds : xr.Dataset
        WRF dataset.
    var : str
        Variable name to plot.
    t_index : int
        Index along the ``Time`` dimension.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If omitted, a new ``PlateCarree`` axis is
        created.
    title : str, optional
        Axes title (defaults to ``var``).
    cmap : str, default "gray_r"
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color-scale limits.

    Returns
    -------
    tuple[matplotlib.collections.QuadMesh, xr.DataArray]
        ``(im, data)`` — the mappable and the extracted 2D field.
    """
    lat, lon = get_latlon(ds)
    data = extract_var(ds, var, t_index)
    return plot_field(data, lat, lon, ax=ax, title=title or var,
                      cmap=cmap, vmin=vmin, vmax=vmax), data


def plot_time(data: dict[str, xr.DataArray], fig_title: str = "",
              plot_size: float = 4, cols: int = 4, cmap: str = "gray_r",
              vmin: float | None = None, vmax: float | None = None,
              time_dim: str = "Time", interval: int = 250,
              track: xr.Dataset | None = None,
              lat: xr.DataArray | None = None,
              lon: xr.DataArray | None = None,
              radius: float | None = None,
              coastlines: bool = True,
              cut_line: bool = False) -> HTML:
    """Build an animated multi-panel ``imshow`` of fields over time.

    If ``track`` is given together with the 2D WRF ``lat``/``lon`` grids,
    a red dot is drawn on every panel at the storm center for each
    frame. If ``radius`` (km) is also given, a light circle of that
    radius is drawn around the center.

    Parameters
    ----------
    data : dict[str, xr.DataArray]
        Mapping of panel title to a 3D DataArray whose first dimension
        is ``time_dim``.
    fig_title : str, default ""
        Figure suptitle.
    plot_size : float, default 4
        Size (in inches) of each panel.
    cols : int, default 4
        Number of panel columns.
    cmap : str, default "gray_r"
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color-scale limits applied to all panels.
    time_dim : str, default "Time"
        Name of the time dimension in each DataArray.
    interval : int, default 250
        Animation frame interval in milliseconds.
    track : xr.Dataset, optional
        Storm track from ``storm_track`` with ``lat``/``lon``/``time``.
    lat, lon : xr.DataArray, optional
        2D WRF latitude/longitude grids; required when ``track`` is
        passed so that the (lat, lon) center can be mapped to a pixel.
    radius : float, optional
        If given (km) and ``track`` is also provided, draws a light
        circle of this radius around the storm center on every panel.

    Returns
    -------
    IPython.display.HTML
        A jshtml animation with scrubbable controls.
    """
    first = next(iter(data.values()))
    n_frames = first.sizes[time_dim]
    assert all(arr.sizes[time_dim] == n_frames for arr in data.values())

    marker_ij = None
    radius_px: float | None = None
    if track is not None:
        assert lat is not None and lon is not None, (
            "`lat` and `lon` are required when passing `track`."
        )
        assert track.sizes["time"] == n_frames, (
            f"Track has {track.sizes['time']} steps, data has {n_frames}."
        )
        lat_np = np.asarray(lat)
        lon_np = np.asarray(lon)
        ctr_lat = np.asarray(track["lat"])
        ctr_lon = np.asarray(track["lon"])
        marker_ij = np.empty((n_frames, 2), dtype=int)
        for t in range(n_frames):
            d2 = (lat_np - ctr_lat[t]) ** 2 + (lon_np - ctr_lon[t]) ** 2
            j, i = np.unravel_index(np.argmin(d2), d2.shape)
            marker_ij[t] = (j, i)

        if radius is not None and radius > 0:
            dx_km = _haversine_km(lat_np[:, :-1], lon_np[:, :-1],
                                  lat_np[:, 1:], lon_np[:, 1:]).mean()
            dy_km = _haversine_km(lat_np[:-1, :], lon_np[:-1, :],
                                  lat_np[1:, :], lon_np[1:, :]).mean()
            radius_px = float(radius / (0.5 * (dx_km + dy_km)))

    panels: list[tuple[str, xr.DataArray, str, float | None, float | None]] = [
        (title, arr, cmap, vmin, vmax) for title, arr in data.items()
    ]
    if len(data) >= 2:
        items = list(data.items())
        (t0, a0), (t1, a1) = items[0], items[1]
        diff = a0 - a1
        dmax = float(np.nanmax(np.abs(np.asarray(diff))))
        panels.append((f"{t0} − {t1}", diff, "RdBu_r", -dmax, dmax))

    cols = min(cols, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig = plt.figure(figsize=(plot_size * cols, plot_size * rows))
    fig.suptitle(fig_title, fontsize=14)

    use_geo = lat is not None and lon is not None
    lat_np = np.asarray(lat) if use_geo else None
    lon_np = np.asarray(lon) if use_geo else None

    ims, arrs, dots, circles, cut_lines = [], [], [], [], []
    for i, (title, arr, panel_cmap, p_vmin, p_vmax) in enumerate(panels):
        if use_geo:
            ax = fig.add_subplot(rows, cols, i + 1,
                                 projection=ccrs.PlateCarree())
            ax.set_title(title)
            im = ax.pcolormesh(lon_np, lat_np, np.asarray(arr.isel({time_dim: 0})),
                               transform=ccrs.PlateCarree(), cmap=panel_cmap,
                               vmin=p_vmin, vmax=p_vmax, shading="auto")
            if coastlines:
                ax.coastlines(linewidth=0.6)
            gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                              color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title)
            im = ax.imshow(arr.isel({time_dim: 0}), origin="lower",
                           cmap=panel_cmap, vmin=p_vmin, vmax=p_vmax)
        plt.colorbar(im, ax=ax, orientation="vertical",
                     pad=0.02, aspect=16, shrink=0.75)
        ims.append(im)
        arrs.append(arr)
        if marker_ij is not None:
            if use_geo:
                lat0 = float(np.asarray(track["lat"])[0])
                lon0 = float(np.asarray(track["lon"])[0])
                (dot,) = ax.plot(lon0, lat0, "o", color="red", markersize=8,
                                 markeredgecolor="white", markeredgewidth=1.2,
                                 transform=ccrs.PlateCarree())
            else:
                j0, i0 = marker_ij[0]
                (dot,) = ax.plot(i0, j0, "o", color="red", markersize=8,
                                 markeredgecolor="white", markeredgewidth=1.2)
            dots.append(dot)
            if radius_px is not None and not use_geo:
                circle = Circle((i0, j0), radius_px, fill=False,
                                edgecolor="red", linewidth=1.2,
                                alpha=0.4, linestyle="--")
                ax.add_patch(circle)
                circles.append(circle)
            elif radius is not None and radius > 0 and use_geo:
                circle = Circle((lon0, lat0), radius / 111.0, fill=False,
                                edgecolor="red", linewidth=1.2,
                                alpha=0.4, linestyle="--",
                                transform=ccrs.PlateCarree())
                ax.add_patch(circle)
                circles.append(circle)
            if cut_line and radius is not None and radius > 0 and use_geo:
                rdeg = radius / 111.0
                (line,) = ax.plot([lon0 - rdeg, lon0 + rdeg], [lat0, lat0],
                                  color="red", linewidth=1.5, alpha=0.8,
                                  transform=ccrs.PlateCarree())
                cut_lines.append(line)

    def update(frame: int) -> list[Any]:
        for im, arr in zip(ims, arrs):
            vals = np.asarray(arr.isel({time_dim: frame}))
            if use_geo:
                im.set_array(vals.ravel())
            else:
                im.set_data(vals)
        if marker_ij is not None:
            if use_geo:
                lat_t = float(np.asarray(track["lat"])[frame])
                lon_t = float(np.asarray(track["lon"])[frame])
                for dot in dots:
                    dot.set_data([lon_t], [lat_t])
                for circle in circles:
                    circle.center = (lon_t, lat_t)
                if cut_lines and radius is not None:
                    rdeg = radius / 111.0
                    for line in cut_lines:
                        line.set_data([lon_t - rdeg, lon_t + rdeg],
                                      [lat_t, lat_t])
            else:
                j, i = marker_ij[frame]
                for dot in dots:
                    dot.set_data([i], [j])
                for circle in circles:
                    circle.center = (i, j)
        return ims + dots + circles + cut_lines

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=interval, blit=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_cross_section(data: dict[str, xr.DataArray], track: xr.Dataset,
                       lat: xr.DataArray, lon: xr.DataArray, radius: float,
                       fig_title: str = "", plot_size: float = 4,
                       cmap: str = "viridis",
                       vmin: float | None = None, vmax: float | None = None,
                       time_dim: str = "Time", level_dim: str = "bottom_top",
                       interval: int = 250) -> HTML:
    """Animated zonal vertical cross-section through the storm center.

    The cut runs parallel to latitudes (constant lat = storm-center lat)
    and spans ``±radius`` km in longitude. Y axis is the model level.
    Only multi-level variables (with ``level_dim``) are accepted. If
    ``data`` has at least two entries, an extra panel showing
    ``first − second`` is appended with a diverging colormap.
    """
    first = next(iter(data.values()))
    n_frames = first.sizes[time_dim]
    n_levels = first.sizes[level_dim]
    assert all(arr.sizes[time_dim] == n_frames for arr in data.values())
    assert all(level_dim in arr.dims for arr in data.values()), (
        f"All variables must have a '{level_dim}' dimension."
    )
    assert track.sizes["time"] == n_frames

    lat_np = np.asarray(lat)
    lon_np = np.asarray(lon)
    ctr_lat = np.asarray(track["lat"])
    ctr_lon = np.asarray(track["lon"])
    rdeg = radius / 111.0

    def slab(arr: xr.DataArray, frame: int) -> tuple[np.ndarray, np.ndarray]:
        d2 = (lat_np - ctr_lat[frame]) ** 2 + (lon_np - ctr_lon[frame]) ** 2
        j, _ = np.unravel_index(np.argmin(d2), d2.shape)
        lon_row = lon_np[j, :]
        mask = np.abs(lon_row - ctr_lon[frame]) <= rdeg
        idx = np.where(mask)[0]
        vals = np.asarray(arr.isel({time_dim: frame}))
        return vals[:, j, idx], lon_row[idx]

    panels: list[tuple[str, xr.DataArray, str, float | None, float | None]] = [
        (title, arr, cmap, vmin, vmax) for title, arr in data.items()
    ]
    if len(data) >= 2:
        items = list(data.items())
        (t0, a0), (t1, a1) = items[0], items[1]
        diff = a0 - a1
        sample, _ = slab(diff, 0)
        dmax = float(np.nanmax(np.abs(sample))) or 1.0
        panels.append((f"{t0} − {t1}", diff, "RdBu_r", -dmax, dmax))

    cols = min(4, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(plot_size * 1.4 * cols, plot_size * rows),
                             squeeze=False)
    fig.suptitle(fig_title, fontsize=14)

    ims, arrs = [], []
    for k, (title, arr, panel_cmap, p_vmin, p_vmax) in enumerate(panels):
        ax = axes[k // cols][k % cols]
        slab0, lon0 = slab(arr, 0)
        im = ax.pcolormesh(lon0, np.arange(n_levels), slab0, cmap=panel_cmap,
                           vmin=p_vmin, vmax=p_vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("model level")
        plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                     aspect=16, shrink=0.85)
        ims.append((im, ax))
        arrs.append(arr)
    for k in range(len(panels), rows * cols):
        axes[k // cols][k % cols].axis("off")

    def update(frame: int) -> list[Any]:
        artists: list[Any] = []
        for (im, ax), arr in zip(ims, arrs):
            vals, lon_row = slab(arr, frame)
            im.set_array(vals.ravel())
            im.set_clim(vmin=im.get_clim()[0], vmax=im.get_clim()[1])
            ax.set_xlim(lon_row.min(), lon_row.max())
            artists.append(im)
        return artists

    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=interval, blit=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def multi_level_vars(ds: xr.Dataset,
                     level_dims: tuple[str, ...] = ("bottom_top",
                                                    "bottom_top_stag")
                     ) -> list[str]:
    """Return names of dataset variables that carry a vertical-level dim."""
    return [v for v in ds.data_vars
            if any(d in ds[v].dims for d in level_dims)]


def _haversine_km(lat_grid: np.ndarray, lon_grid: np.ndarray,
                  lat0: float, lon0: float) -> np.ndarray:
    """Great-circle distance from ``(lat0, lon0)`` to each point on a grid.

    Parameters
    ----------
    lat_grid, lon_grid : np.ndarray
        2D arrays of latitude/longitude in degrees.
    lat0, lon0 : float
        Reference point in degrees.

    Returns
    -------
    np.ndarray
        Distance in kilometres, same shape as ``lat_grid``.
    """
    r = 6371.0
    lat1 = np.radians(lat_grid)
    lat2 = np.radians(lat0)
    dlat = lat2 - lat1
    dlon = np.radians(lon0 - lon_grid)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def plot_track_time(data: dict[str, xr.DataArray], track: xr.Dataset,
                    lat: xr.DataArray, lon: xr.DataArray,
                    fig_title: str = "", plot_size: float = 4,
                    radius: float | None = None,
                    time_dim: str = "Time"
                    ) -> tuple[Figure, dict[str, np.ndarray]]:
    """Line plots of values at (or averaged around) the storm center.

    Parameters
    ----------
    data : dict[str, xr.DataArray]
        Mapping of panel title to a DataArray whose first dim is
        ``time_dim`` and whose spatial dims match ``lat``/``lon``.
    track : xr.Dataset
        Storm track from ``storm_track`` with per-time center
        ``lat``/``lon``.
    lat, lon : xr.DataArray
        2D WRF latitude/longitude grids.
    fig_title : str, default ""
        Figure suptitle prefix.
    plot_size : float, default 4
        Size (in inches) of the plot height; width is 1.6× height.
    radius : float, optional
        If given, average values within this radius (km) of the center.
        If omitted, the nearest-grid-point value is used.
    time_dim : str, default "Time"
        Name of the time dimension in each DataArray.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, np.ndarray]]
        ``(figure, series)`` — the matplotlib Figure and a dict mapping
        panel title to the per-time 1D array of values.
    """
    first = next(iter(data.values()))
    n_frames = first.sizes[time_dim]
    assert all(arr.sizes[time_dim] == n_frames for arr in data.values())
    assert track.sizes["time"] == n_frames, (
        f"Track has {track.sizes['time']} steps, data has {n_frames}."
    )

    lat_np = np.asarray(lat)
    lon_np = np.asarray(lon)
    ctr_lat = np.asarray(track["lat"])
    ctr_lon = np.asarray(track["lon"])
    times = np.asarray(track["time"])

    series = {title: np.empty(n_frames) for title in data}
    for t in range(n_frames):
        dist_km = _haversine_km(lat_np, lon_np, ctr_lat[t], ctr_lon[t])
        if radius is None:
            j, i = np.unravel_index(np.argmin(dist_km), dist_km.shape)
            for title, arr in data.items():
                series[title][t] = float(np.asarray(arr.isel({time_dim: t}))[j, i])
        else:
            mask = dist_km <= radius
            for title, arr in data.items():
                vals = np.asarray(arr.isel({time_dim: t}))
                series[title][t] = float(np.nanmean(vals[mask]))

    fig, ax = plt.subplots(figsize=(plot_size * 1.6, plot_size))
    suffix = " at storm center" if radius is None else f" averaged within {radius} km"
    ax.set_title(fig_title + suffix if fig_title else suffix.strip(), fontsize=14)

    for title, vals in series.items():
        ax.plot(times, vals, label=title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    plt.tight_layout()
    return fig, series


def plot_compare(obs: xr.DataArray, model: xr.DataArray, lat: xr.DataArray,
                 lon: xr.DataArray, var_name: str = "Tb", unit: str = "K",
                 vmin: float | None = None, vmax: float | None = None,
                 cmap: str = "gray_r") -> Figure:
    """Plot observed vs modelled fields side-by-side with a shared colorbar.

    Parameters
    ----------
    obs : xr.DataArray
        Observed 2D field.
    model : xr.DataArray
        Modelled 2D field.
    lat, lon : xr.DataArray
        2D latitude/longitude grids matching both fields.
    var_name : str, default "Tb"
        Variable name shown in panel titles and the colorbar label.
    unit : str, default "K"
        Variable unit shown in the colorbar label.
    vmin, vmax : float, optional
        Color-scale limits applied to both panels.
    cmap : str, default "gray_r"
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure.
    """
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
