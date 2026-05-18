"""Helpers for loading WRF output, extracting a variable, and plotting it."""

from __future__ import annotations

import glob
import math
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
from tqdm.auto import tqdm
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
    for path in tqdm(files, desc="computing slp", unit="file"):
        nc = Dataset(path)
        try:
            slp = getvar(nc, "slp", timeidx=ALL_TIMES)
            if "Time" not in slp.dims:
                slp = slp.expand_dims("Time")
            slp_frames.append(np.asarray(slp))
        finally:
            nc.close()
    slp_arr = np.concatenate(slp_frames, axis=0)
    ds["SLP"] = (("Time", "south_north", "west_east"), slp_arr,
                 {"units": "hPa", "description": "sea level pressure"})
    if all(var in ds for var in ("RAINC", "RAINNC")):
        ds["PRECIP"] = ds["RAINC"] + ds["RAINNC"]
        ds["PRECIP"].attrs["units"] = ds["RAINC"].attrs.get("units", "")
        ds["PRECIP"].attrs["description"] = (
            "Total precipitation from convective and grid-scale processes"
        )

    if "TSK" in ds and "LANDMASK" in ds:
        ds["TSK_OCEAN"] = ds["TSK"].where(ds["LANDMASK"] == 0)
        ds["TSK_OCEAN"].attrs["units"] = ds["TSK"].attrs.get("units", "K")
        ds["TSK_OCEAN"].attrs["description"] = (
            "Skin temperature over ocean only (NaN over land)"
        )

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
              track: xr.Dataset | list[xr.Dataset | None] | None = None,
              lat: xr.DataArray | None = None,
              lon: xr.DataArray | None = None,
              radius: float | None = None,
              coastlines: bool = True,
              cut_line: bool = False,
              show_track_line: bool = False,
              mark_min_slp: bool = False) -> HTML:
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

    if isinstance(track, list):
        assert len(track) == len(data), (
            f"Got {len(track)} tracks for {len(data)} data entries."
        )
        data_tracks: list[xr.Dataset | None] = list(track)
    elif track is None:
        data_tracks = [None] * len(data)
    else:
        data_tracks = [track] * len(data)

    any_track = next((t for t in data_tracks if t is not None), None)
    radius_px: float | None = None
    if any_track is not None:
        assert lat is not None and lon is not None, (
            "`lat` and `lon` are required when passing `track`."
        )
        lat_np = np.asarray(lat)
        lon_np = np.asarray(lon)
        if radius is not None and radius > 0:
            dx_km = _haversine_km(lat_np[:, :-1], lon_np[:, :-1],
                                  lat_np[:, 1:], lon_np[:, 1:]).mean()
            dy_km = _haversine_km(lat_np[:-1, :], lon_np[:-1, :],
                                  lat_np[1:, :], lon_np[1:, :]).mean()
            radius_px = float(radius / (0.5 * (dx_km + dy_km)))

    def _marker_ij_for(tr: xr.Dataset) -> np.ndarray:
        assert tr.sizes["time"] == n_frames, (
            f"Track has {tr.sizes['time']} steps, data has {n_frames}."
        )
        ctr_lat = np.asarray(tr["lat"])
        ctr_lon = np.asarray(tr["lon"])
        mij = np.empty((n_frames, 2), dtype=int)
        for t in range(n_frames):
            d2 = (lat_np - ctr_lat[t]) ** 2 + (lon_np - ctr_lon[t]) ** 2
            j, i = np.unravel_index(np.argmin(d2), d2.shape)
            mij[t] = (j, i)
        return mij

    if any_track is not None:
        times = np.asarray(any_track["time"])
    else:
        times = None
        for coord in first.coords.values():
            if np.issubdtype(coord.dtype, np.datetime64):
                times = np.asarray(coord)
                break

    panels: list[tuple[str, xr.DataArray, str, float | None, float | None, str]] = [
        (title, arr, cmap, vmin, vmax, arr.attrs.get("units", ""))
        for title, arr in data.items()
    ]
    panel_tracks: list[xr.Dataset | None] = list(data_tracks)
    if len(data) >= 2:
        items = list(data.items())
        (_, a0), (_, a1) = items[0], items[1]
        diff = a0 - a1
        dmax = float(np.nanmax(np.abs(np.asarray(diff))))
        diff_unit = a0.attrs.get("units", "")
        panels.append(("Difference", diff, "RdBu_r", -dmax, dmax, diff_unit))
        panel_tracks.append(data_tracks[0])

    cols = min(cols, len(panels))
    rows = math.ceil(len(panels) / cols)
    fig = plt.figure(figsize=(plot_size * cols, plot_size * rows))
    fig.suptitle(fig_title, fontsize=14)
    fig.subplots_adjust(top=0.90, bottom=0.07)
    time_label = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11,
                          style="italic", color="0.3")

    use_geo = lat is not None and lon is not None
    lat_np = np.asarray(lat) if use_geo else None
    lon_np = np.asarray(lon) if use_geo else None

    ims, arrs = [], []
    panel_dots: list[Any] = [None] * len(panels)
    panel_circles: list[Any] = [None] * len(panels)
    panel_cut_lines: list[Any] = [None] * len(panels)
    panel_marker_ij: list[np.ndarray | None] = [None] * len(panels)
    for i, (title, arr, panel_cmap, p_vmin, p_vmax, unit) in enumerate(panels):
        ptrack = panel_tracks[i]
        if ptrack is not None:
            panel_marker_ij[i] = _marker_ij_for(ptrack)
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
        cb = plt.colorbar(im, ax=ax, orientation="vertical",
                          pad=0.02, aspect=16, shrink=0.75)
        if unit:
            cb.set_label(unit)
        ims.append(im)
        arrs.append(arr)
        if ptrack is not None and show_track_line:
            tr_lat = np.asarray(ptrack["lat"])
            tr_lon = np.asarray(ptrack["lon"])
            if use_geo:
                ax.plot(tr_lon, tr_lat, "-", color="lightcoral",
                        linewidth=1.2, alpha=0.7,
                        transform=ccrs.PlateCarree())
            else:
                mij_full = panel_marker_ij[i]
                ax.plot(mij_full[:, 1], mij_full[:, 0], "-",
                        color="lightcoral", linewidth=1.2, alpha=0.7)
        if ptrack is not None:
            if use_geo:
                lat0 = float(np.asarray(ptrack["lat"])[0])
                lon0 = float(np.asarray(ptrack["lon"])[0])
                (dot,) = ax.plot(lon0, lat0, "o", color="red", markersize=8,
                                 markeredgecolor="white", markeredgewidth=1.2,
                                 transform=ccrs.PlateCarree())
            else:
                j0, i0 = panel_marker_ij[i][0]
                (dot,) = ax.plot(i0, j0, "o", color="red", markersize=8,
                                 markeredgecolor="white", markeredgewidth=1.2)
            panel_dots[i] = dot
            if radius_px is not None and not use_geo:
                circle = Circle((i0, j0), radius_px, fill=False,
                                edgecolor="red", linewidth=1.2,
                                alpha=0.4, linestyle="--")
                ax.add_patch(circle)
                panel_circles[i] = circle
            elif radius is not None and radius > 0 and use_geo:
                circle = Circle((lon0, lat0), radius / 111.0, fill=False,
                                edgecolor="red", linewidth=1.2,
                                alpha=0.4, linestyle="--",
                                transform=ccrs.PlateCarree())
                ax.add_patch(circle)
                panel_circles[i] = circle
            if cut_line and radius is not None and radius > 0 and use_geo:
                rdeg = radius / 111.0
                (line,) = ax.plot([lon0 - rdeg, lon0 + rdeg], [lat0, lat0],
                                  color="red", linewidth=1.5, alpha=0.8,
                                  transform=ccrs.PlateCarree())
                panel_cut_lines[i] = line
            if mark_min_slp and "slp_min" in ptrack:
                f_min = int(np.argmin(np.asarray(ptrack["slp_min"])))
                if use_geo:
                    lat_m = float(np.asarray(ptrack["lat"])[f_min])
                    lon_m = float(np.asarray(ptrack["lon"])[f_min])
                    ax.plot(lon_m, lat_m, marker="*", markersize=14,
                            color="gold", markeredgecolor="black",
                            markeredgewidth=1.0, linestyle="None",
                            transform=ccrs.PlateCarree())
                else:
                    j_m, i_m = panel_marker_ij[i][f_min]
                    ax.plot(i_m, j_m, marker="*", markersize=14,
                            color="gold", markeredgecolor="black",
                            markeredgewidth=1.0, linestyle="None")

    def update(frame: int) -> list[Any]:
        if times is not None:
            ts_str = str(times[frame])[:19].replace("T", " ")
            stamp = ts_str
        else:
            stamp = f"{frame + 1} / {n_frames}"
        time_label.set_text(stamp)
        for im, arr in zip(ims, arrs):
            vals = np.asarray(arr.isel({time_dim: frame}))
            if use_geo:
                im.set_array(vals.ravel())
            else:
                im.set_data(vals)
        for p, ptrack in enumerate(panel_tracks):
            if ptrack is None:
                continue
            if use_geo:
                lat_t = float(np.asarray(ptrack["lat"])[frame])
                lon_t = float(np.asarray(ptrack["lon"])[frame])
                if panel_dots[p] is not None:
                    panel_dots[p].set_data([lon_t], [lat_t])
                if panel_circles[p] is not None:
                    panel_circles[p].center = (lon_t, lat_t)
                if panel_cut_lines[p] is not None and radius is not None:
                    rdeg = radius / 111.0
                    panel_cut_lines[p].set_data(
                        [lon_t - rdeg, lon_t + rdeg], [lat_t, lat_t])
            else:
                j, i = panel_marker_ij[p][frame]
                if panel_dots[p] is not None:
                    panel_dots[p].set_data([i], [j])
                if panel_circles[p] is not None:
                    panel_circles[p].center = (i, j)
        extras = [a for a in panel_dots + panel_circles + panel_cut_lines
                  if a is not None]
        return ims + extras

    pbar = tqdm(total=n_frames, desc="rendering plot_time", unit="frame")
    _update = update

    def _tracked(frame: int) -> list[Any]:
        artists = _update(frame)
        pbar.update(1)
        return artists

    ani = animation.FuncAnimation(fig, _tracked, frames=n_frames,
                                  interval=interval, blit=False)
    plt.close(fig)
    html = HTML(ani.to_jshtml())
    pbar.close()
    return html




def plot_at_min_slp(data: dict[str, xr.DataArray],
                    track: xr.Dataset | list[xr.Dataset],
                    lat: xr.DataArray,
                    lon: xr.DataArray,
                    fig_title: str = "",
                    plot_size: float = 7,
                    cmap: str = "RdYlBu_r",
                    cmap_difference: str = "RdBu_r",
                    diff_title: str = "Difference",
                    vmin: float | None = None,
                    vmax: float | None = None,
                    cbar_label: str = "",
                    radius: float = 400.0,
                    time_dim: str = "Time") -> Figure:
    """Snapshot of each field at the frame of its track's minimum SLP.

    For every entry in ``data``, locates the time index where the
    matching track's ``slp_min`` is lowest, then plots that 2D slice on
    a PlateCarree map centered on the storm and zoomed to ``±radius``
    km around it. A black ``x`` marks the storm center; the panel title
    reports the min SLP value and time.

    ``track`` may be a single Dataset (reused for all panels) or a list
    matching ``data`` entry-for-entry. All panels share one horizontal
    colorbar at the bottom of the figure.
    """
    items = list(data.items())
    if isinstance(track, list):
        assert len(track) == len(items), (
            f"Got {len(track)} tracks for {len(items)} data entries."
        )
        tracks = list(track)
    else:
        tracks = [track] * len(items)

    grid_lat = np.asarray(lat)
    grid_lon = np.asarray(lon)
    rdeg = radius / 111.0

    # Build per-panel snapshots: (title, field, cmap, vmin, vmax,
    #                             clat, clon, subtitle, is_diff)
    first_unit = items[0][1].attrs.get("units", "") if items else ""

    def _window(field: np.ndarray, clat: float, clon: float) -> np.ndarray:
        d2 = (grid_lat - clat) ** 2 + (grid_lon - clon) ** 2
        j, i = np.unravel_index(int(np.argmin(d2)), d2.shape)
        dlat = float(np.mean(np.abs(np.diff(grid_lat[:, i]))))
        dlon = float(np.mean(np.abs(np.diff(grid_lon[j, :]))))
        hj = int(rdeg / dlat)
        hi = int(rdeg / dlon)
        jmax, imax = grid_lat.shape
        return field[max(j - hj, 0):min(j + hj + 1, jmax),
                     max(i - hi, 0):min(i + hi + 1, imax)]

    # Shared colour scale derived from the first track's min-SLP snapshot.
    ref_arr, ref_tr = items[0][1], tracks[0]
    ref_fmin = int(np.nanargmin(np.asarray(ref_tr["slp_min"])))
    ref_field = np.asarray(ref_arr.isel({time_dim: ref_fmin}))
    ref_clat = float(np.asarray(ref_tr["lat"])[ref_fmin])
    ref_clon = float(np.asarray(ref_tr["lon"])[ref_fmin])
    ref_win = _window(ref_field, ref_clat, ref_clon)
    shared_vmin = float(np.nanmin(ref_win)) if vmin is None else vmin
    shared_vmax = float(np.nanmax(ref_win)) if vmax is None else vmax

    panels: list[tuple] = []
    for (title, arr), tr in zip(items, tracks):
        assert "slp_min" in tr, f"Track for '{title}' has no 'slp_min'."
        slp_min = np.asarray(tr["slp_min"])
        f_min = int(np.nanargmin(slp_min))
        min_val = float(slp_min[f_min])
        t_str = str(np.asarray(tr["time"])[f_min])[:19].replace("T", " ")
        field = np.asarray(arr.isel({time_dim: f_min}))
        clat = float(np.asarray(tr["lat"])[f_min])
        clon = float(np.asarray(tr["lon"])[f_min])
        var_name = getattr(arr, "name", None) or ""
        head = f"{title}: {var_name}" if var_name else title
        subtitle = f"{head}\nMin SLP: {min_val:.1f} hPa at {t_str}"
        panels.append((title, field, cmap, shared_vmin, shared_vmax,
                       clat, clon, subtitle, False))

    diff_grid: tuple[np.ndarray, np.ndarray] | None = None
    if len(items) >= 2:
        (_, a0), (_, a1) = items[0], items[1]
        tr0, tr1 = tracks[0], tracks[1]
        f0_idx = int(np.nanargmin(np.asarray(tr0["slp_min"])))
        f1_idx = int(np.nanargmin(np.asarray(tr1["slp_min"])))
        f0 = np.asarray(a0.isel({time_dim: f0_idx}))
        f1 = np.asarray(a1.isel({time_dim: f1_idx}))
        clat0 = float(np.asarray(tr0["lat"])[f0_idx])
        clon0 = float(np.asarray(tr0["lon"])[f0_idx])
        clat1 = float(np.asarray(tr1["lat"])[f1_idx])
        clon1 = float(np.asarray(tr1["lon"])[f1_idx])

        def _center_idx(clat: float, clon: float) -> tuple[int, int]:
            d2 = (grid_lat - clat) ** 2 + (grid_lon - clon) ** 2
            return tuple(np.unravel_index(int(np.argmin(d2)), d2.shape))

        j0, i0 = _center_idx(clat0, clon0)
        j1, i1 = _center_idx(clat1, clon1)
        dlat = float(np.mean(np.abs(np.diff(grid_lat[:, i0]))))
        dlon = float(np.mean(np.abs(np.diff(grid_lon[j0, :]))))
        hj = int(rdeg / dlat)
        hi = int(rdeg / dlon)
        sj0_lo, sj0_hi = j0 - hj, j0 + hj + 1
        si0_lo, si0_hi = i0 - hi, i0 + hi + 1
        sj1_lo, sj1_hi = j1 - hj, j1 + hj + 1
        si1_lo, si1_hi = i1 - hi, i1 + hi + 1
        # Clamp windows to grid and align widths between the two.
        jmax, imax = grid_lat.shape
        sj0_lo, sj1_lo = max(sj0_lo, 0), max(sj1_lo, 0)
        si0_lo, si1_lo = max(si0_lo, 0), max(si1_lo, 0)
        sj0_hi, sj1_hi = min(sj0_hi, jmax), min(sj1_hi, jmax)
        si0_hi, si1_hi = min(si0_hi, imax), min(si1_hi, imax)
        h_j = min(sj0_hi - sj0_lo, sj1_hi - sj1_lo)
        h_i = min(si0_hi - si0_lo, si1_hi - si1_lo)
        w0 = f0[sj0_lo:sj0_lo + h_j, si0_lo:si0_lo + h_i]
        w1 = f1[sj1_lo:sj1_lo + h_j, si1_lo:si1_lo + h_i]
        diff = w0 - w1
        dmax = float(np.nanmax(np.abs(diff))) or 1.0
        # Lay the diff on a synthetic grid centered at track 0's center,
        # using the panel-0 spacing.
        lat_w = clat0 + (np.arange(h_j) - (j0 - sj0_lo)) * dlat
        lon_w = clon0 + (np.arange(h_i) - (i0 - si0_lo)) * dlon
        diff_grid = (np.broadcast_to(lat_w[:, None], (h_j, h_i)),
                     np.broadcast_to(lon_w[None, :], (h_j, h_i)))
        t_str = str(np.asarray(tr0["time"])[f0_idx])[:19].replace("T", " ")
        subtitle = f"{diff_title}\nat {t_str}"
        panels.append((diff_title, diff, cmap_difference, -dmax, dmax,
                       clat0, clon0, subtitle, True))

    n = len(panels)
    cols = min(3, n) if n > 2 else min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(plot_size * cols, plot_size * rows),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             squeeze=False)

    for k, (_, field, panel_cmap, p_vmin, p_vmax,
            clat, clon, subtitle, is_diff) in enumerate(panels):
        ax = axes[k // cols][k % cols]
        ax.set_extent([clon - rdeg, clon + rdeg,
                       clat - rdeg, clat + rdeg],
                      crs=ccrs.PlateCarree())
        if is_diff and diff_grid is not None:
            plot_lat, plot_lon = diff_grid
        else:
            plot_lat, plot_lon = grid_lat, grid_lon
        im = ax.pcolormesh(plot_lon, plot_lat, field, cmap=panel_cmap,
                           vmin=p_vmin, vmax=p_vmax,
                           transform=ccrs.PlateCarree(), shading="auto")
        if not is_diff:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.plot(clon, clat, marker="x", color="black",
                markersize=12, markeredgewidth=2.5,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.set_title(subtitle)
        cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                          fraction=0.035, shrink=0.6)
        if is_diff:
            label = cbar_label or first_unit or "difference"
        else:
            label = cbar_label or first_unit
        if label:
            cb.set_label(label)

    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    if fig_title:
        fig.suptitle(fig_title, fontsize=14)
    return fig



def plot_cross_section(data: dict[str, xr.DataArray],
                       track: xr.Dataset | list[xr.Dataset],
                       lat: xr.DataArray, lon: xr.DataArray,
                       radius: float,
                       fig_title: str = "",
                       plot_size: float = 4,
                       cmap: str = "viridis",
                       cmap_difference: str = "RdBu_r",
                       diff_title: str = "Difference",
                       vmin: float | None = None,
                       vmax: float | None = None,
                       time_dim: str = "Time",
                       level_dim: str = "bottom_top",
                       interval: int = 250) -> HTML:
    """Animated zonal vertical cross-section through the storm center.

    The cut runs parallel to latitudes (constant lat = storm-center lat)
    and spans ``±radius`` km in longitude. Y axis is the model level.
    Only multi-level variables (with ``level_dim``) are accepted. If
    ``data`` has at least two entries, an extra panel showing
    ``first − second`` is appended with a diverging colormap.

    ``track`` may be a single Dataset (used for all panels) or a list of
    Datasets — one per entry in ``data`` — to follow a different storm
    center per variable.
    """
    first = next(iter(data.values()))
    n_frames = first.sizes[time_dim]
    n_levels = first.sizes[level_dim]
    assert all(arr.sizes[time_dim] == n_frames for arr in data.values())
    assert all(level_dim in arr.dims for arr in data.values()), (
        f"All variables must have a '{level_dim}' dimension."
    )

    if isinstance(track, list):
        assert len(track) == len(data), (
            f"Got {len(track)} tracks for {len(data)} data entries."
        )
        tracks = list(track)
    else:
        tracks = [track] * len(data)
    assert all(t.sizes["time"] == n_frames for t in tracks)

    lat_np = np.asarray(lat)
    lon_np = np.asarray(lon)
    centers = [(np.asarray(t["lat"]), np.asarray(t["lon"])) for t in tracks]
    rdeg = radius / 111.0

    # Fixed storm-relative longitude offset axis (degrees from center).
    # Width chosen from the first panel's initial frame.
    c_lat0, c_lon0 = centers[0]
    j0, _ = np.unravel_index(
        np.argmin((lat_np - c_lat0[0]) ** 2 + (lon_np - c_lon0[0]) ** 2),
        lat_np.shape,
    )
    n_x = int(np.sum(np.abs(lon_np[j0, :] - c_lon0[0]) <= rdeg))
    n_x = max(n_x, 8)
    off_axis = np.linspace(-rdeg, rdeg, n_x)

    def slab(arr: xr.DataArray, frame: int,
             ctr_lat: np.ndarray, ctr_lon: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
        d2 = (lat_np - ctr_lat[frame]) ** 2 + (lon_np - ctr_lon[frame]) ** 2
        j, _ = np.unravel_index(np.argmin(d2), d2.shape)
        lon_row = lon_np[j, :]
        offsets = lon_row - ctr_lon[frame]
        order = np.argsort(offsets)
        offsets_sorted = offsets[order]
        vals = np.asarray(arr.isel({time_dim: frame}))[:, j, :][:, order]
        out = np.empty((vals.shape[0], n_x), dtype=float)
        for lev in range(vals.shape[0]):
            out[lev] = np.interp(off_axis, offsets_sorted, vals[lev],
                                 left=np.nan, right=np.nan)
        return out, off_axis

    items = list(data.items())
    # Lock colour scales to the values seen at the first track's min-SLP frame
    # (colorbars cannot change during an animation).
    ref_fmin = int(np.nanargmin(np.asarray(tracks[0]["slp_min"])))
    ref_slab, _ = slab(items[0][1], ref_fmin, centers[0][0], centers[0][1])
    shared_vmin = float(np.nanmin(ref_slab)) if vmin is None else vmin
    shared_vmax = float(np.nanmax(ref_slab)) if vmax is None else vmax
    ref_slabs = [(title, slab(arr, ref_fmin, centers[i][0], centers[i][1])[0])
                 for i, (title, arr) in enumerate(items)]
    panels: list[tuple[str, xr.DataArray, str, float | None, float | None,
                       np.ndarray, np.ndarray, str]] = [
        (title, arr, cmap, shared_vmin, shared_vmax,
         centers[i][0], centers[i][1], arr.attrs.get("units", ""))
        for i, (title, arr) in enumerate(items)
    ]
    diff_spec: dict | None = None
    if len(data) >= 2:
        (_, a0), (_, a1) = items[0], items[1]
        c0_lat, c0_lon = centers[0]
        c1_lat, c1_lon = centers[1]
        # Each array sampled along ITS OWN track, then differenced in
        # storm-relative space — so two storms at different locations are
        # still compared centre-to-centre.
        s0_ref, _ = slab(a0, ref_fmin, c0_lat, c0_lon)
        s1_ref, _ = slab(a1, ref_fmin, c1_lat, c1_lon)
        ref_diff_slab = s0_ref - s1_ref
        ref_slabs.append((diff_title, ref_diff_slab))
        dmax = float(np.nanmax(np.abs(ref_diff_slab))) or 1.0
        diff_spec = {
            "title": diff_title,
            "a0": a0, "a1": a1,
            "c0": (c0_lat, c0_lon), "c1": (c1_lat, c1_lon),
            "cmap": cmap_difference,
            "vmin": -dmax, "vmax": dmax,
            "unit": a0.attrs.get("units", ""),
        }

    dbg_n = len(ref_slabs)
    dbg_fig, dbg_axes = plt.subplots(
        1, dbg_n, figsize=(plot_size * 1.4 * dbg_n, plot_size), squeeze=False)
    for ax, (title, s) in zip(dbg_axes[0], ref_slabs):
        is_diff_panel = title == diff_title and len(ref_slabs) > len(items)
        if is_diff_panel:
            m = float(np.nanmax(np.abs(s))) or 1.0
            im = ax.imshow(s, origin="lower", aspect="auto",
                           cmap=cmap_difference, vmin=-m, vmax=m)
        else:
            im = ax.imshow(s, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(f"{title} @ ref_fmin={ref_fmin}")
        dbg_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    dbg_fig.tight_layout()
    plt.show()

    times = np.asarray(tracks[0]["time"])
    if not np.issubdtype(times.dtype, np.datetime64):
        times = None
        for coord in first.coords.values():
            if np.issubdtype(coord.dtype, np.datetime64):
                times = np.asarray(coord)
                break

    total_panels = len(panels) + (1 if diff_spec is not None else 0)
    cols = min(4, total_panels)
    rows = math.ceil(total_panels / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(plot_size * 1.4 * cols, plot_size * rows),
                             squeeze=False)
    fig.suptitle(fig_title, fontsize=14)
    fig.subplots_adjust(top=0.90, bottom=0.10)
    time_label = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=11,
                          style="italic", color="0.3")

    ims, arrs, panel_centers = [], [], []
    for k, (title, arr, panel_cmap, p_vmin, p_vmax,
            c_lat, c_lon, unit) in enumerate(panels):
        ax = axes[k // cols][k % cols]
        slab0, lon0 = slab(arr, 0, c_lat, c_lon)
        im = ax.pcolormesh(lon0, np.arange(n_levels), slab0, cmap=panel_cmap,
                           vmin=p_vmin, vmax=p_vmax, shading="auto")
        ax.set_xlim(-rdeg, rdeg)
        ax.set_title(title)
        ax.set_xlabel("Δ longitude from storm center (deg)")
        ax.set_ylabel("model level")
        cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02,
                          aspect=16, shrink=0.85)
        if unit:
            cb.set_label(unit)
        ims.append((im, ax))
        arrs.append(arr)
        panel_centers.append((c_lat, c_lon))

    diff_im = None
    if diff_spec is not None:
        k = len(panels)
        ax = axes[k // cols][k % cols]
        s0_0, lon0 = slab(diff_spec["a0"], 0,
                          diff_spec["c0"][0], diff_spec["c0"][1])
        s1_0, _ = slab(diff_spec["a1"], 0,
                       diff_spec["c1"][0], diff_spec["c1"][1])
        diff_im = ax.pcolormesh(lon0, np.arange(n_levels), s0_0 - s1_0,
                                cmap=diff_spec["cmap"],
                                vmin=diff_spec["vmin"], vmax=diff_spec["vmax"],
                                shading="auto")
        ax.set_xlim(-rdeg, rdeg)
        ax.set_title(diff_spec["title"])
        ax.set_xlabel("Δ longitude from storm center (deg)")
        ax.set_ylabel("model level")
        cb = plt.colorbar(diff_im, ax=ax, orientation="vertical", pad=0.02,
                          aspect=16, shrink=0.85)
        if diff_spec["unit"]:
            cb.set_label(diff_spec["unit"])

    for k in range(total_panels, rows * cols):
        axes[k // cols][k % cols].axis("off")

    def update(frame: int) -> list[Any]:
        if times is not None:
            stamp = str(times[frame])[:19].replace("T", " ")
        else:
            stamp = f"{frame + 1} / {n_frames}"
        time_label.set_text(stamp)
        artists: list[Any] = [time_label]
        for (im, _), arr, (c_lat, c_lon) in zip(ims, arrs, panel_centers):
            vals, _ = slab(arr, frame, c_lat, c_lon)
            im.set_array(vals.ravel())
            artists.append(im)
        if diff_im is not None and diff_spec is not None:
            s0, _ = slab(diff_spec["a0"], frame,
                         diff_spec["c0"][0], diff_spec["c0"][1])
            s1, _ = slab(diff_spec["a1"], frame,
                         diff_spec["c1"][0], diff_spec["c1"][1])
            diff_im.set_array((s0 - s1).ravel())
            artists.append(diff_im)
        return artists

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pbar = tqdm(total=n_frames, desc="rendering cross-section", unit="frame")
    _update = update

    def _tracked(frame: int) -> list[Any]:
        artists = _update(frame)
        pbar.update(1)
        return artists

    ani = animation.FuncAnimation(fig, _tracked, frames=n_frames,
                                  interval=interval, blit=False)
    plt.close(fig)
    html = HTML(ani.to_jshtml())
    pbar.close()
    return html


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


def plot_track_time(data: dict[str, xr.DataArray],
                    track: xr.Dataset | list[xr.Dataset],
                    lat: xr.DataArray,
                    lon: xr.DataArray,
                    fig_title: str = "",
                    plot_size: float = 4,
                    radius: float | None = None,
                    time_dim: str = "Time",
                    mark_min_slp: bool = False,
                    agg: str = "avg",
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
        Radius (km) around the storm center used by ``agg``. If omitted
        and ``agg='avg'``, falls back to the nearest-grid-point value.
        Required for ``agg='max'`` and ``agg='distance'``.
    time_dim : str, default "Time"
        Name of the time dimension in each DataArray.
    mark_min_slp : bool, default False
        If True, mark the time of each track's lowest ``slp_min`` with
        a star on the corresponding line.
    agg : {"avg", "max", "distance"}, default "avg"
        How to reduce values inside ``radius`` to a single number per
        frame: ``"avg"`` mean, ``"max"`` maximum, ``"distance"`` great-
        circle distance (km) from the storm center to the location of
        the maximum value within the radius.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, np.ndarray]]
        ``(figure, series)`` — the matplotlib Figure and a dict mapping
        panel title to the per-time 1D array of values.
    """
    first = next(iter(data.values()))
    n_frames = first.sizes[time_dim]
    assert all(arr.sizes[time_dim] == n_frames for arr in data.values())

    if isinstance(track, list):
        assert len(track) == len(data), (
            f"Got {len(track)} tracks for {len(data)} data entries."
        )
        data_tracks: list[xr.Dataset] = list(track)
    else:
        data_tracks = [track] * len(data)
    for tr in data_tracks:
        assert tr.sizes["time"] == n_frames, (
            f"Track has {tr.sizes['time']} steps, data has {n_frames}."
        )

    lat_np = np.asarray(lat)
    lon_np = np.asarray(lon)
    times = np.asarray(data_tracks[0]["time"])

    assert agg in ("avg", "max", "distance"), (
        f"agg must be 'avg', 'max', or 'distance' (got {agg!r})."
    )
    if agg in ("max", "distance"):
        assert radius is not None, f"agg={agg!r} requires `radius`."

    series = {title: np.empty(n_frames) for title in data}
    for (title, arr), tr in zip(data.items(), data_tracks):
        ctr_lat = np.asarray(tr["lat"])
        ctr_lon = np.asarray(tr["lon"])
        for t in range(n_frames):
            dist_km = _haversine_km(lat_np, lon_np, ctr_lat[t], ctr_lon[t])
            vals = np.asarray(arr.isel({time_dim: t}))
            if radius is None:
                j, i = np.unravel_index(np.argmin(dist_km), dist_km.shape)
                series[title][t] = float(vals[j, i])
                continue
            mask = dist_km <= radius
            if not np.any(mask):
                series[title][t] = np.nan
                continue
            if agg == "avg":
                series[title][t] = float(np.nanmean(vals[mask]))
            elif agg == "max":
                series[title][t] = float(np.nanmax(vals[mask]))
            else:  # "distance"
                masked_vals = np.where(mask, vals, np.nan)
                if np.all(np.isnan(masked_vals)):
                    series[title][t] = np.nan
                else:
                    j, i = np.unravel_index(np.nanargmax(masked_vals),
                                            masked_vals.shape)
                    series[title][t] = float(dist_km[j, i])

    fig, ax = plt.subplots(figsize=(plot_size * 1.6, plot_size))
    if radius is None:
        suffix = " at storm center"
    elif agg == "avg":
        suffix = f" averaged within {radius} km radius"
    elif agg == "max":
        suffix = f" max within {radius} km radius"
    else:
        suffix = f" distance (km) of max within {radius} km radius"
    ax.set_title(fig_title + suffix if fig_title else suffix.strip(), fontsize=14)

    for ((title, vals), tr, (_, arr)) in zip(series.items(), data_tracks,
                                              data.items()):
        unit = arr.attrs.get("units", "") if agg != "distance" else "km"
        label = f"{title} [{unit}]" if unit else title
        (line,) = ax.plot(times, vals, label=label)
        if mark_min_slp and "slp_min" in tr:
            f_min = int(np.argmin(np.asarray(tr["slp_min"])))
            ax.plot(times[f_min], vals[f_min], marker="*", markersize=14,
                    color=line.get_color(), markeredgecolor="white",
                    markeredgewidth=1.2, linestyle="None")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    plt.tight_layout()
    return fig, series


def plot_track_time_intersect(data: dict[str, xr.DataArray], track: xr.Dataset,
                              lat: xr.DataArray, lon: xr.DataArray,
                              fig_title: str = "", plot_size: float = 4,
                              radius: float | None = None,
                              time_dim: str = "Time"
                              ) -> tuple[Figure, dict[str, np.ndarray]]:
    """Like plot_track_time but also draws a vertical line at the first intersection.

    If the two series cross, a vertical dashed line is drawn at that time with a
    label showing the date/time of the intersection.  If there is no intersection
    the plot is identical to plot_track_time.

    Parameters
    ----------
    data : dict[str, xr.DataArray]
        Mapping of panel title to a DataArray.  Exactly two entries are needed
        for intersection detection; if more (or fewer) are passed the intersection
        marker is silently skipped.
    track : xr.Dataset
        Storm track from ``storm_track``.
    lat, lon : xr.DataArray
        2D WRF latitude/longitude grids.
    fig_title : str, default ""
        Figure suptitle prefix.
    plot_size : float, default 4
        Size (in inches) of the plot height; width is 1.6× height.
    radius : float, optional
        If given, average values within this radius (km) of the center.
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

    # Find and mark the first two intersections (after 2016-10-04) when exactly two series are present.
    if len(series) == 2:
        import datetime as _dt
        _cutoff = _dt.datetime(2016, 10, 4)
        (title_a, vals_a), (title_b, vals_b) = list(series.items())
        diff = vals_a.astype(float) - vals_b.astype(float)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        _marked = 0
        for idx in sign_changes:
            if _marked >= 2:
                break
            d0, d1 = diff[idx], diff[idx + 1]
            frac = abs(d0) / (abs(d0) + abs(d1)) if (abs(d0) + abs(d1)) > 0 else 0.5
            t0_f = float(np.datetime64(times[idx], "ns").astype(np.float64))
            t1_f = float(np.datetime64(times[idx + 1], "ns").astype(np.float64))
            t_cross_dt = np.datetime64(int(t0_f + frac * (t1_f - t0_f)), "ns") \
                           .astype("datetime64[s]").astype(object)
            if t_cross_dt <= _cutoff:
                continue
            ax.axvline(t_cross_dt, color="black", linestyle="--", linewidth=1.2, zorder=3)
            from matplotlib.transforms import ScaledTranslation
            _nudge = ScaledTranslation(5 / 72, 0, fig.dpi_scale_trans)
            ax.text(
                t_cross_dt, 0.02,
                t_cross_dt.strftime('%Y-%m-%d %H:%M'),
                rotation=90, va="bottom", ha="left", fontsize=9, color="black",
                transform=ax.get_xaxis_transform() + _nudge,
            )
            _marked += 1

    return fig, series


def plot_deepening_rate(data: dict[str, xr.DataArray],
                        track: xr.Dataset | list[xr.Dataset],
                        lat: xr.DataArray,
                        lon: xr.DataArray,
                        fig_title: str = "",
                        plot_size: float = 4,
                        radius: float | None = None,
                        time_dim: str = "Time",
                        mark_min_slp: bool = False,
                        agg: str = "avg",
                        cmap: str = "seismic_r",
                        ylim_pad: float = 1.0,
                        ) -> tuple[Figure, dict[str, np.ndarray]]:
    """Per-panel time series colored segment-by-segment by deepening rate.

    Same signature as :func:`plot_track_time` (one DataArray per panel,
    one track per panel or shared, ``radius``/``agg`` to reduce inside
    the storm radius) but each entry gets its own subplot stacked
    vertically, and the line is colored by the local rate of change
    (Δ value / Δ time per hour).

    Markers: orange = max deepening (most negative rate), lime = overall
    minimum value; merged into one pink marker if they coincide. If
    ``mark_min_slp`` is True and the track carries ``slp_min``, the time
    of that track's lowest ``slp_min`` is also marked with a star.

    Returns ``(figure, series)`` like :func:`plot_track_time`.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    import matplotlib.dates as mdates

    first = next(iter(data.values()))
    n_frames = first.sizes[time_dim]
    assert all(arr.sizes[time_dim] == n_frames for arr in data.values())

    if isinstance(track, list):
        assert len(track) == len(data), (
            f"Got {len(track)} tracks for {len(data)} data entries."
        )
        data_tracks: list[xr.Dataset] = list(track)
    else:
        data_tracks = [track] * len(data)
    for tr in data_tracks:
        assert tr.sizes["time"] == n_frames, (
            f"Track has {tr.sizes['time']} steps, data has {n_frames}."
        )

    assert agg in ("avg", "max", "distance"), (
        f"agg must be 'avg', 'max', or 'distance' (got {agg!r})."
    )
    if agg in ("max", "distance"):
        assert radius is not None, f"agg={agg!r} requires `radius`."

    lat_np = np.asarray(lat)
    lon_np = np.asarray(lon)
    times = np.asarray(data_tracks[0]["time"])

    series = {title: np.empty(n_frames) for title in data}
    for (title, arr), tr in zip(data.items(), data_tracks):
        ctr_lat = np.asarray(tr["lat"])
        ctr_lon = np.asarray(tr["lon"])
        for t in range(n_frames):
            dist_km = _haversine_km(lat_np, lon_np, ctr_lat[t], ctr_lon[t])
            vals = np.asarray(arr.isel({time_dim: t}))
            if radius is None:
                j, i = np.unravel_index(np.argmin(dist_km), dist_km.shape)
                series[title][t] = float(vals[j, i])
                continue
            mask = dist_km <= radius
            if not np.any(mask):
                series[title][t] = np.nan
                continue
            if agg == "avg":
                series[title][t] = float(np.nanmean(vals[mask]))
            elif agg == "max":
                series[title][t] = float(np.nanmax(vals[mask]))
            else:
                masked_vals = np.where(mask, vals, np.nan)
                if np.all(np.isnan(masked_vals)):
                    series[title][t] = np.nan
                else:
                    j, i = np.unravel_index(np.nanargmax(masked_vals),
                                            masked_vals.shape)
                    series[title][t] = float(dist_km[j, i])

    cases: list[dict[str, Any]] = []
    for (title, vals), tr, (_, arr) in zip(series.items(), data_tracks,
                                           data.items()):
        unit = "km" if agg == "distance" else arr.attrs.get("units", "")
        valid = np.isfinite(vals)
        t_v = times[valid]; p_v = vals[valid]
        dt_h = (np.diff(t_v).astype("timedelta64[s]").astype(float)) / 3600.0
        dp = np.diff(p_v)
        seg_ok = (dt_h > 0) & np.isfinite(dt_h) & np.isfinite(dp)
        rate = dp[seg_ok] / dt_h[seg_ok]
        x_all = mdates.date2num(t_v)
        x0 = x_all[:-1][seg_ok]; x1 = x_all[1:][seg_ok]
        p0 = p_v[:-1][seg_ok];   p1 = p_v[1:][seg_ok]
        segments = np.stack([np.column_stack([x0, p0]),
                             np.column_stack([x1, p1])], axis=1)
        i_min = int(np.nanargmin(p_v)) if p_v.size else 0
        if rate.size:
            i_max = int(np.nanargmin(rate))
            md_time = t_v[1:][seg_ok][i_max]
            md_val = p_v[1:][seg_ok][i_max]
        else:
            md_time = t_v[i_min] if p_v.size else None
            md_val = float(p_v[i_min]) if p_v.size else np.nan
        cases.append({
            "label": title, "track": tr, "time": t_v, "p": p_v, "rate": rate,
            "segments": segments,
            "max_deepening_time": md_time, "max_deepening_slp": md_val,
            "min_slp_time": t_v[i_min] if p_v.size else None,
            "min_slp": float(p_v[i_min]) if p_v.size else np.nan,
            "unit": unit,
        })

    all_rates = np.concatenate([c["rate"] for c in cases]) if any(
        c["rate"].size for c in cases) else np.array([1.0])
    absmax = float(np.nanmax(np.abs(all_rates))) or 1.0
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

    n = len(cases)
    fig, axes = plt.subplots(n, 1,
                             figsize=(plot_size * 1.6, plot_size * n),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    lc = None
    for ax, c in zip(axes, cases):
        if c["segments"].size:
            lc = LineCollection(c["segments"], cmap=cmap, norm=norm,
                                linewidth=2.5)
            lc.set_array(c["rate"])
            ax.add_collection(lc)
        ax.plot(c["time"], c["p"], color="black", alpha=0.15, linewidth=1)

        same = (c["max_deepening_time"] is not None
                and c["max_deepening_time"] == c["min_slp_time"])
        u = f" {c['unit']}" if c["unit"] else ""
        if same:
            ax.scatter(c["min_slp_time"], c["min_slp"], color="deeppink",
                       edgecolor="black", s=70, zorder=5,
                       label=f"Max deepening & Min: {c['min_slp']:.1f}{u}")
        else:
            if c["max_deepening_time"] is not None:
                ax.scatter(c["max_deepening_time"], c["max_deepening_slp"],
                           color="darkorange", edgecolor="black",
                           s=70, zorder=5, label="Max deepening rate")
            if c["min_slp_time"] is not None:
                ax.scatter(c["min_slp_time"], c["min_slp"], color="lime",
                           edgecolor="black", s=70, zorder=5,
                           label=f"Min: {c['min_slp']:.1f}{u}")
        if mark_min_slp and "slp_min" in c["track"]:
            f_min = int(np.argmin(np.asarray(c["track"]["slp_min"])))
            y = (c["p"][np.argmin(np.abs(c["time"] - times[f_min]))]
                 if c["p"].size else 0)
            ax.scatter(times[f_min], y, marker="*", color="gold",
                       edgecolor="black", s=140, zorder=6,
                       label="Min SLP (track)")

        if c["p"].size:
            ax.set_ylim(c["p"].min() - ylim_pad, c["p"].max() + ylim_pad)
        ylabel = f"{c['label']} [{c['unit']}]" if c["unit"] else c["label"]
        ax.set_ylabel(ylabel)
        ax.set_title(c["label"])
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlim(times.min(), times.max())
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    axes[-1].tick_params(axis="x", rotation=30)

    if fig_title:
        fig.suptitle(fig_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.90, 0.96 if fig_title else 1.0])
    if lc is not None:
        cax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
        cbar = fig.colorbar(lc, cax=cax)
        units_seen = {c["unit"] for c in cases if c["unit"]}
        rate_unit = (f"{next(iter(units_seen))} h$^{{-1}}$"
                     if len(units_seen) == 1 else "unit h$^{-1}$")
        cbar.set_label(f"Rate of change [{rate_unit}]")
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
