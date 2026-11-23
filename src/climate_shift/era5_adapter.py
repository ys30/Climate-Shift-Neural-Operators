"""WeatherBench 2 / ERA5 six-hourly to 24-hour paired-field adapter.

Only reads caller-supplied xarray datasets; cloud access is in the CLI.
No inference about physical variables or forecast skill is performed.
"""
from __future__ import annotations
import numpy as np
import xarray as xr


def make_24h_pairs(ds: xr.Dataset, variable: str = "2m_temperature",
                   start: str = "2019-01-01", end: str = "2019-01-31",
                   lat_bounds: tuple[float, float] = (25.0, 50.0),
                   lon_bounds: tuple[float, float] = (-125.0, -100.0),
                   stride: int = 4, max_samples: int = 256,
                   selection: str = "first", return_source_times: bool = False):
    """Return X/y/target years and target timestamps for exact +24h pairs.

    Assumes a time/latitude/longitude field without extra dimensions.
    Longitudes can be either 0..360 or -180..180. Array values are
    unchanged, normally Kelvin for 2m_temperature. The caller must check
    the physical metadata of their dataset version.
    """
    if variable not in ds:
        raise KeyError(f"Missing variable {variable!r}")
    if stride < 1 or max_samples < 1:
        raise ValueError("stride and max_samples must be positive")
    if selection not in ("first", "uniform"):
        raise ValueError("selection must be first or uniform")
    if not (lat_bounds[0] < lat_bounds[1] and lon_bounds[0] < lon_bounds[1]):
        raise ValueError("Bounds must be increasing and must not cross the dateline")
    da = ds[variable]
    if set(da.dims) != {"time", "latitude", "longitude"} or da.ndim != 3:
        raise ValueError("Expected exactly time, latitude, longitude dimensions")
    times = np.asarray(ds.time.values)
    if times.dtype.kind != "M" or np.isnat(times).any():
        raise ValueError("Requires valid numpy datetime64 time; cftime needs separate calendar handling")
    if not np.all(np.diff(times) > np.timedelta64(0, "ns")):
        raise ValueError("Time must be unique and strictly increasing")
    lat = np.asarray(ds.latitude.values)
    lon_raw = np.asarray(ds.longitude.values)
    lon = (lon_raw + 180) % 360 - 180
    lat_indices = np.flatnonzero((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]))
    lon_indices = np.flatnonzero((lon >= lon_bounds[0]) & (lon <= lon_bounds[1]))
    lon_indices = lon_indices[np.argsort(lon[lon_indices])]
    if len(lat_indices) < 2 or len(lon_indices) < 2:
        raise ValueError("Requested geographic subset is too small or outside input grid")
    start_dt = np.datetime64(start, "ns")
    end_dt = np.datetime64(end, "ns") + np.timedelta64(1, "D")
    within = np.flatnonzero((times >= start_dt) & (times < end_dt))
    if len(within) == 0:
        raise ValueError("No timestamps inside requested date window")
    # Exact timedelta lookup, never row-offset lookup (handles missing timesteps).
    lookup = {int(t.astype("datetime64[ns]").astype("int64")): i for i, t in enumerate(times)}
    dt_ns = int(np.timedelta64(24, "h") / np.timedelta64(1, "ns"))
    pairs = []
    for i in within[::stride]:
        future = int(times[i].astype("datetime64[ns]").astype("int64")) + dt_ns
        j = lookup.get(future)
        if j is not None and times[j] < end_dt:
            pairs.append((int(i), int(j)))
        if selection == "first" and len(pairs) >= max_samples:
            break
    if selection == "uniform" and len(pairs) > max_samples:
        # Spread samples over the entire month rather than selecting only early days.
        chosen = np.linspace(0, len(pairs) - 1, max_samples, dtype=int)
        pairs = [pairs[int(idx)] for idx in chosen]
    if not pairs:
        raise ValueError("No exact 24-hour pairs inside date window")
    x_indices = [i for i, _ in pairs]
    y_indices = [j for _, j in pairs]
    subset = da.isel(latitude=lat_indices, longitude=lon_indices)
    x = subset.isel(time=x_indices).transpose("time", "latitude", "longitude").values.astype("float32")
    y = subset.isel(time=y_indices).transpose("time", "latitude", "longitude").values.astype("float32")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite samples; decide policy before export")
    target_times = times[y_indices].astype("datetime64[ns]")
    years = target_times.astype("datetime64[Y]").astype(int) + 1970
    source_times = times[x_indices].astype("datetime64[ns]")
    if return_source_times:
        return x[:, None], y[:, None], years.astype("int32"), target_times, source_times
    return x[:, None], y[:, None], years.astype("int32"), target_times
