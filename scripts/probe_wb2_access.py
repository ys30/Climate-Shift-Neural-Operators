"""Minimal anonymous read probe for the official WeatherBench 2 ERA5 Zarr store.

Records exact failure stage; never downloads an entire year or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from datetime import datetime, timezone

DEFAULT_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


def probe(store: str = DEFAULT_STORE) -> dict:
    import gcsfs
    import numpy as np
    import xarray as xr

    report = {
        "status": "failure",
        "source": store,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_type": "metadata + two tiny gridded reads; not ERA5 training",
    }
    stage = "gcsfs_anonymous_connect"
    try:
        if not store.startswith("gs://"):
            raise ValueError("Probe only permits a gs:// store")
        fs = gcsfs.GCSFileSystem(token="anon", cache_timeout=0)
        mapper = fs.get_mapper(store.removeprefix("gs://"))
        stage = "xarray_open_zarr"
        ds = xr.open_zarr(mapper, consolidated=None, chunks=None)
        try:
            stage = "metadata"
            report["sizes"] = {k: int(v) for k, v in ds.sizes.items()}
            report["variables"] = list(ds.data_vars)
            if "2m_temperature" not in ds:
                raise KeyError("2m_temperature not present in dataset")
            t = ds["2m_temperature"]
            report["variable_dimensions"] = list(t.dims)
            report["variable_units"] = t.attrs.get("units", "NOT_SPECIFIED")
            if set(t.dims) != {"time", "latitude", "longitude"}:
                raise ValueError("Temperature variable dimensions differ from the E1 adapter contract")
            stage = "coordinate_read"
            times = ds.time.values
            report["time_first"] = str(times[0])
            report["time_last"] = str(times[-1])
            if not np.all(np.diff(times) > np.timedelta64(0, "ns")):
                raise ValueError("Times not strictly ascending")
            stage = "tiny_chunk_read"
            # A 2x2x2 subset only; no unbounded data materialization.
            small = t.isel(time=slice(0, 2), latitude=slice(30, 32),
                           longitude=slice(50, 52)).values
            if small.shape != (2, 2, 2) or not np.isfinite(small).all():
                raise ValueError(f"Unexpected tiny read shape or NaNs: {small.shape}")
            report["tiny_chunk_shape"] = list(small.shape)
            report["tiny_chunk_min"] = float(small.min())
            report["tiny_chunk_max"] = float(small.max())
            report["status"] = "success"
            report["stage"] = "completed"
        finally:
            ds.close()
    except Exception as exc:
        report["stage"] = stage
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)[:1000]
        report["traceback_tail"] = traceback.format_exc(limit=3)[-2000:]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--output", default="outputs/wb2_access_probe.json")
    args = parser.parse_args()
    report = probe(args.store)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
