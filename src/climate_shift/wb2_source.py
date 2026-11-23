"""Open the public WeatherBench 2 store through explicit anonymous GCS access."""
from __future__ import annotations


def open_public_wb2(store: str):
    """Return an xarray Dataset; caller must close it.

    The caller must decide whether the selected dataset is suitable and licensed
    for the intended analysis. No downloads happen until data are accessed.
    """
    if not isinstance(store, str) or not store.startswith("gs://weatherbench2/datasets/"):
        raise ValueError("Refusing non-WeatherBench 2 public GCS path")
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(store.removeprefix("gs://"))
    return xr.open_zarr(mapper, consolidated=None, chunks=None)
