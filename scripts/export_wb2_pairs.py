"""Export a bounded subset of WeatherBench 2 ERA5 as explicit +24h pairs.

Requires separate optional dependencies: pip install -e '.[era5]'
Cloud access is opt-in; no full-bucket sync is attempted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from climate_shift.era5_adapter import make_24h_pairs
from climate_shift.wb2_source import open_public_wb2

DEFAULT_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--start", default="2019-07-01")
    p.add_argument("--end", default="2019-07-07")
    p.add_argument("--output", default="data/processed/wb2_e1_24h.npz")
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        p.error("Output exists; refuse overwrite")
    # Consolidation choice is explicit for WeatherBench 2 Zarr stores.
    ds = open_public_wb2(args.store)
    try:
        x, y, years, target_times = make_24h_pairs(
            ds, start=args.start, end=args.end, max_samples=args.max_samples, stride=args.stride
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, X=x, y=y, years=years, target_times=target_times)
        sha = hashlib.sha256(output.read_bytes()).hexdigest()
        meta = {"source": args.store, "variable": "2m_temperature", "source_time_range": [args.start, args.end],
                "forecast_lead_hours": 24, "lat_bounds": [25.0, 50.0], "lon_bounds": [-125.0, -100.0],
                "stride": args.stride, "samples": len(years), "target_first": str(target_times[0]),
                "target_last": str(target_times[-1]), "array_shape": list(x.shape),
                "unit": ds["2m_temperature"].attrs.get("units", "UNVERIFIED"),
                "sha256_npz": sha, "scientific_status": "data preparation only; no model results"}
        output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2))
    finally:
        ds.close()

if __name__ == "__main__":
    main()
