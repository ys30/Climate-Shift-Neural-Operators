"""Independent audit for E1 NPZ and exporter metadata.

Validation only; passing an audit does not establish climate science validity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import load_npz_pairs


def audit(npz_path: str | Path, meta_path: str | Path | None = None,
          require_timestamps: bool = True) -> dict:
    path = Path(npz_path)
    x, y, years = load_npz_pairs(path)
    with np.load(path, allow_pickle=False) as data:
        times = data["target_times"] if "target_times" in data.files else None
        source_times = data["source_times"] if "source_times" in data.files else None
    if require_timestamps and times is None:
        raise ValueError("Cannot audit provenance: target_times absent")
    if times is not None:
        # For +24h samples the first target is not proof that its predictor
        # existed; the adapter's exact-pairing tests provide that separate check.
        months = times.astype("datetime64[M]")
        month_labels, month_counts = np.unique(months.astype(str), return_counts=True)
        month_map = dict(zip(month_labels.tolist(), month_counts.astype(int).tolist()))
        first, last = str(times[0]), str(times[-1])
    else:
        month_map, first, last = {}, None, None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "status": "STRUCTURAL_AUDIT_ONLY",
        "sha256": digest,
        "samples": int(x.shape[0]),
        "shape_x": list(x.shape),
        "shape_y": list(y.shape),
        "years": {str(int(yr)): int(np.sum(years == yr)) for yr in np.unique(years)},
        "first_target": first,
        "last_target": last,
        "samples_per_month": month_map,
        "timestamps_verified": times is not None,
        "exact_24h_pairs_verified": source_times is not None,
        "first_source": str(source_times[0]) if source_times is not None else None,
        "last_source": str(source_times[-1]) if source_times is not None else None,
    }
    if meta_path is not None:
        metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        expected = metadata.get("sha256", metadata.get("sha256_npz"))
        if expected != digest:
            raise ValueError("Metadata SHA-256 missing or mismatched")
        if metadata.get("n_samples", metadata.get("samples")) != len(years):
            raise ValueError("Metadata sample count mismatch")
        if metadata.get("target_first") != first or metadata.get("target_last") != last:
            raise ValueError("Metadata timestamp bounds mismatch")
        if metadata.get("variable") != "2m_temperature" or metadata.get("forecast_horizon_hours", metadata.get("forecast_lead_hours")) != 24:
            raise ValueError("Unexpected variable or forecast horizon")
        result["metadata_verified"] = True
        result["source"] = metadata.get("source")
        result["unit"] = metadata.get("unit")
    else:
        result["metadata_verified"] = False
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--metadata")
    p.add_argument("--output")
    args = p.parse_args()
    report = audit(args.input, args.metadata)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        out = Path(args.output)
        if out.exists():
            p.error("Output already exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
