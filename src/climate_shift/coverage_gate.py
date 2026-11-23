"""Fail-closed E1 real-data coverage gate before model fitting.

Coverage and provenance checks are structural. Passing does not imply
representative meteorological or climate-shift sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .audit_e1 import audit


def check_coverage(npz_path: str | Path, metadata_path: str | Path,
                   expected_years: list[int] | tuple[int, ...],
                   min_per_month: int = 20) -> dict:
    expected = [int(y) for y in expected_years]
    if not expected or expected != sorted(set(expected)):
        raise ValueError("Expected years must be distinct, sorted and nonempty")
    if min_per_month < 1:
        raise ValueError("min_per_month must be positive")
    report = audit(npz_path, metadata_path, require_timestamps=True)
    card = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if card.get("unit") != "K":
        raise ValueError("Expected Kelvin for 2m_temperature")
    if card.get("selection") != "uniform_across_month_valid_pairs":
        raise ValueError("Expected v1.2+ uniform-within-month extractor provenance")
    if card.get("requested_years") != expected:
        raise ValueError("Declared requested_years disagree with experimental year plan")
    if card.get("actual_years") != expected:
        raise ValueError("Extracted years do not match expected year plan")
    if card.get("skipped_months"):
        raise ValueError("Extractor skipped one or more required calendar months")
    if not report.get("exact_24h_pairs_verified"):
        raise ValueError("Source and target times must be explicitly present and 24h paired")
    counts = report["samples_per_month"]
    insufficient = {}
    for yr in expected:
        for month in range(1, 13):
            key = f"{yr}-{month:02d}"
            count = counts.get(key, 0)
            if count < min_per_month:
                insufficient[key] = count
    if insufficient:
        raise ValueError(f"Insufficient samples in months: {insufficient}")
    return {
        "status": "passed_structural_coverage_gate",
        "years": expected,
        "required_months": 12 * len(expected),
        "min_per_month": min_per_month,
        "total_samples": report["samples"],
        "source_sha256": report["sha256"],
        "pair_time_verified": True,
        "sampling": card["selection"],
        "limitation": "Deterministic sparse monthly sampling is not complete daily coverage",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--min-per-month", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = check_coverage(args.input, args.metadata, args.years, args.min_per_month)
    dest = Path(args.output)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
