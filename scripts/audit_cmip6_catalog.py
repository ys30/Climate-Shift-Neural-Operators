"""Bounded *metadata-only* CMIP6 Historical-to-SSP Day/tas availability audit.

This does not download climate fields, remap grids or evaluate ERA5 models.
No daily model output may be treated as the same target as ERA5 +24h snapshots
without a separately documented target transformation and re-training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

CATALOG_URL = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
SCENARIOS = ("ssp245", "ssp585")
REQUIRED_FIELDS = frozenset({
    "activity_id", "source_id", "experiment_id", "member_id",
    "table_id", "variable_id", "grid_label", "zstore",
})
KELVIN_VARIABLE = "tas"
FREQUENCY = "day"


def read_catalog(url: str = CATALOG_URL, timeout: int = 50,
                 max_bytes: int = 35_000_000) -> tuple[list[dict], str]:
    """Read only small CSV metadata; fail closed on oversized/unavailable sources."""
    if max_bytes < 1000:
        raise ValueError("Metadata byte limit too low")
    req = Request(url, headers={"User-Agent": "ClimateShiftResearch-CMIP6-Metadata/1.0"})
    with urlopen(req, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("CMIP6 metadata catalog exceeds bounded download size")
    source_sha = hashlib.sha256(raw).hexdigest()
    decoded = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))
    if not REQUIRED_FIELDS.issubset(reader.fieldnames or []):
        raise ValueError("CMIP6 catalog lacks required fields")
    rows = list(reader)
    if not rows:
        raise ValueError("Empty CMIP6 catalog")
    return rows, source_sha


def select_pairs(rows: list[dict], source_sha: str,
                 max_models: int = 8) -> dict:
    """Pair historical and both SSPs on model/member/grid EXACTLY."""
    if max_models < 1 or max_models > 20:
        raise ValueError("Require 1–20 metadata-only candidates")
    matching = []
    for r in rows:
        if (r.get("variable_id") != KELVIN_VARIABLE
                or r.get("table_id") != FREQUENCY
                or r.get("experiment_id") not in ("historical", *SCENARIOS)):
            continue
        if not r.get("zstore", "").startswith("gs://cmip6/"):
            continue
        if not all(r.get(col) for col in
                   ("source_id", "member_id", "grid_label", "activity_id")):
            continue
        matching.append(r)
    # Defensive: a model member with several versioned locations is NOT
    # silently accepted as multiple independent models or realizations.
    groups: dict[tuple, dict[str, list[dict]]] = {}
    for row in matching:
        key = (row["source_id"], row["member_id"], row["grid_label"])
        groups.setdefault(key, {}).setdefault(row["experiment_id"], []).append(row)
    eligible = []
    for (model, member, grid), experiments in sorted(groups.items()):
        if not all(x in experiments for x in ("historical", *SCENARIOS)):
            continue
        candidates = {}
        for experiment in ("historical", *SCENARIOS):
            versions = experiments[experiment]
            # Reproducible metadata tie-break; do not mistake versions as ensembles.
            versions = sorted(versions, key=lambda r: (
                str(r.get("version", "")), r["zstore"]))
            candidates[experiment] = {
                "store": versions[-1]["zstore"],
                "activity_id": versions[-1]["activity_id"],
                "version": str(versions[-1].get("version", "")),
                "duplicate_catalog_rows": len(versions),
            }
        if candidates["historical"]["activity_id"] != "CMIP" or any(
            candidates[s]["activity_id"] != "ScenarioMIP" for s in SCENARIOS
        ):
            continue
        eligible.append({
            "source_id": model, "member_id": member, "grid_label": grid,
            "historical": candidates["historical"],
            "ssp245": candidates["ssp245"],
            "ssp585": candidates["ssp585"],
        })
    distinct = []
    seen = set()
    for e in eligible:
        if e["source_id"] in seen:
            continue
        seen.add(e["source_id"])
        distinct.append(e)
        if len(distinct) >= max_models:
            break
    return {
        "status": "catalog_metadata_only_no_climate_fields_or_transfer_skill",
        "catalog_url": CATALOG_URL,
        "catalog_sha256": source_sha,
        "variables": ["tas"],
        "table_id": "day",
        "historical_to_scenarios": list(SCENARIOS),
        "matching_keys": ["source_id", "member_id", "grid_label"],
        "matching_model_members": len(eligible),
        "distinct_matching_models": len({e["source_id"] for e in eligible}),
        "selected_candidates": distinct,
        "requirements_before_real_climate_shift_claim": [
            "Verify time coordinate coverage and native model calendar in each Zarr store",
            "Verify Kelvin units, missing data, grid geometry and longitude convention",
            "Build identical 24h prediction target using same timestamp semantics, or label daily benchmark separately",
            "Do not treat bias-corrected NASA daily tas/tasmax as direct ERA5 snapshot targets",
            "Regrid consistently with method and weights recorded; baseline and neural operator must share grid and lead",
            "Distinguish future SSP scenario signal, model structural differences and downscaling bias",
            "Never estimate climatology or choose hyperparameters using future heldout SSP targets",
        ],
        "note": "Catalog eligibility does NOT verify actual Zarr store access or climate-year coverage.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--max-models", type=int, default=8)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite prior catalog audit")
    rows, digest = read_catalog()
    report = select_pairs(rows, digest, max_models=args.max_models)
    if not report["selected_candidates"]:
        raise SystemExit("No verified metadata Historical/SSP/day/tas trio; do not invent candidates")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "catalog_sha256": report["catalog_sha256"],
        "matching_model_members": report["matching_model_members"],
        "distinct_matching_models": report["distinct_matching_models"],
        "selected_candidates": [
            {"source_id": x["source_id"], "member_id": x["member_id"],
             "grid_label": x["grid_label"]}
            for x in report["selected_candidates"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
