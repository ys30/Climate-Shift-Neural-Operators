"""E1 v5.14: inspect additional independent CMIP6 GCM candidates, bounded.

Discover exact original Zarr URLs from a LIVE SHA-audited public CMIP6 catalog,
then check native Historical/SSP coverage with tiny real-data slices.
This is a candidate feasibility gate, not neural training or model skill.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.audit_cmip6_catalog import read_catalog, select_pairs
from scripts.audit_cmip6_small_slices import audit_store

EXCLUDED=frozenset(("AWI-CM-1-1-MR","BCC-CSM2-MR","ACCESS-CM2"))
PREFERRED=("CAMS-CSM1-0","CESM2","CESM2-WACCM","CMCC-CM2-SR5")
EXPERIMENTS=("historical","ssp245","ssp585")

def select_new_candidates(metadata,max_models=2):
    if not 1<=max_models<=4:
        raise ValueError("Pilot permits 1–4 new GCMs")
    if metadata.get("status")!="catalog_metadata_only_no_climate_fields_or_transfer_skill":
        raise ValueError("Not a live verified CMIP6 catalog manifest")
    pool={r["source_id"]:r for r in metadata.get("selected_candidates",[])
          if r["source_id"] in PREFERRED and r["source_id"] not in EXCLUDED}
    candidates=[pool[g] for g in PREFERRED if g in pool][:max_models]
    if not candidates:
        raise ValueError("No independently catalog-matched third GCM")
    for candidate in candidates:
        if not all(x in candidate for x in EXPERIMENTS):
            raise ValueError("Missing required triple")
        if any(not candidate[e]["store"].startswith("gs://cmip6/")
               or f"/{e}/" not in candidate[e]["store"]
               or "/day/tas/" not in candidate[e]["store"]
               for e in EXPERIMENTS):
            raise ValueError("Invalid source catalog triple")
    return candidates

def coverage_gates(scenarios):
    if set(scenarios)!=set(EXPERIMENTS):
        raise ValueError("Missing experiments")
    if not all(scenarios[e]["status"]=="small_slice_cf_dates_and_kelvin_samples_verified"
               for e in EXPERIMENTS):
        return False,["failed_tiny_source_read"]
    h=scenarios["historical"]
    s245=scenarios["ssp245"]
    s585=scenarios["ssp585"]
    problems=[]
    if h["end_year"]!=2014 or h["start_year"]>1980:
        problems.append("historical_incompatible_with_1980_2014")
    for name,rec in (("ssp245",s245),("ssp585",s585)):
        if rec["start_year"]!=2015 or rec["end_year"]<2080:
            problems.append(name+"_missing_2015_to_2080")
        if rec["tas_units"]!="K":
            problems.append(name+"_not_kelvin")
    if any(scenarios[e]["tas_units"]!="K" for e in EXPERIMENTS):
        problems.append("not_kelvin")
    if len({scenarios[e]["calendar"] for e in EXPERIMENTS})!=1:
        problems.append("calendar_mismatch")
    if len({tuple(scenarios[e]["spatial_grid_shape"]) for e in EXPERIMENTS})!=1:
        problems.append("grid_shape_mismatch")
    if len({tuple(scenarios[e]["spatial_dims"]) for e in EXPERIMENTS})!=1:
        problems.append("spatial_dims_mismatch")
    return not problems,problems

def audit(selected,catalog_sha):
    results=[]
    for candidate in selected:
        observed={}
        for name in EXPERIMENTS:
            observed[name]=audit_store(candidate[name]["store"])
        eligible,reasons=coverage_gates(observed)
        results.append({
            "source_id":candidate["source_id"],
            "member_id":candidate["member_id"],
            "grid_label":candidate["grid_label"],
            "metadata_catalog_sha256":catalog_sha,
            "metadata_provenance":{e:candidate[e] for e in EXPERIMENTS},
            "experiments":observed,
            "passed_small_slice_time_and_geometry_gate":eligible,
            "failed_conditions":reasons,
            "native_regional_grid_still_unverified":True
        })
    return {
        "status":"v514_candidate_third_GCM_native_daily_source_small_slice_eligibility",
        "catalog_sha256":catalog_sha,
        "excluded_previous_sources":sorted(EXCLUDED),
        "models":results,
        "n_passed_metadata_and_small_slice":sum(x["passed_small_slice_time_and_geometry_gate"] for x in results),
        "limitations":[
            "Passing tiny slices is NOT regional lat/lon coordinate or missingness certification",
            "No historical model trained, no frozen neural skill and no projected trend inference",
            "Selected individual model-members are not yet an independent trained-GCM replication",
            "Do not reuse ACCESS-CM2 catalog SSP585 2201–2300 as 2030/2050/2080 future data",
            "No model selection based on future forecasting scores"
        ]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--max-models",type=int,default=2)
    a=parser.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError("Immutable audit destination exists")
    rows,source_sha=read_catalog()
    manifest=select_pairs(rows,source_sha,max_models=12)
    chosen=select_new_candidates(manifest,a.max_models)
    report=audit(chosen,source_sha)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"catalog":source_sha,
        "models":[{"gcm":m["source_id"],"member":m["member_id"],
                   "passed":m["passed_small_slice_time_and_geometry_gate"],
                   "reasons":m["failed_conditions"]} for m in report["models"]]},indent=2))

if __name__=="__main__":main()
