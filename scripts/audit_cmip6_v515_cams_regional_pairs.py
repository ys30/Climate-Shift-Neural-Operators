"""E1 v5.15: third GCM CAMS-CSM1-0 real native Western-US regional +1-day gate.

Bounded two regional source/target date pairs per Historical/SSP experiment,
no training and no publication-grade climate/forecast claim. The exact
source stores come from the original live CMIP6 catalog, cross-checked against
the independently verified v5.14 audit artifact.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.audit_cmip6_catalog import read_catalog,select_pairs
from scripts.audit_cmip6_v514_third_gcm_candidates import select_new_candidates
from scripts.audit_cmip6_bcc_regional_pairs import sample_one

GCM="CAMS-CSM1-0"
EXPS=("historical","ssp245","ssp585")
DATES={"historical":((2010,1,15),(2010,7,15)),
       "ssp245":((2030,1,15),(2050,7,15)),
       "ssp585":((2030,1,15),(2050,7,15))}

def select_verified(metadata,prior):
    if prior.get("status")!="v514_candidate_third_GCM_native_daily_source_small_slice_eligibility":
        raise ValueError("Requires verified v5.14 native source audit")
    matching=[x for x in prior.get("models",[]) if x.get("source_id")==GCM]
    if len(matching)!=1 or matching[0].get("passed_small_slice_time_and_geometry_gate") is not True:
        raise ValueError("CAMS candidate did not pass v5.14")
    candidate=[x for x in metadata.get("selected_candidates",[]) if x["source_id"]==GCM]
    if len(candidate)!=1:
        raise ValueError("Live catalog does not include unique CAMS member-grid")
    selected=candidate[0]
    old=matching[0]
    if (old["member_id"]!=selected["member_id"] or old["grid_label"]!=selected["grid_label"]
        or prior["catalog_sha256"]!=metadata["catalog_sha256"]):
        raise ValueError("Live catalog changed from v5.14 provenance")
    for e in EXPS:
        if old["metadata_provenance"][e]["store"]!=selected[e]["store"]:
            raise ValueError("Source store changed after v5.14")
    return selected

def validate_regional(results):
    if set(results)!=set(EXPS):
        raise ValueError("All three experiments must be present")
    shapes={tuple(results[e]["grid_shape"]) for e in EXPS}
    grids={(results[e]["latitude_sha256"],results[e]["longitude_sha256"]) for e in EXPS}
    calendars={results[e]["calendar"] for e in EXPS}
    if len(shapes)!=1 or len(grids)!=1 or len(calendars)!=1:
        raise ValueError("Historical and SSP regional geometry/calendar changed")
    shape=next(iter(shapes))
    if shape[0]<4 or shape[1]<4 or shape[0]*shape[1]>2000:
        raise ValueError("Unsupported native grid geometry")
    for e in EXPS:
        r=results[e]
        if r["status"]!="actual_bcc_native_one_day_pair_sanity_only" or len(r["pairs"])!=2:
            raise ValueError("Missing exactly two real native forecast pairs")
        if any(not p["finite_values"] or p["n_cells"]!=shape[0]*shape[1] for p in r["pairs"]):
            raise ValueError("Nonfinite or mismatched regional cells")
        if any(p["persistence_weighted_rmse_K_sanity_only"]<0 for p in r["pairs"]):
            raise ValueError("Invalid persistence sanity score")
    return shape

def audit(prior,source_manifest):
    item=select_verified(source_manifest,prior)
    actual={e:sample_one(item[e]["store"],DATES[e]) for e in EXPS}
    shape=validate_regional(actual)
    return {
      "status":"v515_CAMS_native_western_US_regional_one_day_real_source_gate",
      "model":GCM,"member":item["member_id"],"grid":item["grid_label"],
      "catalog_sha256":source_manifest["catalog_sha256"],
      "native_regional_shape":list(shape),
      "experiments":actual,
      "no_neural_training":True,
      "limitations":[
          "Only two source/target date pairs per experiment; not a data-completeness audit",
          "Same native grid and calendar verified only on sampled fields",
          "No 1980–2014 export, third-GCM historical fitting or future skill yet",
          "No cross-grid neural transfer or SSP response claim",
          "CMIP6 future scenario fields are projections, not future observations"
      ]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError("Do not overwrite third GCM audit")
    rows,sha=read_catalog()
    metadata=select_pairs(rows,sha,max_models=12)
    result=audit(json.loads(Path(a.prior).read_text()),metadata)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
      "status":result["status"],"model":result["model"],
      "native_regional_shape":result["native_regional_shape"],
      "calendar":result["experiments"]["historical"]["calendar"],
      "pairs_by_experiment":{e:len(r["pairs"]) for e,r in result["experiments"].items()}
    },indent=2))

if __name__=="__main__":
    main()
