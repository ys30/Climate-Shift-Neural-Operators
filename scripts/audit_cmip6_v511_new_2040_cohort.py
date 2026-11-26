"""E1 v5.11: new fixed-date CMIP6 source-only 2040 regional cohort.

This audit is a distinct, predeclared supplementary sample (no 8th/20th dates).
It does NOT evaluate trained forecasts and cannot establish full-year means,
heatwave duration or independent out-of-sample performance on its own.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.audit_cmip6_awi_daily_pairs import contiguous_span, exact_daily_pair_indices
from scripts.evaluate_cmip6_v48_frozen_ssp import manifest_entry
from scripts.audit_cmip6_bcc_regional_pairs import get_model
from scripts.audit_cmip6_v59_source_warming import weighted_mean

YEAR=2040
DAYS=(3,7,11,15,19,23,27)
SCENARIOS=("ssp245","ssp585")
SOURCES={"AWI":("proleptic_gregorian",(26,27)),
         "BCC":("noleap",(23,23))}

def schedule():
    return tuple((YEAR,m,d) for m in range(1,13) for d in DAYS)

def validate_schedule(s):
    if len(s)!=84 or len(set(s))!=84:
        raise ValueError("Exactly 84 unique source dates required")
    if any(y!=YEAR or m not in range(1,13) or d not in DAYS or d in (8,20)
           for y,m,d in s):
        raise ValueError("Non-frozen date plan or prior sampled-day overlap")

def extract(store,source,scenario):
    import fsspec
    import xarray as xr
    if source not in SOURCES or scenario not in SCENARIOS:
        raise ValueError("Unknown source or scenario")
    if not store.startswith("gs://cmip6/") or f"/{scenario}/" not in store or "/day/tas/" not in store:
        raise ValueError("Unapproved GCM SSP native daily Kelvin data source")
    cal,shape=SOURCES[source]
    ds=xr.open_zarr(fsspec.get_mapper(store,anon=True,token="anon"),
                    consolidated=True,decode_times=False,chunks=None,mask_and_scale=True)
    try:
        v=ds["tas"]
        if v.attrs.get("units")!="K" or set(v.dims)!={"time","lat","lon"}:
            raise ValueError("Expected native Kelvin daily tas")
        lat=np.asarray(ds["lat"].values,dtype=float)
        lon=np.asarray(ds["lon"].values,dtype=float)%360
        lati=contiguous_span(lat,25,50)
        loni=contiguous_span(lon,235,260)
        reglat=lat[lati];reglon=lon[loni]
        if (len(reglat),len(reglon))!=shape:
            raise ValueError("Unexpected native region geometry")
        t=ds["time"]
        nativecal=str(t.attrs.get("calendar",t.encoding.get("calendar","standard")))
        units=str(t.attrs.get("units",""))
        if ((source=="AWI" and nativecal!="proleptic_gregorian") or
            (source=="BCC" and nativecal not in ("noleap","365_day")) or
            "days since" not in units):
            raise ValueError("Unverified native time calendar")
        numeric=np.asarray(t.values,dtype=float)
        dates=schedule();validate_schedule(dates)
        indices=[]
        for year,month,day in dates:
            i,j,sd,td=exact_daily_pair_indices(numeric,(year,month,day),units,nativecal)
            if sd[:10]!=f"{year:04}-{month:02}-{day:02}" or td[:4]!=str(year):
                raise ValueError("Native source +1 target date mismatch")
            indices.append((i,j))
        rows=[]
        # Read small remote slices in modest batches rather than full grid or century.
        for start in range(0,len(indices),14):
            pairs=indices[start:start+14]
            si=[i for i,_ in pairs];ti=[j for _,j in pairs]
            xs=np.asarray(v.isel(time=si,lat=lati,lon=loni).values,dtype=np.float32)
            ys=np.asarray(v.isel(time=ti,lat=lati,lon=loni).values,dtype=np.float32)
            if xs.shape!=(len(pairs),*shape) or ys.shape!=xs.shape or not (
                np.isfinite(xs).all() and np.isfinite(ys).all()):
                raise ValueError("Nonfinite or incomplete native regional cohort")
            if min(xs.min(),ys.min())<150 or max(xs.max(),ys.max())>350:
                raise ValueError("Implausible Kelvin values")
            for k in range(len(pairs)):
                year,month,day=dates[start+k]
                rows.append({"year":year,"month":month,"source_day":day,
                             "source_regional_K":weighted_mean(xs[k],reglat),
                             "target_regional_K":weighted_mean(ys[k],reglat),
                             "target_p95_spatial_K":float(np.quantile(ys[k],.95))})
        if len(rows)!=84:raise ValueError("Incomplete 2040 matched cohort")
        return {"gcm":source,"scenario":scenario,"calendar":nativecal,
                "grid_shape":list(shape),"sample_year":YEAR,"n_pairs":len(rows),
                "latitude_sha256":hashlib.sha256(reglat.tobytes()).hexdigest(),
                "longitude_sha256":hashlib.sha256(reglon.tobytes()).hexdigest(),
                "source_store":store,
                "mean_target_regional_K":float(np.mean([r["target_regional_K"] for r in rows])),
                "daily_records":rows}
    finally:
        ds.close()

def validate_records(rows):
    if len(rows)!=4:raise ValueError("Missing GCM × SSP cohort")
    indexed={(r["gcm"],r["scenario"]):r for r in rows}
    if len(indexed)!=4 or set(indexed)!={(g,s) for g in SOURCES for s in SCENARIOS}:
        raise ValueError("Duplicate or incomplete scenario cohort")
    for g in SOURCES:
        x=indexed[g,"ssp245"];y=indexed[g,"ssp585"]
        for field in ("latitude_sha256","longitude_sha256","grid_shape","n_pairs"):
            if x[field]!=y[field]:raise ValueError("SSP source grid mismatch")
        if x["n_pairs"]!=84:
            raise ValueError("Incomplete sample pairs")
        for item in (x,y):
            expected=[(m,d) for m in range(1,13) for d in DAYS]
            actual=[(r["month"],r["source_day"]) for r in item["daily_records"]]
            if actual!=expected:
                raise ValueError("Frozen source-day sequence changed")
    return True

def build(awi_manifest,bcc_manifest):
    awi=manifest_entry(json.loads(Path(awi_manifest).read_text()))
    bcc=get_model(json.loads(Path(bcc_manifest).read_text()))
    rows=[]
    for g,record in (("AWI",awi),("BCC",bcc)):
        for scenario in SCENARIOS:
            rows.append(extract(record[scenario]["store"],g,scenario))
    validate_records(rows)
    return {"status":"v511_verified_new_2040_seven_dates_per_month_source_cohort",
            "year":YEAR,"days_per_month":list(DAYS),"n_source_days_per_scenario":84,
            "original_sparse_days_excluded":[8,20],"source_only":True,
            "rows":rows,
            "limitations":["Only 84 prescribed dates in one year, not a full annual climatology",
                           "No frozen model predictions or SSP fitting were performed",
                           "Future simulation scenarios are not independent observations",
                           "Source calendar and native grid remain GCM-specific",
                           "Future 2040 cohort is not an ensemble significance test"]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--awi-manifest",required=True)
    parser.add_argument("--bcc-manifest",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    dest=Path(a.output)
    if dest.exists():raise FileExistsError("Do not overwrite pilot")
    d=build(a.awi_manifest,a.bcc_manifest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":d["status"],"scenarios":len(d["rows"]),
                      "pairs_per_scenario":84},indent=2))
if __name__=="__main__":main()
