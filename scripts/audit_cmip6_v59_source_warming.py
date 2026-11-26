"""Source-only historical-to-SSP sparse CMIP6 simulated temperature reference.

Historical validation 2005-2014 (target 9th/21st of each month) is
a descriptive reference, NOT an untouched neural-model holdout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_cmip6_v48_frozen_ssp import read_future_once, manifest_entry
from scripts.evaluate_cmip6_bcc_v54_frozen_ssp import get_future_pairs
from scripts.audit_cmip6_bcc_regional_pairs import get_model

SOURCES={"AWI":"ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914",
         "BCC":"a23260e8f14a23fb57483038bcd6ca03ba74c7944a48ba29d62bcc5bd354f452"}
YEARS=(2030,2050,2080)

def weighted_mean(grid,lat):
    grid=np.asarray(grid,dtype=float)
    lat=np.asarray(lat,dtype=float)
    if grid.ndim!=2 or lat.shape!=(grid.shape[0],) or not np.isfinite(grid).all():
        raise ValueError("Invalid native Kelvin grid")
    w=np.cos(np.deg2rad(lat))
    if not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError("Invalid latitudes")
    return float(np.average(grid,weights=np.broadcast_to(w[:,None],grid.shape)))

def compare(hist,mask,future,lat,years):
    hist=np.asarray(hist);mask=np.asarray(mask,dtype=bool)
    future=np.asarray(future);years=np.asarray(years)
    if hist.ndim!=4 or future.ndim!=4 or hist.shape[1:]!=future.shape[1:] or hist.shape[1]!=1:
        raise ValueError("Native grids do not match")
    if mask.shape!=(len(hist),) or int(mask.sum())!=240:
        raise ValueError("Require 240 Historical reference targets")
    if len(future)!=72 or any(int(np.sum(years==y))!=24 for y in YEARS):
        raise ValueError("Require 24 future targets per year")
    if not np.isfinite(hist).all() or not np.isfinite(future).all():
        raise ValueError("Nonfinite Kelvin temperature")
    base=np.mean(hist[mask,0],axis=0)
    base_regional=weighted_mean(base,lat)
    out=[]
    for yr in YEARS:
        projected=np.mean(future[years==yr,0],axis=0)
        change=projected-base
        delta=weighted_mean(change,lat)
        if not np.isclose(delta,weighted_mean(projected,lat)-base_regional,atol=1e-8):
            raise ValueError("Area-weight anomaly identity failed")
        out.append({"year":yr,"n_historical_pairs":240,"n_future_pairs":24,
            "historical_regional_mean_K":base_regional,
            "projected_sparse_regional_mean_K":weighted_mean(projected,lat),
            "sampled_change_vs_historical_K":delta,
            "spatial_anomaly_p05_K":float(np.quantile(change,.05)),
            "spatial_anomaly_p50_K":float(np.median(change)),
            "spatial_anomaly_p95_K":float(np.quantile(change,.95))})
    return out

def historical(path,source):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCES[source]:
        raise ValueError("Frozen Historical NPZ checksum mismatch")
    with np.load(path,allow_pickle=False) as z:
        y=z["y"].copy(); years=z["years"].copy()
        lat=z["latitude"].copy()
        if source=="AWI":
            dates=z["target_times"].astype("datetime64[D]").astype(str)
            mask=(years>=2005)&(years<=2014)
            if any(d[8:10] not in ("09","21") for d in dates[mask]):
                raise ValueError("AWI historical target date mismatch")
            months=np.array([int(d[5:7]) for d in dates])
            expected=(1140,1,26,27)
        else:
            mask=~z["train_mask"].astype(bool)
            months=z["months"].copy()
            if not np.array_equal(mask,(years>=2005)&(years<=2014)):
                raise ValueError("BCC native Historical mask mismatch")
            expected=(1140,1,23,23)
        if y.shape!=expected or len(lat)!=y.shape[2] or mask.sum()!=240:
            raise ValueError("Historical source/grid shape mismatch")
        if any(int(np.sum(mask & (months==m)))!=20 for m in range(1,13)):
            raise ValueError("Missing Historical monthly dates")
        lon=z["longitude_east"].copy() if source=="BCC" else None
    return y,mask,lat,lon

def run(awi_npz,bcc_npz,awi_manifest,bcc_manifest):
    source_data={g:historical(path,g) for g,path in
                 (("AWI",awi_npz),("BCC",bcc_npz))}
    a=manifest_entry(json.loads(Path(awi_manifest).read_text()))
    b=get_model(json.loads(Path(bcc_manifest).read_text()))
    rows=[]
    for g,record in (("AWI",a),("BCC",b)):
        h,mask,lat,lon=source_data[g]
        for scenario in ("ssp245","ssp585"):
            if g=="AWI":
                _,f,yr,_=read_future_once(record[scenario]["store"],scenario,lat)
            else:
                _,f,yr,_=get_future_pairs(record[scenario]["store"],scenario,lat,lon)
            rows.extend(dict(gcm=g,scenario=scenario,**r) for r in compare(h,mask,f,lat,yr))
    if len(rows)!=12:raise ValueError("Incomplete source-only temperature baseline")
    return {"status":"v59_sparse_CMIP6_source_only_temperature_reference",
            "historical_reference":[2005,2014],"future_years":list(YEARS),
            "future_source_days":[8,20],"native_sources_sha256":SOURCES,"rows":rows,
            "limits":["Not observed future temperatures or full annual warming",
             "Historical validation was used for previous model checkpoint selection",
             "24 target days per future year versus 240 reference target days",
             "No model prediction, fitting or selection occurs in this source-only audit",
             "No pooling native GCM grid pixels as independent replications"]}

def main():
    p=argparse.ArgumentParser()
    for flag in ("awi-npz","bcc-npz","awi-manifest","bcc-manifest","output"):
        p.add_argument("--"+flag,required=True)
    a=vars(p.parse_args())
    dest=Path(a["output"])
    if dest.exists():raise FileExistsError("Immutable output already exists")
    result=run(a["awi_npz"],a["bcc_npz"],a["awi_manifest"],a["bcc_manifest"])
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Source-only verified rows:",len(result["rows"]))
if __name__=="__main__":
    main()
