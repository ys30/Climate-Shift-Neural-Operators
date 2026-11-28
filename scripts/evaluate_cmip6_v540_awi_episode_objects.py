"""E1 v5.40 — AWI matched same-pixel high-TAS episodes (exploratory).

An *episode* is a contiguous run of >=3 above-Historical-P95 daily mean TAS
target days in a single native grid pixel. This is NOT a spatial heatwave
object. Match a forecast episode to an observed episode only in the same pixel
with >=1 overlapping day, using deterministic greedy maximum overlap and
one-to-one exclusion (no post hoc timing tolerance). Report episode recall,
false alarms, fragmentation and duration signed error only for uncensored
matched episodes. Strictly reproduce original 3-day event-DAY metrics first.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scripts.evaluate_cmip6_v529_awi_2040_contiguous_holdout import verified_frozen
from scripts.evaluate_cmip6_v524_awi_contiguous_summer import read_awi_91,predict
from scripts.evaluate_cmip6_v48_frozen_ssp import manifest_entry
from scripts.evaluate_cmip6_v534_awi_duration_sensitivity import (
    event,check_match,summarize,YEARS,SSPS,SEEDS,WEIGHTS)

STATUS="v540_AWI_2040_2050_frozen_same_pixel_episode_matching"
MIN_DAYS=3
def segments(binary,min_days=MIN_DAYS):
    """Half-open [start,end) tuples for exact contiguous true runs in one pixel."""
    z=np.asarray(binary,dtype=bool)
    if z.ndim!=1 or min_days<2 or min_days>len(z):
        raise ValueError("Expected one native-pixel daily series and valid event length")
    padded=np.r_[False,z,False]
    starts=np.flatnonzero(np.diff(padded.astype(np.int8))==1)
    ends=np.flatnonzero(np.diff(padded.astype(np.int8))==-1)
    return [(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=min_days]

def match_one_pixel(truth,pred,n_days=91):
    """One-to-one greedy max overlap; return matched pairs and ambiguous links."""
    if any(a<0 or b>n_days or a>=b for a,b in truth+pred):
        raise ValueError("Event intervals must remain inside 91-day observation window")
    links=[]
    for i,(a,b) in enumerate(truth):
        for j,(c,d) in enumerate(pred):
            overlap=min(b,d)-max(a,c)
            if overlap>0:
                links.append((overlap,i,j,abs(a-c)))
    # Deterministic priority: most overlapping days, then nearest onset, then indices.
    links.sort(key=lambda x:(-x[0],x[3],x[1],x[2]))
    observed_used=set();forecast_used=set();pairs=[]
    for overlap,i,j,_ in links:
        if i not in observed_used and j not in forecast_used:
            pairs.append((i,j,overlap))
            observed_used.add(i);forecast_used.add(j)
    # Fragmentation is count of true episodes overlapped by >=2 distinct forecast episodes.
    per_truth={i:set() for i in range(len(truth))}
    for _,i,j,_lag in links:per_truth[i].add(j)
    fragmentation=sum(len(per_truth[i])>=2 for i in per_truth)
    return {"pairs":pairs,"fragmented_truth":fragmentation,
            "matched_truth":len(observed_used),"matched_pred":len(forecast_used),
            "overlapping_candidate_edges":len(links)}

def diagnostic(pred,target,months,p95,latitude):
    if (pred.shape!=target.shape or pred.ndim!=4 or pred.shape[0]!=91
        or pred.shape[1]!=1 or p95.shape!=(12,*pred.shape[2:])
        or np.asarray(months).shape!=(91,) or np.asarray(latitude).shape!=(pred.shape[2],)):
        raise ValueError("Original 91 target-day native geometry required for episode analysis")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Missing native daily mean TAS values")
    threshold=p95[np.asarray(months)-1]
    truth=(target[:,0]>threshold)
    forecast=(pred[:,0]>threshold)
    latw=np.cos(np.deg2rad(latitude))
    if not np.all(latw>0):raise ValueError("Invalid native latitude weights")
    totals={k:0. for k in (
        "observed","predicted","matched","fragmented_truth","boundary_truth",
        "boundary_forecast","matched_interior","duration_signed_error_weighted",
        "duration_absolute_error_weighted","onset_signed_error_weighted")}
    max_overlap=0
    for iy in range(truth.shape[1]):
        w=float(latw[iy])
        for ix in range(truth.shape[2]):
            t=segments(truth[:,iy,ix]);p=segments(forecast[:,iy,ix])
            m=match_one_pixel(t,p)
            totals["observed"]+=w*len(t)
            totals["predicted"]+=w*len(p)
            totals["matched"]+=w*m["matched_truth"]
            totals["fragmented_truth"]+=w*m["fragmented_truth"]
            totals["boundary_truth"]+=w*sum(a==0 or b==91 for a,b in t)
            totals["boundary_forecast"]+=w*sum(a==0 or b==91 for a,b in p)
            max_overlap+=m["overlapping_candidate_edges"]
            for i,j,_days in m["pairs"]:
                a,b=t[i];c,d=p[j]
                # Any boundary-contacting matched event is duration censored.
                if a==0 or b==91 or c==0 or d==91:continue
                totals["matched_interior"]+=w
                totals["duration_signed_error_weighted"]+=w*((d-c)-(b-a))
                totals["duration_absolute_error_weighted"]+=w*abs((d-c)-(b-a))
                totals["onset_signed_error_weighted"]+=w*(c-a)
    obs=totals["observed"];prd=totals["predicted"];matched=totals["matched"]
    interior=totals["matched_interior"]
    return {
      "episode_recall":matched/obs if obs else None,
      "episode_precision":matched/prd if prd else None,
      "area_weighted_observed_episodes":obs,
      "area_weighted_forecast_episodes":prd,
      "area_weighted_matched_episodes":matched,
      "area_weighted_fragmented_true_episodes":totals["fragmented_truth"],
      "fragmentation_rate":totals["fragmented_truth"]/obs if obs else None,
      "area_weighted_boundary_censored_true_episodes":totals["boundary_truth"],
      "area_weighted_boundary_censored_predicted_episodes":totals["boundary_forecast"],
      "area_weighted_matched_interior_episodes":interior,
      "mean_matched_interior_duration_bias_days":totals["duration_signed_error_weighted"]/interior if interior else None,
      "mean_matched_interior_duration_absolute_error_days":totals["duration_absolute_error_weighted"]/interior if interior else None,
      "mean_matched_interior_onset_lag_days":totals["onset_signed_error_weighted"]/interior if interior else None,
      "candidate_overlap_edges":max_overlap,
      "matching_rule":"single pixel, >=1 day overlap, one-to-one greedy maximum overlap; no spatial or temporal tolerance",
      "boundary_policy":"Count edge-contacting episodes for detection; exclude any matched pair touching day 1 or day 91 from duration/onset estimates"}

def audit_3day(pred,target,months,p95,latitude,expected,label):
    scores=event(pred,target,months,p95,latitude,3)
    for k in ("event_day_recall","event_day_precision"):
        check_match(scores[k],expected[k],label+"/"+k)
    return scores

def analyze(frozen,manifest,reference_2040,reference_2050):
    if len(frozen.get("models",{}))!=18 or len(frozen.get("checkpoint_sha256_by_arm",{}))!=18:
        raise ValueError("Require all 18 original SHA-verified Historical-only AWI checkpoints before reading future SSP data")
    sha=frozen["checkpoint_sha256_by_arm"]
    for year,ref in ((2040,reference_2040),(2050,reference_2050)):
        if (ref.get("year")!=year or ref.get("source_model")!="AWI-CM-1-1-MR"
            or ref.get("ssp_fitted") is not False
            or ref.get("checkpoint_sha")!=sha
            or set(ref.get("scenarios",{}))!=set(SSPS)):
            raise ValueError("AWI frozen archive SHA or three-day reference changed")
    row=manifest_entry(manifest)
    import torch
    torch.set_num_threads(1)
    records=[]
    for year,ref in ((2040,reference_2040),(2050,reference_2050)):
        for scenario in SSPS:
            X,Y,months,meta=read_awi_91(row[scenario]["store"],scenario,frozen,year=year)
            if X.shape!=Y.shape!=(91,1,26,27):
                raise ValueError("Original AWI native 91-day source grid missing")
            threshold=frozen["local_p95"];latitude=frozen["latitude"]
            expected=ref["scenarios"][scenario]
            base=audit_3day(X,Y,months,threshold,latitude,expected["persistence"],
                            f"{year}/{scenario}/Persistence")
            records.append({"year":year,"scenario":scenario,"model":"persistence","historical_multiplier":None,
                "seed":None,"three_day_event_day_recall":base["event_day_recall"],
                "three_day_event_day_precision":base["event_day_precision"],
                **diagnostic(X,Y,months,threshold,latitude)})
            for weight in WEIGHTS:
                per_seed=[];recalled=[];precisions=[]
                for seed in SEEDS:
                    forecast=predict(frozen,"fno",weight,seed,X)
                    d=diagnostic(forecast,Y,months,threshold,latitude)
                    eventday=event(forecast,Y,months,threshold,latitude,3)
                    recalled.append(eventday["event_day_recall"])
                    precisions.append(eventday["event_day_precision"])
                    per_seed.append({"year":year,"scenario":scenario,"model":"fno",
                        "historical_multiplier":1+weight,"seed":seed,
                        "three_day_event_day_recall":eventday["event_day_recall"],
                        "three_day_event_day_precision":eventday["event_day_precision"],**d})
                old=expected["models"]["fno"][str(weight)]
                check_match(summarize([{"v":v} for v in recalled],"v"),old["event_day_recall"]["mean"],
                     f"{year}/{scenario}/FNO/{weight}/event-day-Recall")
                check_match(summarize([{"v":v} for v in precisions],"v"),old["event_day_precision"]["mean"],
                     f"{year}/{scenario}/FNO/{weight}/event-day-Precision")
                records.extend(per_seed)
    if len(records)!=28:raise ValueError("Require 4 SSP-year scenarios x (1 persistence + 6 frozen FNO seeds)")
    return {"status":STATUS,"source_model":"AWI-CM-1-1-MR",
        "independent_GCM_count":1,"years":[2040,2050],"scenarios":list(SSPS),
        "three_day_reference_reproduced":True,"ssp_fitted":False,
        "original_frozen_checkpoint_sha":sha,"original_frozen_checkpoint_count":18,
        "records":records,"limitations":[
          "Events are single native-pixel TAS runs and are NOT spatially tracked heatwave objects",
          "One-to-one greedy maximum-overlap is a predefined deterministic convention, not a unique physical match",
          "Same-pixel event matching accepts >=1 overlapping day and no onset tolerance",
          "Boundary-censored matched episodes excluded from duration and onset bias estimates",
          "2040 and 2050 and SSP245/585 share one original independent GCM member",
          "Three neural seeds are optimization repetitions not independent GCM samples",
          "This is a post hoc exploratory mechanism study following three-day event-day analysis",
          "Daily mean model TAS is not observed tasmax, human heat exposure or full heatwave persistence",
          "No future SSP weights, P95 thresholds, model selection or normalization changes"]}

def main():
    p=argparse.ArgumentParser()
    for key in ("historical","arm-dir","manifest","reference-2040","reference-2050","output"):
        p.add_argument("--"+key,required=True)
    a=vars(p.parse_args());target=Path(a["output"])
    if target.exists():raise FileExistsError("Do not overwrite original episode diagnostic evidence")
    f=verified_frozen(a["historical"],a["arm_dir"])
    report=analyze(f,json.loads(Path(a["manifest"]).read_text()),
                   json.loads(Path(a["reference_2040"]).read_text()),
                   json.loads(Path(a["reference_2050"]).read_text()))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":STATUS,"records":len(report["records"]),
       "original_three_day_event_day_metrics_reproduced":report["three_day_reference_reproduced"],
       "independent_GCM_count":1,
       "episode_recall_by_year_ssp":{f"{r['year']}_{r['scenario']}_{r['model']}_{r['historical_multiplier']}_{r['seed']}":
           r["episode_recall"] for r in report["records"]}},indent=2))
if __name__=="__main__":main()
