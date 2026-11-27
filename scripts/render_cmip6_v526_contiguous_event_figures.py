"""E1 v5.26 publication figures from successful immutable v5.25 report only."""
from __future__ import annotations
import argparse,json,csv,math
from pathlib import Path
from scripts.audit_cmip6_v525_three_gcm_contiguous_events import STATUS as SOURCE_STATUS
CASES=[(g,s) for g in ("AWI","BCC","CAMS") for s in ("ssp245","ssp585")]
def verify(source):
    if (source.get("status")!=SOURCE_STATUS or source.get("independent_GCM_count")!=3
        or source.get("scenario_cells")!=6 or source.get("neural_arm_rows")!=36
        or source.get("frozen_checkpoint_count")!=54
        or source.get("ssp_used_for_fitting_or_selection") is not False):
        raise ValueError("Only successful complete v5.25 report permitted")
    rows=source.get("scenario_findings",[])
    if len(rows)!=6 or {(x["gcm"],x["scenario"]) for x in rows}!=set(CASES):
        raise ValueError("Missing independent GCM SSP outcomes")
    index={(x["gcm"],x["scenario"]):x for x in rows}
    for r in rows:
        for key in ("persistence_event_day_recall","standard_fno_event_day_recall",
                    "strong_tail_fno_event_day_recall","standard_fno_relative_rmse_skill_pct",
                    "strong_tail_minus_standard_rmse_K",
                    "standard_fno_event_day_precision","strong_tail_fno_event_day_precision",
                    "persistence_event_day_precision"):
            v=r[key]
            if v is None or not isinstance(v,(int,float)) or not math.isfinite(v):
                raise ValueError("No undefined or unverified manuscript metric: "+key)
            if "recall" in key or "precision" in key:
                if key.endswith(("_recall","_precision")) and not 0<=v<=1:
                    raise ValueError("Invalid event probability")
        p=r["persistence_event_day_recall"]
        a=r["standard_fno_event_day_recall"]
        b=r["strong_tail_fno_event_day_recall"]
        if not math.isclose(100*(p-a),r["persistence_minus_standard_fno_event_recall_pp"],abs_tol=1e-6):
            raise ValueError("Stored event gap changed")
        if not math.isclose(100*(b-a),r["strong_tail_minus_standard_event_recall_pp"],abs_tol=1e-6):
            raise ValueError("Stored event change changed")
    return [index[k] for k in CASES]

def render(rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output)
    if output.exists():raise FileExistsError("Refuse to overwrite v5.26 output")
    output.mkdir(parents=True)
    names=[f"{r['gcm']} {r['scenario'].upper()}" for r in rows]
    loc=np.arange(6)
    # Figure 1: exact source event recall, three fixed treatments per cell.
    fig,ax=plt.subplots(figsize=(10.8,5.0))
    width=.25
    for off,key,title,color in [
        (-1,"persistence_event_day_recall","Persistence","#5A6472"),
        (0,"standard_fno_event_day_recall","FNO (1×)","#3273A8"),
        (1,"strong_tail_fno_event_day_recall","FNO (4×)","#CF6C43")]:
        ax.bar(loc+off*width,[100*r[key] for r in rows],width,label=title,color=color)
    ax.set_xticks(loc,names);ax.set_ylim(0,100)
    ax.set_ylabel("Continuous >=3-day local TAS event-day Recall (%)")
    ax.set_title("2050 summer: same-pixel continuous hot-event detection")
    ax.legend(frameon=False,ncol=3)
    fig.text(.02,.01,"3 independent GCM members; 2 paired SSP scenarios/GCM; 91 consecutive target days/cell; descriptive only.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/"figure_v526_continuous_event_recall.png",dpi=210,bbox_inches="tight");plt.close(fig)
    # Figure 2: native Persistence-relative RMSE skill and extreme event-day Recall gap.
    fig,ax=plt.subplots(figsize=(9.3,5.2))
    for g,color in (("AWI","#3273A8"),("BCC","#DA8B3C"),("CAMS","#458C79")):
        rr=[r for r in rows if r["gcm"]==g]
        ax.scatter([r["standard_fno_relative_rmse_skill_pct"] for r in rr],
                   [r["persistence_minus_standard_fno_event_recall_pp"] for r in rr],
                   color=color,s=85,label=g)
        for r in rr:
            ax.annotate(r["scenario"].upper(),(r["standard_fno_relative_rmse_skill_pct"],
                r["persistence_minus_standard_fno_event_recall_pp"]),
                xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.axhline(0,color="#555555",lw=.8)
    ax.axvline(0,color="#555555",lw=.8)
    ax.set_xlabel("Standard FNO weighted RMSE improvement vs native Persistence (%)")
    ax.set_ylabel("Persistence − FNO continuous event-day Recall (pp)")
    ax.set_title("RMSE gain does not guarantee sustained-hot-event Recall")
    ax.legend(frameon=False)
    fig.text(.02,.01,"Source-specific baseline and historical monthly-local P95; points within the same GCM are paired.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/"figure_v526_rmse_vs_event_recall_gap.png",dpi=210,bbox_inches="tight");plt.close(fig)
    # Figure 3: explicit per-scenario precision and recall effects of 4x training loss.
    fig,ax=plt.subplots(figsize=(9.8,4.7))
    ax.axhline(0,color="#333333",lw=.8)
    ax.plot(loc,[r["strong_tail_minus_standard_event_recall_pp"] for r in rows],
        "o-",label="Event Recall change (pp)",color="#3273A8")
    ax.plot(loc,[100*(r["strong_tail_fno_event_day_precision"]-r["standard_fno_event_day_precision"])
                for r in rows],"s--",label="Event Precision change (pp)",color="#CF6C43")
    ax.set_xticks(loc,names)
    ax.set_ylabel("FNO 4× minus 1×, percentage points")
    ax.set_title("Greater hot-event sensitivity and potential false-positive cost")
    ax.legend(frameon=False)
    fig.text(.02,.01,"The trained checkpoints remain frozen; no 2050 tuning or independent ensemble inferential test.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/"figure_v526_tail_weight_precision_recall.png",dpi=210,bbox_inches="tight");plt.close(fig)
    with (output/"table_v526_three_gcm_2050_continuous_events.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    draft=["# E1 v5.26 — Manuscript continuous event-day results (source verified)","",
      "Three independent native-grid CMIP6 GCMs were evaluated for the 2050 summer season using original, historically trained and frozen neural checkpoints. Each GCM contributes two correlated SSP scenarios (not six independent GCM replicates). Events are defined as at least three consecutive daily-mean simulated TAS exceedances of GCM-specific Historical-training local monthly P95 at the same grid cell. All comparisons are descriptive.","",
      "**Full six-case results**","",
      "| GCM | SSP | Persistence Recall % | FNO 1× Recall % | FNO 4× Recall % | RMSE skill 1× % | 4×−1× RMSE K | 4×−1× Precision pp |",
      "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        prec=100*(r["strong_tail_fno_event_day_precision"]-r["standard_fno_event_day_precision"])
        draft.append(f"| {r['gcm']} | {r['scenario']} | {100*r['persistence_event_day_recall']:.2f} | {100*r['standard_fno_event_day_recall']:.2f} | {100*r['strong_tail_fno_event_day_recall']:.2f} | {r['standard_fno_relative_rmse_skill_pct']:.2f} | {r['strong_tail_minus_standard_rmse_K']:+.3f} | {prec:+.2f} |")
    draft+=["","Across all six GCM–SSP cells, standard-loss FNO reduced weighted next-day TAS RMSE relative to its own persistence baseline, but the persistence forecast attained higher sustained-hot-event-day Recall. Increasing the Historical hot-target multiplier from 1× to 4× improved event-day Recall in all six cells without overturning persistence's Recall advantage. The RMSE response to the stronger tail-weight was not consistent: it increased in some cells and decreased in others. Precision shifts must be reported concurrently because higher Recall can result from more predicted high-event days.",
      "","**Scope limitations:** One GCM member each, one exploratory 2050 summer, simulated daily-mean TAS rather than tasmax, same-cell same-day event labels rather than physically tracked heatwaves, censored boundary events, grid/time dependence and no inferential p-values. No future training or checkpoint selection.","",
      "Source audit: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45948547935",""]
    (output/"E1_v526_manuscript_continuous_event_results.md").write_text("\n".join(draft),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output-dir",required=True)
    a=ap.parse_args()
    rows=verify(json.loads(Path(a.input).read_text()))
    render(rows,a.output_dir)
    print(json.dumps({"status":"v526_three_GCM_verified_contiguous_manuscript_figures",
       "three_independent_GCMs":3,"six_correlated_SSP_cells":6,
       "cases_standard_FNO_lower_RMSE":sum(r["standard_fno_relative_rmse_skill_pct"]>0 for r in rows),
       "cases_persistence_higher_hot_event_Recall":sum(r["persistence_minus_standard_fno_event_recall_pp"]>0 for r in rows),
       "cases_4x_higher_hot_event_Recall":sum(r["strong_tail_minus_standard_event_recall_pp"]>0 for r in rows)},indent=2))
if __name__=="__main__":main()
