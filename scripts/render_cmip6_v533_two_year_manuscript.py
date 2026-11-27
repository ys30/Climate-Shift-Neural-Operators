"""E1 v5.33: publication-ready paired 2040/2050 plots from audited v5.32 only."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

SRC="v532_three_GCM_two_summer_years_frozen_event_replication"
GCMS=("AWI","BCC","CAMS")
SSPS=("ssp245","ssp585")

def verify(report):
    if (report.get("status")!=SRC or report.get("independent_GCM_count")!=3
        or report.get("distinct_summer_years")!=[2040,2050]
        or report.get("year_GCM_SSP_cells")!=12 or report.get("descriptive_neural_arm_rows")!=72
        or report.get("original_frozen_neural_checkpoint_count")!=54
        or report.get("checkpoint_SHA_equal_between_years") is not True
        or report.get("ssp_used_for_fitting") is not False):
        raise ValueError("Source must be the complete frozen v5.32 two-year audit")
    rows=report.get("scenario_findings",[])
    expected={(g,y,s) for g in GCMS for y in (2040,2050) for s in SSPS}
    if len(rows)!=12 or {(r["gcm"],r["year"],r["scenario"]) for r in rows}!=expected:
        raise ValueError("Missing/duplicate model-year-SSP results")
    for r in rows:
        for k in ("persistence_event_recall","standard_fno_event_recall",
                  "strong_tail_fno_event_recall","standard_fno_rmse_skill_pct",
                  "four_x_minus_one_x_precision_pp",
                  "four_x_minus_one_x_rmse_K",
                  "persistence_minus_fno_recall_pp","four_x_minus_one_x_recall_pp"):
            v=r.get(k)
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v):
                raise ValueError("Missing/nonfinite original source: "+k)
        for k in ("persistence_event_recall","standard_fno_event_recall","strong_tail_fno_event_recall"):
            if not 0<=r[k]<=1:raise ValueError("Invalid source event Recall")
        if not math.isclose(100*(r["persistence_event_recall"]-r["standard_fno_event_recall"]),
                r["persistence_minus_fno_recall_pp"],abs_tol=1e-6):
            raise ValueError("Persistence-vs-FNO event gap corrupted")
        if not math.isclose(100*(r["strong_tail_fno_event_recall"]-r["standard_fno_event_recall"]),
                r["four_x_minus_one_x_recall_pp"],abs_tol=1e-6):
            raise ValueError("Strong-tail event Recall change corrupted")
    if len(report.get("complete_neural_arms",[]))!=72 or len(report.get("paired_year_findings",[]))!=6:
        raise ValueError("Original all-arm and paired-year outputs incomplete")
    return sorted(rows,key=lambda r:(GCMS.index(r["gcm"]),r["scenario"],r["year"]))

def make_plots(rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(output)
    if out.exists():raise FileExistsError("Do not overwrite independent v5.33 manuscript outputs")
    out.mkdir(parents=True)
    colors={"persistence":"#59656E","fno_1x":"#256A9E","fno_4x":"#C96A42"}

    # 2 rows of scenarios × 3 columns of independent climate models:
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),sharey=True)
    for c,g in enumerate(GCMS):
        for i,s in enumerate(SSPS):
            ax=axes[i,c]
            cases=[next(r for r in rows if (r["gcm"],r["scenario"],r["year"])==(g,s,y)) for y in (2040,2050)]
            for name,k,color,marker in [
              ("Persistence","persistence_event_recall",colors["persistence"],"s"),
              ("FNO 1×","standard_fno_event_recall",colors["fno_1x"],"o"),
              ("FNO 4×","strong_tail_fno_event_recall",colors["fno_4x"],"^")]:
                ax.plot([2040,2050],[100*r[k] for r in cases],marker=marker,
                    label=name,color=color,lw=2,markersize=7)
            ax.set_xticks([2040,2050]);ax.set_ylim(0,100)
            ax.set_title(f"{g} / {s.upper()}")
            ax.grid(alpha=.16)
    for a in axes[:,0]:a.set_ylabel("Same-cell >=3-day event Recall (%)")
    for a in axes[-1]:a.set_xlabel("Simulated future summer")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,.97),frameon=False)
    fig.suptitle("Frozen neural operators: summer hot-event Recall in 2040 vs 2050",y=1.015)
    fig.text(.04,.01,"3 independent GCMs, 2 correlated SSP pathways per GCM, original Historical-only monthly P95; descriptive comparison.",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.93))
    fig.savefig(out/"figure_v533_three_GCM_two_year_event_recall.png",dpi=200,bbox_inches="tight")
    plt.close(fig)

    # A deliberately transparent display of all RMSE/Recall/Precision effects.
    fig,axes=plt.subplots(1,3,figsize=(15.0,5.0))
    cases=[f"{r['gcm']} {r['scenario'][-3:]} {r['year']}" for r in rows]
    x=np.arange(len(rows))
    series=[
      ("Standard FNO vs Persistence: RMSE skill (%)","standard_fno_rmse_skill_pct"),
      ("FNO 4× minus 1×: event Recall (pp)","four_x_minus_one_x_recall_pp"),
      ("FNO 4× minus 1×: event Precision (pp)","four_x_minus_one_x_precision_pp")]
    for ax,(label,key) in zip(axes,series):
        ys=[r[key] for r in rows]
        ax.axhline(0,color="#444444",lw=.8)
        ax.bar(x,ys,color=["#256A9E" if y>=0 else "#C96A42" for y in ys])
        ax.set_title(label,fontsize=10)
        ax.set_xticks(x,cases,rotation=75,ha="right",fontsize=7)
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("Accuracy–event-detection tradeoffs across two summers")
    fig.text(.02,.01,"All 12 correlated GCM×year×SSP cells shown, including exceptions; do not pool cells as independent replicates.",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.94))
    fig.savefig(out/"figure_v533_RMSE_recall_precision_tradeoffs.png",dpi=200,bbox_inches="tight")
    plt.close(fig)

    with (out/"table_v533_12_two_year_event_cases.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    except_cases=[(r["gcm"],r["year"],r["scenario"],r["four_x_minus_one_x_precision_pp"])
                  for r in rows if r["four_x_minus_one_x_precision_pp"]>=0]
    text=["# E1 v5.33 — Paired-year multi-objective manuscript Results","",
      "Three independently simulated CMIP6 climate-model members (AWI, BCC, CAMS) were assessed in 2040 and 2050 across SSP245 and SSP585 using original fixed Historical-only checkpoints (54 total across the three models). Each GCM–year–scenario cell contains 91 consecutive +1-day native-grid simulated daily-mean TAS targets, with a hot event defined by at least three consecutive days above that GCM's Historical-training monthly local P95 at the same grid cell.","",
      "| Metric direction | GCM–year–SSP cells |",
      "|---|---:|",
      "| Standard FNO reduces overall RMSE relative to Persistence | 12/12 |",
      "| Standard FNO event Recall falls below Persistence | 12/12 |",
      "| FNO 4× Historical high-TAS loss increases event Recall vs FNO 1× | 12/12 |",
      "| FNO 4× event Precision falls vs FNO 1× | 11/12 |",
      "| FNO 4× increases RMSE vs FNO 1× | 6/12 |","",
      "In these two distinct simulated summers, standard FNO consistently improved weighted next-day temperature RMSE relative to native persistence, while missing more sustained hot-event-days. A stronger Historical high-TAS training weight improved event-day sensitivity in every evaluated cell, but generally lowered event-day Precision. There is no uniform RMSE penalty from the stronger high-TAS weighting; six of 12 cells saw RMSE increase, and six did not.","",
      "## Precision counterexample (must be retained)",""]
    for g,y,s,val in except_cases:
        text.append(f"- {g}, {y}, {s}: FNO 4× minus 1× event Precision {val:+.3f} percentage points.")
    if not except_cases:text.append("- None in this source.")
    text+=["","## Interpretation and limitations","",
      "The 12 evaluation cells do NOT represent 12 independent climate-model replicates; there are three independently simulated GCM members, with years and SSP scenarios correlated within each. The 2040 follow-up was planned after inspecting the 2050 results, so cross-year agreement is promising descriptive replication, not preregistered confirmatory evidence from independent GCM samples. Events represent simulated daily-mean TAS persistence in native grid cells rather than observed tasmax heatwaves, tracked heatwave systems, or human heat-exposure risk. Summer edges censor events. No p-values or independence-based significance claims should be inferred from the 12 cells, 91 days, spatial pixels, or the three neural initialization seeds.","",
      "Source: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46080073542",""]
    (out/"E1_v533_two_year_three_GCM_results.md").write_text("\n".join(text),encoding="utf-8")
    return except_cases

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    args=p.parse_args()
    rows=verify(json.loads(Path(args.input).read_text()))
    exceptions=make_plots(rows,args.output_dir)
    print(json.dumps({"status":"v533_three_GCM_two_year_manuscript_figures",
       "cases":len(rows),"precision_counterexamples":exceptions},indent=2))
if __name__=="__main__":main()
