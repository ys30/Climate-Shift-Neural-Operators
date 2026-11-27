"""E1 v5.28: manuscript Pareto / precision–recall figures from real v5.27 audit.

No model fitting, no data-dependent checkpoint selection, no p-values.
All chart data must be the immutable successfully audited v5.27 archive.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
SRC="v527_three_GCM_contiguous_event_multiobjective_descriptive_audit"
GCMS=("AWI","BCC","CAMS")
SCENARIOS=("ssp245","ssp585")
FAMILIES=("persistence","fno","unet")
def verified(source):
    if (source.get("status")!=SRC
        or source.get("independent_GCM_count")!=3
        or source.get("scenario_cells")!=6
        or source.get("candidate_rows")!=42
        or source.get("original_neural_checkpoint_count")!=54
        or source.get("no_future_refit") is not True):
        raise ValueError("Only complete v5.27 frozen Pareto source is allowed")
    records=source.get("candidate_dominance",[])
    if len(records)!=42:
        raise ValueError("Missing original seven candidates for six scenarios")
    keyset=set()
    for r in records:
        if (r.get("gcm") not in GCMS or r.get("scenario") not in SCENARIOS
            or r.get("model") not in FAMILIES):
            raise ValueError("Unexpected model/SSP/GCM")
        weight=r.get("loss_multiplier")
        if r["model"]=="persistence":
            if weight is not None:raise ValueError("Persistence cannot have training weight")
        elif weight not in (1,2,4):raise ValueError("Changed frozen loss arm")
        key=(r["gcm"],r["scenario"],r["model"],weight)
        if key in keyset:raise ValueError("Duplicate frozen model choice")
        keyset.add(key)
        for name in ("rmse_K","event_day_recall","event_day_precision"):
            val=r.get(name)
            if isinstance(val,bool) or not isinstance(val,(float,int)) or not math.isfinite(val):
                raise ValueError("Missing/nonfinite source score")
            if name=="rmse_K" and val<=0:raise ValueError("Nonpositive RMSE")
            if name!="rmse_K" and not 0<=val<=1:raise ValueError("Invalid event probability")
        if not isinstance(r.get("pareto_nondominated"),bool):
            raise ValueError("Non-boolean original Pareto status")
    expected={(g,s,m,w) for g in GCMS for s in SCENARIOS for m in FAMILIES
              for w in ((None,) if m=="persistence" else (1,2,4))}
    if keyset!=expected:raise ValueError("Missing expected original model arm")
    # Verify the archived Pareto memberships against unmodified native scores.
    def dominates(a,b):
        return (a["rmse_K"]<=b["rmse_K"]+1e-9
            and a["event_day_recall"]>=b["event_day_recall"]-1e-9
            and a["event_day_precision"]>=b["event_day_precision"]-1e-9
            and (a["rmse_K"]<b["rmse_K"]-1e-9
              or a["event_day_recall"]>b["event_day_recall"]+1e-9
              or a["event_day_precision"]>b["event_day_precision"]+1e-9))
    for g in GCMS:
        for s in SCENARIOS:
            sub=[r for r in records if r["gcm"]==g and r["scenario"]==s]
            if len(sub)!=7:raise ValueError("Missing scenario comparison")
            for r in sub:
                calculated=not any(dominates(other,r) for other in sub if other is not r)
                if calculated!=r["pareto_nondominated"]:
                    raise ValueError("Original Pareto label inconsistent with metrics")
    return records

def draw(records,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(out)
    if out.exists():raise FileExistsError("Refusing to overwrite scientific figures")
    out.mkdir(parents=True)

    # Figure A: six independent within-cell Pareto comparisons; x=event recall,
    # y=precision, point size distinguishes RMSE. No pooling across native grids.
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),sharex=True,sharey=True)
    styles={"persistence":("#4A5565","s"),"fno":("#2677A8","o"),"unet":("#C97C45","^")}
    for ri,scenario in enumerate(SCENARIOS):
        for ci,gcm in enumerate(GCMS):
            ax=axes[ri,ci]
            subset=[r for r in records if r["gcm"]==gcm and r["scenario"]==scenario]
            for row in subset:
                color,marker=styles[row["model"]]
                size=42+75*(1/(row["rmse_K"]))
                if row["pareto_nondominated"]:
                    ax.scatter(100*row["event_day_recall"],100*row["event_day_precision"],
                        s=size,marker=marker,c=color,alpha=.9,zorder=3,
                        edgecolors="#222222",linewidths=.7)
                else:
                    ax.scatter(100*row["event_day_recall"],100*row["event_day_precision"],
                        s=size,marker=marker,facecolors="none",edgecolors=color,
                        linewidths=1.3,alpha=.85)
                label=("P" if row["model"]=="persistence"
                       else ("F" if row["model"]=="fno" else "U")+str(row["loss_multiplier"]))
                ax.annotate(label,(100*row["event_day_recall"],100*row["event_day_precision"]),
                            xytext=(4,4),textcoords="offset points",fontsize=7)
            ax.set_title(f"{gcm} / {scenario.upper()}")
            ax.grid(alpha=.17)
    for ax in axes[1]:ax.set_xlabel("Continuous event-day Recall (%)")
    for ax in axes[:,0]:ax.set_ylabel("Continuous event-day Precision (%)")
    fig.suptitle("Pareto screening of frozen neural arms vs native persistence (2050 summer)")
    fig.text(.04,.025,"Filled = Pareto non-dominated; open = dominated. Labels: P=Persistence, F/U=FNO/U-Net and 1×/2×/4× weight. Size loosely inversely proportional to RMSE.",fontsize=8)
    fig.text(.04,.005,"One native CMIP6 member/GCM; correlated SSP cases; plotted scores descriptive, not independent out-of-sample model selection.",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(out/"figure_v528_six_case_precision_recall_pareto.png",dpi=190,bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.6,4.9))
    cases=[f"{g} {s.upper()}" for g in GCMS for s in SCENARIOS]
    count=[]
    for g in GCMS:
        for s in SCENARIOS:
            count.append(sum(r["pareto_nondominated"] for r in records if r["gcm"]==g and r["scenario"]==s))
    ax.bar(cases,count,color="#47769D")
    ax.set_ylim(0,7.7)
    ax.set_ylabel("Non-dominated alternatives among 7 candidates")
    ax.set_title("Large within-scenario Pareto frontiers indicate objective tradeoffs")
    for i,v in enumerate(count):ax.text(i,v+.10,f"{v}/7",ha="center",fontsize=9)
    fig.text(.04,.01,"Three GCMs, two paired SSP scenarios each; neither cells nor training seeds are independent climate-model replicates.",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(out/"figure_v528_pareto_frontier_sizes.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    with (out/"table_v528_original_42_candidate_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
    draft=["# E1 v5.28 — Pareto screening Results and interpretation","",
        "All comparisons use the original 54 Historical-only frozen neural checkpoints and independently verified native-grid 2050 summer source outputs. Persistence and the six pre-existing FNO/U-Net training arms were screened separately for each GCM×SSP cell on weighted RMSE, continuous event-day Recall, and continuous event-day Precision.","",
        "| GCM | Scenario | Pareto-front configurations / 7 |",
        "|---|---|---:|"]
    for g in GCMS:
        for s in SCENARIOS:
            n=sum(r["pareto_nondominated"] for r in records if r["gcm"]==g and r["scenario"]==s)
            draft.append(f"| {g} | {s} | {n} |")
    draft+=["","Across the six GCM–scenario cells, native persistence and both standard-loss and strong-hot-weight FNO solutions remain Pareto non-dominated. Thus improvements in one objective alone cannot establish general superiority in high-temperature detection and overall temperature prediction. The 4× historical hot-loss arm increased sustained-hot-event Recall but also lowered event Precision in every evaluated cell. The result supports examining operational thresholding and calibration alongside architecture and loss design, but should not be interpreted as a pre-registered model-selection result.",
       "","**Evidence boundary:** The model members are simulated climate futures, not observed heatwaves; local monthly P95 thresholds and native grids differ among GCMs; each SSP is paired within a GCM. One 2050 summer of 91 target days and this exploratory Pareto screening cannot establish uncertainty significance or generalization to different years or climate models. Tests compare same-cell event-day labels, not tracked durations or heatwave objects.","",
       "Source: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45951478573",""]
    (out/"E1_v528_manuscript_Pareto_results.md").write_text("\n".join(draft),encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output-dir",required=True)
    args=p.parse_args()
    rows=verified(json.loads(Path(args.input).read_text()))
    draw(rows,args.output_dir)
    print(json.dumps({"status":"v528_three_GCM_immutable_Pareto_figures",
       "GCMs":3,"SSP_cases":6,"original_model_choices":42,
       "frontier_sizes":[sum(r["pareto_nondominated"] for r in rows if r["gcm"]==g and r["scenario"]==s)
              for g in GCMS for s in SCENARIOS]},indent=2))
if __name__=="__main__":main()
