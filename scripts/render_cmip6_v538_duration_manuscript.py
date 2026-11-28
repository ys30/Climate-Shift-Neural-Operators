"""E1 v5.38 — immutable v5.37 duration-sensitivity manuscript figures.

Explicitly plot paired within-GCM and SSP 2->3->5 day Recall gaps and 4x gain.
The counts of 12/12 do NOT prove that the magnitude grows with duration;
show all 12 paired differences, reversals, and null values without imputation.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
GCMS=("AWI","BCC","CAMS")
SSPS=("ssp245","ssp585")
YEARS=(2040,2050)
DUR=(2,3,5)
STATUS="v538_three_GCM_paired_duration_manuscript_figures"

def validate(report):
    if (report.get("status")!="v537_three_GCM_two_year_two_three_five_day_sensitivity_audit"
        or report.get("independent_GCM_count")!=3
        or report.get("GCM_year_SSP_cells")!=12
        or report.get("GCM_year_SSP_duration_cells")!=36
        or report.get("original_frozen_checkpoints")!=54
        or report.get("original_GCM_duration_arm_rows")!=108
        or report.get("all_three_day_archived_references_reproduced") is not True
        or report.get("no_SSP_fitting") is not True):
        raise ValueError("Require complete source-verified frozen v5.37 archive")
    rows=report.get("scenario_duration_findings",[])
    keys={(g,y,s,d) for g in GCMS for y in YEARS for s in SSPS for d in DUR}
    if len(rows)!=36 or {(r["gcm"],r["year"],r["scenario"],r["min_days"]) for r in rows}!=keys:
        raise ValueError("Missing or duplicated original 36 duration cases")
    paired=report.get("paired_two_vs_five_day_changes",[])
    if len(paired)!=12 or {(r["gcm"],r["year"],r["scenario"]) for r in paired}!={
            (g,y,s) for g in GCMS for y in YEARS for s in SSPS}:
        raise ValueError("Missing paired two/five-day contrasts")
    for r in rows:
        for key in ("persistence_minus_fno_1x_recall_pp","fno_4x_minus_1x_recall_pp",
                    "fno_4x_minus_1x_precision_pp"):
            z=r.get(key)
            if z is not None and (isinstance(z,bool) or not isinstance(z,(int,float)) or not math.isfinite(z)):
                raise ValueError("Nonfinite source delta: "+key)
        for a,b,k in (("persistence_event_recall","fno_1x_event_recall","persistence_minus_fno_1x_recall_pp"),
                      ("fno_4x_event_recall","fno_1x_event_recall","fno_4x_minus_1x_recall_pp")):
            x,y,v=r.get(a),r.get(b),r.get(k)
            if x is None or y is None:
                if v is not None:raise ValueError("Undefined source delta must stay null")
            elif not math.isclose(100*(x-y),v,rel_tol=0,abs_tol=1e-7):
                raise ValueError("Original source Recall delta inconsistent")
    rows=sorted(rows,key=lambda x:(GCMS.index(x["gcm"]),x["year"],x["scenario"],x["min_days"]))
    for r in paired:
        m={d:next(x for x in rows if (x["gcm"],x["year"],x["scenario"],x["min_days"])==(
              r["gcm"],r["year"],r["scenario"],d)) for d in (2,5)}
        for d,key in ((2,"recall_gap_2day_pp"),(5,"recall_gap_5day_pp")):
            a=m[d]["persistence_minus_fno_1x_recall_pp"]
            b=r[key]
            if (a is None)!=(b is None) or a is not None and not math.isclose(a,b,abs_tol=1e-7):
                raise ValueError("Archived paired Recall gap changed")
        diff=None if m[2]["persistence_minus_fno_1x_recall_pp"] is None or m[5]["persistence_minus_fno_1x_recall_pp"] is None else m[5]["persistence_minus_fno_1x_recall_pp"]-m[2]["persistence_minus_fno_1x_recall_pp"]
        if (diff is None)!=(r["five_minus_two_day_gap_change_pp"] is None) or diff is not None and not math.isclose(diff,r["five_minus_two_day_gap_change_pp"],abs_tol=1e-7):
            raise ValueError("Archived paired duration effect changed")
    return rows,paired

def render(rows,paired,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path=Path(path)
    if path.exists():raise FileExistsError("Do not overwrite paper figures")
    path.mkdir(parents=True)
    colors={"AWI":"#377FA5","BCC":"#BF704B","CAMS":"#56886C"}
    metrics=[("persistence_minus_fno_1x_recall_pp","Persistence minus standard FNO event Recall (pp)",
              "figure_v538_duration_recall_deficit.png"),
             ("fno_4x_minus_1x_recall_pp","FNO 4× minus FNO 1× event Recall (pp)",
              "figure_v538_duration_hotweight_gain.png"),
             ("fno_4x_minus_1x_precision_pp","FNO 4× minus FNO 1× event Precision (pp)",
              "figure_v538_duration_precision_cost.png")]
    for key,header,filename in metrics:
        fig,ax=plt.subplots(figsize=(9.5,5.8))
        for g in GCMS:
            for y in YEARS:
                for s in SSPS:
                    subset=[next(r for r in rows if (r["gcm"],r["year"],r["scenario"],r["min_days"])==(g,y,s,d))
                            for d in DUR]
                    ys=[r[key] for r in subset]
                    ax.plot(DUR,[float("nan") if z is None else z for z in ys],
                       color=colors[g],ls="-" if y==2040 else "--",
                       marker="o" if s=="ssp245" else "^",markersize=5,alpha=.9,
                       label=f"{g} {y} {s.upper()}")
        ax.axhline(0,color="#444444",lw=.9)
        ax.set_xticks(DUR,[f"≥{d} days" for d in DUR])
        ax.set_xlabel("Minimum consecutive Historical P95 exceedance days")
        ax.set_ylabel(header)
        ax.grid(alpha=.17)
        ax.set_title(header+"\n3 GCM members × 2 correlated summers × 2 SSPs")
        ax.legend(loc="best",ncol=2,fontsize=7,frameon=False)
        fig.text(.015,.01,"Descriptive paired event-day sensitivity. Native-grid TAS simulations; 12 correlated lines, only 3 independent GCM members.",fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        fig.savefig(path/filename,dpi=190,bbox_inches="tight")
        plt.close(fig)

    with (path/"table_v538_36_original_duration_cases.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (path/"table_v538_12_paired_duration_changes.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)

    changes=[r["five_minus_two_day_gap_change_pp"] for r in paired]
    gain_changes=[r["five_minus_two_day_hotweight_gain_change_pp"] for r in paired]
    def counts(values):
        return {"defined":sum(x is not None for x in values),
                "positive":sum(x is not None and x>1e-9 for x in values),
                "negative":sum(x is not None and x< -1e-9 for x in values),
                "unchanged":sum(x is not None and abs(x)<=1e-9 for x in values),
                "undefined":sum(x is None for x in values)}
    result={"status":STATUS,"original_independent_GCM_members":3,
       "GCM_year_SSP_duration_cells":36,"paired_two_five_day_scenarios":12,
       "persistence_minus_FNO_recall_gap_5d_minus_2d":counts(changes),
       "FNO_4x_gain_5d_minus_2d":counts(gain_changes),
       "no_SSP_training_or_model_selection":True}
    (path/"v538_paired_duration_direction_counts.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    fmt=lambda x: "undefined" if x is None else f"{x:+.2f}"
    lines=["# E1 v5.38 — Two-, three-, five-day paired sustained-hot TAS event sensitivity","",
       "Three independent GCM members each evaluated two summers (2040,2050) and two correlated SSP pathways. Historical-only neural checkpoints (54 total) and historical month-local P95 were frozen; no future SSP fitting or selection. Each 91-day native daily-mean TAS evaluation labels days belonging to same-cell >=2/3/5 consecutive exceedance runs.","",
       "## Key direction counts of 5-day minus 2-day changes","",
       f"- Persistence minus standard FNO Recall deficit: {counts(changes)}.",
       f"- FNO 4× minus 1× Recall advantage: {counts(gain_changes)}.","",
       "A positive 5−2-day change means the gap or training-weight advantage grew when the minimum event duration was raised from two to five days. The sign is evaluated within each identical GCM–year–SSP case; it is not a significance test. An undefined result is retained rather than imputed.","",
       "| GCM | Year | SSP | 2d Recall gap (pp) | 5d Recall gap (pp) | Gap change (pp) | 4× gain change (pp) |",
       "|---|---:|---|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['gcm']} | {r['year']} | {r['scenario']} | {fmt(r['recall_gap_2day_pp'])} | {fmt(r['recall_gap_5day_pp'])} | {fmt(r['five_minus_two_day_gap_change_pp'])} | {fmt(r['five_minus_two_day_hotweight_gain_change_pp'])} |")
    lines+=["","## Scientific interpretation","",
       "Direction consistency at every event definition does not imply a monotonically growing event Recall deficit or hotweight gain. The qualifying observed event-day cohorts and denominators differ by minimum duration; run masks at 91-day summer boundaries censor longer events. These are area-weighted same-pixel event-day diagnostics for simulated daily-mean TAS, not tracked heatwave-object duration or observed heat-health hazards. The independent climate-model count is three, NOT 12 or 36; no pseudo-replicated significance claims. The 2- and 5-day checks were conducted after inspecting the earlier 3-day results.","",
       "Source: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/46117981473",""]
    (path/"E1_v538_three_GCM_duration_manuscript_results.md").write_text("\n".join(lines),encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    rows,paired=validate(json.loads(Path(a.input).read_text()))
    print(json.dumps(render(rows,paired,a.output_dir),indent=2))
if __name__=="__main__":main()
