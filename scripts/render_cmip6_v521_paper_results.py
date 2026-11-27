"""E1 v5.21 three-GCM publication figure and results tables.

Render only archived successful v5.19 descriptive results.
Never derive an inferential p-value or pool correlated SSP cells as independent.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

GCM=("AWI","BCC","CAMS")
SSPS=("ssp245","ssp585")
STATUS="v519_three_independent_GCM_frozen_sparse_SSP_audit"

def audited(source):
    if source.get("status")!=STATUS:
        raise ValueError("Only the successful v5.19 frozen SSP audit is eligible")
    if (source.get("independent_GCM_count")!=3
         or source.get("scenario_cells")!=6
         or source.get("ssp_used_for_checkpoint_selection") is not False
         or len(source.get("gcm_ssp_findings",[]))!=6
         or len(source.get("arms",[]))!=36):
        raise ValueError("Incomplete or unverified three-GCM report")
    rows=source["gcm_ssp_findings"]
    observed={(x["gcm"],x["ssp"]) for x in rows}
    if len(observed)!=6 or observed!={(g,s) for g in GCM for s in SSPS}:
        raise ValueError("Missing or repeated GCM×SSP cells")
    fields=("persistence_rmse_K","fno1_rmse_K","fno1_rmse_skill_pct",
            "fno4_recall_minus_fno1_pp","fno4_rmse_minus_fno1_K",
            "persistence_recall_minus_fno4_pp")
    for row in rows:
        if any(not isinstance(row[k],(int,float)) or not math.isfinite(row[k]) for k in fields):
            raise ValueError("Missing or invalid real outcome")
        expected=100*(1-row["fno1_rmse_K"]/row["persistence_rmse_K"])
        if not math.isclose(expected,row["fno1_rmse_skill_pct"],abs_tol=1e-7):
            raise ValueError("Broken native persistence-relative skill")
    return rows

def write_tables(rows,target):
    with target.open("w",newline="",encoding="utf-8") as f:
        keys=["gcm","ssp","persistence_rmse_K","fno1_rmse_K","fno1_rmse_skill_pct",
              "fno4_recall_minus_fno1_pp","fno4_rmse_minus_fno1_K",
              "persistence_recall_minus_fno4_pp"]
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def render(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    categories=[f"{r['gcm']} / {r['ssp'].upper()}" for r in rows]
    values=[r["fno1_rmse_skill_pct"] for r in rows]
    y=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.barh(y,values,color="#47769d",height=.65)
    ax.set_yticks(y,categories)
    ax.invert_yaxis()
    ax.axvline(0,color="#333333",lw=1)
    ax.set_xlabel("Standard-loss FNO RMSE skill vs same-source persistence (%)")
    ax.set_title("Three native-grid CMIP6 models · sparse SSP test dates")
    for i,v in enumerate(values):
        ax.text(v+0.2,i,f"{v:.1f}%",va="center",fontsize=9)
    ax.set_xlim(min(0,min(values)-1),max(values)+4)
    fig.text(.02,.01,"3 independent GCMs, 2 correlated SSP scenarios each; 72 +1-day pairs per cell. Descriptive only.",
             fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(out/"fig_v521_fno_relative_rmse_skill.png",dpi=210,bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    recall=np.array([r["fno4_recall_minus_fno1_pp"] for r in rows])
    cost=np.array([r["fno4_rmse_minus_fno1_K"] for r in rows])
    ax.scatter(cost,recall,s=75,color="#47769d")
    ax.axhline(0,color="#777777",lw=.8)
    ax.axvline(0,color="#777777",lw=.8)
    for x,z,name in zip(cost,recall,categories):
        ax.annotate(name,(x,z),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xlabel("4× minus 1× loss-weight FNO RMSE (K; positive = worse)")
    ax.set_ylabel("4× minus 1× loss-weight FNO P95 Recall (percentage points)")
    ax.set_title("Historical tail-weight tradeoff under frozen SSP evaluation")
    fig.text(.02,.01,"One GCM member per model; sparse dates and no future model re-fitting.",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(out/"fig_v521_recall_rmse_tradeoff.png",dpi=210,bbox_inches="tight")
    plt.close(fig)

def results_markdown(rows):
    lines=["# E1 v5.21 — Three-GCM Results: descriptive SSP predictions","",
       "**Status:** Source-verified Results draft based only on successful v5.19 immutable outputs.",
       "Not a climate-warming attribution, statistical generalization estimate, or fully continuous event validation.","",
       "## Main figure data",
       "| GCM | SSP | Persistence RMSE (K) | FNO RMSE (K) | FNO RMSE skill (%) | 4×−1× FNO hot Recall (pp) | 4×−1× FNO RMSE (K) | Persistence−4× FNO Recall (pp) |",
       "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['gcm']} | {r['ssp']} | {r['persistence_rmse_K']:.3f} | {r['fno1_rmse_K']:.3f} | {r['fno1_rmse_skill_pct']:.2f} | {r['fno4_recall_minus_fno1_pp']:+.2f} | {r['fno4_rmse_minus_fno1_K']:+.3f} | {r['persistence_recall_minus_fno4_pp']:+.2f} |")
    lines+=["","## Manuscript-ready Results (English)","",
       "Three independently calibrated native-grid CMIP6 general circulation models (AWI-CM-1-1-MR, BCC-CSM2-MR, and CAMS-CSM1-0) were evaluated using historical-only trained FNO and U-Net models. Each GCM contributed one realization for SSP2-4.5 and SSP5-8.5; each GCM–scenario cell contained 72 fixed +1-day native daily-mean temperature pairs sampled across 2030, 2050, and 2080. For the standard-loss FNO, area-weighted RMSE was lower than the same GCM–scenario persistence baseline in all six cells. Relative RMSE improvements were 11.00% and 11.10% for AWI, 10.41% and 11.38% for BCC, and 16.23% and 16.04% for CAMS (SSP2-4.5 and SSP5-8.5, respectively).",
       "",
       "Increasing the FNO historical hot-target loss multiplier from 1× to 4× increased grid-local, historical-training monthly-P95 recall in all six GCM–scenario cells (improvements of 2.30–7.48 percentage points). This gain coincided with higher RMSE in all six cells (+0.012 to +0.044 K, by source and scenario). Persistence nevertheless attained higher local-P95 recall than the 4× FNO in every cell (gaps of 0.99–8.68 percentage points), highlighting a repeatable distinction between overall next-day RMSE and extreme exceedance detection.",
       "",
       "These comparisons are descriptive across three independently trained GCMs, **not six independent climate-model replications**; SSP scenarios are paired within each GCM. The dates are sparse, projected CMIP6 daily-mean `tas` is not an observed future reference, and historical validation years were used for checkpoint selection. Accordingly, neither heatwave persistence nor long-term forced climate signal fidelity can be inferred from these results. The separately designed continuous-day pilot is exploratory and must be reported distinctly.","",
       "## Source provenance",
       "- AWI frozen original SSP run: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174",
       "- BCC frozen original SSP run: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45872811683",
       "- CAMS frozen original SSP run: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45905955958",
       "- Three-GCM comparison audit: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45908305246",
       "","## Figure captions","",
       "**Figure A.** Persistence-relative weighted next-day temperature RMSE skill for historical-only standard-loss FNO by GCM and SSP. Positive values indicate FNO improvement; each source–scenario comparison uses its own native-grid persistence baseline.",
       "",
       "**Figure B.** Changes in historical local-month P95 Recall versus weighted RMSE when the fixed hot-target training loss multiplier is increased from 1× to 4×. Points represent GCM–SSP cells; correlated SSP cells do not imply six independent climate-model samples.",""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output-dir",required=True)
    a=p.parse_args()
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError("Do not overwrite archived manuscript figures")
    rows=audited(json.loads(Path(a.input).read_text()))
    out.mkdir(parents=True)
    write_tables(rows,out/"table_v521_three_GCM_SSP.csv")
    render(rows,out)
    (out/"E1_v521_three_GCM_results_draft.md").write_text(results_markdown(rows),encoding="utf-8")
    print(json.dumps({"status":"v521_immutable_three_GCM_paper_visualization",
          "number_of_independent_GCMs":3,"number_of_SSP_cells":len(rows),
          "rmse_advantage_all":sum(r["fno1_rmse_skill_pct"]>0 for r in rows),
          "tail_gain_all":sum(r["fno4_recall_minus_fno1_pp"]>0 for r in rows),
          "rmse_tradeoff_all":sum(r["fno4_rmse_minus_fno1_K"]>0 for r in rows)},indent=2))
if __name__=="__main__":main()
