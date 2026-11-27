# E1 v5.23 — Independent BCC 2050 contiguous summer event replication

**Validated prerequisite:** v5.22 CAMS contiguous SSP two-scenario report run 45927052977 completed successfully. CAMS standard FNO event-day Recall was 45.78% / 44.25%; Persistence 79.72% / 80.78% for SSP245/SSP585. CAMS 4× loss increased event-day Recall by +12.46 / +18.73 percentage points relative to 1×. These are one-GCM exploratory findings, not climate-population inference.

**Second GCM:** BCC-CSM2-MR r1i1p1f1 gn, frozen original independent Historical-only 1980–2004 training, 2005–2014 validation. Use exact original BCC 18 checkpoints from run 45870654954 and native noleap Historical dataset from run 45868844059. Validate all 18 SHA256 and Historical-derived P95 before any SSP read. No new training.

**Matched continuous experiment:** 2050-06-01 to 2050-08-30 source (91 consecutive dates), 2050-06-02 to 2050-08-31 +1-day targets (91 consecutive dates); both SSP245 and SSP585. Bounded region 25–50°N, 235–260°E using BCC's native 23×23 grid. Process 92 contiguous Kelvin regional fields per scenario, never entire global field.

**Event definition:** At same pixel, at least three consecutive native daily-mean `tas` target days strictly above BCC's own Historical-training monthly local P95. Report area-weighted event-day Recall, Precision, daily P95 Recall/Precision, area-weighted RMSE/MAE/Bias for native Persistence and original 2×3×3 frozen FNO/U-Net model arms. The event code is shared with CAMS, while spatial grid and thresholds remain independently GCM-specific.

**Caveats:** The source is simulated daily mean TAS rather than observed future tasmax. Summer boundary events are censored. SSP cells within BCC are correlated. Expanding from CAMS to BCC provides a second native-GCM replication, not formal significance or full forced climate-response fidelity. Model protocol is exploratory because designed after observing sparse SSP outcomes.

**Next:** Repeat for AWI-CM-1-1-MR using its own Gregorian original Historical-only 18 checkpoints and historical training P95. Then compare all three GCMs, preserving per-GCM persistence baselines.
