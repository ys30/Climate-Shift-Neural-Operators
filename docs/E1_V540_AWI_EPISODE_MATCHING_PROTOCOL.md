# E1 v5.40 — Auditable same-pixel episode-level diagnostics

**Status: exploratory protocol, not independently confirmed model results until GitHub Actions succeeds.** The study extends the successfully archived v5.38 *event-day* duration sensitivity without altering the original source model, frozen checkpoints, or historical threshold.

## Source and reproduction gate

Single AWI-CM-1-1-MR member, 2040 and 2050, SSP245 and SSP585, 91 native continuous +1-day simulated Kelvin daily-mean TAS targets each. 26 × 27 native regional cells; each cell uses AWI Historical train-only month-specific local TAS P95. Original 18 Historical-only FNO/U-Net checkpoints (three hot loss weights × three fixed seeds) are verified before SSP access. Required original three-day event-day Recall/Precision scores (v5.24 #45937473743 and v5.29 #45954950987) must be reproduced for Persistence and original FNO 1× and 4× (seed average) before any new episode result is archived.

## Episode definitions and matching — fixed before first event-level field read

1. A qualifying **single-pixel episode** is a *maximal contiguous run* of at least three above-Historical-P95 target days. Intervals are half-open [first target index, last target index + 1), 91 days long per case.
2. Predicted and observed episodes are linked if they overlap by **at least one target day in the same native grid pixel**. No spatial-neighbor matching, timing grace period, or gap filling.
3. Greedy deterministic **one-to-one maximum-day-overlap** matching. Sort all overlapping true/predicted candidate pairs by descending shared days, then absolute onset lag, then true/predicted indices. Accept only if neither episode is already matched. This is a reproducible convention, not necessarily maximum-cardinality or optimal event tracking.
4. **Episode Recall** is area-weighted matched true episodes / area-weighted true episodes, **episode Precision** is area-weighted matched predicted episodes / area-weighted forecast episodes. They are not event-*day* Recall/Precision. Null if denominator is zero.
5. **Fragmentation** is the area-weighted fraction of true episodes overlapping **two or more different forecast episodes**, assessed before one-to-one assignment, including cases where one forecast is matched to another truth.
6. For duration and onset evaluation, keep only one-to-one matched pairs **where neither the true nor forecast run touches day 1 or day 91**. Report mean predicted-minus-true duration, absolute duration error, and predicted-minus-true onset lag in days. Incomplete edge-touching events remain eligible for episode detection counts but are excluded from duration estimates. Report included matched count and censored counts for transparency.
7. All episode counts and aggregation are weighted by latitude cosine *per native pixel*; do not aggregate 26 × 27 AWI grid with other GCM pixels. Reports use all three actual original neural seeds separately, rather than falsely treating an averaged multi-seed forecast as a single physical episode.

## Interpretation / limitations

This protocol operationalizes *temporal runs in each grid pixel*. They are not multi-pixel contiguous spatial heatwave objects and do not establish observed heatwave duration, heat-related risk, tasmax extremes or model-independent climate projections. Exact overlap matching may underestimate meteorologically relevant offset episodes; no tolerance is inferred from the data after inspecting outcomes. The 2040 and 2050 SSP targets are not independent climate models, and all analyses are exploratory follow-ups to v5.38. Do not make significance claims from pixel-level pseudoreplication.

**Next:** confirm AWI archive completion and original-day metric reproduction, then extend identical matching definitions to BCC/CAMS without tailoring the overlap rule to results.
