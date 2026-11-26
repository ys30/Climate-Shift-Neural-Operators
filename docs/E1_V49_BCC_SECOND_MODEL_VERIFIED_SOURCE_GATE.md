# E1 v4.9 — VERIFIED second CMIP6 source-model compatibility: BCC-CSM2-MR

[Successful independent second-source data audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45865322183) and [full machine diagnostics](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45865322183/artifacts/15139922001).

The BCC-CSM2-MR member `r1i1p1f1` grid `gn` was sourced strictly from the [successful v4.0 Pangeo CMIP6 catalog audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45758551504), not guessed URLs. All **three actual Zarr stores opened successfully**, passed `day/tas` Kelvin sanity reads, native CF date decoding and compatible coordinate/grid schema.

| Source | Actual native first day | Actual native last day | Eligibility |
|---|---|---|---|
| BCC-CSM2-MR Historical | 1850-01-01 | 2014-12-31 | Compatible |
| BCC-CSM2-MR SSP245 | 2015-01-01 | 2100-12-31 | Compatible |
| BCC-CSM2-MR SSP585 | 2015-01-01 | 2100-12-31 | Compatible |

**This is strictly an actual source availability, calendar, unit and bounded temperature check, not a second-model forecast or independent replication of AWI skill.**

## Scientific and engineering next gates

1. Inspect actual BCC native Western-US regional coordinates and grid dimensions; BCC native grids can differ from AWI's 26×27. Native-calendar pair time may also require separate CF date handling.
2. Verify six tiny predeclared daily regional source-target pairs (Historical, SSP245 and SSP585) on BCC, no global Zarr loads. Then audit Historical/SSP full date support and sample completeness.
3. Freeze BCC-specific Historical train/validation daily extraction and training-only local gridcell P95 reference. **Do not reuse AWI Historical normalization, absolute-temperature calibration, model checkpoints or native pixel geometry on a BCC grid.**
4. Train matching architecture families and training loss multipliers on BCC Historical only and score BCC SSP245/585 using frozen BCC checkpoints, the same scientific experiment protocol but separately audited source-model geometry and timestamps.
5. Analyze **between-model** rather than only 3-seed uncertainty; BCC and AWI are distinct GCMs but only one member each, so this is an incremental replication not a CMIP6 ensemble-wide generalization claim.

The [v4.8 AWI first successful real frozen SSP forecast results](E1_V48_VERIFIED_FROZEN_SSP245_SSP585_RESULTS.md) remain unchanged; no hyperparameter re-selection on those SSP scores is appropriate.
