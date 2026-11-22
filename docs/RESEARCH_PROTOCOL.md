# Research protocol (draft v0.1)

## Scope and hypotheses

**H1 — Distribution shift:** Relative performance of FNO/AFNO and non-operator baselines changes under geographic, temporal, and thermal shifts. The direction is **not assumed**.

**H2 — Tail fidelity:** Aggregate skill does not necessarily imply calibrated extreme-event magnitude, frequency, duration, or spatial footprint.

**H3 — Conditioning:** Explicit climate-state information may improve response fidelity, but any gain must be compared against *equally conditioned* non-operator models.

## Task design

**E1 Historical skill:** Start with fixed-lead prediction of 2-m temperature fields and clearly specified dynamical inputs from ERA5 / WeatherBench 2. Define grid resolution, time sampling, lead time, normalization, and target alignment *before training*. Use chronological validation/test years; geographic holdouts must avoid spatial leakage and should distinguish extrapolation from zero-shot prediction.

**E2 Climate-response fidelity:** Use internally consistent CMIP6 historical and SSP2-4.5/SSP5-8.5 model/member/scenario data. Specify historical baseline and future windows; assess changes in seasonal means and extreme-event statistics. Never label CMIP6 simulated temperatures as observed truth. A single scenario split does not automatically test generalization to an independent climate model.

**E3 Methods:** Add climate-state conditioning to the operator; compare to an identical-input baseline and perform ablations for conditioning signal, physical regularization, and training coverage.

## Metrics (pre-register definitions)

- Point prediction: MAE, RMSE, bias, regional and seasonal stratification
- Extremes: warm-tail quantile bias, exceedance frequency, event duration and footprint
- Spatial fidelity: anomaly correlation and scale-dependent power spectra
- Change signal: error in future-minus-historical temperature response, plus uncertainty across models/members
- Computational cost: parameter count, training/inference time and hardware

## Controls

1. Prevent look-ahead through normalization and split-specific preprocessing.
2. Record climate calendar conventions (including 360-day calendars), units, variable definitions, and regridding methods.
3. Use region/time blocked tests and report uncertainty with dependence-aware intervals (e.g. block bootstrap).
4. Avoid training/test overlap from closely related CMIP6 ensembles.
5. Compare matched baselines and quantify sensitivity to seeds/hyperparameters.
6. Present null and negative results transparently.

## Initial exit criteria

A reproducible data card; frozen split manifest; working persistence, U-Net and FNO smoke tests; and one validated metrics pipeline before any large training run.
