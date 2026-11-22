# E1 data contract — v0.3 (provisional)

**Status:** interface and synthetic smoke tests only; NO ERA5 download/analysis is claimed.

## Frozen interface for any future real-data adapter

- File: NPZ with exactly `X`, `y`, `years`; arrays are `[samples, channels, lat, lon]`, years are target timestamps.
- Units and variable identity: must be documented in a separate dataset card. No implicit Kelvin/Celsius conversion.
- Predictor and target times: adapter must document lead-time alignment and ensure predictors precede targets.
- Missing values: reject rather than silently impute; an explicit missing-data policy will be required.
- Normalization: fit on training partition exclusively; keep frozen for validation/test.
- Evaluation: separate out-of-time tests, geographically blocked tests, and hot-condition *stratification*. Selecting hot test samples does not itself demonstrate anthropogenic climate-shift extrapolation.
- Climate response: E2 needs matched CMIP6 model/member/scenario comparisons and reference baseline windows; E1 weather forecast performance alone cannot establish long-term warming fidelity.

## First experiment conventions (to be frozen after metadata audit)

| Choice | Initial candidate | Status |
| --- | --- | --- |
| Region | Continental Western US subset | Candidate, not yet fixed |
| Variables | 2m temperature target; available contemporaneous predictors | Candidate |
| Forecast horizon | 24 h | Candidate |
| Split | Chronological years with embargo; no shuffled dates | Requirement |
| Spatial test | Distinct regions with explicit buffer | Requirement |
| Metrics | MAE/RMSE/bias; fixed training-reference extreme threshold | Provisional |
| Baseline | Persistence and equally informed FNO / U-Net | Planned |

## Usage

`pip install -e '.[test,torch]'`

`python scripts/synthetic_smoke.py --steps 10`

The smoke run uses deterministic sinusoidal synthetic fields and a local averaging target; its outputs may validate computation only. They are **not scientific results** and must never appear as evidence in the manuscript.
