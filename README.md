# Climate-Shift Neural Operators

**Research project A — Climate-Shift Generalization of Neural Operators**

Working question: **Do neural operators generalize to a warmer climate?** This project evaluates out-of-distribution performance under temporal, geographic, and climate-state shifts, focusing on extreme-event fidelity and physical consistency.

> Status: **research design / repository initialization**. No model results, climate-change projections, or validation claims have been established yet.

## Core questions

1. How do FNO-like models perform on unseen years, regions, and hotter-than-training conditions?
2. Does strong in-distribution forecast skill translate to heatwave and climate-response fidelity?
3. Can climate-state conditioning improve robustness under controlled, leakage-free tests?

## Planned experiments

| Stage | Domain | Goal |
| --- | --- | --- |
| E1 | ERA5 / WeatherBench 2 | Historical out-of-time and out-of-region weather prediction |
| E2 | CMIP6 historical + future scenarios | Warming-response and extreme-event fidelity within consistent model domains |
| E3 | Controlled ablations | Climate-conditioned operator vs comparably conditioned baselines |

**Important:** ERA5 reanalysis and CMIP6 simulations are different domains. Their labels and splits will not be pooled naively. CMIP6 evaluation must distinguish scenario, climate-model, and member dependencies.

## Proposed baseline families

Persistence/climatology, U-Net/CNN, FNO/AFNO, and a competitive transformer baseline, using comparable data access, training budgets, and evaluation metrics. Exact implementation choices will be pinned before results are produced.

## Reproducibility principles

- Train/validation/test splits frozen before final evaluation; spatial buffer and temporal embargo when appropriate
- At least several independent seeds for meaningful model comparisons
- Evaluate aggregate RMSE/MAE **and** tail bias, heatwave duration, spatial structure, and warming response
- Version source datasets and record units, calendars, grid definitions, preprocessing, and licenses
- No manuscript claims without reproducible scripts and traceable output artifacts

See [Research protocol](docs/RESEARCH_PROTOCOL.md) and [Experiment registry](docs/EXPERIMENT_MATRIX.md).

## Relationship to other projects

This project is independent of [Western-US-Weather-JAX](https://github.com/ys30/Western-US-Weather-JAX) and the PyPSA climate-grid research series. Existing code/data can inform design, but results must be independently verified and provenance documented.

## Project status

- [x] Research question and scope drafted
- [x] Initial experimental protocol drafted
- [ ] Data access and license audit
- [ ] Frozen train/validation/test splits
- [ ] Baseline implementation and smoke tests
- [ ] ERA5 historical experiments
- [ ] CMIP6 future-climate experiments
- [ ] Statistical and physical diagnostics
- [ ] Manuscript and archival release
