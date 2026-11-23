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

## E1 v0.3: executable synthetic smoke test

A small PyTorch 2-D FNO baseline, strict NPZ input contract, chronology checks, and CPU smoke test have been added. **No ERA5, CMIP6, or model-performance finding is claimed.** The prototype FNO is not SFNO and does not yet use climate-state conditioning.

```bash
python -m pip install -e '.[test,torch]'
python -m pytest
python scripts/synthetic_smoke.py --steps 10
```

Synthetic targets are local averages of fabricated sinusoidal fields; smoke metrics only confirm the code executes and are not evidence for climate generalization. See [E1 data contract](docs/E1_DATA_CONTRACT.md) and [synthetic smoke CI](.github/workflows/synthetic-smoke.yml). The production data adapter, U-Net/transformer baselines, area-weighted metrics, full train/validation/test pipeline, and independent CMIP6 experiments are not yet implemented.

## E1 v0.5: multi-year baseline trainer (development)

Added `src/climate_shift/train_e1.py`, CLI `scripts/train_e1.py`, two pipeline tests, and [E1 run protocol](docs/E1_RUN_PROTOCOL.md). The trainer rejects incomplete chronological year partitions; computes train-only normalization; chooses the FNO checkpoint using validation loss; evaluates held-out years against persistence; and writes a reproducibility JSON with data SHA-256.

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
# Replace example years with audited, preregistered actual data availability:
python scripts/train_e1.py --input data/processed/multi_year_pairs.npz \
  --train-end 2015 --val-end 2018 --test-end 2022 \
  --embargo-years 1 --epochs 10 --output outputs/e1_baseline.json
```

**Do not use the single-week 2019 adapter output for training:** a short single-year extraction is insufficient for all three held-out chronological partitions. Multi-year extraction and scientific partition freeze are still outstanding. No ERA5 performance numbers or climate-change conclusions have been produced.
