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

## E1 v0.6 — bounded multi-year extraction and exploratory three-way model comparison

New components:
- [Monthly-stratified multi-year WeatherBench 2 exporter](scripts/export_wb2_multiyear.py)
- [Compact U-Net](src/climate_shift/unet.py)
- [FNO / U-Net / persistence comparison](src/climate_shift/compare_e1.py)
- [Reproduction and methodological caveats](docs/E1_V06_RUNBOOK.md)

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/export_wb2_multiyear.py --years 2010 2015 2020 --max-per-month 8 --stride 4 --output data/processed/e1_multiyear_dev.npz
python scripts/compare_e1.py --input data/processed/e1_multiyear_dev.npz --train-end 2010 --val-end 2015 --test-end 2020 --epochs 4 --width 8 --modes 4 --output outputs/e1_v06_dev.json
```

The three years and 8 samples/month are *development examples*, not a preregistered climate study. This pipeline operates on a bounded in-memory NPZ, not streaming storage. ERA5 cloud access and CI success have not yet been confirmed. **Do not publish synthetic or tiny-extract model comparisons as research findings.**

## E1 v0.7 — extreme-grid diagnostics and independent seeds

The [heat diagnostics module](src/climate_shift/extremes.py) fits a threshold from the training target distribution only and measures exceedance frequency bias, conditional hot-temperature bias, recall, and precision for FNO, U-Net, and persistence. Undefined cases are JSON `null`. The [multi-seed runner](scripts/run_e1_multiseed.py) produces separate seed reports and a descriptive summary. See [v0.7 protocol](docs/E1_V07_EXTREMES_PROTOCOL.md).

```bash
python -m pip install -e '.[test,torch,era5]'
python -m pytest
python scripts/run_e1_multiseed.py --input data/processed/e1_multiyear_dev.npz \
  --output-dir outputs/e1_v07_multiseed --train-end 2010 --val-end 2015 --test-end 2020 \
  --seeds 11 22 33 --epochs 4 --width 8 --modes 4
```

**Research status:** Development code, not climate-science findings. The current global percentile threshold is not a multi-day heatwave definition. The v0.7 comparison has no latitude coordinates for area weighting; real ERA5 extraction, future climate data, blocked uncertainty and full CI verification remain outstanding.

## E1 v0.8 — auditable timestamps and source manifests

New development exports persist `target_times` alongside `X`, `y`, `years`. Existing training utilities remain backward compatible, but timestamps in new files are checked for strict ordering and consistency with target years.

[Independent audit tool](src/climate_shift/audit_e1.py) and [tests](tests/test_audit_e1.py) check timestamps, source metadata, month/year counts, and SHA-256. Run:

```bash
python scripts/audit_e1.py --input data/processed/e1_multiyear_dev.npz \
  --metadata data/processed/e1_multiyear_dev.json \
  --output outputs/e1_v08_audit.json
```

See [v0.8 audit protocol](docs/E1_V08_DATA_AUDIT.md). Public source documentation is confirmed, but **remote dataset extraction and real ERA5 numerical results remain unverified**. Current tests use fabricated fields. This audit does not prove correct predictor-target pairing; the adapter and raw source metadata require separate checks.

## E1 v0.9 — real-data connectivity gate

A bounded [WeatherBench 2 access probe](scripts/probe_wb2_access.py) and [GitHub Actions workflow](.github/workflows/wb2-access-probe.yml) now test anonymous Google Cloud access, verify the variable/time schema, and attempt an actual tiny temperature chunk read. A JSON diagnostic is uploaded even when the action fails. Both the single-week and multi-year exporters now use the shared [anonymous GCS reader](src/climate_shift/wb2_source.py).

**Important:** The existence of this workflow is not evidence that it succeeded; inspect the specific Actions run and `wb2-access-probe` artifact. See [v0.9 data-access gate](docs/E1_V09_REMOTE_DATA_GATE.md). Full-year ERA5 training and OOD evidence remain pending.

## E1 v1.0 — real ERA5 access verified; real-data pilot initiated

The [WeatherBench 2 real-data probe run 45103779035](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45103779035) **passed**. It read a small genuine ERA5 temperature field and confirmed unit K, 1959-01-01 to 2023-01-10 timestamps, and the expected variable. This verifies remote access only, not full training or scientific skill.

The [bounded real-data CI pilot](.github/workflows/e1-realdata-pilot.yml) now runs a tiny 3-year extraction, data provenance audit, and FNO/U-Net/persistence comparison with archived diagnostics. Inspect the workflow status/artifacts before claiming it completed. See [v1.0 source verification and pilot protocol](docs/E1_V10_VERIFIED_SOURCE_AND_PILOT.md).

**No credible climate-shift or climate-change model results have yet been established.**

## E1 v1.1 — actual ERA5 pilot evidence, then a manually gated larger test

The [real ERA5 integration workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45109019800) completed **successfully** (data extraction, timestamp audit and three-model calculation on actual observations/reanalysis). See [audited v1.0 numerical results](docs/E1_V10_ACTUAL_ERA5_PILOT_RESULTS.md): 72 17×17 samples from 2010/2015/2020; only two training epochs. Persistence outperformed insufficiently trained neural models in this tiny test, which is **not evidence of climate-shift generalization or architectural quality**.

The next [contiguous-year development workflow](.github/workflows/e1-contiguous-dev.yml) is **manual-only** to control GitHub Actions compute costs. It proposes 2008–2010 training, 2011 validation and 2012 test, up to 24 daily source times per month, five epochs, a strict 18-minute workflow timeout, and metadata-only artifact upload. This is still a development subset; no workflow success or result is claimed before its actual execution.
