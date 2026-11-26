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

## E1 v1.2 — corrected within-month data selection

A sampling-bias audit found that the bounded monthly exporter previously kept only the **earliest** valid days in each month. As of v1.2 it selects evenly distributed, exact +24-hour valid pairs throughout the month; metadata identifies the new method. See [sampling audit](docs/E1_V12_SAMPLING_AUDIT.md) and regression test in `tests/test_multiyear.py`.

This is a **development sampling fix**, not an unbiased climate-extreme sampling design. Do not compute multi-day heatwave duration from sparse monthly extracts. Existing v1.0 results remain reproducible as historical integration evidence, but must not be treated as equivalent to v1.2 resampled data. The cost-controlled contiguous-year workflow remains manual-only.

## E1 v1.3 — independent 24-hour pairing provenance

New ERA5 exports now record **actual predictor/source timestamps**, read directly from the selected ERA5 input coordinates, plus target timestamps. The NPZ loader rejects records with non-increasing or invalid timestamps or a lead time other than exactly 24 hours. The audit reports `exact_24h_pairs_verified`; older NPZ files remain readable but cannot claim this stronger check. See [E1 v1.3 provenance protocol](docs/E1_V13_PAIR_PROVENANCE.md).

The last v1.2 test passed: [GitHub Actions run 45245083243](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45245083243). The contiguous-year development benchmark has not yet run; it remains manual-only to control compute costs.

## E1 v1.4 — strict chronological fold boundaries

A [timestamp-isolation guard](src/climate_shift/split_isolation.py) now audits chronological train, validation and test folds by comparing actual input `source_times` against previous-fold `target_times` before fitting FNO or U-Net. The JSON experiment report records the boundary gaps and verification status. Legacy arrays without both timestamp fields remain readable for older development examples, but their isolation is explicitly **unverified**. The [manual contiguous-year workflow](.github/workflows/e1-contiguous-dev.yml) now requires the audit to pass (`--require-pair-times`).

See [v1.4 scientific caveats and regression checks](docs/E1_V14_SPLIT_ISOLATION.md). This stricter chronology is not a substitute for statistically independent geographic or climate-shift tests, and no additional ERA5 model results are claimed by this code revision.

## E1 v1.5 — contiguous-year coverage preflight

The previous syntax repair has passed in [GitHub Actions run 45282547649](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45282547649). The [manual contiguous-year development workflow](.github/workflows/e1-contiguous-dev.yml) now runs an additional [data coverage gate](src/climate_shift/coverage_gate.py) before training. It checks all expected 2008–2012 months, at least 20 pairs/month, uniform sampling metadata, complete source/target +24h timestamps, and matching provenance SHA-256. The [v1.5 protocol](docs/E1_V15_COVERAGE_GATE.md) describes usage and limits.

**Status:** continuous-year ERA5 results are still unavailable until a new manual workflow run completes; no additional model result is claimed.

## E1 v1.6 — actual contiguous-year ERA5 evidence and three-seed plan

The [2008–2012 real ERA5 benchmark](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45292352961) **completed successfully**, including 1,440 paired observations, 60/60 covered months, true +24h pairing, 24h gaps at train/validation and validation/test boundaries, and FNO/U-Net/persistence evaluation. See [the archived observed metrics and limitations](docs/E1_V16_REAL_ERA5_CONTIGUOUS_RESULTS.md). The actual 2012 test RMSE was FNO 3.2346 K, U-Net 3.2641 K, and persistence 3.5369 K. Crucially, learned models showed **lower hot-gridcell recall than persistence**, despite better global RMSE. These are one-seed, 5-epoch, sparse within-month *development* results, not research conclusions.

A separate [manual-only 3-seed ERA5 workflow](.github/workflows/e1-three-seed-dev.yml) now repeats the bounded extraction and structural coverage audit, then evaluates seeds 11/22/33 using the same train/validation/test split and explicit +24h timestamp isolation. It archives per-seed RMSE/MAE/bias and heat-tail recall. The new workflow is **not yet executed**; no multi-seed metrics or climate-warming generalization claims are made.

## E1 v1.7 — area-weighted real-ERA5 comparison (code ready)

[Observed-latitude area weighting](docs/E1_V17_AREA_WEIGHTED_EVALUATION.md) adds actual ERA5 row coordinates to new exports and checks their alignment against sidecar metadata. Evaluation now separately reports existing equal-gridcell skill and cosine-latitude-weighted MAE/RMSE/bias and high-temperature exceedance diagnostics. Three-seed reports also summarize weighted metrics when supported. Prior 2008–2012 [real ERA5 results](docs/E1_V16_REAL_ERA5_CONTIGUOUS_RESULTS.md) remain **unweighted**. A new manual three-seed ERA5 run is required before reporting real weighted scores.

## E1 v1.8 — real 3-seed replication now verified

The [three-seed ERA5 replication](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45295324701) completed successfully using v1.6 code, 1,440 sampled ERA5 temperature field pairs and seeds 11/22/33. **Unweighted** test RMSE across seeds: FNO **3.2444 ± 0.0217 K**, U-Net **3.1893 ± 0.0208 K**, Persistence **3.5369 K**. Hot-gridcell recall was approximately **0.545** (FNO), **0.502** (U-Net), and **0.733** (Persistence). See [the verified run report](docs/E1_V18_THREE_SEED_REAL_ERA5_RESULTS.md) for per-seed values and limitations.

The multi-seed summary now includes explicit paired FNO-vs-Persistence, U-Net-vs-Persistence and FNO-vs-U-Net differences plus Recall mean/SD, and cosine-latitude-weighted differences **only when actual latitude coordinates are present**. This new reporting logic has tests but must not be mistaken for observed weighted metrics from the already completed v1.6 run. A **new manual run** on updated main is needed to generate weighted ERA5 results.

## E1 v1.9 — training-only month-local temperature extremes

The existing global training-P95 hot-gridcell threshold can conflate seasonal climatology and geographic temperature differences. The [v1.9 local monthly P95 protocol](docs/E1_V19_MONTHLY_LOCAL_P95.md) adds a complementary **12-month × gridcell** training-only threshold and computes test-year exceedance recall, precision, frequency bias and hot-conditional temperature bias. Actual latitude weighting is used when coordinates are present. Inadequate training month coverage fails gracefully with an explicit unavailable reason; the global-P95 outputs remain unchanged for historical comparisons.

**Version caution:** the [verified earlier real three-seed ERA5 run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45295324701) used pre-v1.7 code. It does not provide area-weighted or monthly local-P95 numerical results. The [manual three-seed workflow](.github/workflows/e1-three-seed-dev.yml) must run again on updated main after passing CI before reporting these new diagnostics.

## E1 v2.0 quality gate for the next real ERA5 run

The [v1.9 train-only local-month P95 experiment](docs/E1_V19_MONTHLY_LOCAL_P95.md) now has a full synthetic end-to-end regression test covering +24h origins, chronological split isolation, true latitude weighting, and monthly/grid-local high-temperature classification. The [manual three-seed ERA5 workflow](.github/workflows/e1-three-seed-dev.yml) now **fails closed** unless its actual real-data output contains area-weighted metrics, 12-month local-P95 recall for all three models, and weighted paired model differences. This safeguard does not itself run ERA5 training; results must come from a new manual run on updated `main`.

To reduce unnecessary GitHub Actions usage, the general unit-test workflow skips documentation-only pushes, while retaining automated tests on code, workflow and dependency changes. The previous [v1.9 CI passed](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45443685789). Check the current v2.0 CI before starting any additional compute.

## E1 v2.1 — independently verified real-data weighted + seasonal-local findings

The [latest real ERA5 3-seed experiment](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45478166832) **completed successfully on v2.0 code**, including audited sample coverage, actual latitude coordinates, area-weighted skill and training-only month-local hot-gridcell P95 metrics. See [the verified v2.1 evidence report](docs/E1_V21_VERIFIED_WEIGHTED_SEASONAL_ERA5_RESULTS.md) and [diagnostic artifacts](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45478166832/artifacts/14961610828).

Across seeds 11/22/33, **area-weighted 2012 test RMSE** was Persistence **3.4613 K**, FNO **3.1756 ± 0.0225 K**, U-Net **3.1264 ± 0.0201 K**. **Month-local area-weighted hot-gridcell recall** was Persistence **0.5430**, FNO **0.4260 ± 0.0413**, U-Net **0.4118 ± 0.0512**. Thus a reproducible small-sample **mean-skill versus extreme-recall tradeoff** appears in the historical development holdout. This is **not** established climate-shift generalization; longer independent years and CMIP6 controlled shifts remain required.

## E1 v2.2 — audited parameter counts and convergence experiment

The [parameter audit tool](scripts/audit_model_parameters.py) reports both PyTorch elements and independent real scalar degrees of freedom. For the **current width=8, modes=4, depth=2, 17×17** baselines: FNO 4,353 PyTorch elements including 4,096 *complex* spectral elements (8,449 real degrees of freedom); U-Net 6,753 real elements; Persistence 0. These are **architecture/accounting** numbers, not compute-matched parameter budgets.

E1 evaluation now records **per-epoch minibatch training MSE and independent validation MSE**, selects the best checkpoint by validation error and records `best_epoch`; additionally it includes grouped real/complex model parameter counts. The [manual 50-epoch real ERA5 convergence workflow](.github/workflows/e1-convergence-dev.yml) compares FNO and U-Net on the fixed 2008–2012 cohort, seed 11, preserves strict +24h and latitude provenance, and archives training/validation histories. This workload is **manual only**. No 50-epoch numeric result is claimed until it runs and its artifacts are audited. Existing [verified three-seed real data results](docs/E1_V21_VERIFIED_WEIGHTED_SEASONAL_ERA5_RESULTS.md) remain the current empirical evidence.

## E1 v2.5–v2.6 — 100-epoch replication and matched-capacity pilot

[Verified real ERA5 100-epoch 3-seed experiment](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45556189440): area-weighted RMSE Persistence **3.4613 K**, FNO **2.8473 ± 0.0051 K**, U-Net **2.8096 ± 0.0438 K**. Monthly grid-local train-only P95 hot-gridcell Recall: Persistence **0.5430**, FNO **0.4428 ± 0.0024**, U-Net **0.4199 ± 0.0211**. The [full validated results report](docs/E1_V25_100EPOCH_THREE_SEED_REAL_ERA5_RESULTS.md) explains why model ranking is seed-dependent and why these are **not climate-shift generalization results**.

The [capacity comparison workflow](.github/workflows/e1-capacity-pilot.yml) adds an independent `--unet-width` argument and uses the [predeclared real-DOF parameter-near configurations](docs/E1_V23_CAPACITY_MATCHED_DESIGN.md). It performs 50-epoch single-seed FNO/U-Net tests on the same historical ERA5 cohort, and archives actual model parameter counts plus area-weighted scores. No capacity experiment numerical result is asserted until its separate workflow succeeds.

## E1 v2.7–v2.8 — capacity pilot verified; independent held-out years running

The [real ERA5 matched-capacity pilot](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45661879900) succeeded (50 epochs, seed 11): FNO-vs-U-Net weighted 2012 RMSE (K) **Small 2.878 vs 2.783**, **Medium 2.801 vs 2.805**, **Large 2.767 vs 2.729**. Details and scientific limitations are documented in [the capacity pilot evidence report](docs/E1_V27_REAL_ERA5_MATCHED_CAPACITY_RESULTS.md).

The new [2012–2014 independent held-out year experiment](.github/workflows/e1-heldout-years.yml) is **one-time push triggered** ([run 45664389336](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45664389336)). Training is frozen at 2008–2010 and validation at 2011; 2012, 2013 and 2014 are separately scored without refitting or test-year-specific checkpoint selection. For each held-out year, results include true latitude-weighted RMSE, unweighted skill and grid-local training-only monthly P95 exceedance diagnostics, separately for Persistence, FNO and U-Net. The workflow uses 100 epochs and seeds 11, 22, 33. **The multi-year result is pending until this Actions run succeeds; no per-year scores are assumed.**

## E1 v2.9–v3.0 — verified three-year generalization and descriptive shift audit

The [independent 2012–2014 test-year run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45679648002) passed and generated three-seed year-specific **cos(latitude)-weighted** metrics. In **every individual held-out year**, FNO and U-Net improved RMSE compared to Persistence but had lower training-only *monthly grid-local P95* temperature exceedance Recall. Detailed observed numbers, caveats and provenance are in [v2.9 report](docs/E1_V29_VERIFIED_YEARWISE_2012_2014.md).

The [v3.0 historical distribution shift audit](.github/workflows/e1-historical-shift.yml) is an independent low-cost, one-time experiment based on exactly the existing ERA5 extract, with **no training**. It measures held-out annual temperature anomalies and standardized-tail frequencies using **2008–2010 training targets only** as a local monthly climatology reference; 2011 validation and 2012–2014 tests cannot influence the fitted reference. This is a **descriptive historical distribution comparison, not an attribution of anthropogenic climate change or future SSP generalization**. Track [GitHub Actions run 45708056678](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45708056678); numerical results must not be asserted until completed successfully.

## E1 v3.0–v3.1 — measured historical distribution shifts and conditional forecast diagnostics

[Verified train-only ERA5 distribution audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45711150645): relative to local monthly 2008–2010 training climatology, the sampled **2012 / 2013 / 2014** weighted mean temperature anomalies were **+1.197 / +0.175 / +1.016 K**; the fractions exceeding local training **+2σ** were **3.32% / 2.25% / 8.74%**, respectively. All numbers, caveats, and side-by-side forecast skill values are archived in [v3.0 evidence report](docs/E1_V30_VERIFIED_HISTORICAL_SHIFT_AND_SKILL.md). These are *historical sampled-cohort differences*, not anthropogenic climate attribution.

New [training-only conditional error decomposition](src/climate_shift/conditional_skill.py) reports **weighted RMSE/MAE/bias by five standardized temperature-anomaly bins** for each heldout year and each Persistence/FNO/U-Net forecast. It fits monthly local-gridcell means and scales on training targets only. [Run 45715908140](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45715908140) is the one-time real ERA5 100-epoch three-seed evaluation including this diagnostic. **Conditional numerical results are pending successful run validation.**

## E1 v3.2–v3.3 — verified tail-loss intervention and moderate-weight sensitivity

The [successful controlled real ERA5 tail-loss run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45718431425) used 100 epochs, FNO and U-Net, three paired seeds, and 2012–2014 heldout years. Tail-weight=3 raised mean **monthly local-P95 Recall** for FNO **0.5035→0.5444** and U-Net **0.4575→0.5433**, but reduced **Precision** (**0.6306→0.6089**; **0.6637→0.6142**) and slightly worsened pooled **weighted RMSE** (**2.8586→2.8759 K**; **2.8408→2.8606 K**). The two neural models improved *conditional* z>2 RMSE and signed heat-tail bias in every test year; [full evidence and design limitations](docs/E1_V32_VERIFIED_TAIL_AWARE_MSE_ABLATION.md).

A separately predeclared, smaller training tail-weight=1 (hot multiplier 2) [sensitivity run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45751577655) has been triggered to examine mean-skill vs recall/precision tradeoffs. This new run is not yet a verified result and its weight is not selected using the test years. Neither result establishes future SSP/CMIP6 climate-shift generalization.

## E1 v3.3–v3.4 — three measured tail-loss weights, fresh-year confirmation

The real ERA5 [tail-weight=1 (2× hot sample) experiment](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45751577655) **succeeded**. Along with the [standard MSE and tail-weight=3 control](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45718431425), the observed 2012–2014 results are:

| Architecture | Hot train multiplier | Weighted RMSE (K) | Train-month-local P95 recall | P95 precision |
|---|---:|---:|---:|---:|
| FNO | 1× | **2.8586** | 0.5035 | **0.6306** |
| FNO | 2× | 2.8727 | 0.5252 | 0.6150 |
| FNO | 4× | 2.8759 | **0.5444** | 0.6089 |
| U-Net | 1× | 2.8408 | 0.4575 | **0.6637** |
| U-Net | 2× | **2.8308** | 0.4764 | 0.6578 |
| U-Net | 4× | 2.8606 | **0.5433** | 0.6142 |

These are the three-seed means over the same historical heldout years. A small U-Net win at hot multiplier 2 is exploratory; weight 2 was investigated **after** reviewing heldout results at 4×, so it must not be marketed as selected on a fully untouched test cohort. See [complete evidence report](docs/E1_V33_VERIFIED_THREE_WEIGHT_PARETO_RESULTS.md).

A [new independent 2015–2017 historical holdout check](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45754843640) has been launched for **all frozen candidate weights 0/1/3**, with 2008–2010 train, 2011 validation, three seeds and 100 epochs unchanged. No numerical claims about the new years until the run succeeds. Even successful future *historical years* cannot stand in for actual CMIP6/SSP climate-shift evaluation.

## E1 v3.4 — *successful* independent 2015–2017 historical holdout

The **[new-year ERA5 three-arm run succeeded](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45754843640)** (100 epochs, three seeds, 2008–10 train, 2011 validation, previously unused **2015–17** historical holdout). Model settings, loss candidates and seed IDs were fixed before running these years. This corroborates an **exploratory U-Net modest tail-weight=1 (2× hot training samples) mean/tail improvement**, but not full Precision–Recall dominance.

| Model | Hot training sample multiplier | New-year weighted RMSE K | Monthly grid-local P95 recall | P95 precision |
|---|---:|---:|---:|---:|
| FNO | 1× | **2.7952** | 0.5625 | **0.6993** |
| FNO | 2× | 2.7988 | 0.5983 | 0.6866 |
| FNO | 4× | 2.8049 | **0.6159** | 0.6822 |
| U-Net | 1× | 2.7956 | 0.5041 | **0.7417** |
| U-Net | 2× | **2.7713** | 0.5196 | 0.7374 |
| U-Net | 4× | 2.8138 | **0.6004** | 0.6955 |

For U-Net at 2×, all three training seeds reduce RMSE relative to MSE; local P95 Recall rises *on average*, but not for every individual seed. In every heldout year 2015–2017, *conditional observed z>2* hot-gridcell RMSE improves relative to MSE. Full numerics, provenance and limitations: [v3.4 independently confirmed historical-year result](docs/E1_V34_FRESH_2015_2017_THREE_ARM_CONFIRMATION.md). **Historical-year validation must not be described as future climate-change or SSP generalization.** Candidate testing on 2015–17 is now closed; future design changes require a further independent test domain.

## E1 v4.0 — CMIP6 Historical / SSP domain-shift feasibility (metadata-only)

Following successful **six historical ERA5 heldout years** and [independent 2015–2017 tail-loss confirmation](docs/E1_V34_FRESH_2015_2017_THREE_ARM_CONFIRMATION.md), the work is transitioning to an **actual climate-scenario shift design**. The [full preregistered CMIP6 protocol](docs/E1_V40_CMIP6_HISTORICAL_SSP_PROTOCOL.md) distinguishes the current ERA5 **instantaneous +24h** temperature forecasting target from CMIP6 daily `tas` and NASA daily `tasmax`. The tasks **cannot be compared as the same forecast without redefined paired inputs and temporal harmonization**.

The [one-time public Pangeo/ESGF metadata feasibility audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45758461342) searches for strictly matched **day/tas Historical, ssp245 and ssp585** stores by `source_id`, `member_id` and `grid_label`, with verified catalog checksum and complete provenance. This job downloads **catalog metadata only**, no climate fields and no neural training. Dataset compatibility, availability, time calendars and future-climate model skill remain **unverified** until the metadata, store-read and experiment gates pass. Run code in [scripts/audit_cmip6_catalog.py](scripts/audit_cmip6_catalog.py), safeguards in [tests/test_cmip6_catalog.py](tests/test_cmip6_catalog.py).

## E1 v4.1 — CMIP6 catalog verified; live Zarr-store compatibility gate

The [v4.0 public CMIP6 metadata audit succeeded](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45758551504): **238 matching day/tas Historical–SSP245–SSP585 model/member/grid trios across 36 distinct climate models**. This is catalog evidence, not yet field-access evidence. The [archived catalog audit and strict source/member matching](docs/E1_V41_VERIFIED_CATALOG_AND_ZARR_ACCESS_GATE.md) include the catalog SHA256 and selected model entries.

A [read-only v4.1 Zarr metadata probe](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759128555) was launched for the matched ACCESS-CM2 and AWI-CM-1-1-MR historical and SSP stores. It checks actual anonymous store accessibility, daily temperature variable `tas`, Kelvin units, time calendar and grid schema. It does not read full temperature fields, tune models or train neural operators. Actual store results remain **unverified** until that run completes. Daily-mean CMIP6 `tas` is not the same forecast variable/temporal target as the existing 24h ERA5 instantaneous temperature pairs; a separate daily model protocol is required.

## E1 v4.2 — verified six CMIP6 Zarr metadata opens; native dates / first physical samples queued

The [read-only actual public CMIP6 Zarr run](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759128555) **succeeded for both complete model trios**: ACCESS-CM2 and AWI-CM-1-1-MR, each Historical/SSP245/SSP585. All six actual stores support anonymous metadata reads and pass `day/tas`, Kelvin, CF time, latitude/longitude and shape checks. This moves beyond the 36-model **catalog eligibility** audit, but is not yet a climate-forecast benchmark.

A [corrected v4.2 tiny-slice/date probe](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759757258) has been launched: it decodes actual native CF-calendar first/last dates and samples at most 12 temperature values per store. [Protocol and compatibility gates](docs/E1_V42_CMIP6_SMALL_SLICE_CALENDAR_GATE.md) explicitly require successful output before any climate-field extraction or daily FNO/U-Net training. Do not claim v4.2 compatibility or future SSP skill before verifying its successful run and artifact.

## E1 v4.2 correction — true common 2015–2100 SSP coverage is mandatory

The [first actual CF-calendar / tiny-temperature read](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45759757258/artifacts/15089577174) successfully read temperature samples and native calendar dates for all six stores. However, it revealed that the **selected ACCESS-CM2 `ssp585` store spans 2201–2300**, while its Historical store spans 1850–2014 and SSP245 spans 2015–2100. **ACCESS-CM2 is NOT currently eligible for the required common 2015–2100 three-scenario comparison.** The inspected AWI-CM-1-1-MR stores span Historical 1850–2014 and both SSPs 2015–2100.

The original v4.2 job finished as failed: its preliminary scientific eligibility check was too permissive and a downstream year-limit assertion rejected an actual 2300 date. These are now both corrected. The [one-time strict-period rerun](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45760624361) records the ACCESS extension but must mark it **ineligible**, while retaining AWI as a valid candidate only after a successful end-to-end audit. The full actual-coverage evidence and limitations are documented in [v4.2 corrected calendar gate](docs/E1_V42_CMIP6_SMALL_SLICE_CALENDAR_GATE.md). Additional ACCESS-CM2 SSP585 stores covering 2015–2100 may exist; none has yet been inspected for this purpose. **No SSP forecast skill is claimed.**

## E1 v4.2–v4.3 — confirmed real CMIP6 calendar/data gate, AWI native daily-pair preflight

The **[corrected real CMIP6 native CF dates and 12-cell temperature read](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764033503) succeeded**. Complete 2015–2100 `day/tas` Historical+SSP245+SSP585 matches were confirmed for **AWI-CM-1-1-MR r1i1p1f1 gn**; the selected ACCESS-CM2 SSP585 dataset spans **2201–2300** and is explicitly ineligible for this common comparison. Original model grids are AWI 192×384 and ACCESS 144×192, unlike the existing ERA5 17×17 forecast task. [Full source audit and correction](docs/E1_V42_CMIP6_SMALL_SLICE_CALENDAR_GATE.md).

A one-time [v4.3 AWI real Western-US regional +1 native day pair probe](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116) has been triggered. This pilot reads just two predeclared temperature pairs per experiment (Historical, SSP245, SSP585), coordinates `25–50°N, 125–100°W` (235–260°E), and checks exact calendar-native +1 day, finite Kelvin data and consistent native spatial shape. It **does not train neural models or establish climate scenario forecast skill**. [Daily-pair and split protocol](docs/E1_V43_AWI_CMIP6_DAILY_PAIR_REGIONAL_PROTOCOL.md).

## E1 v4.3 — **real** AWI-CMIP6 day/tas native regional daily-pair pilot succeeded

The [v4.3 native +1 day AWI regional-pair workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116) **succeeded**, after the [strict v4.2 Historical/SSP 2015–2100 common-coverage gate](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764033503) also succeeded. AWI-CM-1-1-MR `r1i1p1f1 gn` Historical, SSP245 and SSP585 have been directly opened, with real Celsius-equivalent absolute Kelvin `day/tas` grids sampled. Western US `25–50N, 125–100W` maps onto a native **26×27** grid (not ERA5's 17×17). Six date-pairs (two per scenario) were confirmed **exactly +1 native calendar day**, with all 702 temperature cells per sampled daily frame finite. [Complete verified provenance and bounded Persistence sanity checks](docs/E1_V43_VERIFIED_REAL_AWI_DAILY_REGIONAL_PAIRS.md) and [original workflow artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116/artifacts/15091098317).

**This is a six-date-pair source/target *data feasibility* pilot, not real CMIP6 future neural-operator skill.** No CMIP6 FNO/U-Net training has run. No scenario-wide RMSE, future heatwave skill, OOD generalization or warming-retention claim is justified yet. The new daily-mean task is separate from ERA5 instantaneous +24h forecasting.

## E1 v4.4 — AWI native daily-mean temperature multiyear sparse cohort audit

After the [successful real v4.3 +1-day regional Zarr paired-data pilot](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45764223116), the [v4.4 real multiyear audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45854938557) is testing a **fixed 312-pair sparse cohort**: AWI-CM-1-1-MR r1i1p1f1 gn Historical training prototypes (1980, 1990, 2000, 2004, **96 pairs**), Historical validation prototypes (2005, 2010, 2014, **72 pairs**), and each SSP245/SSP585 test prototype (2030, 2050, 2080, **72 pairs/scenario**). Every selected year/month uses exactly days **8 and 20**, a true +1 native-calendar-day lead, the same 26×27 native-Western-US grid, and finite Kelvin checks. Results are archived as aggregate diagnostics only; SSP targets are never fit.

This preliminary cohort has **only eight observations per training calendar month**, less than the E1 ERA5 training-only monthly P95 minimum of **20**. Thus **no neural tail-loss training can be justified from these v4.4 samples**; the next stage must expand a separately audited full or sufficiently dense daily Historical training set. See [frozen data QA protocol](docs/E1_V44_SPARSE_MULTIYEAR_COHORT_PROTOCOL.md) and [source audit code](scripts/audit_cmip6_awi_multiyear_cohort.py). A first CI attempt failed solely at Python module import; it was repaired without changing any data cohort or scientific constraints. The corrected run's final real-data outcome is pending.

## E1 v4.4 VERIFIED and v4.5 historical fit-data expansion

The [real AWI-CMIP6 multiyear sample audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45854938557) **succeeded** for all **312** exact native-calendar +1 day Western-US 26×27 temperature pairs (Historical train-prototype 96; Historical validation-prototype 72; future SSP245 72; SSP585 72). The selected-month/day subset is sparse, not a complete daily time series. Descriptive Persistence cos(lat) RMSE in this cohort was Historical **2.342443 K**, SSP245 **2.171081 K**, SSP585 **2.118762 K**; **not neural future forecast skill**. [Full source verification and limitations](docs/E1_V45_HISTORICAL_DENSE_MONTH_P95_GATE.md).

The [v4.5 Historical-only data extraction workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857281051) has been triggered. It freezes **25 training years (1980–2004), with three dates/month/year = 900 daily pairs, 75 training targets per calendar month**, plus **10 Historical validation years (2005–2014), two dates/month/year = 240 daily pairs**. The exporter verifies the exact daily target lead, no Historical train/validation time boundary overlap, 26×27 native grids, finite Kelvin, matching source/target times and post-export input SHA, and successfully fits local monthly P95 **training-only** *if* the run completes. No SSP data are read in this phase. This is a **sparse daily** Historical fit-data cohort, not 25 years of complete daily records. Do not report v4.5 as verified until its CI output succeeds.

Full unit testing for native-calendar cases now declares `cftime` in the climate-data optional dependencies. The v4.5 Historical artifact (if generated) is a separate train/validation cohort; the original `compare_e1.py` three-fold ERA5 trainer must not silently conflate SSP experiments into a third chronological fold. Future scenario evaluation will need a separate explicit Historical-fitted model and strictly read-only SSP test harness.

## E1 v4.5 verified Historical training artifact; v4.6 blinded Historical checkpoint smoke

The [real v4.5 AWI CMIP6 Historical-only extraction succeeded](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857281051): **1,140 daily-mean regional `tas(t)→tas(t+1 day)` pairs**, of which **900 train (1980–2004)** and **240 Historical validation (2005–2014)**. Each train calendar month has **75** targets; historical train-only grid-local monthly P95 fit and exact +24h pair/split/finite-K checks **passed**. Verification SHA256 is `ac0a354032836c5de5463b122d26fad2270fd5ba2ebb48f43074105c8ce7b914`; data [artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857281051/artifacts/15138645070). **Historical validation-only Persistence weighted RMSE = 2.32978 K**; this is not future scenario skill.

The [v4.6 checksum-locked Historical-only two-epoch FNO and U-Net checkpoint smoke](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45859545722) is underway. It downloads the exact v4.5 data artifact, performs no new CMIP6 remote dataset reads, selects checkpoints only on Historical validation, and does **not** access SSP targets. See [scientific split/SSP-blind protocol](docs/E1_V46_HISTORICAL_FIT_AND_SSP_BLIND_TEST_PROTOCOL.md) and [v4.5 verified audit](docs/E1_V45_HISTORICAL_DENSE_MONTH_P95_GATE.md). Neither the smoke results nor SSP forecast skill are verified until successful artifact inspection. The repo's latest default [Unit test CI](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45857475991) is successful following the `cftime` dependency fix.

## E1 v4.6 Historical neural smoke **verified**; v4.7 three-weight/three-seed experiment underway

The [v4.6 genuine AWI Historical training smoke](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45859545722) **passed every job and artifact check**. Fixed seed 11, 2 epochs: FNO and U-Net weighted RMSE were respectively **2.48197 K** and **2.72977 K** on Historical **2005–2014 validation**, versus **2.32978 K** for Persistence. Both neural models *underperform* Persistence in this deliberately short smoke; checkpoints were written/reloaded, input SHA256 and three years/target data guards passed, **no SSP data read**. [Verified 2-epoch checkpoints](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45859545722/artifacts/15137711474).

The [v4.7 fixed Historical-only experiment](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303) has been triggered with **3 independent seeds × 3 fixed loss-weight arms (0/1/3) × 100 epochs × FNO/U-Net**, checkpoint selection only using Historical validation. Three workflow jobs track the three weights independently and all checkpoints/source hashes must be archived. **Status is running, not a result.** [Predeclared full v4.7 method and no-SSP safeguards](docs/E1_V47_FROZEN_HISTORICAL_THREE_WEIGHT_PROTOCOL.md). This Historical validation set is used for checkpoint selection and cannot be described as an untouched future climate SSP holdout. The next actual SSP experiment must use only frozen Historical-fitted checkpoints and predeclared future sample dates.

## E1 v4.7 Historical-only loss comparison **verified**, v4.8 future SSP test launched

The [full 100-epoch, 3-seed × 3-weight AWI-CMIP6 Historical FNO/U-Net workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45861178303) **succeeded for all three loss arms**, producing **18 saved checksum-verified checkpoints**. Historical 2005–2014 *validation* RMSE (K, three-seed mean) was **FNO 2.0343 / 2.0393 / 2.0500** and **U-Net 2.0486 / 2.0487 / 2.0540** for training hot multipliers **1× / 2× / 4×**, versus **Persistence 2.3298**. FNO local train-based monthly P95 Recall **0.5512 / 0.5538 / 0.5915**; U-Net **0.5286 / 0.5149 / 0.5531**. These Historical results are not SSP forecasts and use the same validation set for checkpoint selection; they cannot be described as an untouched climate-shift test. [Exact three-seed Historical result table, source hashes and limitations](docs/E1_V48_FROZEN_SSP_FIRST_EVALUATION_PROTOCOL.md).

The [v4.8 one-time SSP245/SSP585 no-refit evaluation](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174) has started. It loads **all 18 frozen** Historical models, first verifies their source dataset and checkpoint hashes, then evaluates the previously fixed **2030/2050/2080, days 8/20 monthly** schedule (72 pairs per scenario) on real AWI native daily mean temperature projections. It uses **only Historical training** for normalization and local monthly P95; no SSP target fitting or future model selection. **No future SSP metrics should be claimed until the v4.8 workflow and archived report actually pass.** Results are sparse simulated scenario target comparisons, not independently observed future weather or heatwave duration tests.

## E1 v4.8 — first **verified** frozen Historical → SSP245 / SSP585 neural evaluation

The [complete first v4.8 CMIP6 future-simulation test workflow](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174) **succeeded**, with exact pre-SSP checks of **all 18 immutable v4.7 Historical-only checkpoints**, source SHA256, training-only normalization and grid-local monthly P95. Frozen weights were applied to native AWI-CM-1-1-MR `day/tas` future projections in **SSP245 and SSP585**, on predeclared 2030/2050/2080 dates (8th/20th each month, 72 forecast pairs/scenario). **No SSP training, tuning or re-selection took place.**

| Predictor | Hot train multiplier | SSP245 weighted RMSE K | SSP585 weighted RMSE K | SSP245 local P95 Recall | SSP585 local P95 Recall |
|---|---:|---:|---:|---:|---:|
| Persistence | — | 2.1711 | 2.1188 | 0.7286 | 0.7962 |
| FNO | 1× | **1.9323** | **1.8836** | 0.679 | 0.763 |
| FNO | 2× | 1.9409 | 1.8889 | 0.681 | 0.764 |
| FNO | 4× | 1.9599 | 1.9049 | **0.708** | **0.786** |
| U-Net | 1× | **1.9552** | 2.0180 | 0.611 | 0.628 |
| U-Net | 2× | 1.9625 | **2.0087** | 0.624 | 0.641 |
| U-Net | 4× | 1.9635 | 2.0187 | **0.651** | **0.663** |

Three training-initialization seeds are included in each neural mean. FNO standard-MSE RMSE is about **11%** below Persistence in each sampled SSP. Strong high-tail weighting raises Recall but also raises RMSE; Persistence remains higher on hot Recall than the neural models. Fixed Historical training-month P95 exceedance fraction grows with projected scenario/year (SSP585: 2030 **14.9%**, 2080 **49.2%** of sampled gridcell-days). This is a simulated threshold exceedance, **not a heatwave frequency or duration measure**.

**[Full verified annual results, calibration tradeoffs and scientific caveats](docs/E1_V48_VERIFIED_FROZEN_SSP245_SSP585_RESULTS.md)**; [unaltered Actions result artifact](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45864317174/artifacts/15140346151). The study uses only one model member and two days/month in three future years; SSP data were previously seen in source QA but **never used to fit or select predictors**. Simulated CMIP6 future temperatures are not independently observed future truth. Broader CMIP6-member and denser temporal holdouts remain necessary before strong future-climate generalization claims.

## E1 v4.9 — second CMIP6 model BCC-CSM2-MR source audit passed

The [real-data BCC-CSM2-MR r1i1p1f1 gn three-scenario feasibility audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45865322183) **succeeded**. Native `day/tas` Kelvin Historical Zarr spans **1850–2014**, SSP245 and SSP585 both span **2015–2100**; all three Zarr stores opened and passed small-slice native CF calendar/physical temperature sanity checks. Unlike the initially selected ACCESS-CM2 extended SSP585 store (2201–2300), this BCC candidate is eligible for next-stage common-period regional pairing. [Exact source archive and follow-up gates](docs/E1_V49_BCC_SECOND_MODEL_VERIFIED_SOURCE_GATE.md). **No BCC neural operator training or future SSP score has yet been produced**, and AWI-trained checkpoints must not be applied to BCC grids without a separately defined transfer experiment.

## E1 v5.0 — second GCM BCC native Western-US regional source-pair audit launched

The [v4.9 successful BCC-CSM2-MR real source audit](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45865322183) verified all three actual `day/tas` Kelvin Zarr stores and exact Historical 1850–2014 / SSP245+SSP585 2015–2100 coverage. A separate [v5.0 real **regional** BCC +1 native day pair gate](https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45868440215) has been triggered: it samples six prescribed paired daily grids in Western US (25–50°N, 125–100°W), checks identical actual coordinate **values** and native calendar across scenarios, finite Kelvin temperatures and exact temporal alignment. The BCC native regional grid size and source-pair results are **pending** until this job succeeds. [Detailed second-GCM replication design and safeguards](docs/E1_V50_BCC_REGIONAL_INDEPENDENT_REPLICATION_GATE.md).

The verified AWI Historical→SSP frozen-score experiment remains unchanged. No BCC trained FNO/U-Net checkpoints or SSP predictions have been generated; AWI normalization, P95 thresholds and trained weights will **not** be reused as if they were BCC-native training.
