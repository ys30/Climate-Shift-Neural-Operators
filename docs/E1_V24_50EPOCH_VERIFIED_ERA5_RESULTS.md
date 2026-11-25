# E1 v2.4 — observed 50-epoch real ERA5 convergence benchmark

**Verified run:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45554909392  
**Artifact:** https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45554909392/artifacts/14995272086  
**Run commit:** `fb5ecaa16c1c8ca8908f28a09d690b75a043ec4f`.  
**Status:** SUCCESS. All syntax, unit tests, real ERA5 extraction, 60-month coverage, +24h pair and temporal isolation, 50-epoch training, metadata and diagnostics gates passed.  
**Input file SHA-256:** `21ba61d319f0732508861418db8c676e3da73b38c59f45749d897fc424a74554`, matches earlier three-seed weighted real ERA5 run.  
**Spatial domain:** 25–50°N, −125 to −100°W, 17×17 ERA5 2m temperature, +24h paired forecast.  
**Split:** train 2008–10 (864 source/target pairs), validation 2011 (288), test 2012 (288). **Training seed:** 11. **Model widths/modes:** FNO width=8, modes=4, depth=2; U-Net width=8; learning rate 0.001; batch size 8; 50 epochs, validation-selected checkpoint. Data contain 24 sampled pairs/month, not continuous daily climate series.

## True observed test scores

| Model | Weighted MAE (K) | Weighted RMSE (K) | Unweighted RMSE (K) | Best validation epoch |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 2.318198 | 3.461329 | 3.536877 | N/A |
| FNO | 2.035454 | 2.877988 | 2.946331 | 48 |
| U-Net | 2.016156 | 2.857513 | 2.925577 | 42 |

For same fixed seed 11, the earlier 5-epoch weighted baseline RMSE from run 45478166832 was **FNO 3.154129 K** and **U-Net 3.129488 K**. These decreased by approximately **8.76%** and **8.69%**, respectively, at 50-epoch validation-selected checkpoints. The exact improvement depends on choosing the model checkpoint based on validation only; this is not matched compute across architectures.

FNO trainable parameter accounting: **4,353 PyTorch elements**, **8,449 real scalar degrees of freedom** (4,096 complex spectral elements). U-Net: **6,753 real scalar trainable parameters**. See `scripts/audit_model_parameters.py`.

## High-temperature recall and conditional bias

| Model | Global train-P95 recall | Local monthly train-P95 recall | Local monthly hot-conditional bias (K) |
| --- | ---: | ---: | ---: |
| Persistence | 0.738879 | 0.542995 | −1.458796 |
| FNO | 0.692915 | 0.424125 | −2.195769 |
| U-Net | Read the complete JSON artifact | 0.406570 | −2.293472 |

**Interpretation:** Extended training markedly improves overall area-weighted RMSE; global-P95 recall for FNO also increases compared to the five-epoch seed-11 experiment. However monthly grid-local P95 recall remains below Persistence for both trained models, so increasing training iterations alone did not resolve the high-tail fidelity deficit.

## Scientific interpretation and next gate

- **Five epochs were insufficient under this exact fixed pilot setup**: 50-epoch models substantially improved validation-selected test predictions. FNO best checkpoint was at epoch 48/50; further learning may still matter. U-Net best epoch 42/50.
- **Still one training seed and one historical test year**. Do not interpret a difference of 0.0205 K in model weighted RMSE as an architecture-level conclusion.
- **Not an event-based heatwave evaluation**: sparse sampling cannot support event duration/persistence.
- **Not demonstrated climate-shift generalization**: no future-warming/CMIP6 held-out shift yet.
- The proposed next step is a controlled longer-duration **3-seed** convergence test (50–100 epochs) and more than one independent year, with identical data and fixed train-only/local-P95 methodology.
