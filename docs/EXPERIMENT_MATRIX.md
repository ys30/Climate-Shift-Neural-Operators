# Experiment matrix — proposed, not yet executed

| ID | Dataset | Split / intervention | Model candidates | Primary outcome | Status |
| --- | --- | --- | --- | --- | --- |
| E1a | ERA5/WB2 | Chronological held-out years | Persistence, U-Net, FNO | MAE/RMSE and tail bias | Planned |
| E1b | ERA5/WB2 | Held-out Western US subregions | U-Net, FNO, transformer | Spatial OOD degradation | Planned |
| E1c | ERA5/WB2 | Hot-condition strata defined by training climatology | U-Net, FNO | Extreme frequency/duration | Planned |
| E2a | CMIP6 | Historical vs SSP2-4.5, within-model | FNO and matched baseline | Delta-temperature fidelity | Planned |
| E2b | CMIP6 | Historical vs SSP5-8.5, model/member holdout | FNO and matched baseline | Cross-model warming response | Planned |
| E3a | E1/E2 | Climate conditioning ablation | Conditional FNO and equally conditioned controls | Generalization gain | Planned |

## Decisions to freeze before runs

- Forecast task: lead time and permitted predictors
- Geography, grid spacing, years and climate models
- Heatwave event threshold and reference climatology
- Calendar mapping and regridding protocol
- Hyperparameter search budget and stopping rule
- Bootstrap/blocking unit and minimum independent seed count

Results must link to code commit, configuration, data-version manifest and output checksum.
