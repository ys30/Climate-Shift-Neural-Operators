# E1 v2.7 — verified parameter-near FNO vs U-Net ERA5 pilot

- **Run**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45661879900
- **Result**: SUCCESS; dependencies and capacity tests, actual ERA5 extraction, coverage, train/validation/test separation, all 3 models/pairs, explicit real-DOF assertions, archival succeeded
- **Artifact**: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45661879900/artifacts/15046047773
- **Commit**: `67fde928e2f6f8f8e2555e3f1e94a9772e128e43`.
- **Protocol**: same 2008–2010 train, 2011 validation, 2012 test; 24 sampled pairs/month, real WB2 ERA5 temperature; 50 epochs, seed 11; best checkpoint selected using validation. FNO depth 2, width/modes as listed. U-Net width varied independently.

| Capacity group | FNO width/modes | FNO real scalar DOF | U-Net width | U-Net real scalar DOF | FNO area-weighted RMSE K | U-Net area-weighted RMSE K |
|---|---|---:|---:|---:|---:|---:|
| Small | 8 / 4 | 8,449 | 9 | 8,488 | 2.877988 | **2.783432** |
| Medium | 12 / 6 | 42,001 | 20 | 40,641 | **2.801073** | 2.805354 |
| Large | 16 / 8 | 131,969 | 36 | 130,177 | 2.766654 | **2.728935** |

## Constrained interpretation

- Larger parameter-near configurations have lower 2012 RMSE than small ones for both model families *in this single-seed trial*.
- No consistent architecture winner: U-Net beats FNO in Small and Large, FNO narrowly beats U-Net in Medium.
- DOF matching is **not compute, optimized convergence, data assimilation or tensor-operation matching**. A given FNO width and Fourier truncation may have different optimization dynamics than the corresponding U-Net.
- **Do not interpret** pairwise differences from one random seed and one 2012 test year as statistically reliable.
- This remains historical ERA5 interpolation/forecasting, **not future climate generalization**.
- Before extending architecture claims, report *per held-out test year* RMSE and local P95 Recall for 2012–2014 using a frozen training baseline (2008–2010), 2011 validation; keep test years out of checkpoint selection.
