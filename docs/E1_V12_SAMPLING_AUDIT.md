# E1 v1.2 — within-month sampling correction

## Confirmed earlier results

- The bounded real-data ERA5 pilot succeeded: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45109019800
- Unit-test run on v1.1 main: https://github.com/ys30/Climate-Shift-Neural-Operators/actions/runs/45233298119 — success.
- The manual-only contiguous-year workflow exists but **has not run as of the v1.2 patch**.

## Sampling flaw found and corrected

Earlier `make_24h_pairs` scanned valid 24h pairs in chronological order and stopped as soon as `max_samples` was reached. With `stride=4` on 6-hourly ERA5 data and 24 samples/month, this selected only the **beginning** of each month. Such samples are not spread uniformly across calendar days and could bias development evaluation toward early-month processes.

The adapter now accepts `selection="uniform"` while preserving the default `"first"` behavior for callers relying on it. The multi-year exporter explicitly uses `"uniform"`, selecting approximately equally spaced valid timestamp pairs through the month. Only actual available +24-hour pairs are considered; missing timestamps are not filled or inferred. A regression test verifies that a 2-sample January extraction covers the late part of January rather than the first two source days.

Dataset metadata now records `selection: uniform_across_month_valid_pairs`. Data produced before v1.2 **must not be silently combined** with v1.2 data: recalculate the sample SHA-256, rerun audit and document sampling-method version.

## Limits

- Uniformly spaced sampling is deterministic but **not random sampling**. The approach is useful for bounded engineering tests, not unbiased statistical inference about climate extremes.
- Sparse temporal samples break sequences of consecutive days. **Do not compute heatwave duration or persistence from the sparse development NPZ.**
- Training/validation/test are still years chosen for development, not a pre-registered final OOD experiment.
- For the scientific heatwave analysis, build continuous daily fields, training-period daily climatology, >=3 consecutive-day event rules, latitude weighting and dependence-aware uncertainty.
- Keep large GitHub Actions workflows manual-only to control costs.

## Next gate

Confirm the new unit tests pass; then execute the manually gated contiguous-year workflow when compute budget is acceptable. Any model results from that workflow must be labeled development-only.
