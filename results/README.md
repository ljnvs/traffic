# Frozen result summaries

These small tables reproduce the numerical claims in the preprint without redistributing the underlying traffic records.

`v8_test_summary.csv` reports seven-day arithmetic means. `v8_daily_results.csv` retains paired date-level outcomes for the five final control modes. `v8_uncertainty.csv` uses 50,000 fixed-seed paired bootstrap resamples over seven dates; all intervals are exploratory. `v7_audit.json` records effective action freedom and the synthetic positive control.

The main effect is a ratio of seven-day arithmetic means. For lower-is-better metrics, improvement is `(baseline mean - candidate mean) / baseline mean`; positive values mean improvement. Different historical control versions are diagnostic and should not be treated as one common ablation scale.
