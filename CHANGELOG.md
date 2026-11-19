# Changelog

Notable changes to the `x13` crate. The crate follows
[Semantic Versioning](https://semver.org/).

## 0.1.0 (2026-11-19)

First release: a pure-Rust translation of Census X-13ARIMA-SEATS Version 1.1
Build 62 for monthly and quarterly series, checked against Build 62 on a
corpus of FRED series.

- regARIMA preprocessing: explicit or automatic ARIMA models, identity or log
  transformation, trading-day, Easter and other calendar effects, user
  regressors, outlier detection, missing-value estimation, forecasts and
  backcasts.
- X-11, standalone or after regARIMA, with its seasonal and trend filter
  options, extreme-value weighting, legacy Easter adjustment and quality,
  seasonality and spectral diagnostics.
- SEATS after regARIMA, with its components, forecasts and diagnostics.
- An in-memory API: `adjust` takes a borrowed `Series`, in which NaN marks a
  missing observation, and a `Spec`.
- Typed errors, sorted by cause into `ErrorKind`, including errors where
  Build 62 reaches an undefined state.
