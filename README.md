# x13

Seasonal adjustment in pure Rust: a translation of X-13ARIMA-SEATS, the U.S.
Census Bureau's seasonal adjustment program, for monthly and quarterly series.
It covers regARIMA preprocessing (explicit or automatic models, identity or log
transformation, calendar and supplied regressors, outlier detection, missing
values), X-11 and SEATS.

Calls run in memory: no Fortran, executable, spec files or subprocesses. The
only dependency is `thiserror`.

The reference is Census X-13ARIMA-SEATS Version 1.1 Build 62. This crate is an
independent translation, not produced or endorsed by the Census Bureau.

## Usage

```sh
cargo add x13
```

Adjust five years of a synthetic monthly series with standalone X-11:

```rust
use x13::{Period, Series, Spec, adjust, x11};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A rising level with a fixed seasonal swing.
    let values: Vec<f64> = (0..60)
        .map(|i| (100.0 + 0.5 * f64::from(i)) * (1.0 + 0.02 * f64::from(i % 12)))
        .collect();
    let series = Series::new(Period::monthly(2015, 1)?, &values);
    assert_eq!(series.periods()?.last(), Some(Period::monthly(2019, 12)?));

    let adjustment = adjust(series, &Spec::x11(x11::Options::new()))?;
    // Every component is a `Series` dated like the input.
    assert_eq!(adjustment.seasonally_adjusted().start(), series.start());
    assert_eq!(adjustment.seasonally_adjusted().len(), values.len());
    Ok(())
}
```

Each `Spec` constructor selects a pipeline:

| Constructor | Pipeline |
|---|---|
| `Spec::x11` | X-11 on the series as given |
| `Spec::x11_regarima` | regARIMA preprocessing, then X-11 |
| `Spec::seats` | regARIMA preprocessing, then SEATS |

The `guide` module of the [API documentation](https://docs.rs/x13) walks
through each pipeline and its limits.

## Checked against Build 62

The crate's results are compared with Build 62's over a table of supported
configurations, the call table, on about 500 public FRED series, gaps included.
For every row and every series it applies to:

- the discrete choices (transform, model orders, outlier types and dates)
  match exactly, and the numbers agree within fixed tolerances;
- where Build 62 rejects the input, this crate returns an error;
- where Build 62 reaches an undefined state, such as a hang or a nonfinite
  result, this crate returns a typed error instead. Each such case is listed.

The comparison covers the seasonal, seasonally adjusted, trend and irregular
series, the SEATS transitory component, the transform, the model and its
coefficients, the outliers and the point forecasts and backcasts. It does not
cover diagnostics, standard errors, likelihoods or warnings. A configuration
the API accepts but no row covers is not verified.

In the source repository, `docs/parity-call-table.md` is the table and
`CONTRIBUTING.md` explains how to run it.

## Platform

The crate is plain Rust, and its own tests run on Linux and macOS. The
comparison with Build 62 is run only on Apple silicon macOS (aarch64-darwin).
Build 62 is compiled there with gfortran `-O2` and links the system math
library, and this crate's `ln`, `exp`, `powf` and trigonometric calls use it
too. Rust does not require those functions to be correctly rounded, so results
on other platforms can differ in the last bits, and near a decision boundary
in a discrete choice. On Linux with glibc, some unit-test results differ from
Apple silicon's by up to about 1e-8 relative. The comparison has not been run
on another platform.

## Differences from X-13 defaults

Each request type starts from Build 62's defaults, except:

- SEATS forecasts: Build 62 forecasts three years (36 months or 12 quarters)
  under SEATS; `RegArimaSpec` defaults to one year for every pipeline. Request
  `ForecastHorizon::Periods(36)` or `Periods(12)` to match.
- X-11 mode after regARIMA with no transform: Build 62 with no `transform`
  spec runs multiplicative X-11; `Spec::x11_regarima` with no mode set resolves
  identity to additive, Build 62's rule for an explicit `function = none`.
  Setting `x11::Mode::Multiplicative` with `x11::Options::with_mode` matches it
  only without calendar, outlier, regressor or missing-value effects; with any
  of them it returns `Error::InvalidSpec`.

Options Build 62 has but this crate does not expose (`pickmdl`, Box-Cox powers
other than log, SEATS tuning options, sliding spans, revision history and
others) keep their Build 62 defaults where they apply.

## Missing values and errors

- NaN marks a missing observation. regARIMA requests estimate it; direct X-11
  rejects it with `Error::MissingValue`. Every finite value is an observation,
  including `-99999.0`, which Build 62 reads as its missing-value code. Infinity
  is invalid.
- `Error::kind()` sorts every error into an `ErrorKind`: invalid input, invalid
  spec, unsupported, too short, unsuitable data, numerical failure or internal
  error. Route on it rather than on `Error`'s variants, which are
  non-exhaustive.

## Minimum supported Rust version

Rust 1.88.

## Contributing

See `CONTRIBUTING.md` in the source repository. It explains how to report a
result that differs from Build 62 and how to run the checks.

## License

Licensed under either the Apache License, Version 2.0 (`LICENSE-APACHE`) or the
MIT license (`LICENSE-MIT`), at your option. `NOTICE` carries the attributions
for Build 62 and the routines it builds on.
