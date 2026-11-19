//! Pure-Rust X-13ARIMA-SEATS seasonal adjustment for monthly and quarterly series.
//!
//! X-13ARIMA-SEATS is the U.S. Census Bureau's seasonal adjustment program.
//! This crate is an independent translation of its Version 1.1 Build 62,
//! checked against that release (see [Accuracy](#accuracy)); it is not
//! produced or endorsed by the Census Bureau. It covers regARIMA preprocessing (explicit or automatic
//! models, identity or log transformation, calendar and user regressors,
//! outlier detection, missing values), X-11 and SEATS.
//!
//! Calls run in memory: no Fortran, executable, spec files or subprocesses.
//! Every request, result and error type is `Send + Sync`, and repeated calls
//! are deterministic and independent, so many series can be adjusted in
//! parallel.
//!
//! # Example
//!
//! Adjust five years of a synthetic monthly series with standalone X-11:
//!
//! ```
//! use x13::{Period, Series, Spec, adjust, x11};
//!
//! // A rising level with a fixed seasonal swing.
//! let values: Vec<f64> = (0..60)
//!     .map(|i| (100.0 + 0.5 * f64::from(i)) * (1.0 + 0.02 * f64::from(i % 12)))
//!     .collect();
//! let series = Series::new(Period::monthly(2015, 1)?, &values);
//! assert_eq!(series.periods()?.last(), Some(Period::monthly(2019, 12)?));
//!
//! let adjustment = adjust(series, &Spec::x11(x11::Options::new()))?;
//! assert_eq!(adjustment.seasonally_adjusted().len(), values.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Choosing a request
//!
//! [`adjust`] takes a [`Series`] (the observations and the period of the
//! first one) and a [`Spec`]. Each [`Spec`] constructor selects a pipeline:
//!
//! | Constructor | Pipeline |
//! |---|---|
//! | [`Spec::x11`] | X-11 on the series as given |
//! | [`Spec::x11_regarima`] | regARIMA preprocessing, then X-11 |
//! | [`Spec::seats`] | regARIMA preprocessing, then the SEATS model-based decomposition |
//!
//! A [`RegArimaSpec`] describes the preprocessing: [`RegArimaSpec::new`]
//! fixes the ARIMA orders, and [`RegArimaSpec::automatic_model`] selects
//! them as the program's automatic modeling does. Its setters add a transform,
//! calendar effects ([`regarima::CalendarSpec`]), outlier detection
//! ([`regarima::OutlierSpec`]),
//! user regressors and forecasts. [`x11::Options`] sets the X-11 mode and
//! filters.
//!
//! The [`guide`] module walks through each pipeline with examples.
//!
//! # Input
//!
//! A [`Series`] borrows its observations, which are consecutive periods
//! starting at its first period ([`Period::monthly`] or [`Period::quarterly`]).
//!
//! NaN marks a missing observation: regARIMA requests estimate it, and direct
//! X-11 rejects it. Every finite value is an observation, including `-99999.0`
//! (the missing-value code of X-13ARIMA-SEATS), and infinity is invalid. No
//! model is estimated when a series' observed values are all equal and its
//! first value is observed, so each gap takes that value and the result is
//! that of the same series without gaps.
//!
//! X-11 needs at least three years of observations, and SEATS at least 36
//! months or 16 quarters. The reference program's capacity sets the upper
//! limits: a span of 85 calendar years including forecasts and backcasts, and
//! under SEATS at most 780 monthly or 325 to 328 quarterly observations,
//! depending on the starting quarter.
//!
//! # Results
//!
//! [`Adjustment`] owns everything a call produces:
//!
//! - the components, each a [`Series`] dated from the input's first period:
//!   [`Adjustment::seasonally_adjusted`], [`Adjustment::seasonal`],
//!   [`Adjustment::trend_cycle`] and [`Adjustment::irregular`];
//! - the fitted regARIMA model, any automatic selections, and its forecasts and
//!   backcasts, which are dated as well: [`Adjustment::model`],
//!   [`Adjustment::model_selection`], [`Adjustment::transform_selection`],
//!   [`Adjustment::forecasts`] and [`Adjustment::backcasts`];
//! - the estimated calendar effects and outliers:
//!   [`Adjustment::calendar_effects`] and [`Adjustment::outliers`];
//! - X-11, SEATS and spectral diagnostics: [`Adjustment::diagnostics`];
//! - non-fatal conditions met along the way: [`Adjustment::warnings`].
//!
//! # Errors
//!
//! An invalid request or an unusable fit returns an [`Error`]. Its variants
//! carry the detail and are non-exhaustive; [`Error::kind`] sorts every one
//! into an [`ErrorKind`] to route on:
//!
//! ```
//! use x13::{ErrorKind, Period, Series, Spec, adjust, x11};
//!
//! // Two years is too short for X-11.
//! let values: Vec<f64> = (0..24).map(|i| 100.0 + f64::from(i % 12)).collect();
//! let series = Series::new(Period::monthly(2020, 1)?, &values);
//!
//! let error = adjust(series, &Spec::x11(x11::Options::new())).unwrap_err();
//! match error.kind() {
//!     ErrorKind::InvalidInput | ErrorKind::InvalidSpec => { /* fix the request */ }
//!     ErrorKind::TooShort => { /* wait for more data */ }
//!     ErrorKind::UnsuitableData => { /* this series cannot be adjusted this way */ }
//!     ErrorKind::Unsupported | ErrorKind::Numerical => { /* try another spec */ }
//!     _ => {}
//! }
//! assert_eq!(error.kind(), ErrorKind::TooShort);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The detail an [`Error`] variant carries is in the [`error`] module.
//!
//! # Accuracy
//!
//! Results are checked against X-13ARIMA-SEATS Version 1.1 Build 62, built
//! from the Census Bureau's source archive, on a fixed set of public-domain
//! FRED series: monthly and quarterly, some with missing values. For each
//! supported configuration:
//!
//! - the components (seasonal, seasonally adjusted, trend-cycle, irregular and
//!   the SEATS transitory component), the ARMA and regression coefficients,
//!   and the point forecasts and backcasts agree within fixed tolerances;
//! - discrete choices (transform, model orders, outlier types and dates) agree
//!   exactly;
//! - an input the reference program rejects is an [`Error`] here, and so is
//!   one where it reaches an undefined state, such as a hang or a nonfinite
//!   result.
//!
//! Agreement is within tolerance, not bit for bit. Diagnostics, standard
//! errors, forecast intervals, likelihoods, information criteria and warnings
//! are not compared. The source repository lists the verified configurations
//! and the tolerances, and contains the test that runs the comparison. A
//! configuration the API accepts but that list does not cover is not verified;
//! [`guide::coverage`] gives the status of each capability of the reference
//! program.
//!
//! The crate's own tests run on Linux and macOS. The comparison runs on
//! aarch64-darwin (Apple silicon macOS) only. Results use the platform libm,
//! which Rust does not require to be correctly rounded, so on other platforms
//! they can differ in the last bits, and a discrete choice that falls close to
//! its threshold can go the other way.
//!
//! # Differences from X-13ARIMA-SEATS defaults
//!
//! Options the reference program has but this crate does not expose
//! (`pickmdl`, Box-Cox powers other than log, SEATS tuning options, sliding
//! spans, revision history and others) keep their default values where they
//! apply. Two defaults differ: under SEATS, X-13ARIMA-SEATS forecasts three
//! years while [`RegArimaSpec`] defaults to one; and with no transform,
//! [`Spec::x11_regarima`] runs additive X-11 unless a mode is set, where
//! the X-13ARIMA-SEATS default is multiplicative.
//!
//! # Stability
//!
//! Public enums are `#[non_exhaustive]`, so a minor release can add options
//! and failure reasons, except where the set of cases is fixed, as in
//! [`regarima::Weekday`]. The minimum supported Rust version is 1.88.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::must_use_candidate, clippy::return_self_not_must_use)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::panic_in_result_fn,
        clippy::todo,
        clippy::unimplemented,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::dbg_macro,
    )
)]

// The README's Rust example runs as a doctest, so it cannot drift from the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod api;
mod arima;
mod automdl;
pub mod diagnostics;
mod easter_dates;
mod engine;
pub mod error;
pub mod guide;
mod numerics;
mod outliers;
pub mod regarima;
mod regression;
pub mod seats;
mod time;
mod transform;
pub mod x11;

pub use api::{
    Adjustment, ArimaModel, ArimaOrder, Backcasts, Diagnostics, Error, ErrorKind, Forecasts,
    Method, ModelResult, PeriodSpan, RegArimaSpec, Spec, Transform, Warning, adjust,
};
pub use time::{Frequency, Period, PeriodError, Periods, Series};

// Callers adjust many series on worker threads, so every request, result and
// error type is `Send + Sync`. An accidental `Rc`, `Cell` or raw pointer fails
// to compile here instead of in a caller's thread pool.
const _: () = {
    const fn assert<T: Send + Sync>() {}
    assert::<Adjustment>();
    assert::<Backcasts>();
    assert::<regarima::CalendarEffects>();
    assert::<regarima::CalendarSpec>();
    assert::<Diagnostics>();
    assert::<Error>();
    assert::<ErrorKind>();
    assert::<regarima::ForecastSpec>();
    assert::<Forecasts>();
    assert::<regarima::ManualOutlier>();
    assert::<ModelResult>();
    assert::<regarima::ModelSelectionResult>();
    assert::<regarima::OutlierSpec>();
    assert::<Period>();
    assert::<PeriodError>();
    assert::<PeriodSpan>();
    assert::<Periods>();
    assert::<RegArimaSpec>();
    assert::<diagnostics::SpectralPeaks>();
    assert::<diagnostics::Unavailable>();
    assert::<seats::Diagnostics>();
    assert::<seats::Model>();
    assert::<seats::Transitory>();
    assert::<Series<'static>>();
    assert::<Spec>();
    assert::<regarima::TransformSelectionResult>();
    assert::<regarima::UserRegressor>();
    assert::<Warning>();
    assert::<x11::Diagnostics>();
    assert::<x11::EasterEffect>();
    assert::<x11::Options>();
    assert::<diagnostics::QualityMeasures>();
    assert::<diagnostics::SeasonalityTests>();
    assert::<x11::SpectralDiagnostics>();
};
