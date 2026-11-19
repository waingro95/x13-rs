//! Guides to seasonal adjustment with this crate.
//!
//! The item documentation describes each type on its own. These guides show
//! how the pieces fit together, task by task:
//!
//! | Guide | Read it to |
//! | --- | --- |
//! | [`x11`] | adjust with X-11, directly or after a regARIMA model |
//! | [`regarima`] | choose the model and transformation, and estimate calendar effects, outliers, missing values and forecasts |
//! | [`seats`] | adjust with SEATS, the model-based method |
//! | [`coverage`] | see which X-13ARIMA-SEATS capabilities the crate covers, and which are verified |
//!
//! The rest of this page covers what every request shares: the input, the
//! result and the errors.
//!
//! # Choose a method
//!
//! - **Direct X-11** ([`Spec::x11`]) filters the series as given. It
//!   needs no model and is the fastest, but it cannot fill gaps, fit
//!   outliers or trading-day effects, or extend the series with forecasts.
//! - **X-11 with regARIMA** ([`Spec::x11_regarima`]) first fits
//!   a regression model with ARIMA errors. The model extends the series and
//!   removes calendar effects, outliers and gaps before X-11 filters it. This
//!   is the usual X-13ARIMA-SEATS setup.
//! - **SEATS** ([`Spec::seats`]) fits the same kind of model and
//!   derives the components from it. It accepts a narrower set of models and
//!   series lengths.
//!
//! # Choose a start period
//!
//! [`adjust`](crate::adjust) takes a [`Series`]: the values, borrowed as a
//! slice, and the [`Period`] of the first one ([`Period::monthly`] or
//! [`Period::quarterly`]). The values fill consecutive periods from there.
//! Monthly and quarterly are the only frequencies, and the crate does not
//! aggregate monthly values into quarters.
//!
//! [`Series::periods`] lists the period of every value, and
//! [`Period::shift`] moves a period by a number of steps, which is how results
//! such as outliers and forecasts are dated.
//!
//! ```
//! use x13::{Period, Series};
//!
//! let values = [1.0, 2.0, 3.0, 4.0];
//! let start = Period::quarterly(2019, 3)?;
//! let series = Series::new(start, &values);
//! assert_eq!(series.periods()?.nth(2), Some(Period::quarterly(2020, 1)?));
//! assert_eq!(start.shift(2)?.to_string(), "2020Q1");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Prepare the values
//!
//! - NaN marks a missing value. Requests with a regARIMA model estimate it;
//!   direct X-11 rejects it ([`Error::MissingValue`]).
//! - Infinity is invalid ([`Error::NonFiniteValue`]), and so is a magnitude of
//!   1e19 or more ([`Error::ObservationTooLarge`]).
//! - Every other value is an observation. X-13ARIMA-SEATS reads `-99999` as
//!   missing; this crate does not, so convert such codes to NaN first.
//! - An empty or all-zero series is rejected.
//!
//! Each method adds its own limits on length, sign and frequency; the guides
//! list them.
//!
//! # Read the result
//!
//! A successful call returns an [`Adjustment`] that owns its data:
//!
//! - the components, one value per input value, each a
//!   [`Series`](crate::Series) dated from the input's first period:
//!   [`seasonally_adjusted`](crate::Adjustment::seasonally_adjusted),
//!   [`seasonal`](crate::Adjustment::seasonal),
//!   [`trend_cycle`](crate::Adjustment::trend_cycle) and
//!   [`irregular`](crate::Adjustment::irregular);
//! - the regARIMA results, when there is a model:
//!   [`model`](crate::Adjustment::model),
//!   [`forecasts`](crate::Adjustment::forecasts),
//!   [`outliers`](crate::Adjustment::outliers),
//!   [`calendar_effects`](crate::Adjustment::calendar_effects) and the
//!   automatic choices;
//! - [`diagnostics`](crate::Adjustment::diagnostics), with an X-11 or a SEATS
//!   part;
//! - [`warnings`](crate::Adjustment::warnings). Check them on every result:
//!   they report fallbacks and conditions that X-13ARIMA-SEATS warns about
//!   while still producing an adjustment.
//!
//! # Handle errors
//!
//! Every failure is a typed [`Error`]. [`Error::kind`] sorts it into one of
//! seven classes ([`ErrorKind`]), which is usually all a caller needs to
//! decide what to do:
//!
//! | Kind | Meaning | Typical response |
//! | --- | --- | --- |
//! | [`InvalidInput`](crate::ErrorKind::InvalidInput) | The values or supplied regressors break the input rules | Fix the data |
//! | [`InvalidSpec`](crate::ErrorKind::InvalidSpec) | The request is invalid or does not fit the series | Fix the request |
//! | [`Unsupported`](crate::ErrorKind::Unsupported) | A valid request outside what this crate runs | Try another model or method |
//! | [`TooShort`](crate::ErrorKind::TooShort) | The series has too few observations for this pipeline | Skip it until it has more data |
//! | [`UnsuitableData`](crate::ErrorKind::UnsuitableData) | This series cannot be adjusted this way | Skip it, or use another method |
//! | [`Numerical`](crate::ErrorKind::Numerical) | Estimation or decomposition failed for this model | Try another model |
//! | [`Internal`](crate::ErrorKind::Internal) | A bug in this crate | Report it |
//!
//! More kinds may be added, so match with a wildcard arm. When adjusting many
//! series, a common pattern is to try automatic modeling, fall back to the
//! airline model when that fails, and skip series that cannot be adjusted:
//!
//! ```
//! use x13::{
//!     Adjustment, ArimaModel, Error, ErrorKind, Period, RegArimaSpec, Series, Spec, Transform,
//! };
//!
//! fn adjust_one(series: Series<'_>) -> Result<Option<Adjustment>, Error> {
//!     let automatic = RegArimaSpec::automatic_model().with_transform(Transform::Log);
//!     let error = match x13::adjust(series, &Spec::seats(automatic)) {
//!         Ok(adjustment) => return Ok(Some(adjustment)),
//!         Err(error) => error,
//!     };
//!     match error.kind() {
//!         // Another model may succeed where the selected one failed.
//!         ErrorKind::Unsupported | ErrorKind::Numerical => {
//!             let fallback =
//!                 RegArimaSpec::new(ArimaModel::airline()).with_transform(Transform::Log);
//!             x13::adjust(series, &Spec::seats(fallback)).map(Some)
//!         }
//!         // Too short, nonpositive under log, and so on: skip this series.
//!         ErrorKind::TooShort | ErrorKind::UnsuitableData => Ok(None),
//!         // Bad input, a bad request or a bug in the crate: report it.
//!         _ => Err(error),
//!     }
//! }
//!
//! let start = Period::monthly(2012, 1)?;
//! let two_years: Vec<f64> = (0..24).map(|i| 100.0 + f64::from(i % 12)).collect();
//! assert!(adjust_one(Series::new(start, &two_years))?.is_none()); // SEATS needs three years
//!
//! let error = adjust_one(Series::new(start, &[1.0, f64::INFINITY, 3.0])).unwrap_err();
//! assert_eq!(error.kind(), ErrorKind::InvalidInput);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What is verified
//!
//! Results are checked against the US Census Bureau's X-13ARIMA-SEATS
//! program, Version 1.1 Build 62, on a set of public-domain FRED series and
//! within fixed tolerances; the [crate documentation](crate#accuracy)
//! describes how. Some options and combinations of options that the API
//! accepts have not been checked this way: they run, but their results have
//! not been compared with the reference program. The [`coverage`] guide marks
//! which are verified.
//!
//! [`Adjustment`]: crate::Adjustment
//! [`Error`]: crate::Error
//! [`Error::MissingValue`]: crate::Error::MissingValue
//! [`Error::NonFiniteValue`]: crate::Error::NonFiniteValue
//! [`Error::ObservationTooLarge`]: crate::Error::ObservationTooLarge
//! [`Error::kind`]: crate::Error::kind
//! [`ErrorKind`]: crate::ErrorKind
//! [`Period`]: crate::Period
//! [`Period::monthly`]: crate::Period::monthly
//! [`Period::quarterly`]: crate::Period::quarterly
//! [`Period::shift`]: crate::Period::shift
//! [`Series`]: crate::Series
//! [`Series::periods`]: crate::Series::periods
//! [`Spec::seats`]: crate::Spec::seats
//! [`Spec::x11`]: crate::Spec::x11
//! [`Spec::x11_regarima`]: crate::Spec::x11_regarima

#[doc = include_str!("coverage.md")]
pub mod coverage {}

#[doc = include_str!("regarima.md")]
pub mod regarima {}

#[doc = include_str!("seats.md")]
pub mod seats {}

#[doc = include_str!("x11.md")]
pub mod x11 {}
