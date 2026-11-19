Model-based seasonal adjustment with SEATS.

SEATS splits the fitted ARIMA model itself into trend, seasonal and irregular
models, then estimates each component with the filter its model implies. It
always runs after regARIMA preprocessing, so a request is
[`Spec::seats`] with a [`RegArimaSpec`]. The
[regARIMA guide](crate::guide::regarima) covers the model, transformation, calendar
effects, outliers and missing values; this guide covers what differs under
SEATS.

# Adjust with SEATS

```rust
# fn synthetic(years: usize) -> Vec<f64> {
#     let mut season = [0.90, 0.94, 1.00, 1.03, 1.06, 1.09, 1.08, 1.05, 1.01, 0.98, 0.93, 0.93];
#     let mut state: u32 = 1;
#     let mut noise = move || {
#         state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
#         f64::from(state >> 8) / f64::from(1u32 << 24) - 0.5
#     };
#     let mut level = 100.0;
#     (0..12 * years)
#         .map(|i| {
#             level += 0.4 + noise();
#             season[i % 12] += 0.01 * noise();
#             season[i % 12] * level * (1.0 + 0.02 * noise())
#         })
#         .collect()
# }
use x13::regarima::ResolvedArimaModel;
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform};

let values = synthetic(8);
let airline = ArimaModel::airline();
let preprocessing = RegArimaSpec::new(airline).with_transform(Transform::Log);
let spec = Spec::seats(preprocessing);
let adjustment = x13::adjust(Series::new(Period::monthly(2012, 1)?, &values), &spec)?;

// Under log, the seasonal component is a factor and the adjusted series is in
// series units.
let seasonal = adjustment.seasonal().expect("the airline model has a seasonal part");
let adjusted = adjustment.seasonally_adjusted();
for ((value, factor), adjusted) in values.iter().zip(seasonal.values()).zip(adjusted.values()) {
    assert!((adjusted * factor - value).abs() < 1e-9 * value);
}
assert!(adjustment.trend_cycle().is_some());

let seats = adjustment.diagnostics().seats().expect("a SEATS adjustment");
assert_eq!(seats.model().orders(), ResolvedArimaModel::from(airline));
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Choose a supported model

SEATS decomposes only some models. An explicit model's orders must be one of
these:

| Seasonal | Regular |
| --- | --- |
| (0 1 1) | (0 1 0), (0 1 1), (0 1 2), (0 2 1), (1 1 0), (2 1 0), (3 1 0), (1 1 1), (2 1 1), (3 1 1), (1 1 2) or (2 1 2) |
| (0 0 1) | (1 1 1), (2 1 1) or (3 1 1) |
| (1 1 0) | (0 1 1) |
| (0 0 0) | (0 1 1), (0 1 2), (1 1 0), (1 1 1), (1 1 2), (2 1 1), (2 1 2) or (3 1 1) |
| up to (1 1 1) | other orders up to (3 2 2) |

The last row takes the remaining orders with at most five AR and MA
coefficients in all, or six when the model has a seasonal AR term or no
differencing. Pure differencing models, such as `(0 1 0)(0 1 0)`, are also
accepted. Other orders return an error of kind [`ErrorKind::Unsupported`].

[`RegArimaSpec::automatic_model`] may select a model outside this set. Then
`adjust` returns an error of kind [`ErrorKind::Unsupported`] that names the
selected model, and an explicit model such as the airline model may still work
on that series. A constant series has no estimated model to decompose, with an
explicit or automatic model, and returns [`Error::DegenerateSeries`].

```rust
# fn synthetic(years: usize) -> Vec<f64> {
#     let mut season = [0.90, 0.94, 1.00, 1.03, 1.06, 1.09, 1.08, 1.05, 1.01, 0.98, 0.93, 0.93];
#     let mut state: u32 = 1;
#     let mut noise = move || {
#         state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
#         f64::from(state >> 8) / f64::from(1u32 << 24) - 0.5
#     };
#     let mut level = 100.0;
#     (0..12 * years)
#         .map(|i| {
#             level += 0.4 + noise();
#             season[i % 12] += 0.01 * noise();
#             season[i % 12] * level * (1.0 + 0.02 * noise())
#         })
#         .collect()
# }
use x13::{ArimaModel, ArimaOrder, ErrorKind, Period, RegArimaSpec, Series, Spec};

let model = ArimaModel::new(ArimaOrder::new(3, 2, 3)?, ArimaOrder::new(0, 1, 1)?)?;
let spec = Spec::seats(RegArimaSpec::new(model));
let error = x13::adjust(Series::new(Period::monthly(2012, 1)?, &synthetic(8)), &spec).unwrap_err();
assert_eq!(error.kind(), ErrorKind::Unsupported);
assert_eq!(
    error.to_string(),
    "SEATS does not support ARIMA orders outside the SEATS models this crate supports"
);
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Check the series length

A monthly series needs 36 to 780 observations. A quarterly series needs at
least 16, and at most 328, 327, 326 or 325 when it starts in the first,
second, third or fourth quarter. A shorter series returns
[`Error::TooShort`] and a longer one [`Error::SeatsTooLong`].
The first year must not be negative ([`Error::SeatsNegativeYear`]).

Missing values work as in the [regARIMA guide](crate::guide::regarima#fill-missing-values).

# Request forecasts

Forecasts default to one year, as for X-11. X-13ARIMA-SEATS defaults to three
years under SEATS; ask for that with [`ForecastHorizon::Periods`] (36 months
or 12 quarters). Longer horizons return an error of kind
[`ErrorKind::Unsupported`], and zero gives no forecasts.

SEATS ignores backcasts, as in X-13ARIMA-SEATS: [`Adjustment::backcasts`] is
`None` and [`Warning::SeatsBackcastsIgnored`] reports how many were requested.

[`Adjustment::forecasts`] come from the regARIMA fit, even when SEATS changes
the final model.

```rust
# fn synthetic(years: usize) -> Vec<f64> {
#     let mut season = [0.90, 0.94, 1.00, 1.03, 1.06, 1.09, 1.08, 1.05, 1.01, 0.98, 0.93, 0.93];
#     let mut state: u32 = 1;
#     let mut noise = move || {
#         state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
#         f64::from(state >> 8) / f64::from(1u32 << 24) - 0.5
#     };
#     let mut level = 100.0;
#     (0..12 * years)
#         .map(|i| {
#             level += 0.4 + noise();
#             season[i % 12] += 0.01 * noise();
#             season[i % 12] * level * (1.0 + 0.02 * noise())
#         })
#         .collect()
# }
use x13::regarima::{ForecastHorizon, ForecastSpec};
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, Warning};

let preprocessing = RegArimaSpec::new(ArimaModel::airline())
    .with_transform(Transform::Log)
    .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(36), 12)?);
let spec = Spec::seats(preprocessing);
let adjustment = x13::adjust(Series::new(Period::monthly(2012, 1)?, &synthetic(8)), &spec)?;

assert_eq!(adjustment.forecasts().map(|forecasts| forecasts.series().len()), Some(36));
assert!(adjustment.backcasts().is_none());
assert!(adjustment
    .warnings()
    .iter()
    .any(|warning| matches!(warning, Warning::SeatsBackcastsIgnored { periods: 12, .. })));
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Set preprocessing options

Every [`RegArimaSpec`] setting applies under SEATS, with these differences:

- At most three years of forecasts (36 months or 12 quarters); backcasts are
  ignored with a warning (see [Request forecasts](#request-forecasts)).
- An empty [`CalendarSpec`] returns an error of kind [`ErrorKind::Unsupported`].
- Supplied regressors must cover three years past the end of the series (36
  months or 12 quarters), whatever the forecast horizon, because the signal
  extraction uses that span.

# Read the components

[`Adjustment::seasonally_adjusted`] and [`Adjustment::irregular`] cover the
observed span (X-13ARIMA-SEATS tables S11 and S13). Every component is a
[`Series`] starting at [`Adjustment::start`], the input's first period.
[`Adjustment::seasonal`] (S10) and [`Adjustment::trend_cycle`] (S12) return an
`Option`, because X-13ARIMA-SEATS omits them in two cases: the seasonal for a
model without a seasonal part under the identity transformation, and the trend
when it is zero everywhere.

Under the identity transformation every component is in series units. Under
log, the seasonal and irregular are factors around one, and the adjusted
series and trend are in series units.

[`Adjustment::seats_transitory`] returns the transitory component (S14) when
the final model has one, for example from regular AR roots that belong to
neither the trend nor the seasonal. Otherwise it is `None`.

Fitted outliers stay in the adjusted series. Level shifts go to the trend, and
additive outliers and temporary changes to the irregular. Calendar effects are
removed from the adjusted series; [`Adjustment::calendar_effects`] reports
them.

# Read the SEATS model

[`Adjustment::model`] is the regARIMA fit. [`Diagnostics::seats`] returns the
[`seats::Diagnostics`], whose [`model`](crate::seats::Diagnostics::model)
is the model SEATS decomposed. It can differ from the fit: SEATS may bound a
coefficient, refit the model or drop the mean, and
[`refitted`](crate::seats::Diagnostics::refitted),
[`coefficients_at_bound`](crate::seats::Diagnostics::coefficients_at_bound)
and [`mean_included`](crate::seats::Diagnostics::mean_included) say which. A
[`seats::Model`] gives its orders and coefficients, each part separately or all
together, in the X-13ARIMA-SEATS sign convention: `(1 − θB)` for MA and
`(1 − φ₁B − φ₂B² − …)` for AR.

```rust
# fn synthetic(years: usize) -> Vec<f64> {
#     let mut season = [0.90, 0.94, 1.00, 1.03, 1.06, 1.09, 1.08, 1.05, 1.01, 0.98, 0.93, 0.93];
#     let mut state: u32 = 1;
#     let mut noise = move || {
#         state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
#         f64::from(state >> 8) / f64::from(1u32 << 24) - 0.5
#     };
#     let mut level = 100.0;
#     (0..12 * years)
#         .map(|i| {
#             level += 0.4 + noise();
#             season[i % 12] += 0.01 * noise();
#             season[i % 12] * level * (1.0 + 0.02 * noise())
#         })
#         .collect()
# }
use x13::seats::TransitoryScale;
use x13::{ArimaModel, ArimaOrder, Period, RegArimaSpec, Series, Spec, Transform};

let values = synthetic(8);
let model = ArimaModel::new(ArimaOrder::new(2, 1, 1)?, ArimaOrder::new(0, 1, 1)?)?;
let preprocessing = RegArimaSpec::new(model).with_transform(Transform::Log);
let spec = Spec::seats(preprocessing);
let adjustment = x13::adjust(Series::new(Period::monthly(2012, 1)?, &values), &spec)?;

let seats = adjustment.diagnostics().seats().expect("a SEATS adjustment");
assert_eq!(seats.model().regular_ar().len(), 2);
assert_eq!(seats.model().seasonal_ma().len(), 1);

// The AR part gives this model a transitory component.
let transitory = adjustment.seats_transitory().expect("a transitory component");
assert_eq!(transitory.scale(), TransitoryScale::Multiplicative);
assert_eq!(transitory.series().len(), values.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Handle SEATS errors

When SEATS would re-estimate the fitted model with smaller orders, as
X-13ARIMA-SEATS does when a fitted AR root reaches the unit circle, this crate
does not follow and returns an error of kind [`ErrorKind::Unsupported`]. When
the fitted model has no admissible decomposition, a regARIMA forecast passed to
SEATS is not finite, or the decomposition's arithmetic fails, the error is of
kind [`ErrorKind::Numerical`]; X-13ARIMA-SEATS also stops without a
decomposition when the model is inadmissible. For both kinds another model,
such as the airline model, may succeed; the
[guide index](crate::guide#handle-errors) shows a fallback.

# Not available

Component standard errors, the uncertainty of the final parameters, SEATS
spectra, the combined adjustment factors (S16), SEATS tuning options and the
X-13ARIMA-SEATS report output are not available.

[`Adjustment::backcasts`]: crate::Adjustment::backcasts
[`Adjustment::calendar_effects`]: crate::Adjustment::calendar_effects
[`Adjustment::forecasts`]: crate::Adjustment::forecasts
[`Adjustment::irregular`]: crate::Adjustment::irregular
[`Adjustment::model`]: crate::Adjustment::model
[`Adjustment::seasonal`]: crate::Adjustment::seasonal
[`Adjustment::seasonally_adjusted`]: crate::Adjustment::seasonally_adjusted
[`Adjustment::seats_transitory`]: crate::Adjustment::seats_transitory
[`Adjustment::start`]: crate::Adjustment::start
[`Adjustment::trend_cycle`]: crate::Adjustment::trend_cycle
[`CalendarSpec`]: crate::regarima::CalendarSpec
[`Diagnostics::seats`]: crate::Diagnostics::seats
[`Error::DegenerateSeries`]: crate::Error::DegenerateSeries
[`Error::SeatsNegativeYear`]: crate::Error::SeatsNegativeYear
[`Error::SeatsTooLong`]: crate::Error::SeatsTooLong
[`Error::TooShort`]: crate::Error::TooShort
[`ErrorKind::Numerical`]: crate::ErrorKind::Numerical
[`ErrorKind::Unsupported`]: crate::ErrorKind::Unsupported
[`ForecastHorizon::Periods`]: crate::regarima::ForecastHorizon::Periods
[`RegArimaSpec`]: crate::RegArimaSpec
[`RegArimaSpec::automatic_model`]: crate::RegArimaSpec::automatic_model
[`Series`]: crate::Series
[`Spec::seats`]: crate::Spec::seats
[`Warning::SeatsBackcastsIgnored`]: crate::Warning::SeatsBackcastsIgnored
[`seats::Diagnostics`]: crate::seats::Diagnostics
[`seats::Model`]: crate::seats::Model
