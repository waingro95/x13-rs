Preprocessing with a regARIMA model.

A [`RegArimaSpec`] describes a regression model with ARIMA errors, fitted
before the decomposition. It extends the series with forecasts, estimates
calendar effects, outliers, supplied regressors and missing values, and removes
those effects before X-11 or SEATS runs. The same spec feeds
[`Spec::x11_regarima`] and [`Spec::seats`]. Where the fitted effects land in the
components is up to the method: see the [X-11](crate::guide::x11) and
[SEATS](crate::guide::seats) guides.

Each `with_*` method on [`RegArimaSpec`] sets one option and leaves the others
alone. The spec owns its settings and is `Clone`.

# Choose a model

[`RegArimaSpec::new`] takes an explicit [`ArimaModel`]: a regular
`(p d q)` and a seasonal `(P D Q)` order. Regular orders go up to `(3 2 3)` and
seasonal orders up to `(2 1 2)`; larger orders return
[`Error::InvalidSpec`]. [`ArimaModel::airline`] builds the airline
model, `(0 1 1)(0 1 1)`, the usual starting point. An explicit model starts
with the identity transformation, no constant, one year of forecasts, no
backcasts, no calendar effects and no outlier search.
[`RegArimaSpec::with_constant`] adds a constant. To have the orders chosen for
you instead, see
[Choose a model automatically](crate::guide::regarima#choose-a-model-automatically).

# Choose a model automatically

[`RegArimaSpec::automatic_model`] runs the X-13ARIMA-SEATS automatic model
search with its default settings. It chooses the orders and whether to include
a constant, so calling `with_constant` on it returns
[`Error::Unsupported`]. Its other defaults match an explicit model.
[`RegArimaSpec::with_transform`] with [`TransformSelection::automatic`] also
chooses between no transform and logs
([Transform the series](crate::guide::regarima#transform-the-series)). With
both, you name neither the orders nor the transform. Calendar effects,
outliers and regressors are still yours to ask for.

[`Adjustment::model_selection`] reports the search:
[`ModelSelectionResult::Selected`] holds the first model identified and
whether its mean was significant, and the final orders are in
[`ModelResult::model`].

Under X-11, a constant series skips preprocessing: the result has no model
and carries [`Warning::ConstantSeries`], and an automatic search reports
[`ModelSelectionResult::ConstantSeries`]. SEATS rejects a constant series.

`adjust` returns [`Error::Unsupported`] for these combinations:

- [`CalendarSelection::AicTest`] with an automatic model, an automatic
  transform or an outlier search. The test needs a fixed model and transform
  and no outlier search;
  [Estimate calendar effects](crate::guide::regarima#estimate-calendar-effects)
  gives its other limits.
- Manual outliers with an automatic model, an automatic transform or an
  outlier search.
- A constant setting on an automatic model.

Under X-11, an automatic transform also needs the X-11 mode left unset; with
[`x11::Options::with_mode`] the request returns [`Error::InvalidSpec`].

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
use x13::regarima::{
    CalendarSelection, CalendarSpec, Easter, ModelSelectionResult, TransformSelection,
    TransformSelectionResult,
};
use x13::{Error, Period, RegArimaSpec, Series, Spec, Transform, x11};

let values = synthetic(8);
let series = Series::new(Period::monthly(2012, 1)?, &values);
let preprocessing =
    RegArimaSpec::automatic_model().with_transform(TransformSelection::automatic());
let spec = Spec::x11_regarima(preprocessing.clone(), x11::Options::new());
let adjustment = x13::adjust(series, &spec)?;

assert!(matches!(adjustment.model_selection(), Some(ModelSelectionResult::Selected { .. })));
let model = adjustment.model().expect("a fitted model");
println!("selected {}", model.model()); // for example "(0 1 1)(1 1 0)"

// The series grows multiplicatively, so log wins.
let choice = adjustment.transform_selection().expect("automatic transformation");
assert!(matches!(choice, TransformSelectionResult::Compared { .. }));
assert_eq!(choice.selected(), Transform::Log);
assert_eq!(model.transform(), Transform::Log);

// Testing calendar effects needs the model and transform fixed first.
let calendar = CalendarSpec::new()
    .with_easter(Easter::new(8)?)
    .with_selection(CalendarSelection::AicTest);
let refused = Spec::x11_regarima(preprocessing.with_calendar(calendar), x11::Options::new());
assert!(matches!(x13::adjust(series, &refused), Err(Error::Unsupported(_))));
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Transform the series

The identity transformation is the default and accepts any finite values.
[`RegArimaSpec::with_transform`] with [`Transform::Log`] fits the model to the
logarithm of the series; every value must then be positive
([`Error::NonPositiveValue`]). Log suits series whose seasonal swings grow with
their level.

The same method with [`TransformSelection::automatic`] chooses between the
two by AICC. Log is chosen when
`log_aicc + aic_difference < no_transform_aicc`, and the default difference,
−2, leans towards log; pass an [`AutoTransformOptions`] with
[`AutoTransformOptions::with_aic_difference`] to change it. The choice and
both scores are in [`Adjustment::transform_selection`]. If
any value is zero or negative, identity is chosen without comparing, with
[`Warning::AutomaticTransformNonPositiveInput`].

Gaps lean the choice the other way. As in X-13ARIMA-SEATS, `log_aicc` counts
each gap as an observation of the internal placeholder, which adds about 41
per gap, so one gap can outweigh a clear preference for log. Gaps among the
first observations, which differencing uses up, don't count. Set the transform
explicitly on a series with gaps if the choice matters.

Box-Cox powers other than the logarithm are not available.

# Request forecasts and backcasts

[`ForecastSpec::new`] sets how many periods to forecast and backcast, up to 120
of each ([`Error::InvalidSpec`]). [`ForecastHorizon::OneYear`]
means twelve months or four quarters, and is the default; there are no
backcasts by default. Under X-11 the forecasts and backcasts extend the series
before filtering. SEATS takes at most three years of forecasts and ignores
backcasts; see the [SEATS guide](crate::guide::seats).

[`Adjustment::forecasts`] and [`Adjustment::backcasts`] return what the model
produced. When X-11 cannot use them, such as a nonpositive forecast under
multiplicative X-11, it adjusts the observed series alone and warns
([`Warning::ForecastsNotUsed`] or [`Warning::BackcastsNotUsed`]); both are
still returned. Each
holds point values on the original scale, 95% limits, and the values and
standard errors on the transformed scale. [`Forecasts`] also splits the
forecast variance into regression and ARIMA parts when the model has
regressors. Log forecasts are the exponential of the transformed forecast,
without a bias correction. Forecasts use the estimated model as it stands, as
in X-13ARIMA-SEATS, even when an MA root lies just inside the unit circle,
which leaves the model barely noninvertible.

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
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, x11};

let start = Period::monthly(2012, 1)?;
let values = synthetic(8);
let preprocessing = RegArimaSpec::new(ArimaModel::airline())
    .with_transform(Transform::Log)
    .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(24), 12)?);
let spec = Spec::x11_regarima(preprocessing, x11::Options::new());
let adjustment = x13::adjust(Series::new(start, &values), &spec)?;

let forecasts = adjustment.forecasts().expect("forecasts were requested");
let point = forecasts.series();
assert_eq!(point.start(), Period::monthly(2020, 1)?); // the period after the data
assert_eq!(point.len(), 24);
assert_eq!(forecasts.lower_95().start(), point.start());
let lower = forecasts.lower_95().values();
let upper = forecasts.upper_95().values();
for ((value, low), high) in point.values().iter().zip(lower).zip(upper) {
    assert!(low < value && value < high);
}

let backcasts = adjustment.backcasts().expect("backcasts were requested");
assert_eq!(backcasts.series().start(), start.shift(-12)?);
assert_eq!(backcasts.series().len(), 12);
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Estimate calendar effects

A [`CalendarSpec`] adds calendar regressors. It holds at most one
[`CalendarBasis`] and any of three moving holidays:

- **Flow trading days** ([`FlowTradingDay`]): six day-of-week contrasts, or one
  weekday-versus-weekend contrast ([`TradingDayContrast`]). By default they
  carry the X-13ARIMA-SEATS leap-year treatment; [`FlowLengthTreatment`] can
  drop the length effect or use the length of the month or quarter instead.
- **Stock trading days** ([`StockTradingDay`]), for stocks measured on one day
  of the month. Monthly series only.
- **Length alone** ([`LengthEffect`]): a leap-year or length-of-period effect
  without trading days.
- **Easter** ([`Easter`]), with a window of 0 to 25 days before Easter.
- **Labor Day** ([`LaborDay`]), window 1 to 25 days. Monthly only.
- **Thanksgiving** ([`Thanksgiving`]), offset −8 to −1 or 1 to 17 days.
  Monthly only.

A quarterly series with a monthly-only effect returns
[`Error::InvalidSpec`]. Every date, including forecasts and
backcasts, must fall inside the calendar's years: from 1776 for trading days,
1901 to 2100 for Easter and Labor Day, and 1939 to 2100 for Thanksgiving
([`Error::CalendarYearOutOfRange`]).

By default every configured effect stays in the model.
[`CalendarSelection::AicTest`] tests the trading days, then Easter, against
leaving them out, at the default AICC difference of zero. It needs an
explicit model and transformation, no outlier search, and a trading-day or
Easter effect. It cannot test a length effect on its own or the
length-of-period treatment. Other combinations return
[`Error::Unsupported`]. It does not search over Easter windows.

[`Adjustment::calendar_effects`] returns the fitted effects over the observed
span: differences in series units under identity, factors under log
([`CalendarEffectScale`]). Under log, the default length treatment of flow
trading days is a fixed leap-year factor rather than a regressor, as in
X-13ARIMA-SEATS; that factor is [`CalendarEffects::fixed_length`]. The
coefficients are in [`ModelResult::regression_coefficients`].

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
use x13::regarima::{
    CalendarBasis, CalendarSelection, CalendarSpec, Easter, FlowTradingDay, RegressionVariable,
    TradingDayContrast,
};
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, x11};

let calendar = CalendarSpec::new()
    .with_basis(CalendarBasis::FlowTradingDay(FlowTradingDay::new(
        TradingDayContrast::SixCoefficient,
    )))
    .with_easter(Easter::new(8)?)
    .with_selection(CalendarSelection::AicTest);
let preprocessing = RegArimaSpec::new(ArimaModel::airline())
    .with_transform(Transform::Log)
    .with_calendar(calendar);
let spec = Spec::x11_regarima(preprocessing, x11::Options::new());
let adjustment = x13::adjust(Series::new(Period::monthly(2012, 1)?, &synthetic(8)), &spec)?;

// The synthetic series has no trading-day effect, so the test drops it.
let model = adjustment.model().expect("a fitted model");
assert!(!model
    .regression_coefficients()
    .iter()
    .any(|coefficient| matches!(coefficient.variable(), RegressionVariable::TradingDay(_))));
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Find outliers

[`RegArimaSpec::with_outliers`] turns on the automatic outlier search.
[`OutlierSpec::new`] is the X-13ARIMA-SEATS default: additive outliers and
level shifts, added one at a time, with critical values set from the series
length.
[`OutlierSpec::with_types`] chooses the types, for example adding temporary
changes ([`OutlierTypes`]),
[`OutlierSpec::with_method`] switches to adding all candidates at once,
[`OutlierSpec::with_critical_values`] fixes the cutoffs, and
[`OutlierSpec::with_span`] limits the search to part of the series.

For outliers at known dates, [`RegArimaSpec::with_manual_outliers`] fits the
listed [`ManualOutlier`]s instead. Each date must be in the series and have its
frequency, a level shift cannot fall on the first or last observation, and an
outlier cannot repeat ([`Error::InvalidManualOutlier`],
[`Error::PeriodFrequencyMismatch`]). They need an explicit model and
transformation and no outlier search ([`Error::Unsupported`]).

A temporary change decays geometrically. The default rate is `0.7^(12/s)` for
`s` periods per year; [`RegArimaSpec::with_temporary_change_rate`] sets one
rate for every temporary change, searched or specified.

[`Adjustment::outliers`] lists every fitted outlier, searched or specified,
with its estimate, standard error and t-value.

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
use x13::regarima::{ManualOutlier, OutlierType};
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, x11};

let start = Period::monthly(2012, 1)?;
let mut values = synthetic(8);
for value in &mut values[60..] {
    *value *= 1.2; // the level jumps 20% in January 2017
}

let shift = Period::monthly(2017, 1)?;
let preprocessing = RegArimaSpec::new(ArimaModel::airline())
    .with_transform(Transform::Log)
    .with_manual_outliers(&[ManualOutlier::new(OutlierType::LevelShift, shift)]);
let spec = Spec::x11_regarima(preprocessing, x11::Options::new());
let adjustment = x13::adjust(Series::new(start, &values), &spec)?;

let outlier = &adjustment.outliers()[0];
assert_eq!((outlier.kind(), outlier.period()), (OutlierType::LevelShift, shift));
assert!((outlier.estimate() - 1.2_f64.ln()).abs() < 0.03); // on the log scale
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Fill missing values

Mark a missing value with NaN. The model estimates each one with a regressor
of its own ([`RegressionVariable::MissingValue`]), and every component is
returned at that period too. The regressor's coefficient is an offset from an
internal placeholder, not the estimated value.

Gaps have three limits. Every calendar month or quarter needs at least one
observation ([`Error::MissingSeason`]). A gap outside the model span returns
[`Error::MissingOutsideModelSpan`]. Under X-11 the missing-value regressors
count as regression effects, so they follow the mode rules in the
[X-11 guide](crate::guide::x11#choose-a-decomposition-mode). Direct X-11 has no
model and rejects gaps.

No model is estimated when a series' observed values are all equal and its first
value is observed. Each gap takes that value, and the result is that of the same
series without gaps; Build 62 leaves its internal placeholder in the gaps
instead. The limits above still apply.

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
use x13::regarima::RegressionVariable;
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, x11};

let start = Period::monthly(2012, 1)?;
let mut values = synthetic(8);
values[30] = f64::NAN; // July 2014
values[31] = f64::NAN; // August 2014

let preprocessing = RegArimaSpec::new(ArimaModel::airline()).with_transform(Transform::Log);
let spec = Spec::x11_regarima(preprocessing, x11::Options::new());
let adjustment = x13::adjust(Series::new(start, &values), &spec)?;

let gaps: Vec<_> = adjustment
    .model()
    .expect("a fitted model")
    .regression_coefficients()
    .iter()
    .filter_map(|coefficient| match coefficient.variable() {
        RegressionVariable::MissingValue { period } => Some(period),
        _ => None,
    })
    .collect();
assert_eq!(gaps, [start.shift(30)?, start.shift(31)?]);
assert!(adjustment.seasonally_adjusted().values().iter().all(|value| value.is_finite()));
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Restrict the model span

[`RegArimaSpec::with_model_span`] fits the model on an inclusive
[`PeriodSpan`] only. The fitted effects still apply to the whole series, and
forecasts still start after the last observation. A span that starts after the
first observation cancels backcasts ([`Warning::ModelSpanBackcastsIgnored`]).
A span outside the series returns [`Error::InvalidModelSpan`].

# Add your own regressors

[`RegArimaSpec::with_user_regressors`] adds dated columns built with
[`UserRegressor::new`]. [`UserRegressorType`] says where the fitted effect
goes: with the trading-day effects, with the holiday effects, or as a separate
regression effect. Values are used as given: they are not centred or
extended. Each column must cover the series and every forecast and backcast,
so one extra year under the default extensions
([`Error::UserRegressorCoverage`]); SEATS needs three extra years. The fitted
coefficient is identified by [`RegressionVariable::User`], which carries the
column's position in the list.

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
use x13::regarima::{RegressionVariable, UserRegressor, UserRegressorType};
use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, x11};

let start = Period::monthly(2012, 1)?;
let mut values = synthetic(8);
values[40] *= 0.8; // a strike cuts output in May 2015

// One value per observation, plus the default year of forecasts.
let mut strike = vec![0.0; values.len() + 12];
strike[40] = 1.0;
let strike = UserRegressor::new(UserRegressorType::User, Series::new(start, &strike))?;

let preprocessing = RegArimaSpec::new(ArimaModel::airline())
    .with_transform(Transform::Log)
    .with_user_regressors(&[strike]);
let spec = Spec::x11_regarima(preprocessing, x11::Options::new());
let adjustment = x13::adjust(Series::new(start, &values), &spec)?;

let coefficient = &adjustment.model().expect("a fitted model").regression_coefficients()[0];
assert!(matches!(coefficient.variable(), RegressionVariable::User { index: 0, .. }));
assert!((coefficient.estimate() - 0.8_f64.ln()).abs() < 0.05);
# Ok::<(), Box<dyn std::error::Error>>(())
```

# Read the fitted model

[`Adjustment::model`] returns the [`ModelResult`], or `None` for direct X-11
and for a constant series under X-11. It holds:

- [`transform`](crate::ModelResult::transform) and
  [`model`](crate::ModelResult::model), the transformation and orders used;
- [`parameters`](crate::ModelResult::parameters), the AR and MA coefficients
  with standard errors, signed as in X-13ARIMA-SEATS: `(1 − θB)` for MA and
  `(1 − φB)` for AR;
- [`regression_coefficients`](crate::ModelResult::regression_coefficients),
  one [`RegressionEstimate`] per regressor, named as X-13ARIMA-SEATS names
  them;
- [`includes_constant`](crate::ModelResult::includes_constant),
  [`innovation_variance`](crate::ModelResult::innovation_variance),
  [`log_likelihood`](crate::ModelResult::log_likelihood) and
  [`information_criteria`](crate::ModelResult::information_criteria) (AIC,
  AICC, Hannan–Quinn and BIC);
- the coefficient covariances, the residuals, the residual spectrum and the
  iteration counts.

# Not available

Tuning the automatic model search, the alternative model-picking procedure
(`pickmdl`), Box-Cox powers other than the logarithm, and conditional
likelihood estimation are not implemented.

[`Adjustment::backcasts`]: crate::Adjustment::backcasts
[`Adjustment::calendar_effects`]: crate::Adjustment::calendar_effects
[`Adjustment::forecasts`]: crate::Adjustment::forecasts
[`Adjustment::model`]: crate::Adjustment::model
[`Adjustment::model_selection`]: crate::Adjustment::model_selection
[`Adjustment::outliers`]: crate::Adjustment::outliers
[`Adjustment::transform_selection`]: crate::Adjustment::transform_selection
[`ArimaModel`]: crate::ArimaModel
[`ArimaModel::airline`]: crate::ArimaModel::airline
[`AutoTransformOptions`]: crate::regarima::AutoTransformOptions
[`AutoTransformOptions::with_aic_difference`]: crate::regarima::AutoTransformOptions::with_aic_difference
[`CalendarBasis`]: crate::regarima::CalendarBasis
[`CalendarEffectScale`]: crate::regarima::CalendarEffectScale
[`CalendarEffects::fixed_length`]: crate::regarima::CalendarEffects::fixed_length
[`CalendarSelection::AicTest`]: crate::regarima::CalendarSelection::AicTest
[`CalendarSpec`]: crate::regarima::CalendarSpec
[`Easter`]: crate::regarima::Easter
[`Error::CalendarYearOutOfRange`]: crate::Error::CalendarYearOutOfRange
[`Error::InvalidManualOutlier`]: crate::Error::InvalidManualOutlier
[`Error::InvalidModelSpan`]: crate::Error::InvalidModelSpan
[`Error::InvalidSpec`]: crate::Error::InvalidSpec
[`Error::MissingOutsideModelSpan`]: crate::Error::MissingOutsideModelSpan
[`Error::MissingSeason`]: crate::Error::MissingSeason
[`Error::NonPositiveValue`]: crate::Error::NonPositiveValue
[`Error::PeriodFrequencyMismatch`]: crate::Error::PeriodFrequencyMismatch
[`Error::Unsupported`]: crate::Error::Unsupported
[`Error::UserRegressorCoverage`]: crate::Error::UserRegressorCoverage
[`FlowLengthTreatment`]: crate::regarima::FlowLengthTreatment
[`FlowTradingDay`]: crate::regarima::FlowTradingDay
[`ForecastHorizon::OneYear`]: crate::regarima::ForecastHorizon::OneYear
[`ForecastSpec::new`]: crate::regarima::ForecastSpec::new
[`Forecasts`]: crate::Forecasts
[`LaborDay`]: crate::regarima::LaborDay
[`LengthEffect`]: crate::regarima::LengthEffect
[`ManualOutlier`]: crate::regarima::ManualOutlier
[`ModelResult`]: crate::ModelResult
[`ModelResult::model`]: crate::ModelResult::model
[`ModelResult::regression_coefficients`]: crate::ModelResult::regression_coefficients
[`ModelSelectionResult::ConstantSeries`]: crate::regarima::ModelSelectionResult::ConstantSeries
[`ModelSelectionResult::Selected`]: crate::regarima::ModelSelectionResult::Selected
[`OutlierSpec::new`]: crate::regarima::OutlierSpec::new
[`OutlierSpec::with_critical_values`]: crate::regarima::OutlierSpec::with_critical_values
[`OutlierSpec::with_method`]: crate::regarima::OutlierSpec::with_method
[`OutlierSpec::with_span`]: crate::regarima::OutlierSpec::with_span
[`OutlierSpec::with_types`]: crate::regarima::OutlierSpec::with_types
[`OutlierTypes`]: crate::regarima::OutlierTypes
[`PeriodSpan`]: crate::PeriodSpan
[`RegArimaSpec`]: crate::RegArimaSpec
[`RegArimaSpec::automatic_model`]: crate::RegArimaSpec::automatic_model
[`RegArimaSpec::new`]: crate::RegArimaSpec::new
[`RegArimaSpec::with_constant`]: crate::RegArimaSpec::with_constant
[`RegArimaSpec::with_manual_outliers`]: crate::RegArimaSpec::with_manual_outliers
[`RegArimaSpec::with_model_span`]: crate::RegArimaSpec::with_model_span
[`RegArimaSpec::with_outliers`]: crate::RegArimaSpec::with_outliers
[`RegArimaSpec::with_temporary_change_rate`]: crate::RegArimaSpec::with_temporary_change_rate
[`RegArimaSpec::with_transform`]: crate::RegArimaSpec::with_transform
[`RegArimaSpec::with_user_regressors`]: crate::RegArimaSpec::with_user_regressors
[`RegressionEstimate`]: crate::regarima::RegressionEstimate
[`RegressionVariable::MissingValue`]: crate::regarima::RegressionVariable::MissingValue
[`RegressionVariable::User`]: crate::regarima::RegressionVariable::User
[`Spec::seats`]: crate::Spec::seats
[`Spec::x11_regarima`]: crate::Spec::x11_regarima
[`StockTradingDay`]: crate::regarima::StockTradingDay
[`Thanksgiving`]: crate::regarima::Thanksgiving
[`TradingDayContrast`]: crate::regarima::TradingDayContrast
[`Transform::Log`]: crate::Transform::Log
[`TransformSelection::automatic`]: crate::regarima::TransformSelection::automatic
[`UserRegressor::new`]: crate::regarima::UserRegressor::new
[`UserRegressorType`]: crate::regarima::UserRegressorType
[`Warning::AutomaticTransformNonPositiveInput`]: crate::Warning::AutomaticTransformNonPositiveInput
[`Warning::BackcastsNotUsed`]: crate::Warning::BackcastsNotUsed
[`Warning::ConstantSeries`]: crate::Warning::ConstantSeries
[`Warning::ForecastsNotUsed`]: crate::Warning::ForecastsNotUsed
[`Warning::ModelSpanBackcastsIgnored`]: crate::Warning::ModelSpanBackcastsIgnored
[`x11::Options::with_mode`]: crate::x11::Options::with_mode
