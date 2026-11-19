//! What a SEATS adjustment reports, which models SEATS runs, and the checks
//! on a SEATS request.

/// A SEATS request this crate does not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SeatsUnsupportedOption {
    /// An explicit model outside the models SEATS runs (see
    /// [`Spec::seats`](crate::Spec::seats)).
    UnsupportedModel,
    /// A calendar request SEATS does not run.
    Calendar(SeatsUnsupportedCalendar),
    /// More than three forecast years (requested SEATS backcasts are ignored).
    RequestedExtensions,
}

impl std::fmt::Display for SeatsUnsupportedOption {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedModel => {
                formatter.write_str("ARIMA orders outside the SEATS models this crate supports")
            }
            Self::Calendar(calendar) => calendar.fmt(formatter),
            Self::RequestedExtensions => formatter.write_str("more than three years of forecasts"),
        }
    }
}

/// A calendar request SEATS does not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SeatsUnsupportedCalendar {
    /// A calendar specification that requests no regressor.
    Empty,
}

display_phrases!(SeatsUnsupportedCalendar {
    Empty => "a calendar specification that requests no regressor",
});

/// Why SEATS does not run this calendar request, or `None` when it does.
pub(crate) fn unsupported_calendar(
    calendar: super::CalendarSpec,
) -> Option<SeatsUnsupportedCalendar> {
    // chkadj/prtref place stock TD in Ftd and all three holidays in Fhol,
    // the groups the fitted effects keep, so every nonempty calendar runs.
    calendar
        .is_empty()
        .then_some(SeatsUnsupportedCalendar::Empty)
}

/// A regARIMA forecast output that SEATS found nonfinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SeatsForecastField {
    /// Forecast on the original scale.
    Original,
    /// Forecast on the transformed scale.
    Transformed,
    /// Standard error of a transformed-scale forecast.
    TransformedStandardError,
    /// Lower 95% bound of a forecast.
    Lower95,
    /// Upper 95% bound of a forecast.
    Upper95,
    /// Regression part of a forecast's variance.
    RegressionVariance,
    /// Stochastic (ARIMA) part of a forecast's variance.
    StochasticVariance,
}

display_phrases!(SeatsForecastField {
    Original => "forecast",
    Transformed => "transformed forecast",
    TransformedStandardError => "transformed forecast standard error",
    Lower95 => "forecast lower 95% bound",
    Upper95 => "forecast upper 95% bound",
    RegressionVariance => "forecast regression variance",
    StochasticVariance => "forecast stochastic variance",
});

/// Units of the SEATS transitory component (table S14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitoryScale {
    /// Additive: the transitory component is in the units of the series.
    Additive,
    /// Multiplicative: the transitory component is a factor around one.
    Multiplicative,
}

/// The SEATS transitory component (table S14) over the observed span, as
/// Build 62 saves it; not the raw extracted cycle.
#[derive(Debug, Clone, PartialEq)]
pub struct Transitory {
    pub(crate) start: crate::Period,
    pub(crate) scale: TransitoryScale,
    pub(crate) values: Vec<f64>,
}

impl Transitory {
    /// Units of [`series`](Self::series).
    #[must_use]
    pub const fn scale(&self) -> TransitoryScale {
        self.scale
    }
    /// Transitory component over the observed span, dated from the input's
    /// first period.
    #[must_use]
    pub fn series(&self) -> crate::Series<'_> {
        crate::Series::new(self.start, &self.values)
    }
}

/// Final SEATS model, separate from the original fitted regARIMA model:
/// import clipping and a SEATS refit can change its coefficients.
///
/// Coefficients use the polynomial sign convention. The regular AR factor is
/// `(1 - phi1 B - ... - phip B^p)` and the regular MA factor is
/// `(1 - theta1 B - ... - thetaq B^q)`; the seasonal factors are the same in
/// `B^s`. Each accessor returns one coefficient per order, so a part the model
/// lacks is empty.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    frequency: crate::Frequency,
    orders: super::ResolvedArimaModel,
    /// Regular AR, seasonal AR, regular MA, then seasonal MA.
    coefficients: Vec<f64>,
}

impl Model {
    /// `coefficients` are the regular AR, seasonal AR, regular MA, then
    /// seasonal MA coefficients, one per order.
    pub(crate) fn new(
        frequency: crate::Frequency,
        regular: [u8; 3],
        seasonal: [u8; 3],
        coefficients: &[f64],
    ) -> Result<Self, super::Error> {
        let orders = super::ResolvedArimaModel::new(regular, seasonal)?;
        let model = Self {
            frequency,
            orders,
            coefficients: coefficients.to_vec(),
        };
        if model.part_lengths().iter().sum::<usize>() != coefficients.len() {
            return Err(super::Error::internal(
                "the SEATS model's coefficients do not match its orders",
            ));
        }
        Ok(model)
    }
    /// Frequency of the adjusted series.
    #[must_use]
    pub const fn frequency(&self) -> crate::Frequency {
        self.frequency
    }
    /// Regular `(p d q)` and seasonal `(P D Q)` orders.
    #[must_use]
    pub const fn orders(&self) -> super::ResolvedArimaModel {
        self.orders
    }
    /// Every coefficient: regular AR, seasonal AR, regular MA, then seasonal MA.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    /// Regular AR coefficients phi1, ..., phip.
    #[must_use]
    pub fn regular_ar(&self) -> &[f64] {
        self.part(0)
    }
    /// Seasonal AR coefficients Phi1, ..., PhiP.
    #[must_use]
    pub fn seasonal_ar(&self) -> &[f64] {
        self.part(1)
    }
    /// Regular MA coefficients theta1, ..., thetaq.
    #[must_use]
    pub fn regular_ma(&self) -> &[f64] {
        self.part(2)
    }
    /// Seasonal MA coefficients Theta1, ..., ThetaQ.
    #[must_use]
    pub fn seasonal_ma(&self) -> &[f64] {
        self.part(3)
    }
    /// Coefficient counts in storage order: `[p, P, q, Q]`.
    fn part_lengths(&self) -> [usize; 4] {
        let (regular, seasonal) = (self.orders.regular(), self.orders.seasonal());
        [regular.p(), seasonal.p(), regular.q(), seasonal.q()].map(usize::from)
    }
    /// The `index`th part in storage order.
    fn part(&self, index: usize) -> &[f64] {
        let lengths = self.part_lengths();
        let start = lengths[..index].iter().sum();
        &self.coefficients[start..start + lengths[index]]
    }
}

/// The coordinate count of a shared ARMA strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArmaCoordinates {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
}

impl ArmaCoordinates {
    /// `None` above six coordinates.
    pub(crate) const fn new(count: u8) -> Option<Self> {
        match count {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            4 => Some(Self::Four),
            5 => Some(Self::Five),
            6 => Some(Self::Six),
            _ => None,
        }
    }
}

/// The engine strategy a family runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeatsStrategy {
    Airline,
    SecondDifference,
    SingleMa,
    Ar1,
    Ar2,
    /// The shared ARMA strategy for this many coordinates.
    Arma(ArmaCoordinates),
}

/// The SEATS model families, by the orders of the regARIMA model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeatsFamily {
    IntegratedArma(ArmaCoordinates),
    NonseasonalArma(ArmaCoordinates),
    Arma12,
    Arma22,
    Ma2Seasonal,
    NonseasonalArma11,
    NonseasonalAr,
    NonseasonalMa,
    Airline,
    SecondDifference,
    SeasonalMa,
    Ar1SeasonalMa,
    Arma11,
    Arma21,
    Arma31,
    SeasonalAr,
    Ar2SeasonalMa,
    Ar3SeasonalMa,
}

impl SeatsFamily {
    /// The engine strategy that runs this family.
    pub(crate) const fn strategy(self) -> SeatsStrategy {
        match self {
            Self::Airline => SeatsStrategy::Airline,
            Self::SecondDifference => SeatsStrategy::SecondDifference,
            Self::SeasonalMa | Self::NonseasonalMa => SeatsStrategy::SingleMa,
            Self::Ar1SeasonalMa | Self::NonseasonalAr => SeatsStrategy::Ar1,
            Self::Ar2SeasonalMa | Self::Ar3SeasonalMa => SeatsStrategy::Ar2,
            Self::IntegratedArma(count) | Self::NonseasonalArma(count) => {
                SeatsStrategy::Arma(count)
            }
            Self::SeasonalAr | Self::NonseasonalArma11 => SeatsStrategy::Arma(ArmaCoordinates::Two),
            Self::Arma11 | Self::Ma2Seasonal => SeatsStrategy::Arma(ArmaCoordinates::Three),
            Self::Arma21 | Self::Arma12 => SeatsStrategy::Arma(ArmaCoordinates::Four),
            Self::Arma31 | Self::Arma22 => SeatsStrategy::Arma(ArmaCoordinates::Five),
        }
    }
}

/// A validated SEATS request: the family of an explicit model, or automatic
/// model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SeatsRequest {
    Explicit(SeatsFamily),
    Automatic,
}

/// What SEATS reports about an adjustment, from
/// [`x13::Diagnostics::seats`](crate::Diagnostics::seats): the final SEATS
/// model, its mean, and whether SEATS re-estimated the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    pub(crate) mean_requested: bool,
    pub(crate) mean_included: bool,
    pub(crate) model: Model,
    pub(crate) refitted: bool,
    pub(crate) mean_temporarily_removed: bool,
    pub(crate) coefficients_at_bound: Vec<bool>,
}

impl Diagnostics {
    /// Whether SEATS was asked to estimate a mean, which it is exactly when
    /// the regARIMA model has a constant term.
    #[must_use]
    pub const fn mean_requested(&self) -> bool {
        self.mean_requested
    }
    /// Whether the final SEATS model has a mean.
    #[must_use]
    pub const fn mean_included(&self) -> bool {
        self.mean_included
    }
    /// The final SEATS model, in the SEATS polynomial sign convention.
    #[must_use]
    pub const fn model(&self) -> &Model {
        &self.model
    }
    /// Whether SEATS re-estimated the model with its own optimizer.
    #[must_use]
    pub const fn refitted(&self) -> bool {
        self.refitted
    }
    /// Whether a SEATS pass removed the mean and repeated.
    #[must_use]
    pub const fn mean_temporarily_removed(&self) -> bool {
        self.mean_temporarily_removed
    }
    /// Which coefficients the SEATS optimizer held at a bound when it
    /// finished, one flag per coefficient in the order of
    /// [`Model::coefficients`]. These are not the regARIMA model's
    /// fixed coefficients.
    #[must_use]
    pub fn coefficients_at_bound(&self) -> &[bool] {
        &self.coefficients_at_bound
    }
}

/// Checks a SEATS request against the series and returns the family of its
/// explicit model, or [`SeatsRequest::Automatic`].
pub(super) fn validate(
    values: &[f64],
    axis: crate::time::TimeAxis,
    regarima: super::RegArimaSpec,
) -> Result<SeatsRequest, super::Error> {
    use super::{Error, ModelSelection, UnsupportedDetail};
    let unsupported = |option| Error::from(UnsupportedDetail::SeatsOption(option));
    let request = match regarima.model() {
        ModelSelection::Automatic => SeatsRequest::Automatic,
        ModelSelection::Explicit(model) => SeatsRequest::Explicit(
            selected_family(model.into())
                .ok_or_else(|| unsupported(SeatsUnsupportedOption::UnsupportedModel))?,
        ),
    };
    validate_request(values, axis, regarima)?;
    Ok(request)
}

/// The SEATS family that runs a model of these orders, or `None` when SEATS
/// does not run it.
pub(crate) fn selected_family(model: super::ResolvedArimaModel) -> Option<SeatsFamily> {
    let order = |o: super::ResolvedArimaOrder| (o.p(), o.d(), o.q());
    Some(match (order(model.regular()), order(model.seasonal())) {
        ((0, 1, 2), (0, 1, 1)) => SeatsFamily::Ma2Seasonal,
        ((p @ 0..=2, 1, 2), (0, 0, 0)) => {
            SeatsFamily::NonseasonalArma(ArmaCoordinates::new(p + 2)?)
        }
        ((p @ 2..=3, 1, 1), (0, 0, 0)) => {
            SeatsFamily::NonseasonalArma(ArmaCoordinates::new(p + 1)?)
        }
        ((1, 1, 2), (0, 1, 1)) => SeatsFamily::Arma12,
        ((2, 1, 2), (0, 1, 1)) => SeatsFamily::Arma22,
        ((0, 1, 1), (0, 1, 1)) => SeatsFamily::Airline,
        ((0, 2, 1), (0, 1, 1)) => SeatsFamily::SecondDifference,
        ((0, 1, 0), (0, 1, 1)) => SeatsFamily::SeasonalMa,
        ((0, 1, 1), (0, 0, 0)) => SeatsFamily::NonseasonalMa,
        ((1, 1, 0), (0, 1, 1)) => SeatsFamily::Ar1SeasonalMa,
        ((1, 1, 0), (0, 0, 0)) => SeatsFamily::NonseasonalAr,
        ((1, 1, 1), (0, 0, 0)) => SeatsFamily::NonseasonalArma11,
        ((1, 1, 1), (0, 0 | 1, 1)) => SeatsFamily::Arma11,
        ((2, 1, 1), (0, 0 | 1, 1)) => SeatsFamily::Arma21,
        ((3, 1, 1), (0, 0 | 1, 1)) => SeatsFamily::Arma31,
        ((0, 1, 1), (1, 1, 0)) => SeatsFamily::SeasonalAr,
        ((2, 1, 0), (0, 1, 1)) => SeatsFamily::Ar2SeasonalMa,
        ((3, 1, 0), (0, 1, 1)) => SeatsFamily::Ar3SeasonalMa,
        ((0, 1..=2, 0), (0, 0..=1, 0)) => SeatsFamily::IntegratedArma(ArmaCoordinates::Zero),
        ((p @ 0..=3, 0..=2, q @ 0..=2), (1, 0..=1, bq @ 0..=1)) if p + 1 + q + bq <= 6 => {
            SeatsFamily::IntegratedArma(ArmaCoordinates::new(p + 1 + q + bq)?)
        }
        ((p @ 0..=3, 0, q @ 0..=2), (0, 0, bq @ 0..=1)) => {
            SeatsFamily::IntegratedArma(ArmaCoordinates::new(p + q + bq)?)
        }
        ((p @ 0..=3, 0, q @ 0..=2), (0, 1, bq @ 0..=1)) if (1..=5).contains(&(p + q + bq)) => {
            SeatsFamily::IntegratedArma(ArmaCoordinates::new(p + q + bq)?)
        }
        ((p @ 0..=3, 1..=2, q @ 0..=2), (0, 0..=1, bq @ 0..=1))
            if (1..=5).contains(&(p + q + bq)) =>
        {
            SeatsFamily::IntegratedArma(ArmaCoordinates::new(p + q + bq)?)
        }
        _ => return None,
    })
}

/// Checks the options, dates and length of a SEATS request, and that its
/// regressors and holidays cover the forecasts SEATS needs, whatever the
/// model.
pub(crate) fn validate_request(
    values: &[f64],
    axis: crate::time::TimeAxis,
    regarima: super::RegArimaSpec,
) -> Result<(), super::Error> {
    use super::{Error, UnsupportedDetail};
    let unsupported = |option| Error::from(UnsupportedDetail::SeatsOption(option));
    if let Some(calendar) = regarima.calendar()
        && let Some(reason) = unsupported_calendar(calendar)
    {
        return Err(unsupported(SeatsUnsupportedOption::Calendar(reason)));
    }
    if regarima.forecast().backcast_periods() != 0
        || regarima.forecast().forecast_periods(axis.frequency())
            > 3 * u16::from(axis.frequency().periods_per_year())
    {
        return Err(unsupported(SeatsUnsupportedOption::RequestedExtensions));
    }
    if axis.start().year() < 0 {
        return Err(Error::SeatsNegativeYear {
            year: axis.start().year(),
        });
    }
    let lengths =
        crate::engine::seats::seats_length_range(axis.frequency(), axis.start().subperiod());
    if values.len() < *lengths.start() {
        return Err(Error::TooShort {
            method: super::Method::Seats,
            actual: values.len(),
            minimum: *lengths.start(),
        });
    }
    if values.len() > *lengths.end() {
        return Err(Error::SeatsTooLong {
            actual: values.len(),
            maximum: *lengths.end(),
        });
    }
    let s = usize::from(axis.frequency().periods_per_year());
    axis.periods(values.len() + 3 * s)?;
    regarima.validate(values, axis)?;
    // editor/regvar need the extended SEATS horizon even when the request
    // asks for only one year of forecasts. A supplied regression column is
    // not extrapolated to fill that span.
    regarima.validate_user_regressors(
        axis,
        values.len(),
        crate::regarima::ForecastSpec::new(
            crate::regarima::ForecastHorizon::Periods((3 * s).max(12) as u16),
            0,
        )?,
    )?;
    // The automatic paths evaluate moving holidays over that horizon too. Past
    // 2100, adestr.f (also adlabr.f, adthnk.f) reads `kdate(year)` beyond its
    // 1901..2100 table, so reject up front rather than mid-fit.
    let internal = crate::regarima::ForecastSpec::new(
        crate::regarima::ForecastHorizon::Periods(3 * s as u16),
        0,
    )?;
    // An explicit model with an explicit transform generates holidays only over
    // the requested horizon, so it is not checked here.
    if let Some(calendar) = regarima.calendar() {
        if regarima.model() == super::ModelSelection::Automatic {
            calendar
                .without_basis()
                .validate(axis, values.len(), internal)?;
        } else if let (super::TransformSelection::Automatic(_), Some(easter)) =
            (regarima.transform(), calendar.easter())
            && !values.iter().any(|&value| value <= 0.0)
        {
            let span = regarima.model_range(axis, values.len())?;
            super::CalendarSpec::new().with_easter(easter).validate(
                axis.shift(span.start as i64)?,
                span.len(),
                internal,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
