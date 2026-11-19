//! Forecast and backcast horizons.

use crate::api::ExtensionDirection;
use crate::error::SpecProblem;
use crate::time::TimeAxis;
use crate::{Error, Frequency};

const MAX_EXTENSION_HORIZON: u16 = 120;

/// How far the regARIMA model forecasts past the last observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForecastHorizon {
    /// One year: 12 months or 4 quarters (the default).
    OneYear,
    /// An explicit number of periods, at most 120; zero turns forecasts off.
    Periods(u16),
}

impl ForecastHorizon {
    pub(super) const fn resolve(self, frequency: Frequency) -> u16 {
        match self {
            Self::OneYear => frequency.periods_per_year() as u16,
            Self::Periods(periods) => periods,
        }
    }
}

/// Forecast and backcast horizons for the regARIMA model, set with
/// [`RegArimaSpec::with_forecast`](crate::RegArimaSpec::with_forecast).
///
/// The default is one year of forecasts and no backcasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForecastSpec {
    forecast_horizon: ForecastHorizon,
    backcast_periods: u16,
}

impl ForecastSpec {
    /// Constructs horizons of at most 120 periods in each direction,
    /// otherwise [`Error::InvalidSpec`].
    pub fn new(forecast_horizon: ForecastHorizon, backcast_periods: u16) -> Result<Self, Error> {
        if let ForecastHorizon::Periods(periods) = forecast_horizon {
            validate_horizon(ExtensionDirection::Forecast, periods)?;
        }
        validate_horizon(ExtensionDirection::Backcast, backcast_periods)?;
        Ok(Self {
            forecast_horizon,
            backcast_periods,
        })
    }

    /// Requested forecast policy.
    #[must_use]
    pub const fn horizon(self) -> ForecastHorizon {
        self.forecast_horizon
    }

    /// Resolves the forecast policy for a concrete frequency.
    #[must_use]
    pub const fn forecast_periods(self, frequency: Frequency) -> u16 {
        self.forecast_horizon.resolve(frequency)
    }

    /// Explicit number of requested backcasts.
    #[must_use]
    pub const fn backcast_periods(self) -> u16 {
        self.backcast_periods
    }

    /// The axis and length that `count` observations from `axis` span with
    /// these backcasts before them and forecasts after. The length overflows
    /// only past `usize::MAX`, which no observed slice reaches, so callers
    /// that checked it before or after the shift fail identically.
    pub(crate) fn extended_span(
        self,
        axis: TimeAxis,
        count: usize,
    ) -> Result<(TimeAxis, usize), Error> {
        let start = axis.shift(-i64::from(self.backcast_periods))?;
        let count = usize::from(self.backcast_periods)
            .checked_add(count)
            .and_then(|count| {
                count.checked_add(usize::from(self.forecast_periods(axis.frequency())))
            })
            .ok_or(Error::internal("extended series length overflowed"))?;
        Ok((start, count))
    }

    /// The horizons used once the model span is resolved:
    /// editor.f sets `Nbcst=0` when the model span starts after the first observation
    /// (`model_start > 0`). Forecasts are unchanged.
    pub(crate) const fn for_model_start(self, model_start: usize) -> Self {
        if model_start == 0 {
            self
        } else {
            Self {
                forecast_horizon: self.forecast_horizon,
                backcast_periods: 0,
            }
        }
    }
}

impl Default for ForecastSpec {
    fn default() -> Self {
        Self {
            forecast_horizon: ForecastHorizon::OneYear,
            backcast_periods: 0,
        }
    }
}

fn validate_horizon(direction: ExtensionDirection, periods: u16) -> Result<(), Error> {
    if periods > MAX_EXTENSION_HORIZON {
        Err(Error::from(SpecProblem::ExtensionHorizon {
            direction,
            periods,
            maximum: MAX_EXTENSION_HORIZON,
        }))
    } else {
        Ok(())
    }
}
