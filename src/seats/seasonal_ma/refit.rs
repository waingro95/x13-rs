//! The single-MA refit: the one-coordinate SEARCH, started from STAVAL's
//! value for the autocorrelation at the coefficient's lag.

use super::SingleMaShape;
use super::residual::SeasonalMaObjective;
use crate::Frequency;
use crate::seats::autocorrelation::AutoResult;
use crate::seats::refit::one_coordinate::{
    LOWER, OneCoordinateObjective, SeasonalMaRefit, search_one,
};
use crate::seats::refit::start::{contract_start, degree_one_start, seasonal_start};
use crate::seats::residual::ResidualResult;
use crate::seats::{Error, Failure, Stage};

/// Refits the MA coefficient on the differenced series `wd`, whose
/// autocorrelations `auto` give the starting value.
pub(crate) fn refit_seasonal_ma(
    frequency: Frequency,
    shape: SingleMaShape,
    wd: &[f64],
    auto: &AutoResult,
) -> Result<SeasonalMaRefit, Error> {
    let period = shape.lag(frequency);
    if auto.active_lags < period || auto.active_lags > auto.acf.len() {
        return Err(Failure::Unusable {
            quantity: "starting-value autocorrelation lags",
            index: None,
            value: auto.active_lags as f64,
        }
        .at(Stage::Mean));
    }
    // With one MA group, STAVAL (ANSUB1 2533–2568) starts it alone.
    let r = auto.acf[period - 1];
    let start = match shape {
        SingleMaShape::Seasonal => seasonal_start(r),
        SingleMaShape::Nonseasonal => degree_one_start(r),
    };
    search_one(
        SeasonalMaObjective::new(frequency, shape, wd),
        contract_start(start),
    )
}

impl OneCoordinateObjective for SeasonalMaObjective<'_> {
    type Error = Error;
    fn evaluate_one(&mut self, x: f64) -> Result<ResidualResult, Self::Error> {
        self.evaluate([x])
    }
    fn evaluations(&self) -> usize {
        self.evaluations()
    }
    fn fixed_lower(&self) -> f64 {
        LOWER
    }
}
