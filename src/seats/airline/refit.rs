//! The airline refit: SEARCH over both MA coefficients, started from the
//! autocorrelations of the differenced series.

use crate::Frequency;
use crate::seats::autocorrelation::AutoResult;
use crate::seats::refit::search::{BoundRule, CovarianceEstimate, SearchObjective};
use crate::seats::refit::start::{contract_start, degree_one_start, seasonal_start};
use crate::seats::residual::{AirlineObjective, ResidualResult};
use crate::seats::{Error, Failure, Stage};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RefitCovariance {
    Available {
        standard_errors: [f64; 2],
        correlation: [[f64; 2]; 2],
    },
    UnavailableAllFixed,
}

impl RefitCovariance {
    /// A two-coordinate SEARCH estimates the covariance unless every
    /// coordinate ended fixed.
    pub(crate) fn from_search(estimate: Option<CovarianceEstimate<2>>) -> Self {
        match estimate {
            Some(CovarianceEstimate {
                standard_errors,
                correlation,
            }) => Self::Available {
                standard_errors,
                correlation,
            },
            None => Self::UnavailableAllFixed,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AirlineRefitResult {
    pub(crate) ma: [f64; 2],
    pub(crate) scaled_residuals: Vec<f64>,
    pub(crate) determinant_scale: f64,
    pub(crate) objective: f64,
    pub(crate) bound_mask: [bool; 2],
    pub(crate) covariance: RefitCovariance,
}

/// Refits both MA coefficients on the differenced series `wd`, whose
/// autocorrelations `auto` give the starting values.
pub(crate) fn refit_prepared_airline(
    frequency: Frequency,
    wd: &[f64],
    auto: &AutoResult,
) -> Result<AirlineRefitResult, Error> {
    let period = usize::from(frequency.periods_per_year());
    if auto.active_lags < period || auto.active_lags > auto.acf.len() {
        return Err(Failure::Unusable {
            quantity: "starting-value autocorrelation lags",
            index: None,
            value: auto.active_lags as f64,
        }
        .at(Stage::Mean));
    }
    let x = starting_parameters(&auto.acf, frequency).map(contract_start);
    search(frequency, wd, x)
}

pub(crate) fn starting_parameters(acf: &[f64; 36], frequency: Frequency) -> [f64; 2] {
    let seasonal_r = acf[usize::from(frequency.periods_per_year()) - 1];
    [degree_one_start(acf[0]), seasonal_start(seasonal_r)]
}

impl SearchObjective<2> for AirlineObjective<'_> {
    const ESTIMATES_COVARIANCE: bool = true;
    fn calcfx(&mut self, point: [f64; 2]) -> Result<ResidualResult, Error> {
        self.evaluate(point)
    }
    fn evaluations(&self) -> usize {
        self.evaluations()
    }
}

/// SEARCH for (0,1,1)(0,1,1).
pub(crate) fn search(
    frequency: Frequency,
    wd: &[f64],
    start: [f64; 2],
) -> Result<AirlineRefitResult, Error> {
    let (result, _) = crate::seats::refit::search::search(
        AirlineObjective::new(frequency, wd),
        start,
        [false; 2],
        BoundRule::from_orders(crate::seats::Orders::new([0, 1, 1], [0, 1, 1])),
    )?;
    Ok(AirlineRefitResult {
        ma: result.parameters,
        scaled_residuals: result.scaled_residuals,
        determinant_scale: result.determinant_scale,
        objective: result.objective,
        bound_mask: result.bound_mask,
        covariance: RefitCovariance::from_search(result.covariance),
    })
}
