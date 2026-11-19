//! The AR(1) refit: STAVAL's starting values, then SEARCH over the AR and
//! seasonal-MA coordinates, or over the AR coordinate alone when the model
//! has no seasonal MA.

use super::residual::Ar1SeasonalMaObjective;
use crate::Frequency;
use crate::seats::airline::refit::RefitCovariance;
use crate::seats::autocorrelation::AutoResult;
use crate::seats::failure::{Error, Failure, Stage, usable};
use crate::seats::refit::search::{BoundRule, SearchObjective};
use crate::seats::refit::start::contract_start;
use crate::seats::residual::ResidualResult;

/// SEARCH's covariance estimate, in the shape of the search that ran.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Ar1RefitCovariance {
    Seasonal(RefitCovariance),
    Nonseasonal(crate::seats::refit::one_coordinate::SeasonalMaCovariance),
}

/// SEARCH's result: the AR and seasonal MA coefficients, zero for an
/// absent seasonal MA, and CALCFX's residuals at them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ar1SeasonalMaRefit {
    pub(crate) parameters: [f64; 2],
    pub(crate) residuals: ResidualResult,
    pub(crate) bound_mask: [bool; 2],
    pub(crate) covariance: Ar1RefitCovariance,
}

impl SearchObjective<2> for Ar1SeasonalMaObjective<'_> {
    const ESTIMATES_COVARIANCE: bool = true;
    fn calcfx(&mut self, point: [f64; 2]) -> Result<ResidualResult, Error> {
        self.evaluate(point)
            .map_err(|error| error.restage(Stage::Refit))
    }
    fn evaluations(&self) -> usize {
        self.evaluations()
    }
}

/// SEARCH for (1,1,0)(0,1,1).
pub(crate) fn search(
    frequency: Frequency,
    wd: &[f64],
    start: [f64; 2],
) -> Result<Ar1SeasonalMaRefit, Error> {
    let (result, _) = crate::seats::refit::search::search(
        Ar1SeasonalMaObjective::new(frequency, wd, true),
        start,
        [false; 2],
        BoundRule::from_orders(crate::seats::Orders::new([1, 1, 0], [0, 1, 1])),
    )?;
    let mut normalized = Vec::with_capacity(result.scaled_residuals.len());
    for (i, &value) in result.scaled_residuals.iter().enumerate() {
        let normalized_value = usable(
            "normalized optimizer residual",
            Some(i + 1),
            value / result.determinant_scale,
        )
        .map_err(|failure| failure.at(Stage::Refit))?;
        normalized.push(normalized_value);
    }
    Ok(Ar1SeasonalMaRefit {
        parameters: result.parameters,
        residuals: ResidualResult {
            scaled_residuals: result.scaled_residuals,
            determinant_scale: result.determinant_scale,
            objective: result.objective,
            normalized_residuals: normalized,
        },
        bound_mask: result.bound_mask,
        covariance: Ar1RefitCovariance::Seasonal(RefitCovariance::from_search(result.covariance)),
    })
}

/// Refits the model on the differenced series `wd`, whose autocorrelations
/// `auto` give the starting values.
pub(crate) fn refit_prepared_ar1_seasonal_ma(
    frequency: Frequency,
    wd: &[f64],
    auto: &AutoResult,
    seasonal: bool,
) -> Result<Ar1SeasonalMaRefit, Error> {
    let required = match (frequency, seasonal) {
        (Frequency::Monthly, true) => 13,
        (Frequency::Monthly, false) => 3,
        (Frequency::Quarterly, _) => 10,
    };
    if auto.active_lags < required || auto.active_lags > auto.acf.len() {
        return Err(Failure::Unusable {
            quantity: "active autocorrelation lags",
            index: None,
            value: auto.active_lags as f64,
        }
        .at(Stage::Refit));
    }
    if !seasonal {
        // Without a seasonal MA, STAVAL still starts the AR coefficient from
        // its three ratios; only the seasonal conditioning and coordinate go.
        let start = crate::seats::refit::start::regular_start(&auto.acf, frequency)?;
        let result = crate::seats::refit::one_coordinate::search_one(
            Ar1SeasonalMaObjective::new(frequency, wd, false),
            contract_start(start),
        )?;
        let covariance = Ar1RefitCovariance::Nonseasonal(result.covariance);
        return Ok(Ar1SeasonalMaRefit {
            parameters: [result.ma[0], 0.0],
            residuals: result.residuals,
            bound_mask: [result.bound_mask[0], false],
            covariance,
        });
    }
    let start = starting_parameters(&auto.acf, frequency)?;
    search(frequency, wd, start.map(contract_start))
}

impl crate::seats::refit::one_coordinate::OneCoordinateObjective for Ar1SeasonalMaObjective<'_> {
    type Error = Error;
    fn evaluate_one(&mut self, x: f64) -> Result<ResidualResult, Error> {
        self.evaluate([x, 0.0])
            .map_err(|error| error.restage(Stage::Refit))
    }
    fn evaluations(&self) -> usize {
        self.evaluations()
    }
    fn fixed_lower(&self) -> f64 {
        -1.0
    }
}

/// STAVAL's starting AR and seasonal MA coefficients.
fn starting_parameters(acf: &[f64; 36], frequency: Frequency) -> Result<[f64; 2], Error> {
    let start = crate::seats::refit::start::starting_parameters(acf, frequency)?;
    Ok([start.regular, start.seasonal_ma])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonseasonal_refit_uses_staval_three_ratio_start() {
        let wd = (0..64)
            .map(|i| (i as f64 * 0.31).sin() + (i as f64 * 0.17).cos())
            .collect::<Vec<_>>();
        for frequency in [Frequency::Monthly, Frequency::Quarterly] {
            let mut acf = [0.0; 36];
            acf[..3].copy_from_slice(&[0.6, 0.2, -0.1]);
            acf[4..6].copy_from_slice(&[0.6, 0.2]);
            acf[8..10].copy_from_slice(&[0.2, -0.1]);
            let auto = AutoResult {
                active_lags: 36,
                acf,
                qstat: 0.0,
            };
            // STAVAL (ansub1.f:2490–2509) starts at single-precision .2 when
            // the ratios change sign, with or without a seasonal MA.
            let expected = crate::seats::refit::one_coordinate::search_one(
                Ar1SeasonalMaObjective::new(frequency, &wd, false),
                contract_start(f64::from(0.2_f32)),
            )
            .unwrap();
            let actual = refit_prepared_ar1_seasonal_ma(frequency, &wd, &auto, false).unwrap();
            assert_eq!(actual.parameters, [expected.ma[0], 0.0]);
            assert_eq!(actual.residuals, expected.residuals);
        }
    }
}
