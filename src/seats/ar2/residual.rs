//! CALCFX for the AR(2)/AR(3) family: the conditional AR prefilter and the
//! SEARCH objective.

use super::transform::{Ar2Coefficients, Ar2Coordinates, regular_coefficients};
use crate::Frequency;
use crate::seats::decomposition::root_allocation::RegularArCoefficients;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::residual::{ResidualResult, ar_prefilter, finite};
use crate::seats::seasonal_ma::residual::seasonal_constrained_residuals;

/// The conditional AR prefilter: the differenced series `wd` less its
/// regular AR part, from the first value that has every lag.
pub(crate) fn ar2_prefilter(ar: RegularArCoefficients, wd: &[f64]) -> Result<Vec<f64>, Error> {
    let ar = ar.as_slice();
    debug_assert!(wd.len() > ar.len());
    for (index, &value) in ar.iter().enumerate() {
        finite(value, "regular AR coefficient", Some(index))?;
    }
    for (index, &value) in wd.iter().enumerate() {
        finite(value, "differenced series", Some(index))?;
    }
    // ANSUB1:1116–1131: one subtraction per lag, in Pstar's order.
    ar_prefilter(ar, wd, |lag, index, value| {
        let quantity = ["Phi1 prefilter", "Phi2 prefilter", "Phi3 prefilter"][lag];
        finite(value, quantity, Some(index))
    })
}

/// The AR(2)/AR(3) SEARCH objective: CALCFX's residuals at trial
/// coordinates.
pub(crate) struct Ar2Objective<'a> {
    frequency: Frequency,
    wd: &'a [f64],
    coefficients: Ar2Coefficients,
    previous: Option<Ar2Coordinates>,
    evaluations: usize,
}

impl<'a> Ar2Objective<'a> {
    /// The objective on the differenced series `wd`. `regular` gives the AR
    /// degree only.
    pub(crate) fn new(frequency: Frequency, wd: &'a [f64], regular: RegularArCoefficients) -> Self {
        Self {
            frequency,
            wd,
            coefficients: Ar2Coefficients {
                regular: regular.map(|_| 0.0),
                seasonal_ma: 0.0,
            },
            previous: None,
            evaluations: 0,
        }
    }

    pub(crate) fn evaluate(
        &mut self,
        coordinates: Ar2Coordinates,
    ) -> Result<ResidualResult, Error> {
        let degree = coordinates.regular.as_slice().len();
        for (index, value) in coordinates.values().into_iter().enumerate() {
            if !value.is_finite() || value.abs() > 1.0 {
                let (quantity, index) = if index < degree {
                    ("regular AR coordinate", Some(index))
                } else {
                    ("seasonal MA coordinate", None)
                };
                return Err(Failure::Unusable {
                    quantity,
                    index,
                    value,
                }
                .at(Stage::Residual));
            }
        }
        // CALCFX keeps a group's coefficients from the last trial when the
        // new coordinates are within TRANSC's threshold of the old ones.
        let mut coefficients = self.coefficients;
        if self.previous.is_none_or(|old| {
            old.regular
                .as_slice()
                .iter()
                .zip(coordinates.regular.as_slice())
                .any(|(a, b)| (a - b).abs() >= 3.834e-20)
        }) {
            coefficients.regular = regular_coefficients(coordinates.regular)
                .map_err(|error| error.restage(Stage::Residual))?
                .coefficients;
        }
        if self
            .previous
            .is_none_or(|old| (old.seasonal_ma - coordinates.seasonal_ma).abs() >= 3.834e-20)
        {
            coefficients.seasonal_ma = coordinates.seasonal_ma;
        }
        let u = ar2_prefilter(coefficients.regular, self.wd)?;
        let result = seasonal_constrained_residuals(
            usize::from(self.frequency.periods_per_year()),
            coefficients.seasonal_ma,
            &u,
        )?;
        self.coefficients = coefficients;
        self.previous = Some(coordinates);
        self.evaluations += 1;
        Ok(result)
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// The coefficients of the last trial.
    pub(crate) fn coefficients(&self) -> Ar2Coefficients {
        self.coefficients
    }
}
