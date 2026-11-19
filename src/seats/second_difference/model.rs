//! The second-difference model `(0 2 1)(0 1 1)` and its import from regARIMA.
//!
//! [`import_second_difference`] follows NMLMDL as the airline import does:
//! each MA coefficient goes through TRANS0's degree-one clip, and ANALTS then
//! reverses both signs.

use crate::Frequency;
use crate::seats::failure::{Error, Stage, usable};
use crate::seats::import::{MA_BOUND, trans0_degree_one};

/// The regARIMA estimate of a second-difference model: the nonseasonal and
/// seasonal MA coefficients, finite.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SecondDifferenceInput {
    frequency: Frequency,
    parameters: [f64; 2],
}

impl SecondDifferenceInput {
    pub(crate) fn new(frequency: Frequency, parameters: [f64; 2]) -> Result<Self, Error> {
        for (index, value) in parameters.into_iter().enumerate() {
            usable("fitted model coefficient", Some(index), value)
                .map_err(|failure| failure.at(Stage::Import))?;
        }
        Ok(Self {
            frequency,
            parameters,
        })
    }
}

/// A second-difference model in the SEATS convention: the nonseasonal and
/// seasonal MA coefficients of the imported model, or of a later refit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSecondDifference {
    pub(crate) frequency: Frequency,
    pub(crate) parameters: [f64; 2],
}

impl ResolvedSecondDifference {
    pub(crate) const fn frequency(&self) -> Frequency {
        self.frequency
    }
    pub(crate) const fn ma_coefficients(&self) -> [f64; 2] {
        self.parameters
    }
}

pub(crate) fn import_second_difference(input: &SecondDifferenceInput) -> ResolvedSecondDifference {
    // nmlmdl.f:119–175 subtracts each coefficient from zero and sends each MA
    // group through TRANS0/TRANS2; analts.f:1128–1132 reverses the signs.
    ResolvedSecondDifference {
        frequency: input.frequency,
        parameters: input
            .parameters
            .map(|v| -trans0_degree_one(0.0 - v, MA_BOUND)),
    }
}

/// The AR and MA polynomials SPECTRUM factors: `(1 − θB)` and the seasonal MA
/// coefficient, on second differences.
pub(super) fn spectral_inputs(
    model: &ResolvedSecondDifference,
) -> crate::seats::decomposition::polynomial::SpectralInputs {
    crate::seats::decomposition::polynomial::spectral_inputs_ordered(
        usize::from(model.frequency.periods_per_year()),
        vec![1.0, -model.parameters[0]],
        model.parameters[1],
        crate::seats::tails::DifferenceOrder::Second,
    )
}
