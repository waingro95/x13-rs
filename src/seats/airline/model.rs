//! The airline model `(0 1 1)(0 1 1)` and its import from regARIMA.
//!
//! [`import_airline`] follows NMLMDL: each MA coefficient goes through TRANS0's
//! degree-one clip, and ANALTS then reverses both signs.

use crate::Frequency;
use crate::seats::failure::{Error, Stage, usable};
use crate::seats::import::{MA_BOUND, trans0_degree_one};

/// The regARIMA estimate of an airline model: both MA coefficients, finite.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AirlineInput {
    frequency: Frequency,
    ma: [f64; 2],
}

impl AirlineInput {
    /// The nonseasonal and seasonal MA coefficients, in regARIMA's order.
    pub(crate) fn new(frequency: Frequency, ma: [f64; 2]) -> Result<Self, Error> {
        for (index, value) in ma.into_iter().enumerate() {
            usable("fitted model coefficient", Some(index), value)
                .map_err(|failure| failure.at(Stage::Import))?;
        }
        Ok(Self { frequency, ma })
    }
}

/// An airline model in the SEATS convention: the nonseasonal and seasonal MA
/// coefficients of the imported model, or of a later refit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedAirline {
    pub(crate) frequency: Frequency,
    pub(crate) ma: [f64; 2],
}

impl ResolvedAirline {
    pub(crate) fn frequency(&self) -> Frequency {
        self.frequency
    }
    pub(crate) fn ma_coefficients(&self) -> [f64; 2] {
        self.ma
    }
}

pub(crate) fn import_airline(input: &AirlineInput) -> ResolvedAirline {
    // nmlmdl.f:119–139 subtracts from zero; a unary minus would turn a
    // positive zero into a negative one.
    let mut th = 0.0 - input.ma[0];
    let mut bth = 0.0 - input.ma[1];

    // nmlmdl.f:163–175 calls TRANS0/TRANS2 separately for each MA group.
    th = trans0_degree_one(th, MA_BOUND);
    bth = trans0_degree_one(bth, MA_BOUND);

    // analts.f:1128–1132 reverses both signs into the SEATS convention.
    ResolvedAirline {
        frequency: input.frequency,
        ma: [-th, -bth],
    }
}

#[cfg(test)]
mod tests;
