//! SEARCH, the least-squares minimiser SEATS refits a model with, and STAVAL,
//! which gives it starting values. Both are in `ansub1.f`.

pub(crate) mod one_coordinate;
pub(crate) mod search;
pub(crate) mod start;
#[cfg(test)]
pub(in crate::seats) mod test_support;

use super::{Error, Failure, Stage};

/// SEARCH's iteration limit: MAXIT at its SEATS default.
const MAX_ITERATIONS: usize = 20;
/// SEARCH's limit on CALCFX calls: MAXF, which ANALTS sets to ten times MAXIT.
const MAX_EVALUATIONS: usize = 200;

/// SEARCH cannot go on from `value`, the `quantity` it computed.
fn unusable(quantity: &'static str, value: f64) -> Error {
    Failure::Unusable {
        quantity,
        index: None,
        value,
    }
    .at(Stage::Refit)
}

/// `value` when it is finite, else `unusable`.
fn finite(value: f64, quantity: &'static str) -> Result<f64, Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(unusable(quantity, value))
    }
}
