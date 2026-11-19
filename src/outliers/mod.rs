//! Automatic outlier identification (`idotlr.f`): the AO, LS and TC
//! candidate regressors, their critical values and the search that adds and
//! deletes them.

pub(crate) mod candidates;
pub(crate) mod critical;
pub(crate) mod search;

use crate::Error;
use crate::error::NumericalOperation;

/// The error for a search that cannot go on, as with a nonfinite residual or
/// statistic.
const fn outlier_failure() -> Error {
    Error::numerical(NumericalOperation::OutlierSearch)
}

/// [`outlier_failure`], keeping the lower-level failure as its source.
fn outlier_failure_from(source: impl Into<Error>) -> Error {
    Error::numerical_from(NumericalOperation::OutlierSearch, source.into())
}
