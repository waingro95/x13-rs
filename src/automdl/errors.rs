//! The error of a numerical state the selector cannot continue from.

use crate::Error;
use crate::error::NumericalOperation;

/// A numerical state the selector cannot continue from, reported as an
/// estimation failure.
pub(super) fn estimation_failure() -> Error {
    Error::numerical(NumericalOperation::Estimation)
}
