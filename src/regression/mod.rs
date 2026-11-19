//! The regression part of a regARIMA model: its columns, the calendar and
//! holiday regressors, the least-squares solve and the effect each fitted
//! group has on the series.

pub(crate) mod calendar;
pub(crate) mod design;
pub(crate) mod effects;
pub(crate) mod gls;
pub(crate) mod holidays;
pub(crate) mod model;
