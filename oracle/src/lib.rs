//! Development-only tools that check the `x13` crate against the Census
//! X-13ARIMA-SEATS Build 62 executable.
//!
//! - [`parity`] is the parity harness: it runs every row of the call table
//!   over the corpus through both implementations and compares the results.
//! - [`fuzz`] is a fuzz tool for `x13::adjust`, which replays its findings
//!   against Build 62.
//!
//! `README.md` says how to run both.

#![forbid(unsafe_code)]

mod build62;
mod configuration;
mod error;
pub mod fuzz;
pub mod parity;
mod rust_spec;
mod saved_tables;
mod series;
mod spec_file;
mod user_regressors;

pub(crate) use configuration::*;
pub(crate) use error::OracleError;
pub(crate) use rust_spec::RustError;
pub(crate) use saved_tables::{OracleOutlierVariable, SeriesPoint};
pub(crate) use series::{Frequency, Period, Series};
