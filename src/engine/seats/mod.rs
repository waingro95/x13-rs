//! The SEATS pipeline around a fitted regARIMA model.

mod airline;
mod ar1_seasonal_ma;
mod ar2;
mod arma;
mod effects;
mod pipeline;
mod seasonal_ma;
mod second_difference;
mod strategy;

pub(crate) use pipeline::{adjust, finish_prepared, prepare_automatic, seats_length_range};
