//! Parsers for the text files Build 62 saves.

use crate::OracleError;

mod outlier;
mod selected_model;
mod series;

pub(crate) use outlier::{OracleOutlierVariable, parse_outlier_label};
pub(crate) use selected_model::parse_selected_model;
pub(crate) use series::{SeriesPoint, parse_series_table};

pub(super) fn parse_finite(value: &str, context: &str) -> Result<f64, OracleError> {
    let value = value
        .parse::<f64>()
        .or_else(|error| {
            let Some(exponent_position) = missing_exponent_marker(value) else {
                return Err(error);
            };
            format!(
                "{}E{}",
                &value[..exponent_position],
                &value[exponent_position..]
            )
            .parse::<f64>()
        })
        .map_err(|_| invalid_table(format!("{context} has invalid value")))?;
    if !value.is_finite() {
        return Err(invalid_table(format!("{context} has non-finite value")));
    }
    Ok(value)
}

fn missing_exponent_marker(value: &str) -> Option<usize> {
    if value.bytes().any(|byte| matches!(byte, b'e' | b'E')) {
        return None;
    }
    let bytes = value.as_bytes();
    let position =
        bytes.iter().enumerate().rev().find_map(|(index, byte)| {
            (index > 0 && matches!(byte, b'+' | b'-')).then_some(index)
        })?;
    let exponent = &bytes[position + 1..];
    (exponent.len() == 3
        && exponent.iter().all(u8::is_ascii_digit)
        && value[..position].parse::<f64>().is_ok())
    .then_some(position)
}

pub(super) fn invalid_table(message: String) -> OracleError {
    OracleError::InvalidTable(message)
}
