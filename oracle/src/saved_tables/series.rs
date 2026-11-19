use super::parse_finite;
use crate::{Frequency, OracleError, Period};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SeriesPoint {
    pub(crate) period: Period,
    pub(crate) value: f64,
}

pub(crate) fn parse_series_table(
    table: &str,
    frequency: Frequency,
) -> Result<Vec<SeriesPoint>, OracleError> {
    let mut points = Vec::new();
    for (line_number, line) in table.lines().enumerate().skip(2) {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let date = fields.next().ok_or_else(|| {
            OracleError::InvalidTable(format!("line {} has no date", line_number + 1))
        })?;
        let value = fields.next().ok_or_else(|| {
            OracleError::InvalidTable(format!("line {} has no value", line_number + 1))
        })?;
        if date.len() < 6 {
            return Err(OracleError::InvalidTable(format!(
                "line {} has invalid date {date:?}",
                line_number + 1
            )));
        }
        let split = date.len() - 2;
        let year = date[..split].parse::<i32>().map_err(|_| {
            OracleError::InvalidTable(format!("line {} has invalid year", line_number + 1))
        })?;
        let subperiod = date[split..].parse::<u8>().map_err(|_| {
            OracleError::InvalidTable(format!("line {} has invalid subperiod", line_number + 1))
        })?;
        if subperiod == 0 || subperiod > frequency.periods_per_year() {
            return Err(OracleError::InvalidTable(format!(
                "line {} has subperiod {subperiod} for {frequency:?}",
                line_number + 1
            )));
        }
        let value = parse_finite(value, &format!("line {}", line_number + 1))?;
        points.push(SeriesPoint {
            period: Period::new(year, subperiod),
            value,
        });
    }
    if points.is_empty() {
        return Err(OracleError::InvalidTable(
            "table has no data rows".to_owned(),
        ));
    }
    Ok(points)
}
