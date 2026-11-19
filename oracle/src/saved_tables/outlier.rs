use super::invalid_table;
use crate::{Frequency, OracleError, OracleOutlierType, Period};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OracleOutlierVariable {
    pub(crate) outlier_type: OracleOutlierType,
    pub(crate) period: Period,
}

fn parse_outlier_variable(
    value: &str,
    frequency: Frequency,
    line_number: usize,
) -> Result<OracleOutlierVariable, OracleError> {
    if value.len() < 5 {
        return Err(invalid_table(format!(
            "outlier iteration line {line_number} has invalid outlier {value:?}"
        )));
    }
    let (code, dated) = value.split_at(2);
    let outlier_type = match code {
        "AO" => OracleOutlierType::Additive,
        "LS" => OracleOutlierType::LevelShift,
        "TC" => OracleOutlierType::TemporaryChange,
        _ => {
            return Err(invalid_table(format!(
                "outlier iteration line {line_number} has unknown outlier {value:?}"
            )));
        }
    };
    let (year, subperiod) = dated.split_once('.').ok_or_else(|| {
        invalid_table(format!(
            "outlier iteration line {line_number} has invalid outlier date {dated:?}"
        ))
    })?;
    let year = year.parse::<i32>().map_err(|_| {
        invalid_table(format!(
            "outlier iteration line {line_number} has invalid outlier year"
        ))
    })?;
    let subperiod = parse_named_subperiod(subperiod, frequency).ok_or_else(|| {
        invalid_table(format!(
            "outlier iteration line {line_number} has invalid outlier period {subperiod:?}"
        ))
    })?;
    Ok(OracleOutlierVariable {
        outlier_type,
        period: Period::new(year, subperiod),
    })
}

pub(crate) fn parse_outlier_label(
    value: &str,
    frequency: Frequency,
) -> Result<OracleOutlierVariable, OracleError> {
    parse_outlier_variable(value, frequency, 0)
}

fn parse_named_subperiod(value: &str, frequency: Frequency) -> Option<u8> {
    let subperiod = match frequency {
        Frequency::Monthly => match value {
            "Jan" => 1,
            "Feb" => 2,
            "Mar" => 3,
            "Apr" => 4,
            "May" => 5,
            "Jun" => 6,
            "Jul" => 7,
            "Aug" => 8,
            "Sep" => 9,
            "Oct" => 10,
            "Nov" => 11,
            "Dec" => 12,
            _ => return None,
        },
        Frequency::Quarterly | Frequency::Semiannual => value.parse().ok()?,
    };
    (subperiod > 0 && subperiod <= frequency.periods_per_year()).then_some(subperiod)
}
