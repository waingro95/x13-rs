//! The user-defined regressor columns a row supplies, generated from the
//! series so that Build 62 and Rust receive the same values.

use chrono::{Datelike, Days, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

use crate::{Frequency, OracleError, Series};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UserRegressorRequest {
    pub(crate) kind: UserRegressorPolicy,
    pub(crate) future_years: u8,
}

impl UserRegressorRequest {
    pub(crate) fn columns(self, series: &Series) -> Result<Vec<Column>, OracleError> {
        self.kind.generate(
            series,
            u16::from(self.future_years) * u16::from(series.frequency.periods_per_year()),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UserRegressorPolicy {
    WorkingDays,
    OrthodoxEaster,
    Ramp,
    /// Working days then the ramp: two groups of different kinds, any frequency.
    WorkingDaysAndRamp,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    TradingDay,
    Holiday,
    User,
}

impl Kind {
    fn keyword(self) -> &'static str {
        match self {
            Self::TradingDay => "td",
            Self::Holiday => "holiday",
            Self::User => "user",
        }
    }
}

pub(crate) struct Column {
    pub(crate) kind: Kind,
    pub(crate) values: Vec<f64>,
}

impl UserRegressorPolicy {
    pub(crate) fn generate(
        self,
        series: &Series,
        forecasts: u16,
    ) -> Result<Vec<Column>, OracleError> {
        let count = series.values.len() + usize::from(forecasts);
        let mut columns = Vec::new();
        if matches!(
            self,
            Self::WorkingDays | Self::WorkingDaysAndRamp | Self::All
        ) {
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                let first = period_start(series, index)?;
                let months = if series.frequency == Frequency::Quarterly {
                    3
                } else {
                    1
                };
                let mut total = 0.0;
                for offset in 0..months {
                    total += working_days(first.year(), first.month() + offset)?;
                }
                values.push(total);
            }
            center_by_season(series, &mut values);
            columns.push(Column {
                kind: Kind::TradingDay,
                values,
            });
        }
        if matches!(self, Self::OrthodoxEaster | Self::All) {
            if series.frequency != Frequency::Monthly {
                return Err(invalid(
                    "Orthodox Easter supplied regressor requires monthly data",
                ));
            }
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                let first = period_start(series, index)?;
                values.push(orthodox_window_fraction(first.year(), first.month())?);
            }
            center_by_season(series, &mut values);
            columns.push(Column {
                kind: Kind::Holiday,
                values,
            });
        }
        if matches!(self, Self::Ramp | Self::WorkingDaysAndRamp | Self::All) {
            let onset = series.values.len() / 2;
            let width = f64::from(series.frequency.periods_per_year()) / 2.0;
            let values = (0..count)
                .map(|index| {
                    if index < onset {
                        0.0
                    } else {
                        ((index - onset + 1) as f64 / width).min(1.0)
                    }
                })
                .collect();
            columns.push(Column {
                kind: Kind::User,
                values,
            });
        }
        Ok(columns)
    }
}

pub(crate) fn render(series: &Series, columns: &[Column]) -> Result<String, OracleError> {
    let mut text = String::new();
    let names = (1..=columns.len())
        .map(|index| format!("user{index}"))
        .collect::<Vec<_>>();
    writeln!(text, "  user = ({})", names.join(" ")).unwrap();
    let kinds = columns
        .iter()
        .map(|column| column.kind.keyword())
        .collect::<Vec<_>>();
    writeln!(text, "  usertype = ({})", kinds.join(" ")).unwrap();
    writeln!(
        text,
        "  start = {}.{}",
        series.start.year, series.start.subperiod
    )
    .unwrap();
    writeln!(text, "  data = (").unwrap();
    // getreg.f reads a flat X vector; regvar.f/copycl consumes period-major rows.
    // CTOD does not use correctly rounded decimal conversion. Encode for its
    // arithmetic, keeping the generated Rust values unchanged. One value per
    // line keeps even multi-column requests below Build 62's line limit.
    let mut decimals = HashMap::new();
    for index in 0..columns[0].values.len() {
        for column in columns {
            let value = column.values[index];
            let decimal = match decimals.entry(value.to_bits()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(ctod_decimal(value)?)
                }
            };
            writeln!(text, "    {decimal}").unwrap();
        }
    }
    text.push_str("  )\n");
    Ok(text)
}

fn ctod_decimal(value: f64) -> Result<String, OracleError> {
    if !value.is_finite() {
        return Err(invalid("nonfinite supplied regressor value"));
    }
    let original = format!("{value:.17e}");
    if ctod(&original)?.to_bits() == value.to_bits() {
        return Ok(original);
    }
    let (mantissa, exponent) = original.split_once('e').ok_or_else(decimal_error)?;
    let exponent: i32 = exponent.parse().map_err(|_| decimal_error())?;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let coefficient: i64 = mantissa
        .trim_start_matches('-')
        .replace('.', "")
        .parse()
        .map_err(|_| decimal_error())?;
    // Bounded spelling search for these supplied columns, not a general float
    // codec. At most 15 integer digits are exact in f64, so CTOD's multiply/add
    // agrees whether its compiler contracts that expression to an FMA or not.
    // Move the decimal point and adjust the final digits without changing the
    // value CTOD must produce; never round or substitute the Rust input.
    for width in 1..=15 {
        for offset in std::iter::once(0).chain((1..=16).flat_map(|n| [-n, n])) {
            let digits = format!("{:018}", coefficient + offset);
            if digits.len() != 18 {
                continue;
            }
            let candidate = format!(
                "{sign}{}.{}e{}",
                &digits[..width],
                &digits[width..],
                exponent + 1 - width as i32
            );
            if ctod(&candidate)?.to_bits() == value.to_bits() {
                return Ok(candidate);
            }
        }
    }
    Err(invalid(&format!(
        "supplied regressor value {value:.17e} has no exact Build 62 decimal in the bounded spelling search"
    )))
}

fn decimal_error() -> OracleError {
    invalid("invalid generated supplied regressor decimal")
}

fn ctod(text: &str) -> Result<f64, OracleError> {
    let (mantissa, exponent) = text.split_once('e').ok_or_else(decimal_error)?;
    let exponent: i32 = exponent.parse().map_err(|_| decimal_error())?;
    let sign = if mantissa.starts_with('-') { -1.0 } else { 1.0 };
    let mut value = 0.0;
    let mut scale = 1.0;
    let mut fractional = false;
    // ctod.f accumulates each fractional digit separately, after multiplying
    // scl by ten. Standard str::parse rounds the complete decimal instead.
    for byte in mantissa.trim_start_matches('-').bytes() {
        if byte == b'.' {
            fractional = true;
            continue;
        }
        if !byte.is_ascii_digit() {
            return Err(decimal_error());
        }
        let digit = f64::from(byte - b'0');
        if fractional {
            scale *= 10.0;
            value += digit / scale;
        } else {
            value = 10.0 * value + digit;
        }
    }
    // GFortran lowers CTOD's 10D0**expint to integer exponentiation by
    // squaring, with the reciprocal taken last for negative powers. Preserve
    // that order rather than using powf or a decimal power-of-ten literal.
    let mut remaining = exponent.unsigned_abs();
    let mut base = 10.0;
    let mut power = 1.0;
    while remaining > 0 {
        if remaining & 1 != 0 {
            power *= base;
        }
        remaining >>= 1;
        if remaining > 0 {
            base *= base;
        }
    }
    if exponent < 0 {
        power = 1.0 / power;
    }
    Ok((sign * value) * power)
}

fn invalid(message: &str) -> OracleError {
    OracleError::InvalidConfiguration(message.into())
}

fn period_start(series: &Series, index: usize) -> Result<NaiveDate, OracleError> {
    let frequency = usize::from(series.frequency.periods_per_year());
    let offset = usize::from(series.start.subperiod) - 1 + index;
    let year = i32::try_from(i64::from(series.start.year) + (offset / frequency) as i64)
        .map_err(|_| invalid("supplied regressor year out of range"))?;
    let month = (offset % frequency) * (12 / frequency) + 1;
    NaiveDate::from_ymd_opt(year, month as u32, 1)
        .ok_or_else(|| invalid("invalid supplied regressor calendar date"))
}

fn working_days(year: i32, month: u32) -> Result<f64, OracleError> {
    let mut count = 0.0;
    for day in 1..=31 {
        let Some(date) = NaiveDate::from_ymd_opt(year, month, day) else {
            continue;
        };
        if date.weekday().number_from_monday() <= 5
            && !matches!((month, day), (1, 1) | (7, 4) | (11, 11) | (12, 25))
        {
            count += 1.0;
        }
    }
    Ok(count)
}

fn orthodox_easter(year: i32) -> Result<NaiveDate, OracleError> {
    if !(1900..=2099).contains(&year) {
        return Err(invalid(
            "Orthodox Easter formula requires years 1900 through 2099",
        ));
    }
    let a = year % 4;
    let b = year % 7;
    let c = year % 19;
    let d = (19 * c + 15) % 30;
    let e = (2 * a + 4 * b - d + 34) % 7;
    let month = (d + e + 114) / 31;
    let day = (d + e + 114) % 31 + 1;
    NaiveDate::from_ymd_opt(year, month as u32, day as u32)
        .and_then(|date| date.checked_add_days(Days::new(13)))
        .ok_or_else(|| invalid("invalid Orthodox Easter date"))
}

fn orthodox_window_fraction(year: i32, month: u32) -> Result<f64, OracleError> {
    let easter = orthodox_easter(year)?;
    let mut count = 0;
    for offset in 1..=14 {
        let date = easter
            .checked_sub_days(Days::new(offset))
            .ok_or_else(|| invalid("invalid Orthodox Easter window"))?;
        count += usize::from(date.month() == month);
    }
    Ok(count as f64 / 14.0)
}

fn center_by_season(series: &Series, values: &mut [f64]) {
    let frequency = usize::from(series.frequency.periods_per_year());
    let season = |index: usize| (usize::from(series.start.subperiod) - 1 + index) % frequency;
    let mut sums = vec![0.0; frequency];
    let mut counts = vec![0; frequency];
    for (index, &value) in values.iter().enumerate() {
        sums[season(index)] += value;
        counts[season(index)] += 1;
    }
    for (index, value) in values.iter_mut().enumerate() {
        let index = season(index);
        *value -= sums[index] / f64::from(counts[index]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Period;

    fn series(frequency: Frequency, count: usize) -> Series {
        Series {
            series_id: "supplied".into(),
            frequency,
            start: Period::new(2023, 1),
            values: vec![100.0; count],
        }
    }

    #[test]
    fn supplied_calendar_formulas_match_known_dates_and_window_boundaries() {
        assert_eq!(working_days(2023, 1).unwrap(), 22.0); // Jan 1 is a Sunday.
        assert_eq!(working_days(2024, 1).unwrap(), 22.0); // Jan 1 is a Monday.
        assert_eq!(working_days(2023, 7).unwrap(), 20.0);
        assert_eq!(
            orthodox_easter(2024).unwrap(),
            NaiveDate::from_ymd_opt(2024, 5, 5).unwrap()
        );
        assert_eq!(orthodox_window_fraction(2024, 4).unwrap(), 10.0 / 14.0);
        assert_eq!(orthodox_window_fraction(2024, 5).unwrap(), 4.0 / 14.0);
        assert!(orthodox_easter(2100).is_err());
    }

    #[test]
    fn calendars_center_over_observed_and_forecast_periods() {
        for frequency in [Frequency::Monthly, Frequency::Quarterly] {
            let periods = usize::from(frequency.periods_per_year());
            let input = series(frequency, 2 * periods);
            let columns = UserRegressorPolicy::WorkingDays
                .generate(&input, periods as u16)
                .unwrap();
            for season in 0..periods {
                let values = &columns[0].values;
                let sum = values[season] + values[season + periods] + values[season + 2 * periods];
                assert!(sum.abs() < 1e-12);
            }
            let first_raw = if frequency == Frequency::Monthly {
                22.0
            } else {
                22.0 + 20.0 + 23.0
            };
            let next_raw = if frequency == Frequency::Monthly {
                22.0
            } else {
                22.0 + 21.0 + 21.0
            };
            let last_raw = if frequency == Frequency::Monthly {
                22.0
            } else {
                22.0 + 20.0 + 21.0
            };
            assert_eq!(
                columns[0].values[0],
                first_raw - (first_raw + next_raw + last_raw) / 3.0
            );
        }
    }

    #[test]
    fn ramp_uses_observed_midpoint_and_extends_forecast_values() {
        for (frequency, count, width) in [(Frequency::Monthly, 25, 6), (Frequency::Quarterly, 9, 2)]
        {
            let input = series(frequency, count);
            let columns = UserRegressorPolicy::Ramp
                .generate(&input, frequency.periods_per_year().into())
                .unwrap();
            let values = &columns[0].values;
            let onset = count / 2;
            assert!(values[..onset].iter().all(|&value| value == 0.0));
            assert_eq!(values[onset], 1.0 / f64::from(width));
            assert!(
                values[onset + width as usize - 1..]
                    .iter()
                    .all(|&value| value == 1.0)
            );
        }
    }

    #[test]
    fn ctod_reads_a_fractional_ramp_back_exactly() {
        let value = 5.0_f64 / 6.0;
        let old = format!("{value:.17e}");
        assert_ne!(ctod(&old).unwrap().to_bits(), value.to_bits());
        let exact = ctod_decimal(value).unwrap();
        assert_eq!(ctod(&exact).unwrap().to_bits(), value.to_bits());
        assert!(ctod_decimal(f64::NAN).is_err());
        assert!(ctod_decimal(f64::INFINITY).is_err());
    }

    #[test]
    fn the_matrix_is_period_major_and_round_trips_exactly() {
        let input = series(Frequency::Monthly, 24);
        let columns = UserRegressorPolicy::All.generate(&input, 12).unwrap();
        let text = render(&input, &columns).unwrap();
        assert!(text.starts_with(
            "  user = (user1 user2 user3)\n  usertype = (td holiday user)\n  start = 2023.1\n"
        ));
        let rows = text
            .split_once("  data = (\n")
            .unwrap()
            .1
            .strip_suffix("  )\n")
            .unwrap();
        let values = rows
            .split_whitespace()
            .map(|value| ctod(value).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 36 * 3);
        for index in 0..36 {
            for column in 0..3 {
                assert_eq!(
                    values[3 * index + column].to_bits(),
                    columns[column].values[index].to_bits()
                );
            }
        }
    }
}
