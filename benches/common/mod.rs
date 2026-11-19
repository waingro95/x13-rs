//! Corpus loading and specs shared by the benches. It lives in a directory
//! so Cargo does not discover it as a bench target of its own.

// Each bench compiles this module separately and uses only part of it.
#![allow(dead_code)]

use std::{error::Error, fs, path::Path};
use x13::regarima::{
    CalendarBasis, CalendarSpec, Easter, FlowTradingDay, OutlierSpec, OutlierTypes,
    TradingDayContrast, TransformSelection,
};
use x13::{Period, RegArimaSpec, Spec};

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// A gap, as `adjust` takes it.
pub const MISSING: f64 = f64::NAN;

/// One frozen corpus series, gaps included.
pub struct CorpusSeries {
    pub quarterly: bool,
    pub start: Period,
    pub values: Vec<f64>,
}

/// The ids and frequencies listed in `tests/parity/series.toml`.
pub fn corpus_ids(root: &Path) -> Result<Vec<(String, bool)>> {
    let metadata: toml::Value = toml::from_str(&fs::read_to_string(root.join("series.toml"))?)?;
    metadata["series"]
        .as_array()
        .ok_or("series array missing")?
        .iter()
        .map(|meta| {
            let id = meta["id"].as_str().ok_or("series id missing")?;
            let quarterly = meta["frequency"].as_str() == Some("quarterly");
            Ok((id.to_owned(), quarterly))
        })
        .collect()
}

/// Reads `series/<id>.csv` under `root`, filling skipped periods with
/// [`MISSING`].
pub fn load(root: &Path, id: &str, quarterly: bool) -> Result<CorpusSeries> {
    let step = if quarterly { 3 } else { 1 };
    let contents = fs::read_to_string(root.join("series").join(format!("{id}.csv")))?;
    let mut start = None;
    let mut previous = None;
    let mut values = Vec::new();
    for line in contents.lines().skip(1) {
        let (date, value) = line.split_once(',').ok_or("invalid CSV")?;
        let parts = date
            .split('-')
            .map(str::parse::<i32>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if parts.len() != 3
            || parts[2] != 1
            || !(1..=12).contains(&parts[1])
            || (parts[1] - 1) % step != 0
        {
            return Err(format!("invalid period in {id}: {date}").into());
        }
        let month = parts[0] * 12 + parts[1] - 1;
        if let Some(last) = previous {
            if month <= last {
                return Err(format!("unordered dates in {id}").into());
            }
            values.extend(std::iter::repeat_n(
                MISSING,
                ((month - last) / step - 1) as usize,
            ));
        }
        previous = Some(month);
        if start.is_none() {
            start = Some(if quarterly {
                Period::quarterly(parts[0], ((parts[1] - 1) / 3 + 1) as u8)?
            } else {
                Period::monthly(parts[0], parts[1] as u8)?
            });
        }
        let value = match value.trim() {
            "" | "." => MISSING,
            text => match text.parse::<f64>()? {
                parsed if parsed.is_finite() => parsed,
                _ => return Err(format!("nonfinite input in {id}").into()),
            },
        };
        values.push(value);
    }
    Ok(CorpusSeries {
        quarterly,
        start: start.ok_or("empty CSV")?,
        values,
    })
}

/// SEATS after an automatic model with automatic transformation, six-coefficient
/// trading day, Easter\[8\] and every outlier type.
pub fn seats_combo() -> Result<Spec> {
    Ok(Spec::seats(
        RegArimaSpec::automatic_model()
            .with_transform(TransformSelection::automatic())
            .with_calendar(
                CalendarSpec::new()
                    .with_basis(CalendarBasis::FlowTradingDay(FlowTradingDay::new(
                        TradingDayContrast::SixCoefficient,
                    )))
                    .with_easter(Easter::new(8)?),
            )
            .with_outliers(OutlierSpec::new().with_types(OutlierTypes::ALL)),
    ))
}
