//! A series as a case hands it to both implementations.

use serde::{Deserialize, Serialize};

/// Build 62's missing-value code (`gtinpt.f` initializes `Mvcode` to it).
/// [`Series::values`] holds it at each gap, for Build 62's input file; Rust
/// gets NaN at the same positions ([`Series::rust_values`]).
pub(crate) const MISSING_VALUE: f64 = -99999.;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Frequency {
    Monthly,
    Quarterly,
    Semiannual,
}

impl Frequency {
    pub(crate) const fn periods_per_year(self) -> u8 {
        match self {
            Self::Monthly => 12,
            Self::Quarterly => 4,
            Self::Semiannual => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Period {
    pub(crate) year: i32,
    pub(crate) subperiod: u8,
}

impl Period {
    pub(crate) const fn new(year: i32, subperiod: u8) -> Self {
        Self { year, subperiod }
    }
}

/// `values` is the Build 62 side: each gap holds [`MISSING_VALUE`], which the
/// input file, the row filters and the comparison read. Rust takes
/// [`Series::rust_values`] instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Series {
    pub(crate) series_id: String,
    pub(crate) frequency: Frequency,
    pub(crate) start: Period,
    pub(crate) values: Vec<f64>,
}

impl Series {
    /// The observations as `x13::adjust` takes them: NaN at each gap, where
    /// Build 62 reads [`MISSING_VALUE`].
    pub(crate) fn rust_values(&self) -> Vec<f64> {
        rust_values(&self.values)
    }
}

/// Build 62 values to Rust values: [`MISSING_VALUE`] becomes NaN.
pub(crate) fn rust_values(build_62: &[f64]) -> Vec<f64> {
    build_62
        .iter()
        .map(|&value| {
            if value == MISSING_VALUE {
                f64::NAN
            } else {
                value
            }
        })
        .collect()
}

/// Rust values to Build 62 values: NaN becomes [`MISSING_VALUE`].
pub(crate) fn build_62_values(rust: &[f64]) -> Vec<f64> {
    rust.iter()
        .map(|&value| if value.is_nan() { MISSING_VALUE } else { value })
        .collect()
}
