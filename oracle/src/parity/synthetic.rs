//! Generated series for call-table rows the FRED corpus cannot reach.
//!
//! Some divergences need inputs no FRED series has: a constant series, a
//! simulated SEATS model with particular roots, or a real series moved to a
//! calendar-year floor. `tests/parity/synthetic.toml` holds one compact
//! descriptor per series and the harness regenerates the values on every run,
//! so the repository stores a few lines instead of the data.
//!
//! - Every ID starts with `SYN_`. A row selects synthetic series only by
//!   listing them in its `series` field, and rows without one stay FRED-only,
//!   so adding a descriptor changes no other row.
//! - `arima` descriptors use SplitMix64 for uniforms, Box–Muller for normal
//!   innovations, and the ARIMA recursion with a 600-period burn-in. The
//!   descriptor names the polynomial factors in the lag operator, written as
//!   `{ lag = coefficient }` terms of `1 + c B^lag`, with `diff` lags adding
//!   `(1 - B^lag)` factors.
//! - Values are rounded to a fixed number of decimals and parsed back, so the
//!   series is a short decimal text; Build 62 reads the rendered input and
//!   Rust the parsed `f64`, which are the same number.
//! - Box–Muller calls the platform libm (`ln`, `cos`, `sin`, `sqrt`). Rounding
//!   hides last-bit differences, except at a rounding boundary; the parity
//!   cache's platform stamp already confines results to one platform.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::{Frequency, Period, Series};

pub(crate) const PREFIX: &str = "SYN_";
const BURN_IN: usize = 600;
const DECIMALS: usize = 4;

pub(crate) fn is_synthetic(id: &str) -> bool {
    id.starts_with(PREFIX)
}

#[derive(Debug, Deserialize)]
pub(crate) struct Manifest {
    pub(crate) series: Vec<Descriptor>,
}

/// A polynomial factor `1 + Σ c B^lag`, keyed by lag.
type Factor = BTreeMap<String, f64>;

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Descriptor {
    Constant {
        id: String,
        #[serde(default = "monthly")]
        frequency: Frequency,
        #[serde(default = "start")]
        start: (i32, u8),
        length: usize,
        value: f64,
    },
    /// `AR(B) Π(1 - B^d) x = MA(B) e` with standard normal `e`, plus `level`.
    Arima {
        id: String,
        #[serde(default = "monthly")]
        frequency: Frequency,
        #[serde(default = "start")]
        start: (i32, u8),
        length: usize,
        seed: u64,
        #[serde(default)]
        level: f64,
        #[serde(default)]
        ar: Vec<Factor>,
        #[serde(default)]
        ma: Vec<Factor>,
        #[serde(default)]
        diff: Vec<usize>,
    },
    /// The last `last` observations of a FRED series, moved to a new start.
    Relabel {
        id: String,
        source: String,
        start: (i32, u8),
        last: usize,
    },
}

fn monthly() -> Frequency {
    Frequency::Monthly
}

fn start() -> (i32, u8) {
    (2000, 1)
}

impl Descriptor {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Constant { id, .. } | Self::Arima { id, .. } | Self::Relabel { id, .. } => id,
        }
    }

    pub(crate) fn generate(&self, fred: &[Series]) -> Result<Series> {
        let (frequency, (year, subperiod), values) = match self {
            Self::Constant {
                frequency,
                start,
                length,
                value,
                ..
            } => (*frequency, *start, vec![*value; *length]),
            Self::Arima {
                frequency,
                start,
                length,
                seed,
                level,
                ar,
                ma,
                diff,
                ..
            } => {
                let ar = product(ar, diff)?;
                let ma = product(ma, &[])?;
                let values = simulate(&ar, &ma, *seed, *length)
                    .into_iter()
                    .map(|value| round(level + value))
                    .collect::<Result<Vec<_>>>()?;
                (*frequency, *start, values)
            }
            Self::Relabel {
                source,
                start,
                last,
                ..
            } => {
                ensure!(!is_synthetic(source), "relabel source must be FRED");
                let source = fred
                    .iter()
                    .find(|series| series.series_id == *source)
                    .with_context(|| format!("missing relabel source {source}"))?;
                let skipped = source
                    .values
                    .len()
                    .checked_sub(*last)
                    .context("relabel takes more observations than its source has")?;
                (source.frequency, *start, source.values[skipped..].to_vec())
            }
        };
        ensure!(
            !values.is_empty() && values.iter().all(|value| value.is_finite()),
            "{}: empty or nonfinite series",
            self.id()
        );
        ensure!(
            frequency != Frequency::Semiannual
                && (1..=frequency.periods_per_year()).contains(&subperiod),
            "{}: invalid frequency or start period",
            self.id()
        );
        Ok(Series {
            series_id: self.id().into(),
            frequency,
            start: Period::new(year, subperiod),
            values,
        })
    }
}

/// Multiplies factors and unit-root differences into dense coefficients,
/// lag 0 first.
fn product(factors: &[Factor], differences: &[usize]) -> Result<Vec<f64>> {
    let mut dense = Vec::new();
    for factor in factors {
        let mut polynomial = vec![1.];
        for (lag, &coefficient) in factor {
            let lag: usize = lag.parse().context("polynomial lag")?;
            ensure!(lag > 0 && coefficient.is_finite(), "invalid term B^{lag}");
            polynomial.resize(polynomial.len().max(lag + 1), 0.);
            polynomial[lag] = coefficient;
        }
        dense.push(polynomial);
    }
    for &lag in differences {
        ensure!(lag > 0, "difference lag must be positive");
        let mut polynomial = vec![0.; lag + 1];
        polynomial[0] = 1.;
        polynomial[lag] = -1.;
        dense.push(polynomial);
    }
    Ok(dense.iter().fold(vec![1.], |left, right| {
        let mut out = vec![0.; left.len() + right.len() - 1];
        for (i, a) in left.iter().enumerate() {
            for (j, b) in right.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        out
    }))
}

fn simulate(ar: &[f64], ma: &[f64], seed: u64, length: usize) -> Vec<f64> {
    let total = BURN_IN + length;
    let innovations = normals(seed, total);
    let mut values = vec![0.; total];
    for t in 0..total {
        let mut value = 0.;
        for (k, coefficient) in ma.iter().enumerate().take(t + 1) {
            value += coefficient * innovations[t - k];
        }
        for (k, coefficient) in ar.iter().enumerate().take(t + 1).skip(1) {
            value -= coefficient * values[t - k];
        }
        values[t] = value;
    }
    values.split_off(BURN_IN)
}

/// One SplitMix64 step.
pub(super) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn normals(seed: u64, count: usize) -> Vec<f64> {
    let mut state = seed;
    // Mapped to (0, 1] so the logarithm below is finite.
    let mut uniform = || ((splitmix64(&mut state) >> 11) + 1) as f64 / (1_u64 << 53) as f64;
    let mut out = Vec::with_capacity(count + 1);
    while out.len() < count {
        let radius = (-2. * uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * uniform();
        out.extend([radius * angle.cos(), radius * angle.sin()]);
    }
    out.truncate(count);
    out
}

fn round(value: f64) -> Result<f64> {
    Ok(format!("{value:.DECIMALS$}").parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(text: &str) -> Descriptor {
        toml::from_str::<Manifest>(text)
            .unwrap()
            .series
            .pop()
            .unwrap()
    }

    #[test]
    fn arima_series_are_deterministic_rounded_and_seed_dependent() {
        let text = |seed| {
            format!(
                "[[series]]\nid = \"SYN_T\"\nkind = \"arima\"\nlength = 50\nseed = {seed}\n\
                 level = 10\nar = [{{ 1 = -0.5 }}]\nma = [{{ 12 = 0.3 }}]\ndiff = [1]\n"
            )
        };
        let first = descriptor(&text(1)).generate(&[]).unwrap();
        assert_eq!(
            first.values,
            descriptor(&text(1)).generate(&[]).unwrap().values
        );
        assert_ne!(
            first.values,
            descriptor(&text(2)).generate(&[]).unwrap().values
        );
        assert_eq!(first.start, Period::new(2000, 1));
        assert_eq!(first.values.len(), 50);
        for value in &first.values {
            assert_eq!(*value, format!("{value:.4}").parse::<f64>().unwrap());
        }
    }

    #[test]
    fn polynomial_factors_use_the_plus_convention_and_differences() {
        let factor = |terms: &[(&str, f64)]| {
            terms
                .iter()
                .map(|&(lag, c)| (lag.to_owned(), c))
                .collect::<Factor>()
        };
        // (1 + 0.5B)(1 - B^2) = 1 + 0.5B - B^2 - 0.5B^3
        assert_eq!(
            product(&[factor(&[("1", 0.5)])], &[2]).unwrap(),
            [1., 0.5, -1., -0.5]
        );
        assert!(product(&[factor(&[("0", 0.5)])], &[]).is_err());
        // An AR(1) recursion: x_t = 0.5 x_{t-1} + e_t for AR(B) = 1 - 0.5B.
        let x = simulate(&[1., -0.5], &[1.], 3, 4);
        let e = normals(3, BURN_IN + 4);
        assert_eq!(x[1], 0.5 * x[0] + e[BURN_IN + 1]);
    }

    #[test]
    fn constant_and_relabel_descriptors() {
        let constant = descriptor(
            "[[series]]\nid = \"SYN_C\"\nkind = \"constant\"\nlength = 3\nvalue = 100.0\n",
        )
        .generate(&[])
        .unwrap();
        assert_eq!(constant.values, [100.; 3]);
        let source = Series {
            series_id: "FRED1".into(),
            frequency: Frequency::Quarterly,
            start: Period::new(1990, 2),
            values: vec![1., 2., 3., 4.],
        };
        let relabel = |last| {
            descriptor(&format!(
                "[[series]]\nid = \"SYN_R\"\nkind = \"relabel\"\nsource = \"FRED1\"\n\
                 start = [1901, 1]\nlast = {last}\n"
            ))
            .generate(std::slice::from_ref(&source))
        };
        let moved = relabel(2).unwrap();
        assert_eq!(moved.values, [3., 4.]);
        assert_eq!(moved.frequency, Frequency::Quarterly);
        assert_eq!(moved.start, Period::new(1901, 1));
        assert!(relabel(5).is_err());
        assert!(
            toml::from_str::<Manifest>(
                "[[series]]\nid = \"SYN_C\"\nkind = \"constant\"\nlength = 3\nvalue = 1.0\nextra = 1\n"
            )
            .is_err()
        );
    }
}
