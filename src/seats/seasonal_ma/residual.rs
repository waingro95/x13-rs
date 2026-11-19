//! CALCFX for one MA coefficient: the single-MA family's SEARCH objective and
//! residuals, and the one-group solves the AR families run after their AR
//! prefilter.

use super::SingleMaShape;
use crate::Frequency;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::residual::{CalcfxMa, MAXIMUM_OBSERVATIONS, ResidualResult, calcfx};

/// The single-MA SEARCH objective: CALCFX's residuals at a trial coefficient.
pub(crate) struct SeasonalMaObjective<'a> {
    frequency: Frequency,
    shape: SingleMaShape,
    differenced: &'a [f64],
    previous: Option<f64>,
    ma: f64,
    evaluations: usize,
}

impl<'a> SeasonalMaObjective<'a> {
    pub(crate) fn new(frequency: Frequency, shape: SingleMaShape, differenced: &'a [f64]) -> Self {
        Self {
            frequency,
            shape,
            differenced,
            previous: None,
            ma: 0.0,
            evaluations: 0,
        }
    }

    pub(crate) fn evaluate(&mut self, coordinates: [f64; 1]) -> Result<ResidualResult, Error> {
        let value = coordinates[0];
        if !value.is_finite() || value.abs() > 1.0 {
            return Err(Failure::Unusable {
                quantity: "seasonal restart coordinate",
                index: Some(0),
                value,
            }
            .at(Stage::Residual));
        }
        // CALCFX keeps the last trial's coefficient when the new one is
        // within TRANSC's threshold of it.
        if self
            .previous
            .is_none_or(|previous| (previous - value).abs() >= 3.834e-20)
        {
            self.ma = value;
        }
        let result = single_ma_residuals(self.frequency, self.shape, self.ma, self.differenced)?;
        self.previous = Some(value);
        self.evaluations += 1;
        Ok(result)
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }
}

/// CALCFX's residuals of the single-MA model with coefficient `ma`.
pub(crate) fn single_ma_residuals(
    frequency: Frequency,
    shape: SingleMaShape,
    ma: f64,
    differenced: &[f64],
) -> Result<ResidualResult, Error> {
    match shape {
        SingleMaShape::Seasonal => {
            validated_seasonal_residuals(usize::from(frequency.periods_per_year()), ma, differenced)
        }
        SingleMaShape::Nonseasonal => regular_ma_residuals(ma, differenced),
    }
}

/// The seasonal solve, after CALCFX's length condition on the differenced
/// series, which is finite.
fn validated_seasonal_residuals(
    s: usize,
    ma: f64,
    differenced: &[f64],
) -> Result<ResidualResult, Error> {
    let qstar = s;
    let n = differenced.len() + qstar;
    debug_assert!(n + 1 >= 2 * s + 2 && n < MAXIMUM_OBSERVATIONS);
    debug_assert!(differenced.iter().all(|value| value.is_finite()));

    seasonal_constrained_residuals(s, ma, differenced)
}

/// CALCFX with one regular MA coefficient and no seasonal one (Q=1, Bq=0),
/// whose K matrix is one dense column.
pub(crate) fn regular_ma_residuals(ma: f64, conditioned: &[f64]) -> Result<ResidualResult, Error> {
    // CALCFX 10.d0**(-Qstar-3) with Qstar=1; the positive power is exact.
    calcfx(CalcfxMa::SingleRegular { ma }, conditioned, 1.0 / 10_000.0)
}

/// CALCFX with one seasonal MA coefficient and no regular one (Q=0, Bq=1).
/// ANSUB1 1269–1327: there are no regular columns, the first seasonal
/// column starts at q1 and columns 2..s are its shifts.
pub(crate) fn seasonal_constrained_residuals(
    s: usize,
    ma: f64,
    differenced: &[f64],
) -> Result<ResidualResult, Error> {
    // CALCFX 10.d0**(-Qstar-3), Qstar=4/12; both positive powers are exact.
    let threshold = if s == 4 {
        1.0 / 10_000_000.0
    } else {
        1.0 / 1_000_000_000_000_000.0
    };
    let ma = CalcfxMa::Ordered {
        period: s,
        regular: &[],
        seasonal: Some(ma),
    };
    calcfx(ma, differenced, threshold)
}
