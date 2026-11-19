//! CALCFX for the AR(1) family: the AR prefilter of the differenced series,
//! then the seasonal-MA solve or, without a seasonal MA, the sum of squares.

use crate::Frequency;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::residual::{ResidualResult, ar_prefilter, finite};

/// The AR(1) SEARCH objective: CALCFX's residuals at trial coefficients.
pub(crate) struct Ar1SeasonalMaObjective<'a> {
    frequency: Frequency,
    wd: &'a [f64],
    previous: Option<[f64; 2]>,
    parameters: [f64; 2],
    evaluations: usize,
    seasonal: bool,
}

impl<'a> Ar1SeasonalMaObjective<'a> {
    pub(crate) fn new(frequency: Frequency, wd: &'a [f64], seasonal: bool) -> Self {
        Self {
            frequency,
            wd,
            previous: None,
            parameters: [0.0; 2],
            evaluations: 0,
            seasonal,
        }
    }
    pub(crate) fn evaluate(&mut self, coordinates: [f64; 2]) -> Result<ResidualResult, Error> {
        validate_coordinates(coordinates)?;
        // CALCFX keeps each coefficient of the last trial when the new one is
        // within TRANSC's threshold of it, one group at a time.
        for (i, &value) in coordinates.iter().enumerate() {
            if self
                .previous
                .is_none_or(|previous| (previous[i] - value).abs() >= 3.834e-20)
            {
                self.parameters[i] = value;
            }
        }
        let result = ar1_residuals(self.frequency, self.parameters, self.wd, self.seasonal)?;
        self.previous = Some(coordinates);
        self.evaluations += 1;
        Ok(result)
    }
    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }
}

/// Refuses an AR or seasonal MA coordinate that is nonfinite or outside
/// [-1, 1].
fn validate_coordinates(parameters: [f64; 2]) -> Result<(), Error> {
    for (value, quantity) in parameters
        .into_iter()
        .zip(["regular AR coordinate", "seasonal MA coordinate"])
    {
        if !value.is_finite() || value.abs() > 1.0 {
            return Err(Failure::Unusable {
                quantity,
                index: None,
                value,
            }
            .at(Stage::Residual));
        }
    }
    Ok(())
}

/// Whether `wd` is finite and implies an observed length in CheckLen's
/// range.
fn differenced_in_range(frequency: Frequency, wd: &[f64], seasonal: bool) -> bool {
    let s = usize::from(frequency.periods_per_year());
    let observed = wd.len().saturating_add(if seasonal { s + 1 } else { 1 });
    let (minimum, maximum) = match frequency {
        Frequency::Monthly => (36, 780),
        Frequency::Quarterly => (16, 328),
    };
    (minimum..=maximum).contains(&observed) && wd.iter().all(|value| value.is_finite())
}

/// CALCFX's residuals of the AR(1) model with the given coefficients on the
/// differenced series `wd`; `seasonal` says whether it has a seasonal MA.
pub(crate) fn ar1_residuals(
    frequency: Frequency,
    parameters: [f64; 2],
    wd: &[f64],
    seasonal: bool,
) -> Result<ResidualResult, Error> {
    validate_coordinates(parameters)?;
    let s = usize::from(frequency.periods_per_year());
    debug_assert!(differenced_in_range(frequency, wd, seasonal));
    // ansub1.f:1116–1131 with Pstar=1: the subtraction is fused with the AR
    // product, and no AR determinant is added. np=Nw-1, and the solve
    // returns np+s=N-2 residuals.
    let u = ar_prefilter(&parameters[..1], wd, |_, index, value| {
        finite(value, "conditional AR prefilter", Some(index))
    })?;
    if !seasonal {
        // With Qstar=0, CALCFX skips the MA solve and leaves Detpri at 1.
        let mut objective = 0.0;
        for &value in &u {
            objective = value.mul_add(value, objective);
        }
        finite(objective, "pure AR objective", None)?;
        return Ok(ResidualResult {
            normalized_residuals: u.clone(),
            scaled_residuals: u,
            determinant_scale: 1.0,
            objective,
        });
    }
    crate::seats::seasonal_ma::residual::seasonal_constrained_residuals(s, parameters[1], &u)
}
