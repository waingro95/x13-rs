//! SEARCH (`ansub1.f`) with one coordinate: the refit of a model whose only
//! estimated coefficient is one MA or one AR(1) coefficient.
//!
//! With one coordinate SEARCH has only its one-coordinate step: a Newton step,
//! halved and interpolated until the sum of squares drops. A coordinate
//! within 1% of a bound is fixed there and ends the search. Otherwise the
//! search ends with the coefficient's standard error.

use super::{MAX_EVALUATIONS, MAX_ITERATIONS, finite};
use crate::seats::residual::ResidualResult;
use crate::seats::{Failure, Stage};

/// SEARCH's covariance tail for one coordinate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SeasonalMaCovariance {
    Available {
        standard_errors: [f64; 1],
        correlation: [[f64; 1]; 1],
    },
    /// The coordinate ended fixed on a bound.
    UnavailableAllFixed,
}

/// What the one-coordinate SEARCH leaves: the coefficient, CALCFX's
/// residuals there and whether SEARCH fixed it on a bound.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeasonalMaRefit {
    pub(crate) ma: [f64; 1],
    pub(crate) residuals: ResidualResult,
    pub(crate) bound_mask: [bool; 1],
    pub(crate) covariance: SeasonalMaCovariance,
}

/// XMIN and XMAX: the bounds of the free coordinate.
pub(crate) const LOWER: f64 = -0.99;
const UPPER: f64 = 0.99;
const CEPS: f64 = 1e-12;

/// A CALCFX with one coordinate as SEARCH calls it.
pub(crate) trait OneCoordinateObjective {
    type Error: From<crate::seats::Error>;
    fn evaluate_one(&mut self, x: f64) -> Result<ResidualResult, Self::Error>;
    /// IFN: the CALCFX calls so far.
    fn evaluations(&self) -> usize;
    /// Where SEARCH fixes the coordinate near its lower bound: XMIN for an
    /// MA coefficient, -1 for an AR one.
    fn fixed_lower(&self) -> f64;
}

/// SEARCH's variables, named as in `super::search`.
struct Search<O> {
    calcfx: O,
    point: f64,
    base: f64,
    residuals: Vec<f64>,
    objective: f64,
    determinant_scale: f64,
    previous_objective: f64,
    normal: f64,
    gradient: f64,
    iteration: usize,
}

impl<O: OneCoordinateObjective> Search<O> {
    /// CALCFX at `point`, keeping its objective and determinant scale;
    /// returns its scaled residuals.
    fn evaluate(&mut self, point: f64) -> Result<Vec<f64>, O::Error> {
        let result = self.calcfx.evaluate_one(point)?;
        self.objective = result.objective;
        self.determinant_scale = result.determinant_scale;
        Ok(result.scaled_residuals)
    }
    /// FEASI: the fraction of the step from the base to `point` that stays
    /// within the bounds.
    fn feasible_fraction(&self, point: f64) -> f64 {
        if point > UPPER {
            (UPPER - self.base) / (point - self.base)
        } else if point < LOWER {
            (self.base - LOWER) / (self.base - point)
        } else {
            1.0
        }
    }
    /// A Newton step, halved, with interpolation, until the objective drops
    /// below the base. Returns whether it found such a point; when the step
    /// falls below `step_tolerance` first, SEARCH has converged.
    fn step(&mut self, base_objective: f64, step_tolerance: f64) -> Result<bool, O::Error> {
        let mut step = finite(
            self.gradient / self.normal,
            "one-coordinate search direction",
        )?;
        self.point = self.base + step;
        let mut trial = 1;
        let reduction = self.feasible_fraction(self.point);
        if (reduction - 1.0).abs() >= CEPS {
            // SEARCH's `.8` here is single precision.
            let reduced = f64::from(0.8_f32) * reduction * step;
            step = reduced;
            self.point = self.base + step;
        }
        loop {
            if step.abs() < step_tolerance {
                return Ok(false);
            }
            self.residuals = self.evaluate(self.point)?;
            if self.objective < base_objective {
                return Ok(true);
            }
            if trial > 1 {
                let threshold = base_objective.mul_add(0.75, self.previous_objective * 0.25);
                if self.objective < threshold {
                    let trial_objective = self.objective;
                    let lambda = finite(
                        (threshold - trial_objective)
                            / (base_objective.mul_add(0.5, self.previous_objective * 0.5)
                                - trial_objective),
                        "one-coordinate interpolation",
                    )?;
                    self.point = lambda.mul_add(step, self.base);
                    self.residuals = self.evaluate(self.point)?;
                    if self.objective < base_objective {
                        return Ok(true);
                    }
                    if self.objective >= trial_objective {
                        self.previous_objective = trial_objective;
                        step *= 0.5;
                        trial += 1;
                        self.point = self.base + step;
                        continue;
                    }
                    step *= lambda;
                }
                self.previous_objective = self.objective;
            }
            step *= 0.5;
            trial += 1;
            self.point = self.base + step;
        }
    }
    /// The result at the current point, with the covariance tail unless the
    /// coordinate ended `fixed`.
    fn finish(self, fixed: bool) -> Result<SeasonalMaRefit, crate::seats::Error> {
        let covariance = if fixed {
            SeasonalMaCovariance::UnavailableAllFixed
        } else {
            let variance = self.objective / (self.residuals.len() - 1) as f64;
            // SEARCH replaces a P'P below 1e-13 in magnitude by 1e-6.
            let normal = if self.normal.abs() < 1.0e-13 {
                1.0e-6
            } else {
                self.normal
            };
            SeasonalMaCovariance::Available {
                standard_errors: [finite(
                    (variance / normal).sqrt(),
                    "one-coordinate standard error",
                )?],
                correlation: [[1.0]],
            }
        };
        let normalized_residuals = self
            .residuals
            .iter()
            .map(|v| v / self.determinant_scale)
            .collect();
        Ok(SeasonalMaRefit {
            ma: [self.point],
            residuals: ResidualResult {
                scaled_residuals: self.residuals,
                determinant_scale: self.determinant_scale,
                objective: self.objective,
                normalized_residuals,
            },
            bound_mask: [fixed],
            covariance,
        })
    }
}

/// SEARCH with one free coordinate from `start`, which lies within the
/// bounds, until it converges or reaches its iteration or evaluation limit.
pub(crate) fn search_one<O: OneCoordinateObjective>(
    calcfx: O,
    start: f64,
) -> Result<SeasonalMaRefit, O::Error> {
    debug_assert!((LOWER..=UPPER).contains(&start));
    let mut state = Search {
        calcfx,
        point: start,
        base: start,
        residuals: Vec::new(),
        objective: 0.0,
        determinant_scale: 1.0,
        previous_objective: 0.0,
        normal: 0.0,
        gradient: 0.0,
        iteration: 1,
    };
    state.residuals = state.evaluate(start)?;
    let delta = (UPPER - LOWER) * 0.0001;
    let step_tolerance = (UPPER - LOWER) * 0.001;
    while state.iteration < MAX_ITERATIONS && state.calcfx.evaluations() < MAX_EVALUATIONS {
        state.base = state.point;
        if state.objective < 1e-9 {
            // On the first iteration SEARCH jumps to 5020 before `nv = nx`,
            // and its covariance tail reads NV unwritten.
            if state.iteration == 1 {
                return Err(Failure::UnwrittenSearchCount {
                    objective: state.objective,
                }
                .at(Stage::Refit)
                .into());
            }
            state.residuals = state.evaluate(state.base)?;
            state.point = state.base;
            return Ok(state.finish(false)?);
        }
        let base_objective = state.objective;
        let ratio = (state.point - LOWER) / (UPPER - LOWER);
        if ratio < 0.01 || ratio > 0.99 {
            state.base = if ratio < 0.01 {
                state.calcfx.fixed_lower()
            } else {
                UPPER
            };
            state.residuals = state.evaluate(state.base)?;
            state.point = state.base;
            return Ok(state.finish(true)?);
        }
        state.base += delta;
        let shifted = state.evaluate(state.base)?;
        let jacobian = state
            .residuals
            .iter()
            .zip(shifted)
            .map(|(f, f1)| (f - f1) / delta)
            .collect::<Vec<_>>();
        state.base -= delta;
        let normal = jacobian.iter().fold(0.0, |sum, p| p.mul_add(*p, sum));
        state.normal = finite(normal, "P'P")?;
        let gradient = state
            .residuals
            .iter()
            .zip(&jacobian)
            .fold(0.0, |sum, (f, p)| f.mul_add(*p, sum));
        state.gradient = finite(gradient, "F'P")?;
        // A vanishing gradient ends the search like a converged step. P'P is
        // already written, so the covariance tail is defined here.
        if state.gradient.abs() <= CEPS || !state.step(base_objective, step_tolerance)? {
            state.point = state.base;
            state.residuals = state.evaluate(state.point)?;
            return Ok(state.finish(false)?);
        }
        state.iteration += 1;
    }
    Ok(state.finish(false)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frequency;
    use crate::seats::ar1_seasonal_ma::residual::Ar1SeasonalMaObjective;
    use crate::seats::refit::test_support::{alternating, differenced_noise, noise, random_walk};
    use crate::seats::seasonal_ma::SingleMaShape;
    use crate::seats::seasonal_ma::residual::SeasonalMaObjective;

    /// SEARCH for the quarterly seasonal MA(1) on `wd`.
    fn seasonal_ma(wd: &[f64], start: f64) -> Result<SeasonalMaRefit, crate::seats::Error> {
        search_one(
            SeasonalMaObjective::new(Frequency::Quarterly, SingleMaShape::Seasonal, wd),
            start,
        )
    }

    fn standard_error(refit: &SeasonalMaRefit) -> f64 {
        match refit.covariance {
            SeasonalMaCovariance::Available {
                standard_errors, ..
            } => standard_errors[0],
            SeasonalMaCovariance::UnavailableAllFixed => f64::NAN,
        }
    }

    #[test]
    fn an_ma_coefficient_converges_with_its_standard_error() {
        let refit = seasonal_ma(&noise(64), 0.2).unwrap();
        assert_eq!(refit.ma, [0.1668038528965311]);
        assert_eq!(refit.bound_mask, [false]);
        assert_eq!(refit.residuals.objective, 4.585237169942247);
        assert_eq!(refit.residuals.determinant_scale, 1.0008821993062942);
        assert_eq!(refit.residuals.scaled_residuals.len(), 68);
        assert_eq!(refit.residuals.scaled_residuals[0], 0.002796771335610768);
        assert_eq!(standard_error(&refit), 0.12195909247157005);
    }

    #[test]
    fn an_ar_coefficient_converges_with_its_standard_error() {
        let refit = search_one(
            Ar1SeasonalMaObjective::new(Frequency::Quarterly, &noise(64), false),
            0.2,
        )
        .unwrap();
        assert_eq!(refit.ma, [-0.1458597992989875]);
        assert_eq!(refit.bound_mask, [false]);
        assert_eq!(refit.residuals.objective, 4.664559603866967);
        assert_eq!(refit.residuals.determinant_scale, 1.0);
        assert_eq!(refit.residuals.scaled_residuals.len(), 63);
        assert_eq!(refit.residuals.scaled_residuals[0], -0.0017932520407882484);
        assert_eq!(standard_error(&refit), 0.1271576805080247);
    }

    #[test]
    fn a_coefficient_near_the_upper_bound_is_fixed_there() {
        let refit = seasonal_ma(&differenced_noise(64, 4), 0.2).unwrap();
        assert_eq!(refit.ma, [0.99]);
        assert_eq!(refit.bound_mask, [true]);
        assert_eq!(refit.residuals.objective, 5.8285434667212765);
        assert_eq!(refit.residuals.determinant_scale, 1.0872636372312945);
        assert_eq!(refit.residuals.scaled_residuals[0], -0.13878429303457485);
        assert_eq!(refit.covariance, SeasonalMaCovariance::UnavailableAllFixed);
    }

    #[test]
    fn the_lower_fix_is_xmin_for_an_ma_and_minus_one_for_an_ar_coefficient() {
        let wd = alternating(64);
        let ma = seasonal_ma(&wd, -0.4).unwrap();
        assert_eq!(ma.ma, [-0.99]);
        assert_eq!(ma.bound_mask, [true]);
        assert_eq!(ma.residuals.objective, 22.42206203544942);
        assert_eq!(ma.residuals.determinant_scale, 1.0872636372312945);
        assert_eq!(ma.residuals.scaled_residuals[0], 0.2201869929181977);
        assert_eq!(ma.covariance, SeasonalMaCovariance::UnavailableAllFixed);

        let ar = search_one(
            Ar1SeasonalMaObjective::new(Frequency::Quarterly, &wd, false),
            0.2,
        )
        .unwrap();
        assert_eq!(ar.ma, [-1.0]);
        assert_eq!(ar.bound_mask, [true]);
        assert_eq!(ar.residuals.objective, 0.43095735355782455);
        assert_eq!(ar.residuals.determinant_scale, 1.0);
        assert_eq!(ar.residuals.scaled_residuals[0], -0.017239654402201543);
        assert_eq!(ar.covariance, SeasonalMaCovariance::UnavailableAllFixed);
    }

    #[test]
    fn search_stops_at_the_iteration_limit() {
        let refit = seasonal_ma(&random_walk(64), 0.2).unwrap();
        assert_eq!(refit.ma, [-0.6115410114325289]);
        assert_eq!(refit.residuals.objective, 24.844312607571737);
        assert_eq!(refit.residuals.determinant_scale, 1.0147444156165402);
        assert_eq!(refit.residuals.scaled_residuals[0], -0.15967875572783768);
        assert_eq!(standard_error(&refit), 0.10400054637795236);
    }

    #[test]
    fn a_zero_series_is_an_error_where_build_62_reads_nv_unwritten() {
        assert_eq!(
            seasonal_ma(&[0.0; 64], 0.2),
            Err(Failure::UnwrittenSearchCount { objective: 0.0 }.at(Stage::Refit))
        );
    }
}
