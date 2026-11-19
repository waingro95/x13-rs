//! SEARCH (`ansub1.f`) over two or more coordinates: the least-squares
//! minimiser SEATS refits a model with.
//!
//! Each iteration fixes any free coordinate within 1% of a bound, forms the
//! forward-difference Jacobian of CALCFX's residuals, and looks for a point
//! with a smaller sum of squares: first along the Newton step, then on a
//! spiral from it towards steepest descent, then along steepest descent
//! alone. FEASI, which shortens a step that leaves the bounds, is
//! `Search::feasible_fraction`.
//!
//! A model family's CALCFX implements [`SearchObjective`]. [`search`] hands
//! the objective back with the result, so the family can read the CALCFX
//! state its last evaluation left. The ARMA CALCFX can stop an evaluation
//! when it finds a common AR and MA factor; SEARCH then ends with the last
//! completed evaluation. `super::one_coordinate` is SEARCH with one
//! coordinate.

use super::{MAX_EVALUATIONS, MAX_ITERATIONS, finite, unusable};
use crate::seats::residual::ResidualResult;
use crate::seats::{Error, Failure, Stage};

/// XMIN and XMAX: the bounds of a free coordinate.
const LOWER: f64 = -0.99;
const UPPER: f64 = 0.99;
const CEPS: f64 = 1e-12;

/// A model family's CALCFX as SEARCH calls it.
pub(crate) trait SearchObjective<const N: usize> {
    /// Whether SEARCH's covariance tail runs. It does for the two-coordinate
    /// models, (0,1,1)(0,1,1) and (1,1,0)(0,1,1). The tail reads NV and SIG,
    /// which Build 62 can reach before writing them; for these models that
    /// is an error.
    const ESTIMATES_COVARIANCE: bool = false;
    /// SEARCH's first CALCFX call, which runs to completion.
    fn calcfx(&mut self, point: [f64; N]) -> Result<ResidualResult, Error>;
    /// Every later CALCFX call. `Ok(None)` means CALCFX found a common
    /// factor and stopped.
    fn interruptible(&mut self, point: [f64; N]) -> Result<Option<ResidualResult>, Error> {
        self.calcfx(point).map(Some)
    }
    /// IFN: the CALCFX calls so far.
    fn evaluations(&self) -> usize;
}

/// What the bound pass does with one coordinate.
#[derive(Debug, PartialEq)]
enum BoundCandidate {
    /// Left free.
    Skip,
    AlreadyFixed,
    /// Fixed at this value.
    Bound(f64),
}
impl BoundCandidate {
    /// Takes a fixed coordinate out of NV.
    fn decrement_free(&self, free_count: &mut usize) {
        if !matches!(self, Self::Skip) {
            *free_count -= 1;
        }
    }
}

/// SEARCH's bound-fix arguments as ANALTS passes them: FIXVAL=P+BP AR
/// coordinates, of which the first IPR=P are regular, BD and DR=D.
///
/// SEARCH fixes a coordinate within 1% of a bound. Its lower fix depends on
/// these arguments: a regular AR coordinate goes to -UR only while D plus the
/// unit roots already fixed this iteration is below 2, a seasonal AR
/// coordinate only when BD=0, and an MA coordinate goes to XMIN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BoundRule {
    fixval: usize,
    ipr: usize,
    dr: usize,
    bd: usize,
}
impl BoundRule {
    /// From the model's orders; SEARCH sees the regular then seasonal AR.
    pub(crate) fn from_orders(orders: crate::seats::Orders) -> Self {
        Self {
            fixval: usize::from(orders.p) + usize::from(orders.bp),
            ipr: usize::from(orders.p),
            dr: usize::from(orders.d),
            bd: usize::from(orders.bd),
        }
    }
}

/// UR, which ANALTS passes as 1.
const UR: f64 = 1.0;

/// One iteration's bound pass over the coordinates, in order.
struct BoundFixing {
    rule: BoundRule,
    /// nUnitReg: the regular AR coordinates this pass has fixed at -UR.
    regular_lower_count: usize,
}
impl BoundFixing {
    fn new(rule: BoundRule) -> Self {
        Self {
            rule,
            regular_lower_count: 0,
        }
    }
    fn candidate(&mut self, coordinate: usize, fixed: bool, value: f64) -> BoundCandidate {
        if fixed {
            return BoundCandidate::AlreadyFixed;
        }
        let xtest = (value - LOWER) / (UPPER - LOWER);
        if xtest >= 0.01 {
            if xtest <= 0.99 {
                return BoundCandidate::Skip;
            }
            // SEARCH's upper fix is 0.99 for an AR coordinate and XMAX for an
            // MA one; both are UPPER.
            return BoundCandidate::Bound(UPPER);
        }
        let j = coordinate + 1;
        if j > self.rule.fixval {
            return BoundCandidate::Bound(LOWER);
        }
        if j > self.rule.ipr {
            // A seasonal AR coordinate is fixed at -UR only when BD=0.
            return if self.rule.bd == 0 {
                BoundCandidate::Bound(-UR)
            } else {
                BoundCandidate::Skip
            };
        }
        // A regular AR coordinate is fixed at -UR only while D plus the
        // unit roots fixed so far stays below 2.
        if self.rule.dr + self.regular_lower_count < 2 {
            self.regular_lower_count += 1;
            return BoundCandidate::Bound(-UR);
        }
        BoundCandidate::Skip
    }
}

/// Standard errors and correlations of SEARCH's coordinates, 100 where a
/// coordinate ended fixed.
pub(crate) struct CovarianceEstimate<const N: usize> {
    pub(crate) standard_errors: [f64; N],
    pub(crate) correlation: [[f64; N]; N],
}

/// What SEARCH leaves: the final coordinates with CALCFX's residuals,
/// DETPRI and sum of squares there.
pub(crate) struct SearchResult<const N: usize> {
    pub(crate) parameters: [f64; N],
    /// The coordinates held at the start or fixed on a bound.
    pub(crate) bound_mask: [bool; N],
    pub(crate) scaled_residuals: Vec<f64>,
    pub(crate) determinant_scale: f64,
    pub(crate) objective: f64,
    /// Fixed coordinates; 1 when SEARCH fixed all of them.
    pub(crate) fixed: usize,
    /// IFN at the end.
    pub(crate) evaluations: usize,
    pub(crate) all_fixed: bool,
    /// `Some` when the covariance tail ran: the objective asks for it and a
    /// coordinate ended free.
    pub(crate) covariance: Option<CovarianceEstimate<N>>,
}

/// Why a SEARCH step stopped before its normal end.
enum Halt {
    /// CALCFX found a common factor: SEARCH ends with the last completed
    /// evaluation.
    Interrupted,
    Failed(Error),
}
impl From<Error> for Halt {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

/// How an iteration's step ended: at a point with a smaller sum of squares,
/// or with every step below its tolerance.
#[derive(Debug, PartialEq)]
enum Step {
    Accepted,
    Converged,
}

/// The multiple-free step's phases, by SEARCH's labels.
enum Phase {
    /// 5006: the Taylor (Newton) trial, with its interpolation.
    Taylor,
    /// 5008: trial points on the spiral between Newton and steepest descent.
    Spiral,
    /// 5009: halve both directions, or give up on them after five trials.
    Halve,
    /// 5015: steepest descent alone.
    Descent,
}

/// GAM, the angle between the Newton and steepest-descent directions, with
/// CGAM, SGAM and COTB (its cosine, sine and cosecant).
struct Angle {
    gamma: f64,
    cos: f64,
    sin: f64,
    csc: f64,
}
impl Angle {
    /// The spiral's angle at mixing weight `mu`.
    fn theta(&self, mu: f64) -> f64 {
        (mu * self.sin / mu.mul_add(self.cos, 1.0 - mu)).atan()
    }
    /// The spiral's radial factor at `theta`.
    fn factor(&self, theta: f64, mu: f64) -> f64 {
        // gfortran fuses each product of SEARCH's QTH with the subtraction
        // that follows it.
        let curvature = self.gamma.mul_add(self.csc, -1.0);
        theta.sin()
            * curvature.mul_add((theta / self.gamma).powi(2), -theta.mul_add(self.csc, -1.0))
            / (mu * self.sin)
    }
}

/// SEARCH's variables, under their Fortran names where they have one.
struct Search<O, const N: usize> {
    calcfx: O,
    /// X: the current trial point.
    point: [f64; N],
    /// BX: the iteration's base point.
    base: [f64; N],
    /// E: coordinates fixed at the start or on a bound.
    fixed: [bool; N],
    /// F, FI and DETPRI from the last completed evaluation.
    residuals: Vec<f64>,
    objective: f64,
    determinant_scale: f64,
    /// FIT: the previous trial's objective. It survives across iterations,
    /// and a halved one-coordinate trial reads it before writing it.
    previous_objective: f64,
    /// A: P'P over the free coordinates, then its standardized inverse.
    normal: [[f64; N]; N],
    /// G: F'P.
    gradient: [f64; N],
    /// SIG: P'P's column scales, unwritten until a multiple-free step.
    column_scale: Option<[f64; N]>,
    /// NV, unwritten until the first bound pass.
    free_count: Option<usize>,
    /// ITN.
    iteration: usize,
}

impl<O: SearchObjective<N>, const N: usize> Search<O, N> {
    /// CALCFX at `point`, keeping its objective and determinant scale;
    /// returns its scaled residuals.
    fn evaluate(&mut self, point: [f64; N]) -> Result<Vec<f64>, Halt> {
        let result = self
            .calcfx
            .interruptible(point)
            .map_err(Halt::Failed)?
            .ok_or(Halt::Interrupted)?;
        Ok(self.keep(result))
    }
    /// Keeps a completed CALCFX's objective and determinant scale; returns
    /// its scaled residuals.
    fn keep(&mut self, result: ResidualResult) -> Vec<f64> {
        self.objective = result.objective;
        self.determinant_scale = result.determinant_scale;
        result.scaled_residuals
    }
    /// Label 5020: evaluate once more at the base point before finishing.
    fn final_evaluation(&mut self) -> Result<(), Halt> {
        self.point = self.base;
        self.residuals = self.evaluate(self.point)?;
        Ok(())
    }
    /// FEASI: the largest fraction of the step from the base to `point` that
    /// keeps every free coordinate within the bounds.
    fn feasible_fraction(&self, point: [f64; N]) -> f64 {
        let mut reduction = 1.0;
        for (j, &coordinate) in point.iter().enumerate() {
            if self.fixed[j] {
                continue;
            }
            let ratio = if coordinate > UPPER {
                (UPPER - self.base[j]) / (coordinate - self.base[j])
            } else if coordinate < LOWER {
                (self.base[j] - LOWER) / (self.base[j] - coordinate)
            } else {
                continue;
            };
            if ratio < reduction {
                reduction = ratio;
            }
        }
        reduction
    }

    /// The iterations; true when SEARCH ends with every coordinate fixed.
    fn iterate(&mut self, rule: BoundRule) -> Result<bool, Halt> {
        let delta = (UPPER - LOWER) * 0.0001;
        let step_tolerance = [(UPPER - LOWER) * 0.001; N];
        while self.iteration < MAX_ITERATIONS && self.calcfx.evaluations() < MAX_EVALUATIONS {
            self.base = self.point;
            if self.objective < 1e-9 {
                // On the first iteration SEARCH jumps to 5020 before
                // `nv = nx`, and its covariance tail reads NV unwritten.
                if self.free_count.is_none() && O::ESTIMATES_COVARIANCE {
                    return Err(Failure::UnwrittenSearchCount {
                        objective: self.objective,
                    }
                    .at(Stage::Refit)
                    .into());
                }
                self.final_evaluation()?;
                return Ok(false);
            }
            let base_objective = self.objective;
            let (free_count, first_free) = self.fix_bounds(rule)?;
            self.free_count = Some(free_count);
            let Some(first_free) = first_free else {
                self.point = self.base;
                return Ok(true);
            };
            self.derivatives(free_count, delta)?;
            if self.gradient[..free_count].iter().all(|g| g.abs() <= CEPS) {
                // SEARCH's covariance tail then reads SIG, which only a
                // multiple-free step writes.
                if free_count == 2 && self.column_scale.is_none() && O::ESTIMATES_COVARIANCE {
                    return Err(Failure::UnwrittenSearchScale {
                        gradient: [self.gradient[0], self.gradient[1]],
                    }
                    .at(Stage::Refit)
                    .into());
                }
                self.final_evaluation()?;
                return Ok(false);
            }
            let step = if free_count > 1 {
                self.multiple_free(base_objective, step_tolerance, free_count)?
            } else {
                self.one_free(first_free, base_objective, step_tolerance)?
            };
            if step == Step::Converged {
                self.final_evaluation()?;
                return Ok(false);
            }
            self.iteration += 1;
        }
        Ok(false)
    }

    /// Fixes each free coordinate near a bound, evaluating at the fix;
    /// returns NV and the first coordinate still free.
    fn fix_bounds(&mut self, rule: BoundRule) -> Result<(usize, Option<usize>), Halt> {
        let mut free_count = N;
        let mut first_free = None;
        let mut fixing = BoundFixing::new(rule);
        for j in 0..N {
            let candidate = fixing.candidate(j, self.fixed[j], self.point[j]);
            if let BoundCandidate::Bound(bound) = candidate {
                self.base[j] = bound;
                self.fixed[j] = true;
                self.residuals = self.evaluate(self.base)?;
            }
            if candidate == BoundCandidate::Skip {
                first_free.get_or_insert(j);
            }
            candidate.decrement_free(&mut free_count);
        }
        Ok((free_count, first_free))
    }

    /// The forward-difference Jacobian P of the free coordinates, then P'P
    /// (lower triangle) and F'P.
    fn derivatives(&mut self, free_count: usize, delta: f64) -> Result<(), Halt> {
        let mut jacobian = vec![[0.0; N]; self.residuals.len()];
        let mut column = 0;
        for j in 0..N {
            if self.fixed[j] {
                continue;
            }
            self.base[j] += delta;
            let shifted = self.evaluate(self.base)?;
            for (row, (f, f1)) in jacobian.iter_mut().zip(self.residuals.iter().zip(&shifted)) {
                row[column] = (f - f1) / delta;
            }
            self.base[j] -= delta;
            column += 1;
        }
        for j in 0..free_count {
            for k in 0..=j {
                let sum = jacobian
                    .iter()
                    .fold(0.0, |sum, row| row[j].mul_add(row[k], sum));
                self.normal[j][k] = finite(sum, "P'P")?;
            }
        }
        for j in 0..free_count {
            let sum = self
                .residuals
                .iter()
                .zip(&jacobian)
                .fold(0.0, |sum, (f, row)| f.mul_add(row[j], sum));
            self.gradient[j] = finite(sum, "F'P")?;
        }
        Ok(())
    }

    /// One free coordinate, `free_index`: a Newton step along it, halved,
    /// with interpolation, until the objective drops below the base.
    fn one_free(
        &mut self,
        free_index: usize,
        base_objective: f64,
        step_tolerance: [f64; N],
    ) -> Result<Step, Halt> {
        let mut step = finite(
            self.gradient[0] / self.normal[0][0],
            "one-coordinate search direction",
        )?;
        self.point = self.base;
        self.point[free_index] += step;
        let mut trial = 1;
        let reduction = self.feasible_fraction(self.point);
        if (reduction - 1.0).abs() >= CEPS {
            // SEARCH's `.8` here is single precision, unlike the `.8d0` of
            // the multiple-free step.
            let reduced = f64::from(0.8_f32) * reduction * step;
            step = reduced;
            self.point[free_index] = self.base[free_index] + step;
        }
        loop {
            if step.abs() < step_tolerance[free_index] {
                return Ok(Step::Converged);
            }
            self.residuals = self.evaluate(self.point)?;
            if self.objective < base_objective {
                return Ok(Step::Accepted);
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
                    self.point[free_index] = lambda.mul_add(step, self.base[free_index]);
                    self.residuals = self.evaluate(self.point)?;
                    if self.objective < base_objective {
                        return Ok(Step::Accepted);
                    }
                    if self.objective >= trial_objective {
                        self.previous_objective = trial_objective;
                        step *= 0.5;
                        trial += 1;
                        self.point[free_index] = self.base[free_index] + step;
                        continue;
                    }
                    step *= lambda;
                }
                self.previous_objective = self.objective;
            }
            step *= 0.5;
            trial += 1;
            self.point[free_index] = self.base[free_index] + step;
        }
    }

    /// Scales P'P to a unit diagonal and the gradient with it (SIG, GS).
    fn standardize(&mut self, free_count: usize) -> Result<([f64; N], [f64; N]), Error> {
        let mut scale = [0.0; N];
        let mut gradient = [0.0; N];
        for j in 0..free_count {
            scale[j] = self.normal[j][j].sqrt().max(1e-30);
            gradient[j] = finite(self.gradient[j] / scale[j], "standardized gradient")?;
            for k in 0..j {
                self.normal[j][k] /= scale[j] * scale[k];
                self.normal[k][j] = self.normal[j][k];
            }
            self.normal[j][j] = 1.0;
        }
        self.column_scale = Some(scale);
        Ok((gradient, scale))
    }

    /// SEARCH's in-place Gauss–Jordan inversion of the standardized P'P,
    /// then the lower triangle mirrored from the upper.
    fn invert_normal(&mut self, free_count: usize) -> Result<(), Error> {
        let mut determinant = 1.0;
        for i in 0..free_count {
            determinant *= self.normal[i][i];
            if !determinant.is_finite() || determinant <= 0.0 {
                return Err(unusable("matrix determinant", determinant));
            }
            // SEARCH goes on with a small positive determinant; only a small
            // pivot stops it.
            if self.normal[i][i] <= 1e-10 {
                return Err(unusable("matrix diagonal", self.normal[i][i]));
            }
            let pivot = 1.0 / self.normal[i][i];
            self.normal[i][i] = pivot;
            for j in 0..free_count {
                if j < i {
                    let v = pivot * self.normal[j][i];
                    for k in j..free_count {
                        if k < i {
                            self.normal[j][k] = v.mul_add(self.normal[k][i], self.normal[j][k]);
                        }
                        if k > i {
                            self.normal[j][k] = (-v).mul_add(self.normal[i][k], self.normal[j][k]);
                        }
                    }
                    self.normal[j][i] = -v;
                } else if j != i {
                    let v = pivot * self.normal[i][j];
                    for k in j..free_count {
                        self.normal[j][k] = (-v).mul_add(self.normal[i][k], self.normal[j][k]);
                    }
                    self.normal[i][j] = v;
                }
            }
        }
        for j in 0..free_count {
            for k in 0..j {
                self.normal[j][k] = self.normal[k][j];
            }
        }
        Ok(())
    }

    /// Several free coordinates: the Taylor step, the spiral towards steepest
    /// descent, halving, then steepest descent alone.
    fn multiple_free(
        &mut self,
        base_objective: f64,
        step_tolerance: [f64; N],
        free_count: usize,
    ) -> Result<Step, Halt> {
        let (standardized_gradient, column_scale) = self.standardize(free_count)?;
        self.invert_normal(free_count)?;
        let mut standardized_newton = [0.0; N];
        let (mut newton_square, mut gradient_square, mut inner_product) = (0.0, 0.0, 0.0);
        for j in 0..free_count {
            standardized_newton[j] = standardized_gradient[..free_count]
                .iter()
                .enumerate()
                .fold(0.0, |sum, (k, g)| self.normal[j][k].mul_add(*g, sum));
            let (newton, gradient) = (standardized_newton[j], standardized_gradient[j]);
            // gfortran compiles SEARCH's ST, SG and SP sums two terms at a
            // time, each product rounded before the additions, and fuses
            // only an odd last term. TS above is fused throughout.
            if j < free_count / 2 * 2 {
                newton_square += newton * newton;
                gradient_square += gradient * gradient;
                inner_product += gradient * newton;
            } else {
                newton_square = newton.mul_add(newton, newton_square);
                gradient_square = gradient.mul_add(gradient, gradient_square);
                inner_product = gradient.mul_add(newton, inner_product);
            }
        }
        let cos = finite(
            inner_product / (newton_square * gradient_square).sqrt(),
            "angle cosine",
        )?;
        let gamma = finite(cos.acos(), "angle")?;
        let sin = gamma.sin();
        let angle = Angle {
            gamma,
            cos,
            sin,
            csc: 1.0 / sin,
        };
        let steepest_scale = finite((newton_square / gradient_square).sqrt(), "steepest scale")?;
        let mut newton = [0.0; N];
        let mut descent = [0.0; N];
        for j in 0..free_count {
            newton[j] = standardized_newton[j] / column_scale[j];
            descent[j] = (standardized_gradient[j] * steepest_scale) / column_scale[j];
        }
        // Spread the free coordinates' directions over all coordinates, last
        // first, as SEARCH does in place.
        let mut free = free_count;
        for j in (0..N).rev() {
            if self.fixed[j] {
                newton[j] = 0.0;
                descent[j] = 0.0;
            } else {
                free -= 1;
                newton[j] = newton[free];
                descent[j] = descent[free];
            }
        }
        for (j, direction) in newton.iter().enumerate() {
            self.point[j] = self.base[j] + direction;
        }
        let mut mu = [0.0; 9];
        mu[2] = 0.1;
        for j in 3..=8 {
            mu[j] = (2.0 * mu[j - 1]) / (1.0 + mu[j - 1]);
        }
        // FIA, H and S: the spiral's objectives, slopes and steps.
        let mut spiral_objectives = [0.0; 9];
        let mut slopes = [0.0; 9];
        let mut spiral_steps = [[0.0; N]; 9];
        // ICOM: a spiral pass stayed inside the bounds, so later passes reuse
        // its steps unchecked.
        let mut spiral_complete = false;
        let mut trial = 1;
        // Q: the spiral steps' accumulated halving and interpolation.
        let mut spiral_scale = 1.0;
        let reduction = self.feasible_fraction(self.point);
        if (reduction - 1.0).abs() >= CEPS {
            let factor = reduction * 0.8;
            for j in 0..N {
                newton[j] *= factor;
                descent[j] *= factor;
                self.point[j] = self.base[j] + newton[j];
            }
        }
        let mut phase = Phase::Taylor;
        loop {
            phase = match phase {
                Phase::Taylor => {
                    if (0..N).all(|j| newton[j].abs() <= step_tolerance[j]) {
                        return Ok(Step::Converged);
                    }
                    self.residuals = self.evaluate(self.point)?;
                    if self.objective < base_objective {
                        return Ok(Step::Accepted);
                    }
                    let threshold = base_objective.mul_add(0.75, self.previous_objective * 0.25);
                    if trial > 1 && self.objective < threshold {
                        let trial_objective = self.objective;
                        let lambda = finite(
                            (threshold - trial_objective)
                                / (base_objective.mul_add(0.5, self.previous_objective * 0.5)
                                    - trial_objective),
                            "Taylor interpolation",
                        )?;
                        for (j, &direction) in newton.iter().enumerate() {
                            self.point[j] = lambda.mul_add(direction, self.base[j]);
                        }
                        self.residuals = self.evaluate(self.point)?;
                        if self.objective < base_objective {
                            return Ok(Step::Accepted);
                        }
                        if self.objective >= trial_objective {
                            self.previous_objective = trial_objective;
                        } else {
                            for j in 0..N {
                                newton[j] *= lambda;
                                descent[j] *= lambda;
                            }
                            spiral_scale *= lambda;
                            self.previous_objective = self.objective;
                        }
                    } else {
                        self.previous_objective = self.objective;
                    }
                    Phase::Spiral
                }
                Phase::Spiral => {
                    spiral_objectives[1] = self.previous_objective;
                    let mut left_bounds = false;
                    for j in 2..=8 {
                        if !spiral_complete {
                            let theta = finite(angle.theta(mu[j]), "spiral angle")?;
                            let factor = angle.factor(theta, mu[j]);
                            for k in 0..N {
                                spiral_steps[j][k] =
                                    factor * mu[j].mul_add(descent[k], (1.0 - mu[j]) * newton[k]);
                            }
                        }
                        for (k, &direction) in spiral_steps[j].iter().enumerate() {
                            self.point[k] = direction.mul_add(spiral_scale, self.base[k]);
                        }
                        if !spiral_complete
                            && (self.feasible_fraction(self.point) - 1.0).abs() >= CEPS
                        {
                            left_bounds = true;
                            break;
                        }
                        self.residuals = self.evaluate(self.point)?;
                        if self.objective < base_objective {
                            return Ok(Step::Accepted);
                        }
                        spiral_objectives[j] = self.objective;
                        slopes[j] =
                            (spiral_objectives[j] - spiral_objectives[j - 1]) / (mu[j] - mu[j - 1]);
                        if j >= 3 && slopes[j] > slopes[j - 1] {
                            let minimum = slopes[j].mul_add(
                                mu[j - 1] + mu[j - 2],
                                -slopes[j - 1] * (mu[j] + mu[j - 1]),
                            ) / (2.0 * (slopes[j] - slopes[j - 1]));
                            if minimum < mu[j] && minimum > mu[j - 2] {
                                // Unlike the spiral points, this one ignores Q.
                                let factor = angle.factor(angle.theta(minimum), minimum);
                                for k in 0..N {
                                    let direction =
                                        minimum.mul_add(descent[k], (1.0 - minimum) * newton[k]);
                                    // gfortran fuses SS's product with BX + SS.
                                    self.point[k] = factor.mul_add(direction, self.base[k]);
                                }
                                if !spiral_complete && self.feasible_fraction(self.point) < 1.0 {
                                    left_bounds = true;
                                    break;
                                }
                                self.residuals = self.evaluate(self.point)?;
                                if self.objective < base_objective {
                                    return Ok(Step::Accepted);
                                }
                            }
                        }
                    }
                    if !left_bounds {
                        spiral_complete = true;
                    }
                    Phase::Halve
                }
                Phase::Halve if trial >= 5 => Phase::Descent,
                Phase::Halve => {
                    for j in 0..N {
                        newton[j] *= 0.5;
                        descent[j] *= 0.5;
                        self.point[j] = self.base[j] + newton[j];
                    }
                    spiral_scale *= 0.5;
                    trial += 1;
                    Phase::Taylor
                }
                Phase::Descent => {
                    return self.steepest_descent(descent, base_objective, step_tolerance);
                }
            };
        }
    }

    /// Label 5015: halve the steepest-descent step until the objective drops
    /// below the base or the step converges.
    fn steepest_descent(
        &mut self,
        mut descent: [f64; N],
        base_objective: f64,
        step_tolerance: [f64; N],
    ) -> Result<Step, Halt> {
        for (j, direction) in descent.iter_mut().enumerate() {
            *direction *= 0.5;
            self.point[j] = self.base[j] + *direction;
        }
        let reduction = self.feasible_fraction(self.point);
        if reduction < 1.0 {
            for (j, direction) in descent.iter_mut().enumerate() {
                let reduced = 0.9 * reduction * *direction;
                *direction = reduced;
                self.point[j] = self.base[j] + *direction;
            }
        }
        loop {
            if (0..N).all(|j| descent[j].abs() <= step_tolerance[j] * 0.5) {
                return Ok(Step::Converged);
            }
            self.residuals = self.evaluate(self.point)?;
            if self.objective < base_objective {
                return Ok(Step::Accepted);
            }
            for (j, direction) in descent.iter_mut().enumerate() {
                *direction *= 0.5;
                self.point[j] = self.base[j] + *direction;
            }
        }
    }

    /// SEARCH's covariance tail: the free coordinates' standard errors and
    /// correlations, expanded with 100 for the fixed ones.
    fn covariance(&self, free_count: usize) -> Result<CovarianceEstimate<N>, Error> {
        let variance = self.objective / (self.residuals.len() - free_count) as f64;
        let mut errors = [0.0; N];
        let mut correlation = [[0.0; N]; N];
        if free_count == 1 {
            let a = if self.normal[0][0].abs() < 1e-13 {
                1e-6
            } else {
                self.normal[0][0]
            };
            errors[0] = finite((variance / a).sqrt(), "one-coordinate standard error")?;
            correlation[0][0] = 1.0;
        } else {
            let Some(column_scale) = self.column_scale else {
                return Err(Failure::UnwrittenSearchScale {
                    gradient: [self.gradient[0], self.gradient[1]],
                }
                .at(Stage::Refit));
            };
            for j in 0..free_count {
                for (k, value) in correlation[j].iter_mut().enumerate().take(j + 1) {
                    *value = finite(
                        self.normal[j][k] / (self.normal[j][j] * self.normal[k][k]).sqrt(),
                        "correlation",
                    )?;
                }
                errors[j] = finite(
                    (self.normal[j][j] * variance).sqrt() / column_scale[j],
                    "two-free SE",
                )?;
            }
        }
        let mut standard_errors = [100.0; N];
        let mut expanded = [[100.0; N]; N];
        let free = (0..N).filter(|&i| !self.fixed[i]);
        for (k, i) in free.clone().enumerate() {
            standard_errors[i] = errors[k];
            for (l, j) in free.clone().take_while(|&j| j <= i).enumerate() {
                expanded[i][j] = correlation[k][l];
                expanded[j][i] = expanded[i][j];
            }
        }
        Ok(CovarianceEstimate {
            standard_errors,
            correlation: expanded,
        })
    }

    fn finish(self, all_fixed: bool) -> Result<(SearchResult<N>, O), Error> {
        // An objective that asks for the covariance has NV by now: its first
        // iteration either wrote it or stopped on a tiny objective.
        let covariance = match self.free_count {
            Some(free_count) if !all_fixed && O::ESTIMATES_COVARIANCE => {
                Some(self.covariance(free_count)?)
            }
            _ => None,
        };
        let fixed = if all_fixed {
            1
        } else {
            self.fixed.iter().filter(|&&fixed| fixed).count()
        };
        let result = SearchResult {
            parameters: self.point,
            bound_mask: self.fixed,
            scaled_residuals: self.residuals,
            determinant_scale: self.determinant_scale,
            objective: self.objective,
            fixed,
            evaluations: self.calcfx.evaluations(),
            all_fixed,
            covariance,
        };
        Ok((result, self.calcfx))
    }
}

/// SEARCH from `start`, holding the `fixed` coordinates, until it converges
/// or reaches its iteration or evaluation limit. Returns the objective with
/// the result, for its CALCFX state.
///
/// Each family checks `start` first: every coordinate finite, and a free
/// one within the bounds.
pub(crate) fn search<O: SearchObjective<N>, const N: usize>(
    calcfx: O,
    start: [f64; N],
    fixed: [bool; N],
    rule: BoundRule,
) -> Result<(SearchResult<N>, O), Error> {
    debug_assert!(
        start
            .iter()
            .zip(fixed)
            .all(|(x, fixed)| x.is_finite() && (fixed || (LOWER..=UPPER).contains(x)))
    );
    let mut state = Search {
        calcfx,
        point: start,
        base: start,
        fixed,
        residuals: Vec::new(),
        objective: 0.0,
        determinant_scale: 1.0,
        previous_objective: 0.0,
        normal: [[0.0; N]; N],
        gradient: [0.0; N],
        column_scale: None,
        free_count: None,
        iteration: 1,
    };
    let initial = state.calcfx.calcfx(start)?;
    state.residuals = state.keep(initial);
    let all_fixed = match state.iterate(rule) {
        Ok(all_fixed) => all_fixed,
        Err(Halt::Interrupted) => false,
        Err(Halt::Failed(failure)) => return Err(failure),
    };
    state.finish(all_fixed)
}

#[cfg(test)]
mod tests;
