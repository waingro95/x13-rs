//! MINPACK's Levenberg-Marquardt solver as Build 62 ships it (`lmdif`,
//! `fdjac2`, `qrfac`, `qrsolv`, `lmpar`, `enorm`). ARIMA estimation minimizes
//! the exact-likelihood residuals with it, and the ARMA covariance (`covar`,
//! in `arima::covariance`) reuses the Jacobian and QR.
//!
//! Every f64 operation, its operands and its order follow the Fortran,
//! including the fused multiply-adds gfortran emitted (`mul_add`), `enorm`'s
//! three-scale ladder, `qrfac`'s pivot tie rule and renormalization test,
//! `qrsolv`'s Givens rounding, and `lmdif`'s exit precedence. Matrices are
//! column-major views (`ColMajorMut`) instead of index arithmetic, and each
//! `lmdif` work vector has one role where MINPACK reuses `wa1`..`wa4`.
//!
//! `lmdif` allocates its buffers once per call and the iteration loops
//! allocate nothing: `lmpar` and `qrsolv` work in caller scratch.

use std::ops::{Index, IndexMut};

use super::{BUILD_62_MACHINE_EPSILON, BUILD_62_MIN_POSITIVE, build_62_equal};

const NORM_DWARF: f64 = 3.834e-20;
const NORM_GIANT: f64 = 1.304e19;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumericalError {
    NonFiniteResidual { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LmExit {
    ImproperInput,
    RelativeReduction,
    RelativeStep,
    RelativeReductionAndStep,
    Orthogonal,
    EvaluationOrIterationLimit,
    FunctionToleranceTooSmall,
    ParameterToleranceTooSmall,
    GradientToleranceTooSmall,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LmOutcome {
    pub(crate) exit: LmExit,
    pub(crate) work_diagonal: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LmEvaluation {
    Unconstrained,
    ConstrainedTrial,
}

impl LmExit {
    /// `lmdif`'s convergence tests, infos 1 to 3. The Fortran assigns `info`
    /// in sequence, so the step test overrides the reduction test and both
    /// together give 3.
    const fn converged(relative_reduction: bool, relative_step: bool) -> Option<Self> {
        match (relative_reduction, relative_step) {
            (true, true) => Some(Self::RelativeReductionAndStep),
            (false, true) => Some(Self::RelativeStep),
            (true, false) => Some(Self::RelativeReduction),
            (false, false) => None,
        }
    }

    /// `lmdif`'s stopping tests, infos 5 to 8, where each later test
    /// overrides the earlier ones: 8 over 7 over 6 over 5.
    const fn stalled(
        limit_reached: bool,
        reduction_at_precision: bool,
        step_at_precision: bool,
        gradient_at_precision: bool,
    ) -> Option<Self> {
        if gradient_at_precision {
            Some(Self::GradientToleranceTooSmall)
        } else if step_at_precision {
            Some(Self::ParameterToleranceTooSmall)
        } else if reduction_at_precision {
            Some(Self::FunctionToleranceTooSmall)
        } else if limit_reached {
            Some(Self::EvaluationOrIterationLimit)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LmOptions {
    pub(crate) function_tolerance: f64,
    pub(crate) parameter_tolerance: f64,
    pub(crate) gradient_tolerance: f64,
    pub(crate) maximum_iterations: usize,
    pub(crate) finite_difference_epsilon: f64,
    pub(crate) factor: f64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LmCounters {
    pub(crate) iterations: usize,
    pub(crate) function_evaluations: usize,
}

/// A mutable column-major matrix: entry `(row, column)` is
/// `data[row + column * leading_dimension]`, as in the Fortran arrays.
///
/// `lmpar` and `covar` read the n-by-n R from the top of the m-by-n
/// Jacobian, so the leading dimension is the Jacobian's row count, not R's
/// order.
pub(crate) struct ColMajorMut<'a> {
    data: &'a mut [f64],
    leading_dimension: usize,
}

impl<'a> ColMajorMut<'a> {
    pub(crate) const fn new(data: &'a mut [f64], leading_dimension: usize) -> Self {
        Self {
            data,
            leading_dimension,
        }
    }

    /// Every stored row of `column`.
    pub(crate) fn col(&self, column: usize) -> &[f64] {
        let start = column * self.leading_dimension;
        &self.data[start..start + self.leading_dimension]
    }

    /// Every stored row of `column`, mutably.
    pub(crate) fn col_mut(&mut self, column: usize) -> &mut [f64] {
        let start = column * self.leading_dimension;
        &mut self.data[start..start + self.leading_dimension]
    }

    /// Column `read`, and column `write` mutably; they must differ.
    pub(crate) fn col_and_col_mut(&mut self, read: usize, write: usize) -> (&[f64], &mut [f64]) {
        let rows = self.leading_dimension;
        if read < write {
            let (before, from_write) = self.data.split_at_mut(write * rows);
            (
                &before[read * rows..(read + 1) * rows],
                &mut from_write[..rows],
            )
        } else {
            let (before, from_read) = self.data.split_at_mut(read * rows);
            (
                &from_read[..rows],
                &mut before[write * rows..(write + 1) * rows],
            )
        }
    }

    /// The first `count` diagonal entries.
    pub(crate) fn diagonal(&self, count: usize) -> Vec<f64> {
        (0..count).map(|index| self[(index, index)]).collect()
    }

    fn swap_columns(&mut self, left: usize, right: usize) {
        let rows = self.leading_dimension;
        let (before, from_right) = self.data.split_at_mut(right * rows);
        before[left * rows..(left + 1) * rows].swap_with_slice(&mut from_right[..rows]);
    }
}

impl Index<(usize, usize)> for ColMajorMut<'_> {
    type Output = f64;

    fn index(&self, (row, column): (usize, usize)) -> &f64 {
        &self.data[row + column * self.leading_dimension]
    }
}

impl IndexMut<(usize, usize)> for ColMajorMut<'_> {
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut f64 {
        &mut self.data[row + column * self.leading_dimension]
    }
}

/// MINPACK `enorm.f`: the Euclidean norm, with separately scaled sums of
/// small, intermediate and large components.
pub(crate) fn euclidean_norm(values: &[f64]) -> f64 {
    let mut large_sum = 0.0;
    let mut intermediate_sum = 0.0;
    let mut small_sum = 0.0;
    let mut large_max = 0.0;
    let mut small_max = 0.0;
    let giant_threshold = NORM_GIANT / values.len() as f64;

    for value in values {
        let absolute = value.abs();
        if absolute > NORM_DWARF && absolute < giant_threshold {
            intermediate_sum = absolute.mul_add(absolute, intermediate_sum);
        } else if absolute <= NORM_DWARF {
            if absolute <= small_max {
                if !build_62_equal(absolute, 0.0) {
                    let ratio = absolute / small_max;
                    small_sum = ratio.mul_add(ratio, small_sum);
                }
            } else {
                let ratio = small_max / absolute;
                small_sum = small_sum.mul_add(ratio * ratio, 1.0);
                small_max = absolute;
            }
        } else if absolute <= large_max {
            let ratio = absolute / large_max;
            large_sum = ratio.mul_add(ratio, large_sum);
        } else {
            let ratio = large_max / absolute;
            large_sum = large_sum.mul_add(ratio * ratio, 1.0);
            large_max = absolute;
        }
    }

    if !build_62_equal(large_sum, 0.0) {
        large_max * (large_sum + (intermediate_sum / large_max) / large_max).sqrt()
    } else if build_62_equal(intermediate_sum, 0.0) && small_sum > 0.0 {
        small_max * small_sum.sqrt()
    } else if build_62_equal(small_max, 0.0) && build_62_equal(intermediate_sum.sqrt(), 0.0) {
        0.0
    } else if intermediate_sum >= small_max {
        (intermediate_sum * (small_max / intermediate_sum).mul_add(small_max * small_sum, 1.0))
            .sqrt()
    } else {
        (small_max * small_max.mul_add(small_sum, intermediate_sum / small_max)).sqrt()
    }
}

/// `out[j] = diagonal[j] * values[j]`, the scaled vectors `diag*x` and
/// `diag*p` of `lmdif` and `lmpar`.
fn scale_into(out: &mut [f64], diagonal: &[f64], values: &[f64]) {
    for ((out, &scale), &value) in out.iter_mut().zip(diagonal).zip(values) {
        *out = scale * value;
    }
}

/// `x(ipvt(j)) = w(j)`: undo the QR column pivoting.
fn unpermute_into(out: &mut [f64], permutation: &[usize], permuted: &[f64]) {
    for (&column, &value) in permutation.iter().zip(permuted) {
        out[column] = value;
    }
}

/// `sum(a[i] * b[i])`, accumulated from +0.0 in index order with the fused
/// multiply-add gfortran emits for `sum = sum + a(i)*b(i)`.
fn fused_dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (&left, &right)| left.mul_add(right, sum))
}

/// MINPACK `fdjac2.f`: the forward-difference Jacobian of the residuals,
/// column-major with one column per parameter.
pub(crate) fn finite_difference_jacobian<F, E>(
    parameters: &mut [f64],
    residuals: &[f64],
    jacobian: &mut [f64],
    epsilon_function: f64,
    callback: &mut F,
) -> Result<(), E>
where
    F: FnMut(&[f64], &mut [f64]) -> Result<(), E>,
{
    let step_scale = epsilon_function.max(BUILD_62_MACHINE_EPSILON).sqrt();
    let mut shifted_residuals = vec![0.0; residuals.len()];
    let mut jacobian = ColMajorMut::new(jacobian, residuals.len());

    for parameter_index in 0..parameters.len() {
        let original = parameters[parameter_index];
        let mut step = step_scale * original.abs();
        if build_62_equal(step, 0.0) {
            step = step_scale;
        }
        parameters[parameter_index] = original + step;
        let callback_result = callback(parameters, &mut shifted_residuals);
        parameters[parameter_index] = original;
        callback_result?;

        let column = jacobian.col_mut(parameter_index);
        for ((entry, shifted), residual) in column.iter_mut().zip(&shifted_residuals).zip(residuals)
        {
            *entry = (shifted - residual) / step;
        }
    }

    Ok(())
}

/// What `qrfac.f` returns beside the factored matrix.
pub(crate) struct QrFactors {
    /// `ipvt`: column `j` of R is column `permutation[j]` of the input.
    pub(crate) permutation: Vec<usize>,
    /// `rdiag`: the diagonal of R.
    pub(crate) r_diagonal: Vec<f64>,
    /// `acnorm`: the norms of the input columns, in input order.
    pub(crate) column_norms: Vec<f64>,
    /// `wa`: each column's norm at its last renormalization.
    reference_norms: Vec<f64>,
}

impl QrFactors {
    pub(crate) fn new(column_count: usize) -> Self {
        Self {
            permutation: vec![0; column_count],
            r_diagonal: vec![0.0; column_count],
            column_norms: vec![0.0; column_count],
            reference_norms: vec![0.0; column_count],
        }
    }
}

/// MINPACK `qrfac.f` with column pivoting, the only mode Build 62 uses:
/// Householder QR of `matrix`, whose leading dimension is its row count.
///
/// On return the strict upper triangle of `matrix` holds R's off-diagonal
/// entries, its lower trapezoid the Householder vectors, and `factors`
/// the rest. `factors` must have one slot per column.
pub(crate) fn qr_factor(matrix: &mut ColMajorMut<'_>, factors: &mut QrFactors) {
    let row_count = matrix.leading_dimension;
    let column_count = factors.permutation.len();
    let QrFactors {
        permutation,
        r_diagonal,
        column_norms,
        reference_norms,
    } = factors;

    for column in 0..column_count {
        column_norms[column] = euclidean_norm(matrix.col(column));
        permutation[column] = column;
    }
    r_diagonal.copy_from_slice(column_norms);
    reference_norms.copy_from_slice(column_norms);

    for column in 0..row_count.min(column_count) {
        // Bring the largest remaining column forward; ties keep the earliest.
        let largest = (column..column_count).fold(column, |largest, candidate| {
            if r_diagonal[candidate] > r_diagonal[largest] {
                candidate
            } else {
                largest
            }
        });
        if largest != column {
            matrix.swap_columns(column, largest);
            r_diagonal[largest] = r_diagonal[column];
            reference_norms[largest] = reference_norms[column];
            permutation.swap(column, largest);
        }

        let mut norm = euclidean_norm(&matrix.col(column)[column..]);
        if !build_62_equal(norm, 0.0) {
            let householder = &mut matrix.col_mut(column)[column..];
            if householder[0] < 0.0 {
                norm = -norm;
            }
            for value in householder.iter_mut() {
                *value /= norm;
            }
            householder[0] += 1.0;

            for remaining in column + 1..column_count {
                let (householder, target) = matrix.col_and_col_mut(column, remaining);
                let householder = &householder[column..];
                let transformation = fused_dot(householder, &target[column..]) / householder[0];
                for (value, &reflector) in target[column..].iter_mut().zip(householder) {
                    *value = (-transformation).mul_add(reflector, *value);
                }

                if !build_62_equal(r_diagonal[remaining], 0.0) {
                    let ratio = target[column] / r_diagonal[remaining];
                    r_diagonal[remaining] *= (0.0_f64.max((-ratio).mul_add(ratio, 1.0))).sqrt();
                    let rescale = r_diagonal[remaining] / reference_norms[remaining];
                    if 0.05 * (rescale * rescale) <= BUILD_62_MACHINE_EPSILON {
                        r_diagonal[remaining] = euclidean_norm(&target[column + 1..]);
                        reference_norms[remaining] = r_diagonal[remaining];
                    }
                }
            }
        }
        r_diagonal[column] = -norm;
    }
}

/// MINPACK `qrsolv.f`: solve the regularized least-squares system from a QR
/// factorization.
///
/// `regularization` is the diagonal D of `lmpar`'s scaled problem. On
/// return `solution` holds x, `s_diagonal` the diagonal of S, and `r`'s
/// strict lower triangle the rest of S; R's upper triangle and diagonal are
/// unchanged. `work` is scratch.
fn qr_solve(
    r: &mut ColMajorMut<'_>,
    permutation: &[usize],
    regularization: &[f64],
    q_transpose_b: &[f64],
    solution: &mut [f64],
    s_diagonal: &mut [f64],
    work: &mut [f64],
) {
    let order = permutation.len();

    // Copy R into the lower triangle (its transpose) and save its diagonal.
    for column in 0..order {
        for row in column..order {
            r[(row, column)] = r[(column, row)];
        }
        solution[column] = r[(column, column)];
    }
    work.copy_from_slice(q_transpose_b);

    // Eliminate D with Givens rotations, one row of D at a time.
    for (column, &permuted_column) in permutation.iter().enumerate() {
        if !build_62_equal(regularization[permuted_column], 0.0) {
            s_diagonal[column..].fill(0.0);
            s_diagonal[column] = regularization[permuted_column];
            let mut extra_qtb = 0.0;

            for row in column..order {
                if build_62_equal(s_diagonal[row], 0.0) {
                    continue;
                }
                let r_diagonal = r[(row, row)];
                let (cosine, sine) = if r_diagonal.abs() >= s_diagonal[row].abs() {
                    let tangent = s_diagonal[row] / r_diagonal;
                    let cosine = 0.5 / 0.25_f64.mul_add(tangent * tangent, 0.25).sqrt();
                    (cosine, cosine * tangent)
                } else {
                    let cotangent = r_diagonal / s_diagonal[row];
                    let sine = 0.5 / 0.25_f64.mul_add(cotangent * cotangent, 0.25).sqrt();
                    (sine * cotangent, sine)
                };

                r[(row, row)] = cosine.mul_add(r_diagonal, sine * s_diagonal[row]);
                let value = cosine.mul_add(work[row], sine * extra_qtb);
                extra_qtb = cosine.mul_add(extra_qtb, -(sine * work[row]));
                work[row] = value;

                let below = &mut r.col_mut(row)[row + 1..order];
                for (r_value, s_value) in below.iter_mut().zip(&mut s_diagonal[row + 1..]) {
                    let value = cosine.mul_add(*r_value, sine * *s_value);
                    *s_value = cosine.mul_add(*s_value, -(sine * *r_value));
                    *r_value = value;
                }
            }
        }

        s_diagonal[column] = r[(column, column)];
        r[(column, column)] = solution[column];
    }

    // Back-substitute through S, treating it as singular from its first
    // zero diagonal on.
    let nonsingular = s_diagonal
        .iter()
        .position(|&value| build_62_equal(value, 0.0))
        .unwrap_or(order);
    work[nonsingular..].fill(0.0);
    for column in (0..nonsingular).rev() {
        let below = column + 1..nonsingular;
        let sum = fused_dot(&r.col(column)[below.clone()], &work[below]);
        work[column] = (work[column] - sum) / s_diagonal[column];
    }

    unpermute_into(solution, permutation, work);
}

/// The trust-region subproblem `lmpar.f` solves: given the pivoted QR of A
/// and `Q^T b`, find the step x minimizing `|A x - b|` within
/// `|diagonal * x| <= delta`.
struct TrustRegion<'a> {
    permutation: &'a [usize],
    diagonal: &'a [f64],
    q_transpose_b: &'a [f64],
    delta: f64,
}

/// Scratch for `lm_parameter` and the `qr_solve` it calls, hoisted out of
/// `lmdif`'s loop.
struct LmParameterScratch {
    /// `sdiag`: the diagonal of S from `qr_solve`.
    s_diagonal: Vec<f64>,
    /// `wa1`: the Gauss-Newton, bound, gradient and correction vectors in
    /// turn, and the regularization `qr_solve` reads.
    work: Vec<f64>,
    /// `wa2`: `diagonal * x`, and `qr_solve`'s scratch.
    scaled_solution: Vec<f64>,
}

impl LmParameterScratch {
    fn new(order: usize) -> Self {
        Self {
            s_diagonal: vec![0.0; order],
            work: vec![0.0; order],
            scaled_solution: vec![0.0; order],
        }
    }
}

/// MINPACK `lmpar.f`: find the Levenberg-Marquardt parameter for the trust
/// region, starting from the previous `parameter`, and return it. `solution`
/// receives the step for that parameter.
fn lm_parameter(
    r: &mut ColMajorMut<'_>,
    region: &TrustRegion<'_>,
    mut parameter: f64,
    solution: &mut [f64],
    scratch: &mut LmParameterScratch,
) -> f64 {
    let TrustRegion {
        permutation,
        diagonal,
        q_transpose_b,
        delta,
    } = *region;
    let LmParameterScratch {
        s_diagonal,
        work,
        scaled_solution,
    } = scratch;
    let order = permutation.len();

    // Gauss-Newton direction, treating R as singular from its first zero
    // diagonal on.
    let nonsingular = (0..order)
        .position(|column| build_62_equal(r[(column, column)], 0.0))
        .unwrap_or(order);
    work[..nonsingular].copy_from_slice(&q_transpose_b[..nonsingular]);
    work[nonsingular..].fill(0.0);
    for column in (0..nonsingular).rev() {
        work[column] /= r[(column, column)];
        let value = work[column];
        for (entry, &r_value) in work[..column].iter_mut().zip(&r.col(column)[..column]) {
            *entry = (-r_value).mul_add(value, *entry);
        }
    }
    unpermute_into(solution, permutation, work);

    let mut iteration = 0;
    scale_into(scaled_solution, diagonal, solution);
    let mut scaled_norm = euclidean_norm(scaled_solution);
    let mut function_value = scaled_norm - delta;

    if function_value > 0.1 * delta {
        // A Newton lower bound for the zero of the function, available only
        // when R has full rank.
        let mut lower_bound = 0.0;
        if nonsingular >= order {
            for (entry, &permuted_column) in work.iter_mut().zip(permutation) {
                *entry =
                    diagonal[permuted_column] * (scaled_solution[permuted_column] / scaled_norm);
            }
            for column in 0..order {
                let sum = fused_dot(&r.col(column)[..column], &work[..column]);
                work[column] = (work[column] - sum) / r[(column, column)];
            }
            let norm = euclidean_norm(work);
            lower_bound = ((function_value / delta) / norm) / norm;
        }

        // An upper bound from the scaled gradient.
        for (column, entry) in work.iter_mut().enumerate() {
            let sum = fused_dot(&r.col(column)[..=column], &q_transpose_b[..=column]);
            *entry = sum / diagonal[permutation[column]];
        }
        let gradient_norm = euclidean_norm(work);
        let mut upper_bound = gradient_norm / delta;
        if build_62_equal(upper_bound, 0.0) {
            upper_bound = BUILD_62_MIN_POSITIVE / delta.min(0.1);
        }

        parameter = parameter.max(lower_bound);
        parameter = parameter.min(upper_bound);
        if build_62_equal(parameter, 0.0) {
            parameter = gradient_norm / scaled_norm;
        }

        loop {
            iteration += 1;
            if build_62_equal(parameter, 0.0) {
                parameter = BUILD_62_MIN_POSITIVE.max(0.001 * upper_bound);
            }
            let square_root = parameter.sqrt();
            for (entry, &scale) in work.iter_mut().zip(diagonal) {
                *entry = square_root * scale;
            }
            qr_solve(
                r,
                permutation,
                work,
                q_transpose_b,
                solution,
                s_diagonal,
                scaled_solution,
            );
            scale_into(scaled_solution, diagonal, solution);
            scaled_norm = euclidean_norm(scaled_solution);
            let previous_value = function_value;
            function_value = scaled_norm - delta;

            if function_value.abs() <= 0.1 * delta
                || (build_62_equal(lower_bound, 0.0)
                    && function_value <= previous_value
                    && previous_value < 0.0)
                || iteration == 10
            {
                break;
            }

            // Newton correction.
            for (entry, &permuted_column) in work.iter_mut().zip(permutation) {
                *entry =
                    diagonal[permuted_column] * (scaled_solution[permuted_column] / scaled_norm);
            }
            for column in 0..order {
                work[column] /= s_diagonal[column];
                let value = work[column];
                let below = &r.col(column)[column + 1..order];
                for (entry, &r_value) in work[column + 1..].iter_mut().zip(below) {
                    *entry = (-r_value).mul_add(value, *entry);
                }
            }
            let norm = euclidean_norm(work);
            let correction = ((function_value / delta) / norm) / norm;

            if function_value > 0.0 {
                lower_bound = lower_bound.max(parameter);
            }
            if function_value < 0.0 {
                upper_bound = upper_bound.min(parameter);
            }
            parameter = lower_bound.max(parameter + correction);
        }
    }

    if iteration == 0 { 0.0 } else { parameter }
}

/// MINPACK `lmdif.f`: the Levenberg-Marquardt driver with a finite-difference
/// Jacobian.
pub(crate) fn lmdif<F>(
    parameters: &mut [f64],
    residuals: &mut [f64],
    options: LmOptions,
    counters: &mut LmCounters,
    mut callback: F,
) -> Result<LmOutcome, NumericalError>
where
    F: FnMut(&[f64], &mut [f64], LmEvaluation) -> Result<(), NumericalError>,
{
    let parameter_count = parameters.len();
    let residual_count = residuals.len();
    if parameter_count == 0
        || residual_count < parameter_count
        || options.function_tolerance < 0.0
        || options.parameter_tolerance < 0.0
        || options.gradient_tolerance < 0.0
        || options.factor <= 0.0
    {
        return Ok(LmOutcome {
            exit: LmExit::ImproperInput,
            work_diagonal: None,
        });
    }

    let maximum_evaluations = options.maximum_iterations.max(200) * (parameter_count + 1);
    let starting_iterations = counters.iterations;
    let starting_evaluations = counters.function_evaluations;
    let mut jacobian_values = vec![0.0; residual_count * parameter_count];
    let mut factors = QrFactors::new(parameter_count);
    let mut lm_scratch = LmParameterScratch::new(parameter_count);
    // `diag`: the variable scaling, from the first Jacobian's column norms.
    let mut diagonal = vec![0.0; parameter_count];
    // `diag * x`, for the scaled parameter norm.
    let mut scaled_parameters = vec![0.0; parameter_count];
    // The residuals as Q^T reflects them, and their first n entries.
    let mut reflected_residuals = vec![0.0; residual_count];
    let mut q_transpose_residuals = vec![0.0; parameter_count];
    // The step p (`lmpar` returns -p), x + p, and diag * p.
    let mut step = vec![0.0; parameter_count];
    let mut trial_parameters = vec![0.0; parameter_count];
    let mut scaled_step = vec![0.0; parameter_count];
    let mut trial_residuals = vec![0.0; residual_count];
    // R * (P^T p), for the predicted reduction.
    let mut r_step = vec![0.0; parameter_count];

    callback(parameters, residuals, LmEvaluation::Unconstrained)?;
    counters.function_evaluations += 1;
    validate_residuals(residuals)?;
    let mut residual_norm = euclidean_norm(residuals);
    let mut lm_parameter_value = 0.0;
    let mut delta = 0.0;
    let mut parameter_norm = 0.0;

    loop {
        finite_difference_jacobian(
            parameters,
            residuals,
            &mut jacobian_values,
            options.finite_difference_epsilon,
            &mut |parameters, residuals| {
                callback(parameters, residuals, LmEvaluation::Unconstrained)
            },
        )?;
        counters.function_evaluations += parameter_count;

        let mut jacobian = ColMajorMut::new(&mut jacobian_values, residual_count);
        qr_factor(&mut jacobian, &mut factors);

        if counters.iterations == starting_iterations {
            for (scale, &norm) in diagonal.iter_mut().zip(&factors.column_norms) {
                *scale = if build_62_equal(norm, 0.0) { 1.0 } else { norm };
            }
            scale_into(&mut scaled_parameters, &diagonal, parameters);
            parameter_norm = euclidean_norm(&scaled_parameters);
            delta = options.factor * parameter_norm;
            if build_62_equal(delta, 0.0) {
                delta = options.factor;
            }
        }

        // Apply Q^T to the residuals, then put R's diagonal in place of the
        // Householder vectors' leading entries.
        reflected_residuals.copy_from_slice(residuals);
        for column in 0..parameter_count {
            let householder = &jacobian.col(column)[column..];
            if !build_62_equal(householder[0], 0.0) {
                let tail = &mut reflected_residuals[column..];
                let transformation = -fused_dot(householder, tail) / householder[0];
                for (value, &reflector) in tail.iter_mut().zip(householder) {
                    *value = reflector.mul_add(transformation, *value);
                }
            }
            jacobian[(column, column)] = factors.r_diagonal[column];
            q_transpose_residuals[column] = reflected_residuals[column];
        }

        // The scaled gradient's largest cosine with the residuals.
        let mut gradient_norm: f64 = 0.0;
        if !build_62_equal(residual_norm, 0.0) {
            for (column, &permuted_column) in factors.permutation.iter().enumerate() {
                let column_norm = factors.column_norms[permuted_column];
                if !build_62_equal(column_norm, 0.0) {
                    let sum = jacobian.col(column)[..=column]
                        .iter()
                        .zip(&q_transpose_residuals[..=column])
                        .fold(0.0, |sum, (&r_value, &qtf)| {
                            r_value.mul_add(qtf / residual_norm, sum)
                        });
                    gradient_norm = gradient_norm.max((sum / column_norm).abs());
                }
            }
        }
        if gradient_norm <= options.gradient_tolerance {
            return Ok(finished(LmExit::Orthogonal, &jacobian, parameter_count));
        }

        for (scale, &norm) in diagonal.iter_mut().zip(&factors.column_norms) {
            *scale = scale.max(norm);
        }

        loop {
            lm_parameter_value = lm_parameter(
                &mut jacobian,
                &TrustRegion {
                    permutation: &factors.permutation,
                    diagonal: &diagonal,
                    q_transpose_b: &q_transpose_residuals,
                    delta,
                },
                lm_parameter_value,
                &mut step,
                &mut lm_scratch,
            );

            for value in &mut step {
                *value = -*value;
            }
            for ((trial, &parameter), &step) in
                trial_parameters.iter_mut().zip(&*parameters).zip(&step)
            {
                *trial = parameter + step;
            }
            scale_into(&mut scaled_step, &diagonal, &step);
            let step_norm = euclidean_norm(&scaled_step);
            if counters.iterations == starting_iterations {
                delta = delta.min(step_norm);
            }

            callback(
                &trial_parameters,
                &mut trial_residuals,
                LmEvaluation::ConstrainedTrial,
            )?;
            counters.function_evaluations += 1;
            validate_residuals(&trial_residuals)?;
            let trial_norm = euclidean_norm(&trial_residuals);

            let actual_reduction = if 0.1 * trial_norm < residual_norm {
                let ratio = trial_norm / residual_norm;
                (-ratio).mul_add(ratio, 1.0)
            } else {
                -1.0
            };

            r_step.fill(0.0);
            for (column, &permuted_column) in factors.permutation.iter().enumerate() {
                let value = step[permuted_column];
                for (entry, &r_value) in r_step[..=column]
                    .iter_mut()
                    .zip(&jacobian.col(column)[..=column])
                {
                    *entry = r_value.mul_add(value, *entry);
                }
            }
            let first = euclidean_norm(&r_step) / residual_norm;
            let second = (lm_parameter_value.sqrt() * step_norm) / residual_norm;
            // `lmdif.f` reuses separately rounded squares; do not fuse the
            // first square into either following sum.
            let first_square = first * first;
            let second_square = second * second;
            let predicted_reduction = first_square + second_square / 0.5;
            let directional_derivative = -(first_square + second_square);
            let ratio = if build_62_equal(predicted_reduction, 0.0) {
                0.0
            } else {
                actual_reduction / predicted_reduction
            };

            // Update the trust region and the Levenberg-Marquardt parameter.
            if ratio <= 0.25 {
                let mut scale = if actual_reduction >= 0.0 {
                    0.5
                } else {
                    0.5 * directional_derivative
                        / 0.5_f64.mul_add(actual_reduction, directional_derivative)
                };
                if 0.1 * trial_norm >= residual_norm || scale < 0.1 {
                    scale = 0.1;
                }
                delta = scale * delta.min(step_norm / 0.1);
                lm_parameter_value /= scale;
            } else if build_62_equal(lm_parameter_value, 0.0) || ratio >= 0.75 {
                delta = step_norm / 0.5;
                lm_parameter_value *= 0.5;
            }

            let accepted = ratio >= 0.0001;
            if accepted {
                parameters.copy_from_slice(&trial_parameters);
                scale_into(&mut scaled_parameters, &diagonal, parameters);
                residuals.copy_from_slice(&trial_residuals);
                parameter_norm = euclidean_norm(&scaled_parameters);
                residual_norm = trial_norm;
                counters.iterations += 1;
            }

            let reduction_within = |tolerance: f64| {
                actual_reduction.abs() <= tolerance
                    && predicted_reduction <= tolerance
                    && 0.5 * ratio <= 1.0
            };
            if let Some(exit) = LmExit::converged(
                reduction_within(options.function_tolerance),
                delta <= options.parameter_tolerance * parameter_norm,
            ) {
                return Ok(finished(exit, &jacobian, parameter_count));
            }

            let limit_reached = (options.maximum_iterations > 0
                && counters.iterations >= options.maximum_iterations)
                || counters.function_evaluations - starting_evaluations >= maximum_evaluations;
            if let Some(exit) = LmExit::stalled(
                limit_reached,
                reduction_within(BUILD_62_MACHINE_EPSILON),
                delta <= BUILD_62_MACHINE_EPSILON * parameter_norm,
                gradient_norm <= BUILD_62_MACHINE_EPSILON,
            ) {
                return Ok(finished(exit, &jacobian, parameter_count));
            }

            if accepted {
                break;
            }
        }
    }
}

/// An `lmdif` exit after the first Jacobian, which also returns R's
/// diagonal as Build 62 leaves it in the work array.
fn finished(exit: LmExit, jacobian: &ColMajorMut<'_>, parameter_count: usize) -> LmOutcome {
    LmOutcome {
        exit,
        work_diagonal: Some(jacobian.diagonal(parameter_count)),
    }
}

fn validate_residuals(residuals: &[f64]) -> Result<(), NumericalError> {
    if let Some(index) = residuals.iter().position(|value| !value.is_finite()) {
        Err(NumericalError::NonFiniteResidual { index })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
