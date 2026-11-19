use thiserror::Error;

use crate::numerics::RowMajorMatrix;
use crate::numerics::blas1::dot_strided;
use crate::numerics::packed::{
    checked_packed_len, packed_backward_solve, packed_cholesky, packed_column, packed_diagonal,
    packed_forward_solve, packed_len, packed_rank_product,
};
use crate::numerics::polynomial::{positive_lag_product, rational_expand};
use crate::regarima::ParameterKind;

use super::model::{ExpandedModel, OperatorSpec};
use super::operators::polynomials;
use super::roots::{RootLocation, root_location};

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ExactFilterWorkspace {
    ma_initial_factor: Vec<f64>,
    ar_initial_factor: Vec<f64>,
    initial_cross_filter: Vec<f64>,
}

impl ExactFilterWorkspace {
    pub(crate) fn restore_ma_initial_conditions(&mut self, saved: &Self) {
        // ssprep/restor save and restore Chlgpg, but not Chlvwp or Matd.
        // fcstxy's armafl(Linit=F) must reuse that mixed workspace.
        self.ma_initial_factor.clone_from(&saved.ma_initial_factor);
    }

    fn clear(&mut self) {
        self.ma_initial_factor.clear();
        self.ar_initial_factor.clear();
        self.initial_cross_filter.clear();
    }

    /// Whether the saved factors have the sizes a reusing call with these
    /// lags reads. The AR factor and cross filter matter only with exact AR
    /// initial rows.
    fn fits(&self, lags: FilterLags, keep_ar_initial: bool) -> bool {
        Some(self.ma_initial_factor.len()) == checked_packed_len(lags.ma)
            && (!keep_ar_initial
                || (Some(self.ar_initial_factor.len()) == checked_packed_len(lags.ar)
                    && (lags.ar == 0
                        || lags.ma == 0
                        || self.initial_cross_filter.len() == lags.exact_ma_rows * lags.ar)))
    }
}

/// The operator spans and row counts one filter call's initial-condition
/// routines share.
#[derive(Debug, Clone, Copy)]
struct FilterLags {
    /// The AR span: the number of exact AR initial rows.
    ar: usize,
    /// The MA span: the number of exact MA initial rows.
    ma: usize,
    /// Rows left after differencing and the AR filter.
    ar_filtered_rows: usize,
    /// `ar_filtered_rows + ma`: the rows the exact MA filter returns.
    exact_ma_rows: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExactFilterResult {
    matrix: RowMajorMatrix,
    effective_observations: usize,
    log_determinant_correction: f64,
}

impl ExactFilterResult {
    /// The filtered matrix: any exact AR initial rows, the exact MA initial
    /// rows, then the filtered observations. Its width is the input's.
    pub(crate) const fn matrix(&self) -> &RowMajorMatrix {
        &self.matrix
    }

    pub(crate) fn into_matrix(self) -> RowMajorMatrix {
        self.matrix
    }

    pub(crate) const fn effective_observations(&self) -> usize {
        self.effective_observations
    }

    pub(crate) const fn log_determinant_correction(&self) -> f64 {
        self.log_determinant_correction
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ExactFilterError {
    /// Reused initial-condition factors whose sizes do not fit this call's
    /// lags and rows. [`RowMajorMatrix`] already guarantees the input's
    /// shape.
    #[error("saved initial-condition factors do not match the filter dimensions")]
    InvalidDimensions,
    #[error("expected {expected} ARIMA parameters but received {actual}")]
    InvalidParameterCount { expected: usize, actual: usize },
    /// Counts are post-differencing observations, as the public
    /// `InsufficientEffectiveObservations` reports them.
    #[error("exact filtering needs {minimum} differenced observations; received {actual}")]
    InsufficientObservations { actual: usize, minimum: usize },
    #[error("fitted {kind:?} roots are invalid")]
    InvalidRoots { kind: ParameterKind },
    #[error("MA initial-condition covariance is singular at pivot {pivot}")]
    SingularMaInitialCovariance { pivot: usize },
    #[error("AR initial-condition covariance is singular at pivot {pivot}")]
    SingularArInitialCovariance { pivot: usize },
    #[error("the Euclid covariance recursion rejected the AR polynomial")]
    InvalidArCovariance,
}

/// Controls only AR initial rows; MA initial conditions remain exact in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArInitialization {
    Exact,
    Conditional,
}

/// Whether the filter factors the initial-condition covariances afresh or
/// filters with the factors the workspace kept (`armafl.f`'s `Linit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InitialConditions {
    Compute,
    Reuse,
}

/// Whether the filter rejects parameters whose AR or MA roots are not
/// outside the unit circle before filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootValidation {
    Validate,
    Skip,
}

/// The three independent switches of one [`exact_filter`] call.
///
/// A call names the options of a Build 62 routine ([`Self::FORECAST_TAIL`]
/// for `fcstxy.f`, [`Self::OUTLIER_TEST`] for `idotlr.f`) or uses
/// [`Self::computed`], which always factors afresh: saved factors are valid
/// only for the fit that saved them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FilterOptions {
    ar_initialization: ArInitialization,
    initial_conditions: InitialConditions,
    roots: RootValidation,
}

impl FilterOptions {
    /// `fcstxy.f` reads only the last q residuals, which the initial p AR
    /// rows that `armafl.f` builds from its saved `Matd`/`Chlvwp` do not
    /// feed. Skipping those rows lets a model span shorter than the series
    /// forecast without a longer D matrix or a second MA initial-condition
    /// factor. `fcstxy.f` passes `Lckrts` false, so an estimate that stopped
    /// just outside the invertible region still forecasts.
    pub(crate) const FORECAST_TAIL: Self = Self {
        ar_initialization: ArInitialization::Conditional,
        initial_conditions: InitialConditions::Reuse,
        roots: RootValidation::Skip,
    };

    /// `idotlr.f`: the saved exact initial conditions, without a roots check.
    pub(crate) const OUTLIER_TEST: Self = Self {
        ar_initialization: ArInitialization::Exact,
        initial_conditions: InitialConditions::Reuse,
        roots: RootValidation::Skip,
    };

    /// Fresh initial conditions with the given AR rows and roots check.
    pub(crate) const fn computed(
        ar_initialization: ArInitialization,
        roots: RootValidation,
    ) -> Self {
        Self {
            ar_initialization,
            initial_conditions: InitialConditions::Compute,
            roots,
        }
    }
}

/// Build 62 `armafl.f`: the exact (or conditional) ARMA filter of the
/// differenced matrix.
///
/// Takes the matrix by value so that its buffer holds the result.
pub(crate) fn exact_filter(
    matrix: RowMajorMatrix,
    model: &ExpandedModel,
    parameters: &[f64],
    workspace: &mut ExactFilterWorkspace,
    options: FilterOptions,
) -> Result<ExactFilterResult, ExactFilterError> {
    if parameters.len() != model.parameter_slots().len() {
        return Err(ExactFilterError::InvalidParameterCount {
            expected: model.parameter_slots().len(),
            actual: parameters.len(),
        });
    }

    let keep_ar_initial = options.ar_initialization == ArInitialization::Exact;
    let initialize = options.initial_conditions == InitialConditions::Compute;
    let validate_parameter_roots = options.roots == RootValidation::Validate;
    if validate_parameter_roots && keep_ar_initial {
        validate_roots(model.ar_operators(), parameters, ParameterKind::Ar)?;
    }
    if validate_parameter_roots {
        validate_roots(model.ma_operators(), parameters, ParameterKind::Ma)?;
    }

    let row_count = matrix.row_count();
    let column_count = matrix.column_count();
    let ar_lag = model.ar_span();
    let differencing_loss = model.differencing_loss();
    let ma_lag = model.ma_span();
    // Every guard below fails exactly when the differenced count is under
    // this minimum, so each reports the same counts.
    let insufficient = ExactFilterError::InsufficientObservations {
        actual: row_count.saturating_sub(differencing_loss),
        minimum: if keep_ar_initial {
            ar_lag.max(1)
        } else {
            ar_lag.saturating_add(ma_lag.max(1))
        },
    };
    let differenced_rows = row_count
        .checked_sub(differencing_loss)
        .ok_or(insufficient)?;
    let ar_filtered_rows = differenced_rows.checked_sub(ar_lag).ok_or(insufficient)?;
    let effective_observations = if keep_ar_initial {
        differenced_rows
    } else {
        ar_filtered_rows
    };
    if effective_observations == 0 {
        return Err(insufficient);
    }
    if !keep_ar_initial && effective_observations < ma_lag {
        return Err(insufficient);
    }
    let lags = FilterLags {
        ar: ar_lag,
        ma: ma_lag,
        ar_filtered_rows,
        exact_ma_rows: ar_filtered_rows.checked_add(ma_lag).ok_or(insufficient)?,
    };

    let log_determinant_correction = if initialize {
        workspace.clear();
        let ma_correction = initialize_ma_covariance(
            lags,
            model.ma_operators(),
            parameters,
            &mut workspace.ma_initial_factor,
        )?;
        if keep_ar_initial && ar_lag > 0 {
            ma_correction + initialize_ar_covariance(lags, model, parameters, workspace)?
        } else {
            ma_correction
        }
    } else if workspace.fits(lags, keep_ar_initial) {
        0.0
    } else {
        return Err(ExactFilterError::InvalidDimensions);
    };

    let mut filtered_tail = matrix.into_values();
    filter_operators(
        &mut filtered_tail,
        column_count,
        model.differencing_operators(),
        parameters,
    );

    let initial_element_count = if keep_ar_initial {
        ar_lag * column_count
    } else {
        0
    };
    let mut initial_values = filtered_tail[..initial_element_count].to_vec();
    filter_operators(
        &mut filtered_tail,
        column_count,
        model.ar_operators(),
        parameters,
    );
    exact_ma_filter(
        &mut filtered_tail,
        column_count,
        ma_lag,
        model.ma_operators(),
        parameters,
        &workspace.ma_initial_factor,
    );

    if keep_ar_initial && ar_lag > 0 && ma_lag > 0 {
        for (initial_row, row) in initial_values.chunks_exact_mut(column_count).enumerate() {
            for (column, value) in row.iter_mut().enumerate() {
                // Column `initial_row` of the exact_ma_rows x ar_lag cross
                // filter against column `column` of the filtered tail.
                *value -= dot_strided(
                    workspace.initial_cross_filter[initial_row..]
                        .iter()
                        .step_by(ar_lag),
                    filtered_tail[column..].iter().step_by(column_count),
                );
            }
        }
    }

    if keep_ar_initial && ar_lag > 0 {
        packed_forward_solve(
            &workspace.ar_initial_factor,
            ar_lag,
            column_count,
            &mut initial_values,
        );
    }

    let values = if initial_values.is_empty() {
        filtered_tail
    } else {
        initial_values.extend_from_slice(&filtered_tail);
        initial_values
    };
    Ok(ExactFilterResult {
        matrix: RowMajorMatrix::new(values, column_count),
        effective_observations,
        log_determinant_correction,
    })
}

/// `armafl.f`'s check that the MA is invertible and the exactly filtered AR
/// is stationary.
fn validate_roots(
    operators: &[OperatorSpec],
    parameters: &[f64],
    kind: ParameterKind,
) -> Result<(), ExactFilterError> {
    for operator in operators {
        if root_location(operator.coefficients(parameters)) != RootLocation::Outside {
            return Err(ExactFilterError::InvalidRoots { kind });
        }
    }
    Ok(())
}

/// Build 62 `intgpg.f`: factor G'G from the MA pi weights and return its log
/// determinant.
fn initialize_ma_covariance(
    lags: FilterLags,
    ma_operators: &[OperatorSpec],
    parameters: &[f64],
    factor: &mut Vec<f64>,
) -> Result<f64, ExactFilterError> {
    let ma_lag = lags.ma;
    if ma_lag == 0 {
        return Ok(0.0);
    }

    let operators = polynomials(ma_operators, parameters);
    let pi_weights = rational_expand(&[1.0], lags.exact_ma_rows, &operators);
    let mut squared_weights = pi_weights.clone();
    negative_lag_expand(&mut squared_weights, 1, ma_operators, parameters);
    // The pi weights counted back from the last exact MA row.
    let from_end = |offset: usize| pi_weights[lags.exact_ma_rows - offset];

    // Column by column: row 0 is the squared weight, and each later entry
    // updates the one up and to the left, in the previous column.
    debug_assert!(factor.is_empty());
    factor.reserve(packed_len(ma_lag));
    for (column, &squared_weight) in squared_weights[..ma_lag].iter().enumerate() {
        factor.push(squared_weight);
        for row in 1..=column {
            let up_left = factor[packed_column(column - 1)][row - 1];
            factor.push((-from_end(row)).mul_add(from_end(column), up_left));
        }
    }

    // intgpg.f: any nonzero dppfa info is a failure.
    packed_cholesky(factor, ma_lag).map_err(|failure| {
        ExactFilterError::SingularMaInitialCovariance {
            pivot: failure.column,
        }
    })?;
    Ok(log_determinant(factor, ma_lag))
}

/// `armafl.f`'s `Linit` block: the AR initial-value covariance, its Cholesky
/// factor and log determinant. Reads the workspace's MA initial factor and
/// fills its AR initial factor and cross filter.
fn initialize_ar_covariance(
    lags: FilterLags,
    model: &ExpandedModel,
    parameters: &[f64],
    workspace: &mut ExactFilterWorkspace,
) -> Result<f64, ExactFilterError> {
    let FilterLags {
        ar: ar_lag,
        ma: ma_lag,
        ..
    } = lags;
    let ma_operators = polynomials(model.ma_operators(), parameters);
    let ar_operators = polynomials(model.ar_operators(), parameters);
    let full_ma = positive_lag_product(&[1.0], ma_lag + 1, &ma_operators);
    let psi_weights = rational_expand(&full_ma, ma_lag + 1, &ar_operators);
    let mut autocovariance = ma_autocovariances(&full_ma);
    let full_ar = positive_lag_product(&[1.0], ar_lag + 1, &ar_operators);
    autocovariance.resize(ar_lag.max(ma_lag) + 1, 0.0);
    euclid_covariance(&full_ar, ma_lag, &mut autocovariance)?;
    expand_ar_covariance(&full_ar, &mut autocovariance);
    autocovariance[0] *= 2.0;

    let factor = &mut workspace.ar_initial_factor;
    factor.resize(packed_len(ar_lag), 0.0);
    if ma_lag > 0 {
        let cross_filter = &mut workspace.initial_cross_filter;
        cross_filter.resize(lags.ar_filtered_rows * ar_lag, 0.0);
        for row in 0..ma_lag {
            // sum over k of full_ma[row + 1 + k] * psi_weights[k], from the
            // highest k down.
            let value = full_ma[row + 1..]
                .iter()
                .zip(&psi_weights)
                .rev()
                .fold(0.0, |value, (&ma, &psi)| ma.mul_add(psi, value));
            // The diagonal that ends at (row, last column), up to the left.
            for (diagonal_row, column) in (0..=row).rev().zip((0..ar_lag).rev()) {
                cross_filter[diagonal_row * ar_lag + column] = value;
            }
        }

        exact_ma_filter(
            cross_filter,
            ar_lag,
            ma_lag,
            model.ma_operators(),
            parameters,
            &workspace.ma_initial_factor,
        );
        debug_assert_eq!(cross_filter.len(), lags.exact_ma_rows * ar_lag);
        packed_rank_product(cross_filter, lags.exact_ma_rows, ar_lag, factor);
        for column in 0..ar_lag {
            for (row, entry) in factor[packed_column(column)].iter_mut().enumerate() {
                *entry = autocovariance[column - row] - *entry;
            }
        }
    } else {
        for column in 0..ar_lag {
            for (row, entry) in factor[packed_column(column)].iter_mut().enumerate() {
                *entry = autocovariance[column - row];
            }
        }
    }

    // armafl.f: any nonzero dppfa info is a failure.
    packed_cholesky(factor, ar_lag).map_err(|failure| {
        ExactFilterError::SingularArInitialCovariance {
            pivot: failure.column,
        }
    })?;
    Ok(log_determinant(factor, ar_lag))
}

/// Build 62 `uconv.f`: the autocovariances of an MA polynomial.
///
/// `uconv.f` overwrites lag `k` in place, but each lag reads only higher
/// lags, which are still the input weights when it runs.
fn ma_autocovariances(full_ma: &[f64]) -> Vec<f64> {
    (0..full_ma.len())
        .map(|lag| {
            full_ma[1..]
                .iter()
                .zip(&full_ma[lag + 1..])
                .fold(full_ma[lag], |sum, (&weight, &lagged)| {
                    weight.mul_add(lagged, sum)
                })
        })
        .collect()
}

/// Build 62 `euclid.f`: solve the ARMA autocovariance equation by the Euclid
/// algorithm. `covariance` enters holding the MA autocovariances, padded with
/// zeros to the larger of the two spans.
fn euclid_covariance(
    full_ar: &[f64],
    ma_lag: usize,
    covariance: &mut [f64],
) -> Result<(), ExactFilterError> {
    let ar_lag = full_ar.len() - 1;
    let maximum_lag = covariance.len() - 1;
    let mut reduced_ar = vec![0.0; ar_lag + 1];
    let mut inverse_scale = vec![0.0; ar_lag + 1];
    reduced_ar[1..].copy_from_slice(&full_ar[1..]);

    for degree in (1..=maximum_lag).rev() {
        if degree <= ar_lag {
            let reflection = reduced_ar[degree];
            if reflection.abs() > 1.0 {
                return Err(ExactFilterError::InvalidArCovariance);
            }
            let scale = 1.0 / (-reflection).mul_add(reflection, 1.0);
            inverse_scale[degree] = scale;
            for first in 1..=degree / 2 {
                let last = degree - first;
                let first_value = reduced_ar[first];
                let last_value = reduced_ar[last];
                reduced_ar[first] = (-last_value).mul_add(reflection, first_value) * scale;
                reduced_ar[last] = (-first_value).mul_add(reflection, last_value) * scale;
            }
        }

        if degree > ma_lag {
            covariance[degree] = 0.0;
        } else {
            // Update lags degree - 1, degree - 2, ... with the reduced AR
            // coefficients 1, 2, ..., stopping at the AR span.
            let (lower, upper) = covariance.split_at_mut(degree);
            let top = upper[0];
            let limit = (degree - 1).min(ar_lag);
            for (&coefficient, entry) in reduced_ar[1..=limit].iter().zip(lower.iter_mut().rev()) {
                *entry = (-coefficient).mul_add(top, *entry);
            }
        }
    }

    covariance[0] /= 2.0;
    for (degree, (&coefficient, &scale)) in
        reduced_ar.iter().zip(&inverse_scale).enumerate().skip(1)
    {
        for first in 0..=degree / 2 {
            let last = degree - first;
            let first_value = covariance[first];
            let last_value = covariance[last];
            covariance[first] = (-coefficient).mul_add(last_value, first_value) * scale;
            covariance[last] = (-coefficient).mul_add(first_value, last_value) * scale;
        }
    }
    Ok(())
}

/// Build 62 `xpand.f`: extend the autocovariances through the AR lags, in
/// place, so each lag reads the lower lags already extended.
fn expand_ar_covariance(full_ar: &[f64], covariance: &mut [f64]) {
    // Lag 0 has no AR terms and stays as it is.
    for index in 1..full_ar.len() {
        let (lower, current) = covariance.split_at_mut(index);
        current[0] = full_ar[1..=index]
            .iter()
            .zip(lower.iter().rev())
            .fold(current[0], |sum, (&coefficient, &lagged)| {
                (-coefficient).mul_add(lagged, sum)
            });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum DifferenceError {
    #[error("differencing leaves no observations")]
    InsufficientObservations,
    #[error("differencing input contains a nonfinite value")]
    NonFiniteInput,
}

/// Applies the differencing operators to one series.
pub(crate) fn difference_series(
    series: &[f64],
    model: &ExpandedModel,
) -> Result<Vec<f64>, DifferenceError> {
    if series.iter().any(|value| !value.is_finite()) {
        return Err(DifferenceError::NonFiniteInput);
    }
    difference_unscreened(series, model)
}

/// [`difference_series`] without the finite screen. `rgarma.f`'s check for a
/// series annihilated by differencing reads whatever values differencing
/// leaves, so it must not stop at a nonfinite input.
pub(crate) fn difference_unscreened(
    series: &[f64],
    model: &ExpandedModel,
) -> Result<Vec<f64>, DifferenceError> {
    if series.len() <= model.differencing_loss() {
        return Err(DifferenceError::InsufficientObservations);
    }
    let mut values = series.to_vec();
    filter_operators(&mut values, 1, model.differencing_operators(), &[]);
    Ok(values)
}

/// Build 62 `arflt.f`, once per operator.
///
/// Each operator drops its span of leading rows, so `values` must hold more
/// than the operators' summed span. [`difference_unscreened`] checks the
/// differencing loss and [`exact_filter`] checks `differenced_rows` and
/// `ar_filtered_rows` before calling this.
fn filter_operators(
    values: &mut Vec<f64>,
    column_count: usize,
    operators: &[OperatorSpec],
    parameters: &[f64],
) {
    let mut element_count = values.len();
    for operator in operators {
        let maximum_lag = operator.span() * column_count;
        element_count -= maximum_lag;
        for index in 0..element_count {
            let offset = index + maximum_lag;
            let mut value = values[offset];
            for (coefficient, lag) in operator
                .coefficients(parameters)
                .iter()
                .zip(operator.lags())
            {
                value = (-coefficient).mul_add(values[offset - lag * column_count], value);
            }
            values[index] = value;
        }
    }
    values.truncate(element_count);
}

/// Build 62 `exctma.f`: the exact MA filter, prepending the initial values
/// `-(G'G)^-1 G'Hw`.
fn exact_ma_filter(
    values: &mut Vec<f64>,
    column_count: usize,
    ma_lag: usize,
    ma_operators: &[OperatorSpec],
    parameters: &[f64],
    ma_initial_factor: &[f64],
) {
    if ma_lag == 0 {
        return;
    }

    let input_element_count = values.len();
    let initial_element_count = ma_lag * column_count;
    values.resize(input_element_count + initial_element_count, 0.0);
    values.copy_within(..input_element_count, initial_element_count);

    // Zero initial rows above the observations. The forward and backward
    // expansions leave G'Hw in the initial rows, which the solves turn into
    // (G'G)^-1 G'Hw.
    let mut projection = vec![0.0; values.len()];
    projection[initial_element_count..].copy_from_slice(&values[initial_element_count..]);
    positive_lag_expand_in_place(&mut projection, column_count, ma_operators, parameters);
    negative_lag_expand(&mut projection, column_count, ma_operators, parameters);
    let initial = &mut projection[..initial_element_count];
    packed_forward_solve(ma_initial_factor, ma_lag, column_count, initial);
    packed_backward_solve(ma_initial_factor, ma_lag, column_count, initial);
    for (value, &solved) in values.iter_mut().zip(&*initial) {
        *value = -solved;
    }
    positive_lag_expand_in_place(values, column_count, ma_operators, parameters);
}

/// Build 62 `ratpos.f` over each column of a row-major matrix, in place.
fn positive_lag_expand_in_place(
    values: &mut [f64],
    column_count: usize,
    operators: &[OperatorSpec],
    parameters: &[f64],
) {
    let row_count = values.len() / column_count;
    debug_assert_eq!(values.len() % column_count, 0);
    for operator in operators {
        let lags = operator.lags();
        let coefficients = operator.coefficients(parameters);
        let first_row = if lags.len() > 1 { lags[0] } else { 0 };
        let mut active = 0;
        // Each column is its own recurrence. A lag is applied to the whole
        // row at once, which keeps each column's fused operations and lag
        // order; the lags that reach back past the first row are found once
        // per row.
        for row in first_row..row_count {
            while active < lags.len() && lags[active] <= row {
                active += 1;
            }
            let (previous, current) = values.split_at_mut(row * column_count);
            for (&coefficient, &lag) in coefficients[..active].iter().zip(&lags[..active]) {
                if coefficient.abs() > 1.0e-150 {
                    let lagged = &previous[(row - lag) * column_count..][..column_count];
                    for (sum, &value) in current[..column_count].iter_mut().zip(lagged) {
                        if value.abs() > 1.0e-150 {
                            *sum = coefficient.mul_add(value, *sum);
                        }
                    }
                }
            }
        }
    }
}

/// Build 62 `ratneg.f` over each column of a row-major matrix, in place.
fn negative_lag_expand(
    values: &mut [f64],
    column_count: usize,
    operators: &[OperatorSpec],
    parameters: &[f64],
) {
    // The narrow widths get a fixed-size row buffer, which saves a memcpy
    // call per row. The arithmetic of each column is the same at any width.
    match column_count {
        1 => negative_lag_expand_columns::<1>(values, column_count, operators, parameters),
        2 => negative_lag_expand_columns::<2>(values, column_count, operators, parameters),
        3 => negative_lag_expand_columns::<3>(values, column_count, operators, parameters),
        _ => negative_lag_expand_columns::<0>(values, column_count, operators, parameters),
    }
}

fn negative_lag_expand_columns<const WIDTH: usize>(
    values: &mut [f64],
    column_count: usize,
    operators: &[OperatorSpec],
    parameters: &[f64],
) {
    let column_count = if WIDTH == 0 { column_count } else { WIDTH };
    let row_count = values.len() / column_count;
    debug_assert_eq!(values.len() % column_count, 0);
    let mut local = [0.0; WIDTH];
    let mut wide = Vec::new();
    let sums = if WIDTH != 0 {
        &mut local[..]
    } else {
        wide.resize(column_count, 0.0);
        &mut wide[..]
    };
    for operator in operators {
        let lags = operator.lags();
        let coefficients = operator.coefficients(parameters);
        let mut active = 0;
        for row in (0..row_count - lags[0]).rev() {
            while active < lags.len() && lags[active] < row_count - row {
                active += 1;
            }
            let offset = row * column_count;
            sums.copy_from_slice(&values[offset..offset + column_count]);
            // `exctma.f` sums the negative lags of each column in this order.
            // The row is written only when every lag is in: a sum that is
            // exactly zero or NaN leaves the old element, and a tiny nonzero
            // one stores zero.
            for (&coefficient, &lag) in coefficients[..active].iter().zip(&lags[..active]) {
                let lagged = &values[(row + lag) * column_count..][..column_count];
                for (sum, &value) in sums.iter_mut().zip(lagged) {
                    *sum = coefficient.mul_add(value, *sum);
                }
            }
            for (value, &sum) in values[offset..offset + column_count].iter_mut().zip(&*sums) {
                if sum.abs() > 1.0e-300 {
                    *value = sum;
                } else if sum.abs() > 0.0 {
                    *value = 0.0;
                }
            }
        }
    }
}

/// Build 62 `logdet.f`: the log determinant from a packed Cholesky factor.
fn log_determinant(factor: &[f64], order: usize) -> f64 {
    (0..order).fold(0.0, |total, column| {
        total + 2.0 * factor[packed_diagonal(column)].ln()
    })
}

#[cfg(test)]
mod tests;
