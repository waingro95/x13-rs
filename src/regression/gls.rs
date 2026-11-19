//! The least-squares solve on a filtered `[X:y]` matrix (`olsreg.f`), and the
//! residuals, standard errors and covariance read from it.

use crate::numerics::blas1::dot;
use crate::numerics::packed::{
    packed_backward_solve, packed_cholesky, packed_column, packed_diagonal, packed_inverse,
    packed_len, packed_rank_product,
};

use super::design::RegressionMatrix;
use crate::Error;

/// The regressors are linearly dependent: the Cholesky factorization
/// stopped at zero-based column `pivot`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SingularRegression {
    pub(crate) pivot: usize,
}

/// A solved regression: coefficients, residuals, the packed Cholesky factor
/// of `[X:y]'[X:y]` and the packed inverse of `X'X`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionSolution {
    coefficients: Vec<f64>,
    residuals: Vec<f64>,
    packed_factor: Vec<f64>,
    normal_inverse: Vec<f64>,
    residual_sum_squares: f64,
}

/// What the outlier t-tests (`ttest.f`) read from a fit: the filtered
/// `[X:y]` matrix and its packed Cholesky factor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OutlierTestState<'a> {
    filtered_matrix: &'a RegressionMatrix,
    packed_factor: &'a [f64],
}

impl<'a> OutlierTestState<'a> {
    /// `packed_factor` packs one column per column of `filtered_matrix`.
    pub(crate) fn new(filtered_matrix: &'a RegressionMatrix, packed_factor: &'a [f64]) -> Self {
        debug_assert_eq!(
            packed_factor.len(),
            packed_len(filtered_matrix.column_count())
        );
        Self {
            filtered_matrix,
            packed_factor,
        }
    }

    /// Filtered `[X:y]`, with the response in the final zero-based column.
    pub(crate) const fn filtered_matrix(self) -> &'a RegressionMatrix {
        self.filtered_matrix
    }

    /// Packed Cholesky factor of `[X:y]'[X:y]`.
    ///
    /// Fortran's one-based element `k` is Rust offset `k - 1`; each packed
    /// column contains its preceding rows followed by its diagonal.
    pub(crate) const fn packed_factor(self) -> &'a [f64] {
        self.packed_factor
    }
}

impl RegressionSolution {
    pub(crate) fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub(crate) fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    pub(crate) fn packed_factor(&self) -> &[f64] {
        &self.packed_factor
    }

    pub(crate) fn normal_inverse(&self) -> &[f64] {
        &self.normal_inverse
    }

    pub(crate) const fn residual_sum_squares(&self) -> f64 {
        self.residual_sum_squares
    }

    pub(crate) fn standard_errors(&self, innovation_variance: f64) -> Vec<f64> {
        regression_standard_errors(
            &self.normal_inverse,
            innovation_variance,
            self.coefficients.len(),
        )
    }
}

/// The coefficient standard errors: the root of each diagonal of the packed
/// `normal_inverse`, times the root of the innovation variance.
pub(crate) fn regression_standard_errors(
    normal_inverse: &[f64],
    innovation_variance: f64,
    estimated_columns: usize,
) -> Vec<f64> {
    let root_variance = innovation_variance.sqrt();
    (0..estimated_columns)
        .map(|column| normal_inverse[packed_diagonal(column)].sqrt() * root_variance)
        .collect()
}

/// Build 62 `olsreg.f`: solves the regression of the last column of `matrix`
/// on the others through the packed Cholesky factor of `[X:y]'[X:y]`.
pub(crate) fn solve_filtered_regression(
    matrix: &RegressionMatrix,
) -> Result<RegressionSolution, SingularRegression> {
    let column_count = matrix.column_count();
    let regression_count = column_count - 1;
    let mut packed_factor = vec![0.0; packed_len(column_count)];
    packed_rank_product(
        matrix.values(),
        matrix.row_count(),
        column_count,
        &mut packed_factor,
    );

    // olsreg.f accepts `Info.eq.Ncxy`: a stop at the augmented response
    // column, the last, means y lies in the span of X, a perfect fit, not a
    // singular X.
    if let Err(failure) = packed_cholesky(&mut packed_factor, column_count)
        && failure.column + 1 != column_count
    {
        return Err(SingularRegression {
            pivot: failure.column,
        });
    }

    let mut coefficients =
        packed_factor[packed_column(regression_count)][..regression_count].to_vec();
    // `dppsl.f` backward half only: the response column of the factor already
    // holds the forward-solved right-hand side.
    packed_backward_solve(&packed_factor, regression_count, 1, &mut coefficients);

    let residuals = residuals_of(matrix, &coefficients);
    let residual_sum_squares = dot(&residuals, &residuals);

    let mut normal_inverse = packed_factor[..packed_len(regression_count)].to_vec();
    packed_inverse(&mut normal_inverse, regression_count);

    Ok(RegressionSolution {
        coefficients,
        residuals,
        packed_factor,
        normal_inverse,
        residual_sum_squares,
    })
}

/// The response minus the regressors times `coefficients`, row by row.
pub(crate) fn regression_residuals(
    matrix: &RegressionMatrix,
    coefficients: &[f64],
) -> Result<Vec<f64>, Error> {
    if coefficients.len() + 1 != matrix.column_count() {
        return Err(Error::internal(
            "coefficient count differs from the regression matrix",
        ));
    }
    Ok(residuals_of(matrix, coefficients))
}

/// `coefficients` holds one value per regressor column of `matrix`.
fn residuals_of(matrix: &RegressionMatrix, coefficients: &[f64]) -> Vec<f64> {
    // Each row's residual starts from its response and subtracts the
    // regressors in coefficient order.
    matrix
        .as_matrix()
        .rows()
        .map(|row| {
            let (regressors, response) = row.split_at(coefficients.len());
            regressors
                .iter()
                .zip(coefficients)
                .fold(response[0], |residual, (&value, &coefficient)| {
                    (-coefficient).mul_add(value, residual)
                })
        })
        .collect()
}

/// The packed coefficient covariance: `normal_inverse` scaled by the
/// innovation variance with its degrees-of-freedom correction.
pub(crate) fn regression_covariance(
    normal_inverse: &[f64],
    innovation_variance: f64,
    effective_observations: usize,
    estimated_columns: usize,
) -> Result<Vec<f64>, Error> {
    if estimated_columns >= effective_observations
        || normal_inverse.len() != packed_len(estimated_columns)
    {
        return Err(Error::internal(
            "regression covariance shape differs from the fit",
        ));
    }
    let regression_variance = innovation_variance * effective_observations as f64
        / (effective_observations - estimated_columns) as f64;
    Ok(normal_inverse
        .iter()
        .map(|value| regression_variance * value)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        SingularRegression, regression_covariance, regression_residuals,
        regression_standard_errors, solve_filtered_regression,
    };
    use crate::Error;
    use crate::regression::design::RegressionMatrix;

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn standard_errors_read_packed_diagonals() {
        let normal_inverse = [4.0, 99.0, 9.0, 99.0, 99.0, 16.0];
        assert_eq!(
            regression_standard_errors(&normal_inverse, 4.0, 3),
            [4.0, 6.0, 8.0]
        );
        assert_eq!(
            regression_standard_errors(&normal_inverse, 4.0, 2),
            [4.0, 6.0]
        );
        assert!(regression_standard_errors(&[], 4.0, 0).is_empty());
        assert_eq!(
            bits(&regression_standard_errors(&[-0.0], 4.0, 1)),
            bits(&[-0.0])
        );
        assert!(regression_standard_errors(&[f64::NAN], 4.0, 1)[0].is_nan());
    }

    #[test]
    fn residuals_apply_coefficients_in_column_order() {
        let large = f64::from_bits((1023 + 54) << 52);
        let matrix = RegressionMatrix::from_row_major(
            vec![
                large,
                -large,
                1.0,
                1.0, //
                0.0,
                2.0,
                0.0,
                -0.0, //
                f64::NAN,
                1.0,
                1.0,
                3.0,
            ],
            3,
            4,
        );
        let residuals = regression_residuals(&matrix, &[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(bits(&residuals[..2]), bits(&[-1.0, -2.0]));
        assert!(residuals[2].is_nan());

        let signed = RegressionMatrix::from_row_major(vec![5.0, -0.0, -3.0, -0.0], 2, 2);
        assert_eq!(
            bits(&regression_residuals(&signed, &[0.0]).unwrap()),
            bits(&[-0.0, 0.0])
        );
        let response_only = RegressionMatrix::from_row_major(vec![-0.0, 2.0], 2, 1);
        assert_eq!(
            bits(&regression_residuals(&response_only, &[]).unwrap()),
            bits(&[-0.0, 2.0])
        );
    }

    #[test]
    fn collinear_regressors_return_the_failing_pivot() {
        let matrix = RegressionMatrix::from_row_major(
            vec![1.0, 2.0, 5.0, 2.0, 4.0, 7.0, 3.0, 6.0, 9.0],
            3,
            3,
        );

        assert_eq!(
            solve_filtered_regression(&matrix),
            Err(SingularRegression { pivot: 1 })
        );
    }

    #[test]
    fn response_column_singularity_remains_a_successful_solution() {
        let matrix = RegressionMatrix::from_row_major(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2);

        let solution = solve_filtered_regression(&matrix).unwrap();

        assert_eq!(solution.coefficients(), [2.0]);
        assert_eq!(solution.residuals(), [0.0, 0.0, 0.0]);
        assert_eq!(solution.residual_sum_squares(), 0.0);
    }

    #[test]
    fn mismatched_shapes_are_errors() {
        let matrix = RegressionMatrix::from_row_major(vec![1.0, 2.0], 1, 2);
        assert!(matches!(
            regression_residuals(&matrix, &[]),
            Err(Error::Internal(_))
        ));
        for covariance in [
            regression_covariance(&[1.0], 1.0, 1, 1),
            regression_covariance(&[], 1.0, 2, 1),
        ] {
            assert!(matches!(covariance, Err(Error::Internal(_))));
        }
    }
}
