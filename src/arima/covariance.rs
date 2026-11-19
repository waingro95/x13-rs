//! The covariance matrix of the ARMA parameters, from the Jacobian of the
//! residuals at the estimate.

use crate::numerics::BUILD_62_MACHINE_EPSILON;
use crate::numerics::minpack::{ColMajorMut, QrFactors, finite_difference_jacobian, qr_factor};

/// The cross product's pivot fell below the rank tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Singular;

/// The covariance matrix, or `Singular` when the Jacobian is rank deficient.
/// Either way the diagonal of `(JᵀJ)⁻¹` is returned, before it is multiplied
/// by the innovation variance.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ParameterCovariance {
    Calculated {
        matrix: Vec<f64>,
        unscaled_diagonal: Vec<f64>,
    },
    Singular {
        unscaled_diagonal: Vec<f64>,
    },
}

/// The innovation variance times `(JᵀJ)⁻¹`, where `J` is the forward-difference
/// Jacobian of `callback`'s residuals at `parameters`. `residuals` are the
/// residuals there, and `tolerance` the rank tolerance of MINPACK's `covar.f`.
pub(super) fn parameter_covariance<F>(
    parameters: &mut [f64],
    residuals: &[f64],
    tolerance: f64,
    innovation_variance: f64,
    mut callback: F,
) -> ParameterCovariance
where
    F: FnMut(&[f64], &mut [f64]),
{
    let parameter_count = parameters.len();
    let residual_count = residuals.len();
    let mut jacobian_values = vec![0.0; residual_count * parameter_count];
    let Ok(()) = finite_difference_jacobian::<_, std::convert::Infallible>(
        parameters,
        residuals,
        &mut jacobian_values,
        0.0,
        &mut |candidate, output| {
            callback(candidate, output);
            Ok(())
        },
    );

    let mut jacobian = ColMajorMut::new(&mut jacobian_values, residual_count);
    let mut factors = QrFactors::new(parameter_count);
    qr_factor(&mut jacobian, &mut factors);
    for (column, &value) in factors.r_diagonal.iter().enumerate() {
        jacobian[(column, column)] = value;
    }

    let inversion = invert_cross_product(&mut jacobian, &factors.permutation, tolerance);
    let unscaled_diagonal = jacobian.diagonal(parameter_count);
    if inversion.is_err() {
        return ParameterCovariance::Singular { unscaled_diagonal };
    }

    // Column-major, `parameter_count` square.
    let mut covariance = Vec::with_capacity(parameter_count * parameter_count);
    for column in 0..parameter_count {
        covariance.extend(
            jacobian.col(column)[..parameter_count]
                .iter()
                .map(|&value| innovation_variance * value),
        );
    }
    ParameterCovariance::Calculated {
        matrix: covariance,
        unscaled_diagonal,
    }
}

/// MINPACK `covar.f`: overwrite the n-by-n top of `r` (R from a pivoted QR)
/// with `(R^T R)^-1` in the original column order, full and symmetric.
/// Columns from the first diagonal at or below the rank tolerance on are
/// zeroed, which is an error.
fn invert_cross_product(
    r: &mut ColMajorMut<'_>,
    permutation: &[usize],
    tolerance: f64,
) -> Result<(), Singular> {
    let order = permutation.len();
    let first_diagonal = r[(0, 0)].abs();
    let threshold = if tolerance <= 0.0 {
        BUILD_62_MACHINE_EPSILON * first_diagonal
    } else {
        tolerance * first_diagonal
    };
    // Each pass below changes only its own column, so the test can run on
    // the diagonal as it stands.
    let nonsingular = (0..order)
        .position(|column| r[(column, column)].abs() <= threshold)
        .unwrap_or(order);

    // R^-1 in the upper triangle.
    for column in 0..nonsingular {
        r[(column, column)] = 1.0 / r[(column, column)];
        for preceding in 0..column {
            let value = r[(column, column)] * r[(preceding, column)];
            r[(preceding, column)] = 0.0;
            let (source, target) = r.col_and_col_mut(preceding, column);
            for (target, &source) in target[..=preceding].iter_mut().zip(&source[..=preceding]) {
                *target = (-value).mul_add(source, *target);
            }
        }
    }

    // R^-1 R^-T in the upper triangle.
    for column in 0..nonsingular {
        for preceding in 0..column {
            let value = r[(preceding, column)];
            let (source, target) = r.col_and_col_mut(column, preceding);
            for (target, &source) in target[..=preceding].iter_mut().zip(&source[..=preceding]) {
                *target = value.mul_add(source, *target);
            }
        }
        let diagonal = r[(column, column)];
        for value in &mut r.col_mut(column)[..=column] {
            *value *= diagonal;
        }
    }

    // Unpivot into the strict lower triangle, zeroing the singular columns,
    // and set the unpivoted diagonal aside.
    let mut diagonal = vec![0.0; order];
    for (column, &permuted_column) in permutation.iter().enumerate() {
        let singular = column >= nonsingular;
        for (row, &permuted_row) in permutation[..=column].iter().enumerate() {
            if singular {
                r[(row, column)] = 0.0;
            }
            if permuted_row > permuted_column {
                r[(permuted_row, permuted_column)] = r[(row, column)];
            } else if permuted_row < permuted_column {
                r[(permuted_column, permuted_row)] = r[(row, column)];
            }
        }
        diagonal[permuted_column] = r[(column, column)];
    }

    // Mirror the lower triangle up and restore the diagonal.
    for (column, &value) in diagonal.iter().enumerate() {
        for row in 0..=column {
            r[(row, column)] = r[(column, row)];
        }
        r[(column, column)] = value;
    }

    if nonsingular == order {
        Ok(())
    } else {
        Err(Singular)
    }
}

#[cfg(test)]
mod tests;
