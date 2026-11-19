//! Outlier critical values (`setcv.f`) and the robust residual scale the
//! candidate statistics are compared against (`medabs.f`).

use crate::Error;
use crate::error::NumericalDetail;
use crate::numerics::{BUILD_62_MAD_SCALE, BUILD_62_PI, build_62_equal, shell_sort};
use crate::regarima::{OutlierCriticalValues, OutlierType};

const OUTLIER_ALPHA: f64 = 0.05;

// `setcv.f` calls `ppnd.f` (AS 111) at only two probabilities, both fixed by
// `Cvalfa = 0.05`. The crate keeps Build 62's two outputs instead of
// translating AS 111, whose terms aren't cleared for MIT OR Apache-2.0.
/// Build 62 `ppnd(1 - α/2)` = 1.9599639822884356.
const NORMAL_QUANTILE_0_975: f64 = f64::from_bits(0x3fff_5c03_3154_44a0);
/// Build 62 `ppnd((1 + √(1 - α))/2)` = 2.236476646021485.
const NORMAL_QUANTILE_SQRT_0_95: f64 = f64::from_bits(0x4001_e44d_de27_70bc);
const _: () = assert!(
    OUTLIER_ALPHA == 0.05,
    "the recorded quantiles assume alpha 0.05"
);

/// Build 62 `medabs.f`: the median of the absolute values, without centering.
pub(crate) fn median_absolute(values: &[f64]) -> f64 {
    debug_assert!(!values.is_empty(), "median requires at least one value");
    let mut absolute = values.iter().map(|value| value.abs()).collect::<Vec<_>>();
    shell_sort(&mut absolute);
    let midpoint = absolute.len() / 2;
    if absolute.len().is_multiple_of(2) {
        (absolute[midpoint - 1] + absolute[midpoint]) / 2.0
    } else {
        absolute[midpoint]
    }
}

/// The robust residual standard deviation `idotlr.f` scales its cutoffs by:
/// the median absolute residual over 0.6745.
pub(crate) fn robust_scale(residuals: &[f64]) -> Result<f64, Error> {
    // Mirrors `x11::robust::scale`: a nonfinite residual would make the median
    // and scale nonfinite, so reject it before sorting (`shell_sort` leaves NaN
    // in place).
    if residuals.iter().any(|value| !value.is_finite()) {
        return Err(super::outlier_failure());
    }
    let median = median_absolute(residuals);
    let scale = median / BUILD_62_MAD_SCALE;
    if build_62_equal(scale, 0.0) {
        return Err(Error::from(NumericalDetail::ZeroOutlierRobustScale));
    }
    Ok(scale)
}

/// Resolves the Build 62 automatic or explicit cutoff for one candidate type.
pub(crate) fn critical_value(
    values: OutlierCriticalValues,
    kind: OutlierType,
    observation_count: usize,
) -> Result<f64, Error> {
    let explicit = match values {
        OutlierCriticalValues::Automatic => None,
        OutlierCriticalValues::Shared(value) => Some(value),
        OutlierCriticalValues::PerType {
            additive,
            level_shift,
            temporary_change,
        } => match kind {
            OutlierType::Additive => additive,
            OutlierType::LevelShift => level_shift,
            OutlierType::TemporaryChange => temporary_change,
        },
    };
    explicit.map_or_else(
        || automatic_critical_value(observation_count),
        |value| Ok(value.value()),
    )
}

/// Resolved per-type outlier cutoffs, held in Build 62's AO, LS, TC order.
///
/// `None` marks a type the search does not test, where Build 62 guards each
/// `Critvl` slot with its `Ltstao`/`Ltstls`/`Ltsttc` flag (`automd.f`,
/// `pass2.f`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OutlierCutoffs([Option<f64>; 3]);

impl OutlierCutoffs {
    /// No tested type: automdl's state when the spec has no outlier search.
    pub(crate) const NONE: Self = Self([None; 3]);

    #[cfg(test)]
    pub(crate) const fn in_build_62_order(values: [Option<f64>; 3]) -> Self {
        Self(values)
    }

    /// The tested types and their cutoffs, in Build 62 order.
    pub(crate) fn tested(self) -> impl Iterator<Item = (OutlierType, f64)> {
        OutlierType::BUILD_62_ORDER
            .into_iter()
            .zip(self.0)
            .filter_map(|(kind, cutoff)| Some((kind, cutoff?)))
    }

    /// Applies `update` to each tested type's cutoff.
    pub(crate) fn map(self, mut update: impl FnMut(f64) -> f64) -> Self {
        Self(self.0.map(|cutoff| cutoff.map(&mut update)))
    }
}

impl std::ops::Index<OutlierType> for OutlierCutoffs {
    type Output = Option<f64>;

    fn index(&self, kind: OutlierType) -> &Option<f64> {
        &self.0[kind.build_62_index()]
    }
}

impl std::ops::IndexMut<OutlierType> for OutlierCutoffs {
    fn index_mut(&mut self, kind: OutlierType) -> &mut Option<f64> {
        &mut self.0[kind.build_62_index()]
    }
}

/// The observation counts at which `setcv.f` fits its critical-value curve.
const INTERPOLATION_COUNTS: [f64; 3] = [2.0, 100.0, 200.0];

/// Build 62 `setcv.f` with `Cvalfa=0.05`.
pub(crate) fn automatic_critical_value(observation_count: usize) -> Result<f64, Error> {
    if observation_count == 0 {
        return Err(Error::from(
            NumericalDetail::OutlierCriticalValueComputation,
        ));
    }
    if observation_count == 1 {
        return Ok(NORMAL_QUANTILE_0_975);
    }

    let mut design = [[0.0; 3]; 3];
    let mut targets = [0.0; 3];
    for (index, ((row, target), count)) in design
        .iter_mut()
        .zip(&mut targets)
        .zip(INTERPOLATION_COUNTS)
        .enumerate()
    {
        *target = if index == 0 {
            NORMAL_QUANTILE_SQRT_0_95
        } else {
            extreme_value_critical_value(count as usize)
        };
        let (scale, shift) = extreme_value_terms(count);
        *row = [1.0, shift, scale];
    }
    let beta = solve_three_by_three(design, targets).ok_or(Error::from(
        NumericalDetail::OutlierCriticalValueComputation,
    ))?;
    let (scale, shift) = extreme_value_terms(observation_count as f64);
    Ok(beta[0] + beta[1] * shift + beta[2] * scale)
}

/// The extreme-value normalizing terms `setcv.f` and `setcvl.f` share for a
/// count `n`: `acv = sqrt(2 ln n)` and `(ln ln n + ln 4π) / (2 acv)`.
fn extreme_value_terms(count: f64) -> (f64, f64) {
    let scale = (2.0 * count.ln()).sqrt();
    let shift = (count.ln().ln() + (2.0 * 2.0 * BUILD_62_PI).ln()) / (2.0 * scale);
    (scale, shift)
}

/// Build 62 `setcvl.f`: the extreme-value critical value at one interpolation
/// count.
fn extreme_value_critical_value(observation_count: usize) -> f64 {
    let modified_probability = 2.0 - (1.0 + OUTLIER_ALPHA).sqrt();
    let (scale, shift) = extreme_value_terms(observation_count as f64);
    let location = scale - shift;
    let quantile = -(-0.5 * modified_probability.ln()).ln();
    (quantile / scale) + location
}

/// Build 62 `lassol.f`, specialized to `setcv.f`'s fixed 3x3 system: scaled
/// partial pivoting, then back substitution.
fn solve_three_by_three(matrix: [[f64; 3]; 3], right_hand_side: [f64; 3]) -> Option<[f64; 3]> {
    let mut augmented = [[0.0; 4]; 3];
    for ((augmented_row, matrix_row), right_hand_side) in
        augmented.iter_mut().zip(&matrix).zip(right_hand_side)
    {
        let row_maximum = matrix_row
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let scale = 1.0 / row_maximum;
        for (target, value) in augmented_row.iter_mut().zip(matrix_row) {
            *target = value * scale;
        }
        augmented_row[3] = right_hand_side * scale;
    }

    for pivot in 0..2 {
        // The first strictly largest magnitude wins, as in `lassol.f`.
        let mut largest = 0.0;
        let mut pivot_row = pivot;
        for (row, values) in augmented.iter().enumerate().skip(pivot) {
            let candidate = values[pivot].abs();
            if largest < candidate {
                largest = candidate;
                pivot_row = row;
            }
        }
        if build_62_equal(largest, 0.0) {
            return None;
        }
        // `lassol.f` swaps only columns `pivot..`; the earlier columns of
        // these rows are never read again, so swapping whole rows is the same.
        augmented.swap(pivot, pivot_row);
        let (upper, lower) = augmented.split_at_mut(pivot + 1);
        let pivot_values = &upper[pivot];
        for row in lower {
            let quotient = row[pivot] / pivot_values[pivot];
            for (value, pivot_value) in row[pivot + 1..].iter_mut().zip(&pivot_values[pivot + 1..])
            {
                *value -= quotient * pivot_value;
            }
        }
    }

    if build_62_equal(augmented[2][2], 0.0) {
        return None;
    }
    let mut solution = [0.0; 3];
    solution[2] = augmented[2][3] / augmented[2][2];
    for row in (0..2).rev() {
        let values = &augmented[row];
        let sum = values[row + 1..3]
            .iter()
            .zip(&solution[row + 1..])
            .fold(0.0, |sum, (value, solved)| sum + value * solved);
        solution[row] = (values[3] - sum) / values[row];
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::{
        automatic_critical_value, critical_value, median_absolute, robust_scale,
        solve_three_by_three,
    };
    use crate::Error;
    use crate::error::{NumericalDetail, NumericalFailure, NumericalOperation};
    use crate::regarima::{OutlierCriticalValue, OutlierCriticalValues, OutlierType};

    fn solution_bits(solution: Option<[f64; 3]>) -> Option<[u64; 3]> {
        solution.map(|values| values.map(f64::to_bits))
    }

    #[test]
    fn three_by_three_solve_pivots_and_reports_singular_systems() {
        // Small leading entries force a row swap at both pivot columns.
        assert_eq!(
            [
                solution_bits(solve_three_by_three(
                    [[1.0e-3, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
                    [1.0, 2.0, 3.0],
                )),
                solution_bits(solve_three_by_three(
                    [[0.0, 1.0, 2.0], [-0.0, 3.0, -1.0], [2.0, -0.0, 1.0]],
                    [-0.0, 4.0, 0.0],
                )),
            ],
            [
                Some([
                    0xbfc9_9c38_f509_efd8,
                    0x3fe9_98f1_c2bd_8419,
                    0xbfc9_95aa_9071_186d
                ]),
                Some([
                    0x3fd2_4924_9249_2492,
                    0x3ff2_4924_9249_2492,
                    0xbfe2_4924_9249_2492
                ]),
            ]
        );
        // Identical scaled rows: the second pivot column is all zero.
        assert_eq!(
            solve_three_by_three(
                [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]],
                [1.0, 2.0, 3.0],
            ),
            None
        );
        // Only the final diagonal vanishes.
        assert_eq!(
            solve_three_by_three(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                [1.0, 2.0, 3.0],
            ),
            None
        );
        // `f64::max` skips a NaN entry in the row scale; the NaN then
        // reaches every solution component.
        let with_nan = solve_three_by_three(
            [[f64::NAN, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]],
            [1.0, 1.0, 1.0],
        );
        assert_eq!(
            with_nan.map(|values| values.map(f64::is_nan)),
            Some([true; 3])
        );
    }

    #[test]
    fn median_absolute_uses_uncentered_values_and_build_62_even_order() {
        assert_eq!(median_absolute(&[-3.0, 1.0, -2.0]), 2.0);
        assert_eq!(median_absolute(&[-4.0, 1.0, -3.0, 2.0]), 2.5);
        assert_eq!(median_absolute(&[100.0, -1.0, -2.0]), 2.0);
        assert_eq!(median_absolute(&[2.0, -2.0, 2.0, -2.0]), 2.0);
        assert_eq!(median_absolute(&[1.0, -2.0, 3.0, -4.0, 5.0]), 3.0);
        assert_eq!(median_absolute(&[-5.0, 4.0, -3.0, 2.0, -1.0]), 3.0);
        assert_eq!(median_absolute(&[-7.25]), 7.25);
    }

    #[test]
    fn robust_scale_uses_build_62_approximate_zero_after_scaling() {
        assert_eq!(
            robust_scale(&[1.0e-20, -1.0e-20]),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::ZeroOutlierRobustScale
            )))
        );
        assert!(robust_scale(&[1.0e-18, -1.0e-18]).is_ok());
    }

    #[test]
    fn robust_scale_rejects_nonfinite_residuals() {
        let failure = Err(Error::numerical(NumericalOperation::OutlierSearch));
        assert_eq!(robust_scale(&[1.0, f64::NAN, -2.0, 0.5]), failure);
        assert_eq!(robust_scale(&[1.0, f64::INFINITY, -2.0]), failure);
        assert_eq!(robust_scale(&[f64::NEG_INFINITY]), failure);
    }

    #[test]
    fn robust_scale_divides_by_0_6745_and_rejects_zero() {
        let residuals = [-3.0, 1.0, -2.0];
        assert_eq!(robust_scale(&residuals).unwrap(), 2.0 / 0.6745);
        assert_eq!(
            robust_scale(&[0.0, -0.0, 0.0]),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::ZeroOutlierRobustScale
            )))
        );
    }

    #[test]
    fn automatic_critical_values_match_build_62_diagnostics_at_print_precision() {
        for (count, rendered) in [
            (1, "1.960"),
            (2, "2.236"),
            (12, "3.164"),
            (60, "3.686"),
            (100, "3.810"),
            (144, "3.890"),
            (200, "3.956"),
        ] {
            assert_eq!(
                format!("{:.3}", automatic_critical_value(count).unwrap()),
                rendered
            );
        }
    }

    #[test]
    fn automatic_critical_values_keep_their_bits() {
        // Full-precision values from a line-by-line translation of `setcv.f`,
        // `setcvl.f`, `ppnd.f` and `lassol.f`.
        for (count, bits) in [
            (1, 0x3fff_5c03_3154_44a0),
            (2, 0x4001_e44d_de27_70bd),
            (12, 0x4009_5072_99e4_ecdf),
            (60, 0x400d_7dba_4529_2663),
            (100, 0x400e_7b0f_3165_46e4),
            (144, 0x400f_1e63_42f6_bdac),
            (200, 0x400f_a6d3_9af7_5f24),
        ] {
            assert_eq!(automatic_critical_value(count).unwrap().to_bits(), bits);
        }
    }

    #[test]
    fn no_observations_have_no_critical_value() {
        assert_eq!(
            automatic_critical_value(0),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::OutlierCriticalValueComputation
            )))
        );
    }

    #[test]
    fn explicit_critical_values_override_with_automatic_per_type_fallback() {
        let shared = OutlierCriticalValue::new(4.25).unwrap();
        assert_eq!(
            critical_value(
                OutlierCriticalValues::Shared(shared),
                OutlierType::LevelShift,
                60,
            ),
            Ok(4.25)
        );

        let additive = OutlierCriticalValue::new(3.5).unwrap();
        let temporary_change = OutlierCriticalValue::new(4.75).unwrap();
        let per_type =
            OutlierCriticalValues::per_type(Some(additive), None, Some(temporary_change));
        assert_eq!(critical_value(per_type, OutlierType::Additive, 60), Ok(3.5));
        assert_eq!(
            critical_value(per_type, OutlierType::LevelShift, 60),
            automatic_critical_value(60)
        );
        assert_eq!(
            critical_value(per_type, OutlierType::TemporaryChange, 60),
            Ok(4.75)
        );
    }
}
