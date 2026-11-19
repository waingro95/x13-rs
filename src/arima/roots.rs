use crate::numerics::build_62_equal;

mod details;
mod polynomial;
mod solver;

pub(crate) use details::{DetailedRoots, RootDetails, detailed_roots};
pub(crate) use polynomial::{RootComputationError, has_strictly_inside_root};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootLocation {
    Outside,
    OnUnitCircle,
    Inside,
}

/// Where the roots of `1 - sum c_k B^k` lie, by Schur reduction: each step
/// takes off the highest coefficient and folds the rest together pairwise
/// from both ends, the middle one (if any) once.
pub(crate) fn root_location(coefficients: &[f64]) -> RootLocation {
    let mut reduced = coefficients.to_vec();

    while let Some(last) = reduced.pop() {
        let scale = 1.0 - last * last;
        if scale <= 0.0 {
            return if build_62_equal(last.abs(), 1.0) {
                RootLocation::OnUnitCircle
            } else {
                RootLocation::Inside
            };
        }

        let count = reduced.len();
        for index in 0..count.div_ceil(2) {
            let mirror = count - 1 - index;
            let coefficient = reduced[index];
            let other = reduced[mirror];
            // Unfused: Build 62 rounds `last * other` before the add.
            reduced[index] = (coefficient + last * other) / scale;
            if mirror != index {
                reduced[mirror] = (other + last * coefficient) / scale;
            }
        }
    }

    RootLocation::Outside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_reduction_folds_odd_and_even_degrees_from_both_ends() {
        use RootLocation::{Inside, OnUnitCircle, Outside};
        let nan = f64::NAN;
        for (coefficients, expected) in [
            (&[][..], Outside),
            (&[-0.0], Outside),
            (&[nan], Outside),
            (&[0.2, nan, 0.1], Outside),
            (&[0.3, -0.0, 0.2], Outside),
            (&[1.2, -0.5, 0.1], Outside),
            (&[1.9, -0.95], Outside),
            (&[0.5, -0.3, 0.2, 0.1], Outside),
            (&[-0.6, -0.2, 0.3, 0.5], Outside),
            (&[0.4, 0.4, 0.4, 0.4], Inside),
            (&[0.0, 0.0, 0.0, 1.0], OnUnitCircle),
            (&[0.5, -0.3, 0.2, 0.1, -0.05], Outside),
            (&[0.1, 0.1, 0.1, 0.1, 0.9], Inside),
            (&[0.9, -0.5, 0.4, 0.3, -0.2, 0.1], Inside),
        ] {
            assert_eq!(
                root_location(coefficients),
                expected,
                "coefficients={coefficients:?}"
            );
        }
    }

    #[test]
    fn a_coefficient_of_one_is_on_the_unit_circle() {
        assert_eq!(
            root_location(&[0.999_999_999_999_999_9]),
            RootLocation::Outside
        );
        assert_eq!(root_location(&[1.0]), RootLocation::OnUnitCircle);
        assert_eq!(
            root_location(&[1.000_000_000_000_000_2]),
            RootLocation::Inside
        );
    }

    #[test]
    fn a_stationary_first_coefficient_does_not_decide_a_second_order_operator() {
        assert_eq!(root_location(&[0.5, 0.2]), RootLocation::Outside);
        assert_ne!(root_location(&[0.5, 0.8]), RootLocation::Outside);
    }
}
