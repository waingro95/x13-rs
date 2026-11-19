use super::{AirlineInput, import_airline};
use crate::Frequency;
use crate::seats::{Failure, Stage};

#[test]
fn import_keeps_unclipped_coefficients() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let input = AirlineInput::new(frequency, [0.4, 0.6]).unwrap();
        let imported = import_airline(&input);
        assert_eq!(imported.frequency(), frequency);
        assert_eq!(imported.ma_coefficients(), [0.4, 0.6]);
    }
}

#[test]
fn rejects_each_nonfinite_ma() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            for (ma, index) in [([value, 0.6], 0), ([0.4, value], 1), ([value, value], 0)] {
                assert_eq!(
                    AirlineInput::new(frequency, ma).unwrap_err(),
                    Failure::Unusable {
                        quantity: "fitted model coefficient",
                        index: Some(index),
                        value,
                    }
                    .at(Stage::Import)
                );
            }
        }
    }
}

#[test]
fn import_clips_at_the_trans0_ratio_cutoffs() {
    let positive = 0.9702_f64;
    let negative = -0.9702_f64;
    // TRANS0 clips on its scalar ratio, not on the coefficient itself.
    for (value, expected) in [
        (positive, positive),
        (positive.next_up(), positive.next_up()),
        (positive.next_up().next_up(), 0.99),
        (negative, negative),
        (negative.next_down(), -0.99),
        (0.971, 0.99),
        (-0.971, -0.99),
    ] {
        for frequency in [Frequency::Monthly, Frequency::Quarterly] {
            let input = AirlineInput::new(frequency, [value, value]).unwrap();
            assert_eq!(import_airline(&input).ma_coefficients(), [expected; 2]);
        }
    }
}

#[test]
fn import_maps_either_zero_to_negative_zero() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        for ma in [[0.0, -0.0], [-0.0, 0.0]] {
            let input = AirlineInput::new(frequency, ma).unwrap();
            let actual = import_airline(&input).ma_coefficients().map(f64::to_bits);
            assert_eq!(actual, [(-0.0_f64).to_bits(); 2]);
        }
    }
}

#[test]
fn import_clips_extreme_finite_coefficients() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        for (ma, expected) in [
            ([f64::MAX, -f64::MAX], [0.99, -0.99]),
            ([-f64::MAX, f64::MAX], [-0.99, 0.99]),
        ] {
            let input = AirlineInput::new(frequency, ma).unwrap();
            let imported = import_airline(&input);
            assert_eq!(imported.ma_coefficients(), expected);
            assert!(
                imported
                    .ma_coefficients()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }
}
