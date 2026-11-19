/// Build 62 `shlsrt.f`. The ±0 order depends on its exact swap sequence.
///
/// `shlsrt.f` swaps unless `a <= b`; this swaps only when `a > b`. The two
/// agree on every pair without NaN, so callers, which reject NaN, see the
/// same order. With NaN, `shlsrt.f` keeps swapping it and some inputs never
/// terminate; here NaN never moves, and every swap removes an inversion, so
/// the sort always ends.
pub(crate) fn shell_sort(values: &mut [f64]) {
    let mut gap = values.len() / 2;
    while gap > 0 {
        let mut bottom = 0;
        while bottom + gap < values.len() {
            // Sift the pair down by `gap` while it is out of order.
            // `shlsrt.f` keeps `bottom` where the sift stopped, so the scan
            // resumes there: the outer counter can move backwards.
            while values[bottom] > values[bottom + gap] {
                values.swap(bottom, bottom + gap);
                if bottom < gap {
                    break;
                }
                bottom -= gap;
            }
            bottom += 1;
        }
        gap /= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::shell_sort;

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn shell_sort_orders_signed_zeros_like_build_62() {
        let mut actual = vec![2.0, -0.0, 1.0, 0.0, -0.0, 3.0, 0.0, -1.0, 1.0, 0.5, -0.0];
        shell_sort(&mut actual);
        assert_eq!(
            bits(&actual),
            bits(&[-1.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn shell_sort_never_moves_nan_and_terminates() {
        // `shlsrt.f` swaps [NaN, NaN, 1] forever.
        let nan = f64::NAN;
        for (input, expected) in [
            (vec![nan], vec![nan]),
            (vec![nan, nan, 1.0], vec![nan, nan, 1.0]),
            (
                vec![3.0, nan, 1.0, nan, -0.0, 2.0, 0.0],
                vec![3.0, nan, 1.0, nan, -0.0, 0.0, 2.0],
            ),
        ] {
            let mut actual = input;
            shell_sort(&mut actual);
            assert_eq!(bits(&actual), bits(&expected));
        }
    }

    #[test]
    fn shell_sort_preserves_pre_move_bit_sequences() {
        // Complete pre-move bit sequences, including signed-zero order.
        for (input, expected) in [
            (vec![3.0, -1.0, 2.0], vec![-1.0, 2.0, 3.0]),
            (vec![4.0, 1.0, 3.0, 2.0], vec![1.0, 2.0, 3.0, 4.0]),
            (vec![2.0, 1.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0]),
            (vec![-1.0, 0.0, 2.0], vec![-1.0, 0.0, 2.0]),
            (vec![3.0, 2.0, 1.0], vec![1.0, 2.0, 3.0]),
            (vec![1.0, -0.0, 0.0, -1.0], vec![-1.0, 0.0, -0.0, 1.0]),
            (vec![0.0, -0.0], vec![0.0, -0.0]),
        ] {
            let mut actual = input;
            shell_sort(&mut actual);
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value: &f64| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}
