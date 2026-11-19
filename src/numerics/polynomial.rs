//! Lag polynomials: their products, and multiplying or dividing a series by
//! them (`polyml.f`, `mltpos.f`, `ratpos.f`).

/// The operator `1 - sum c B^lag`, borrowed as its coefficients `c` and
/// their lags, stored ascending.
#[derive(Clone, Copy)]
pub(crate) struct Polynomial<'a> {
    coefficients: &'a [f64],
    lags: &'a [usize],
}

impl<'a> Polynomial<'a> {
    pub(crate) fn new(coefficients: &'a [f64], lags: &'a [usize]) -> Self {
        // `terms` zips these, so a mismatch would otherwise silently drop terms.
        assert_eq!(coefficients.len(), lags.len());
        Self { coefficients, lags }
    }

    fn len(self) -> usize {
        self.coefficients.len()
    }

    /// Each `(coefficient, lag)` term, in storage order.
    pub(crate) fn terms(self) -> impl Iterator<Item = (f64, usize)> + 'a {
        self.coefficients
            .iter()
            .copied()
            .zip(self.lags.iter().copied())
    }

    /// The first lag the recursions start from: `ratpos.f` starts at the
    /// lowest lag only when the operator has more than one term.
    fn first_index(self) -> usize {
        if self.len() > 1 { self.lags[0] } else { 0 }
    }
}

/// An owned [`Polynomial`], such as a product of operators.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct OwnedPolynomial {
    coefficients: Vec<f64>,
    lags: Vec<usize>,
}

impl OwnedPolynomial {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(capacity),
            lags: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn as_polynomial(&self) -> Polynomial<'_> {
        Polynomial {
            coefficients: &self.coefficients,
            lags: &self.lags,
        }
    }

    pub(crate) fn terms(&self) -> impl Iterator<Item = (f64, usize)> + '_ {
        self.as_polynomial().terms()
    }

    /// `polyml.f`'s insertion. Scanning down from the highest lag, the term
    /// goes after the first lag at or below its own, adding into an equal lag;
    /// below every lag it goes first.
    fn add_term(&mut self, coefficient: f64, lag: usize) {
        match self.lags.iter().rposition(|&existing| existing <= lag) {
            Some(index) if self.lags[index] == lag => self.coefficients[index] += coefficient,
            position => {
                let index = position.map_or(0, |index| index + 1);
                self.coefficients.insert(index, coefficient);
                self.lags.insert(index, lag);
            }
        }
    }
}

impl From<Polynomial<'_>> for OwnedPolynomial {
    fn from(polynomial: Polynomial<'_>) -> Self {
        Self {
            coefficients: polynomial.coefficients.to_vec(),
            lags: polynomial.lags.to_vec(),
        }
    }
}

/// `polyml.f`: the product `(1 - left)(1 - right)`, as the terms of
/// `left`, then each term of `right` followed by its products with `left`.
pub(crate) fn multiply_polynomials(left: Polynomial<'_>, right: Polynomial<'_>) -> OwnedPolynomial {
    let mut product =
        OwnedPolynomial::with_capacity(left.len() * right.len() + left.len() + right.len());

    for (coefficient, lag) in left.terms() {
        product.add_term(coefficient, lag);
    }

    for (right_coefficient, right_lag) in right.terms() {
        product.add_term(right_coefficient, right_lag);
        for (left_coefficient, left_lag) in left.terms() {
            product.add_term(
                (-right_coefficient) * left_coefficient,
                right_lag + left_lag,
            );
        }
    }

    product
}

/// `ratpos.f`: divide `input`, extended with zeros to `output_length`, by
/// each operator in turn.
pub(crate) fn rational_expand(
    input: &[f64],
    output_length: usize,
    operators: &[Polynomial<'_>],
) -> Vec<f64> {
    let mut output = vec![0.0; output_length];
    output[..input.len()].copy_from_slice(input);
    let mut input_length = input.len();

    for operator in operators {
        let first_index = operator.first_index();
        // Build 62 zero-fills past the logical length into PMAX-sized storage
        // when a seasonal lag exceeds the requested length; here the fill
        // stops at `output_length` and the loop below is already empty.
        if input_length < first_index {
            output[input_length..first_index.min(output_length)].fill(0.0);
        }

        for index in first_index..output_length {
            let mut sum = if index < input_length {
                output[index]
            } else {
                0.0
            };
            for (coefficient, lag) in operator.terms() {
                if index >= lag {
                    let lagged = output[index - lag];
                    if coefficient.abs() > 1.0e-150 && lagged.abs() > 1.0e-150 {
                        sum = coefficient.mul_add(lagged, sum);
                    }
                }
            }
            output[index] = sum;
        }
        input_length = output_length;
    }

    output
}

/// `mltpos.f`: multiply `input`, extended with zeros to `output_length`, by
/// each operator in turn. The first pass reads only the input's own span.
pub(crate) fn positive_lag_product(
    input: &[f64],
    output_length: usize,
    operators: &[Polynomial<'_>],
) -> Vec<f64> {
    let mut output = vec![0.0; output_length];
    output[..input.len()].copy_from_slice(input);
    let mut product = vec![0.0; output_length];

    for (pass, operator) in operators.iter().enumerate() {
        let known = if pass == 0 {
            input.len()
        } else {
            output_length
        };
        for (index, value) in product.iter_mut().enumerate() {
            let mut sum = if index < known { output[index] } else { 0.0 };
            for (coefficient, lag) in operator.terms() {
                if index >= lag && index - lag < known {
                    sum = (-coefficient).mul_add(output[index - lag], sum);
                }
            }
            *value = sum;
        }
        // Every element of `product` is rewritten on the next pass.
        std::mem::swap(&mut output, &mut product);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::{Polynomial, multiply_polynomials, positive_lag_product, rational_expand};

    /// Bit patterns, with every NaN as `None`: NaN sign and payload depend on
    /// the target's propagation rules, not on this code.
    fn bits(values: &[f64]) -> Vec<Option<u64>> {
        values
            .iter()
            .map(|value| (!value.is_nan()).then(|| value.to_bits()))
            .collect()
    }

    fn product(
        left: (&[f64], &[usize]),
        right: (&[f64], &[usize]),
    ) -> (Vec<Option<u64>>, Vec<usize>) {
        let product = multiply_polynomials(
            Polynomial::new(left.0, left.1),
            Polynomial::new(right.0, right.1),
        );
        (bits(&product.coefficients), product.lags)
    }

    #[test]
    fn multiply_polynomials_merges_ties_in_ascending_lag_order() {
        // Overlapping lags merge by addition; -0.0 survives as a term.
        assert_eq!(
            product((&[0.4, -0.0, 0.3], &[1, 2, 4]), (&[0.5, 0.2], &[2, 3])),
            (
                [
                    0x3fd9_9999_9999_999a,
                    0x3fe0_0000_0000_0000,
                    0x0000_0000_0000_0000,
                    0x3fcc_28f5_c28f_5c28,
                    0x0000_0000_0000_0000,
                    0xbfc3_3333_3333_3333,
                    0xbfae_b851_eb85_1eb8,
                ]
                .map(Some)
                .to_vec(),
                vec![1, 2, 3, 4, 5, 6, 7],
            )
        );
        // Unsorted input lags: each term goes after the last lag at or below
        // it, scanning from the end.
        assert_eq!(
            product((&[0.1, 0.2], &[3, 1]), (&[0.7], &[2])),
            (
                [
                    0x3fc9_9999_9999_999a,
                    0x3fe6_6666_6666_6666,
                    0xbfa4_7ae1_47ae_1478,
                    0xbfb1_eb85_1eb8_51eb,
                ]
                .map(Some)
                .to_vec(),
                vec![1, 2, 3, 5],
            )
        );
        assert_eq!(
            product((&[0.6, f64::NAN], &[1, 2]), (&[-0.0, 0.25], &[1, 12])),
            (
                vec![
                    Some(0x3fe3_3333_3333_3333),
                    None,
                    None,
                    Some(0x3fd0_0000_0000_0000),
                    Some(0xbfc3_3333_3333_3333),
                    None,
                ],
                vec![1, 2, 3, 12, 13, 14],
            )
        );
        // An empty factor leaves the other unchanged.
        let unchanged = (
            vec![Some(0x3fd3_3333_3333_3333), Some(0x3fb9_9999_9999_999a)],
            vec![1, 2],
        );
        assert_eq!(product((&[], &[]), (&[0.3, 0.1], &[1, 2])), unchanged);
        assert_eq!(product((&[0.3, 0.1], &[1, 2]), (&[], &[])), unchanged);
    }

    #[test]
    fn expansions_keep_their_guards_signed_zeros_and_pass_order() {
        let tiny = [1e-151, 0.5, -0.25];
        let tiny_lags = [1, 2, 3];
        let seasonal = [0.4, -0.0];
        let seasonal_lags = [4, 8];
        let operators = [
            Polynomial::new(&tiny, &tiny_lags),
            Polynomial::new(&seasonal, &seasonal_lags),
        ];
        assert_eq!(
            bits(&rational_expand(&[1.0, 2e-151, -0.0, 0.5], 12, &operators)),
            [
                0x3ff0_0000_0000_0000,
                0x20a4_f31f_8832_dd2a,
                0x3fe0_0000_0000_0000,
                0x3fd0_0000_0000_0000,
                0x3fe4_cccc_cccc_cccd,
                0x0000_0000_0000_0000,
                0x3fd0_cccc_cccc_cccd,
                0x3fa3_3333_3333_3334,
                0x3fd2_a3d7_0a3d_70a4,
                0xbfa8_0000_0000_0000,
                0x3fc1_70a3_d70a_3d71,
                0xbf90_a3d7_0a3d_70a3,
            ]
            .map(Some)
        );
        // A NaN coefficient fails the 1e-150 guard and drops out.
        let nan = [f64::NAN];
        let nan_lags = [2];
        assert_eq!(
            bits(&rational_expand(
                &[1.0, -0.0],
                6,
                &[Polynomial::new(&nan, &nan_lags)]
            )),
            [0x3ff0_0000_0000_0000, 0x8000_0000_0000_0000, 0, 0, 0, 0].map(Some)
        );
        assert_eq!(
            bits(&positive_lag_product(&[1.0, -0.0, 0.5], 10, &operators)),
            [
                0x3ff0_0000_0000_0000,
                0xa094_f31f_8832_dd2a,
                0x0000_0000_0000_0000,
                0x3fd0_0000_0000_0000,
                0xbfe4_cccc_cccc_cccd,
                0x3fc0_0000_0000_0000,
                0x0000_0000_0000_0000,
                0xbfb9_9999_9999_999a,
                0x3fb9_9999_9999_999a,
                0xbfa9_9999_9999_999a,
            ]
            .map(Some)
        );
        assert_eq!(
            bits(&positive_lag_product(
                &[-0.0],
                4,
                &[Polynomial::new(&[-0.0], &[1])]
            )),
            [0x8000_0000_0000_0000, 0, 0, 0].map(Some)
        );
        assert!(rational_expand(&[], 0, &operators).is_empty());
        assert!(positive_lag_product(&[], 0, &operators).is_empty());
    }

    #[test]
    fn rational_expand_clamps_zero_fill_when_seasonal_lag_exceeds_output() {
        // A seasonal AR(2) with lags 12 and 24 against an output length of 2
        // (`ma_lag + 1` with no seasonal MA).
        let coefficients = [-0.7, -0.1];
        let lags = [12, 24];
        let operators = [Polynomial::new(&coefficients, &lags)];
        assert_eq!(rational_expand(&[1.0, 0.5], 2, &operators), [1.0, 0.5]);
        assert_eq!(rational_expand(&[1.0], 3, &operators), [1.0, 0.0, 0.0]);
    }
}
