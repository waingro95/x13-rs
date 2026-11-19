use crate::numerics::polynomial::{OwnedPolynomial, Polynomial, multiply_polynomials};

use super::model::OperatorSpec;

/// Each operator as a polynomial, its coefficients drawn from `parameters`.
pub(crate) fn polynomials<'a>(
    operators: &'a [OperatorSpec],
    parameters: &'a [f64],
) -> Vec<Polynomial<'a>> {
    operators
        .iter()
        .map(|operator| polynomial(operator, parameters))
        .collect()
}

fn polynomial<'a>(operator: &'a OperatorSpec, parameters: &'a [f64]) -> Polynomial<'a> {
    Polynomial::new(operator.coefficients(parameters), operator.lags())
}

/// The product of `operators`, as `fcstxy.f` forms its full AR and MA.
pub(crate) fn multiply_operators(
    operators: &[OperatorSpec],
    parameters: &[f64],
) -> OwnedPolynomial {
    let mut factors = operators
        .iter()
        .map(|operator| polynomial(operator, parameters));
    let Some(first) = factors.next() else {
        return OwnedPolynomial::default();
    };
    // fcstxy.f calls polyml(new factor, accumulated product). Reversing
    // these operands changes the order of additions at overlapping lags.
    factors.fold(OwnedPolynomial::from(first), |product, factor| {
        multiply_polynomials(factor, product.as_polynomial())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arima::model::ExpandedModel;
    use crate::{ArimaModel, ArimaOrder, Frequency};

    #[test]
    fn airline_products_match_monthly_and_quarterly_lags() {
        let model = ArimaModel::new(
            ArimaOrder::new(0, 1, 1).unwrap(),
            ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        let parameters = [0.051493974690264, 0.99848699907562];

        for (frequency, period) in [(Frequency::Monthly, 12), (Frequency::Quarterly, 4)] {
            let expanded = ExpandedModel::from_spec(model, frequency);
            let difference = multiply_operators(expanded.differencing_operators(), &parameters);
            assert_eq!(
                difference.terms().collect::<Vec<_>>(),
                [(1.0, 1), (1.0, period), (-1.0, period + 1)]
            );

            let moving_average = multiply_operators(expanded.ma_operators(), &parameters);
            assert_eq!(
                moving_average.terms().collect::<Vec<_>>(),
                [
                    (0.051493974690264, 1),
                    (0.99848699907562, period),
                    (-0.051416064258957624, period + 1)
                ]
            );
        }
    }

    #[test]
    fn ar1_and_ma2_keep_coefficients_in_one_factor() {
        let ar1 = ArimaModel::new(
            ArimaOrder::new(1, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap();
        let expanded = ExpandedModel::from_spec(ar1, Frequency::Monthly);
        let operator = multiply_operators(expanded.ar_operators(), &[0.95]);
        assert_eq!(operator.terms().collect::<Vec<_>>(), [(0.95, 1)]);
        assert_eq!(
            multiply_operators(expanded.ma_operators(), &[0.95])
                .terms()
                .count(),
            0
        );

        let ma2 = ArimaModel::new(
            ArimaOrder::new(0, 0, 2).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap();
        let expanded = ExpandedModel::from_spec(ma2, Frequency::Monthly);
        let operator = multiply_operators(expanded.ma_operators(), &[0.3, -0.2]);
        assert_eq!(operator.terms().collect::<Vec<_>>(), [(0.3, 1), (-0.2, 2)]);
    }
}
