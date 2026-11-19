use std::ops::Range;

use super::roots::{RootComputationError, RootLocation, has_strictly_inside_root, root_location};
use crate::Frequency;
use crate::regarima::{ArimaFactor, ParameterKind, ResolvedArimaModel};

/// `setmdl.f` shrinks an MA factor that has a root within this modulus.
const BUILD_62_NEAR_UNIT_ROOT: f64 = 1.06;
/// The factor, per lag, by which `setmdl.f` shrinks it.
const BUILD_62_MA_SHRINK: f64 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParameterSlot {
    kind: ParameterKind,
    factor: ArimaFactor,
    period: usize,
    lag: usize,
}

impl ParameterSlot {
    pub(crate) const fn new(
        kind: ParameterKind,
        factor: ArimaFactor,
        period: usize,
        lag: usize,
    ) -> Self {
        Self {
            kind,
            factor,
            period,
            lag,
        }
    }
}

#[derive(Debug, Clone)]
enum CoefficientSource {
    Fixed(Vec<f64>),
    Parameters(Range<usize>),
}

#[derive(Debug, Clone)]
pub(crate) struct OperatorSpec {
    coefficients: CoefficientSource,
    lags: Vec<usize>,
    factor: ArimaFactor,
}

impl OperatorSpec {
    fn parameters(range: Range<usize>, lags: Vec<usize>, factor: ArimaFactor) -> Self {
        Self {
            coefficients: CoefficientSource::Parameters(range),
            lags,
            factor,
        }
    }

    /// `(1 - B^period)^order`, stored as the `c_k` of `1 - c_1 B^period - ...`
    /// by the binomial theorem.
    fn difference(order: u8, period: usize, factor: ArimaFactor) -> Self {
        let mut coefficient = -1;
        let mut coefficients = Vec::with_capacity(usize::from(order));
        let mut lags = Vec::with_capacity(usize::from(order));
        for power in 1..=order {
            coefficient = -coefficient * i32::from(order - power + 1) / i32::from(power);
            coefficients.push(f64::from(coefficient));
            lags.push(usize::from(power) * period);
        }
        Self {
            coefficients: CoefficientSource::Fixed(coefficients),
            lags,
            factor,
        }
    }

    pub(crate) fn coefficients<'a>(&'a self, parameters: &'a [f64]) -> &'a [f64] {
        match &self.coefficients {
            CoefficientSource::Fixed(coefficients) => coefficients,
            CoefficientSource::Parameters(range) => &parameters[range.clone()],
        }
    }

    pub(crate) fn lags(&self) -> &[usize] {
        &self.lags
    }

    /// The operator's highest lag, which is the number of leading rows it
    /// consumes. Lags are stored ascending.
    pub(crate) fn span(&self) -> usize {
        self.lags.last().copied().unwrap_or(0)
    }
}

/// The summed span of a product of operators.
pub(crate) fn operator_span(operators: &[OperatorSpec]) -> usize {
    operators.iter().map(OperatorSpec::span).sum()
}

#[derive(Debug, Clone)]
pub(crate) struct ExpandedModel {
    parameter_slots: Vec<ParameterSlot>,
    ar_operators: Vec<OperatorSpec>,
    differencing_operators: Vec<OperatorSpec>,
    ma_operators: Vec<OperatorSpec>,
}

impl ExpandedModel {
    pub(crate) fn from_spec(model: impl Into<ResolvedArimaModel>, frequency: Frequency) -> Self {
        let model = model.into();
        let regular = model.regular();
        let seasonal = model.seasonal();
        let seasonal_period = usize::from(frequency.periods_per_year());
        let mut parameter_slots = Vec::new();
        let mut ar_operators = Vec::new();
        let mut ma_operators = Vec::new();

        add_parameter_operator(
            ParameterKind::Ar,
            ArimaFactor::Regular,
            regular.p(),
            1,
            &mut parameter_slots,
            &mut ar_operators,
        );
        add_parameter_operator(
            ParameterKind::Ar,
            ArimaFactor::Seasonal,
            seasonal.p(),
            seasonal_period,
            &mut parameter_slots,
            &mut ar_operators,
        );

        let mut differencing_operators = Vec::new();
        if regular.d() > 0 {
            differencing_operators.push(OperatorSpec::difference(
                regular.d(),
                1,
                ArimaFactor::Regular,
            ));
        }
        if seasonal.d() > 0 {
            differencing_operators.push(OperatorSpec::difference(
                seasonal.d(),
                seasonal_period,
                ArimaFactor::Seasonal,
            ));
        }

        add_parameter_operator(
            ParameterKind::Ma,
            ArimaFactor::Regular,
            regular.q(),
            1,
            &mut parameter_slots,
            &mut ma_operators,
        );
        add_parameter_operator(
            ParameterKind::Ma,
            ArimaFactor::Seasonal,
            seasonal.q(),
            seasonal_period,
            &mut parameter_slots,
            &mut ma_operators,
        );

        Self {
            parameter_slots,
            ar_operators,
            differencing_operators,
            ma_operators,
        }
    }

    pub(crate) fn parameter_slots(&self) -> &[ParameterSlot] {
        &self.parameter_slots
    }

    pub(crate) fn ar_operators(&self) -> &[OperatorSpec] {
        &self.ar_operators
    }

    pub(crate) fn differencing_operators(&self) -> &[OperatorSpec] {
        &self.differencing_operators
    }

    pub(crate) fn ma_operators(&self) -> &[OperatorSpec] {
        &self.ma_operators
    }

    /// The first MA factor with a root strictly inside the unit circle, which
    /// `setmdl.f` rejects before it shrinks near-unit roots.
    pub(crate) fn first_inside_ma_factor(
        &self,
        parameters: &[f64],
    ) -> Result<Option<ArimaFactor>, RootComputationError> {
        for operator in &self.ma_operators {
            if has_strictly_inside_root(operator.coefficients(parameters))? {
                return Ok(Some(operator.factor));
            }
        }
        Ok(None)
    }

    /// Applies `setmdl.f`'s MA setup to the estimates of a previous fit.
    ///
    /// Build 62 shrinks every coefficient in an MA factor when any root of
    /// that factor is at most 1.06 from the origin. The exponent uses the
    /// coefficient's expanded lag, including the seasonal period.
    pub(crate) fn prepare_previous_parameters(&self, parameters: &mut [f64]) {
        for operator in &self.ma_operators {
            let CoefficientSource::Parameters(range) = &operator.coefficients else {
                continue;
            };
            let scaled = parameters[range.clone()]
                .iter()
                .enumerate()
                .map(|(index, coefficient)| {
                    coefficient * BUILD_62_NEAR_UNIT_ROOT.powi((index + 1) as i32)
                })
                .collect::<Vec<_>>();
            if root_location(&scaled) == RootLocation::Outside {
                continue;
            }
            for (parameter, lag) in parameters[range.clone()].iter_mut().zip(&operator.lags) {
                *parameter *= BUILD_62_MA_SHRINK.powi(*lag as i32);
            }
        }
    }

    pub(crate) fn ar_span(&self) -> usize {
        operator_span(&self.ar_operators)
    }

    pub(crate) fn differencing_loss(&self) -> usize {
        operator_span(&self.differencing_operators)
    }

    pub(crate) fn ma_span(&self) -> usize {
        operator_span(&self.ma_operators)
    }

    /// Observations left after differencing (`rgarma.f`'s `Nefobs`); `None`
    /// when differencing leaves none.
    pub(crate) fn effective_observations(&self, series_len: usize) -> Option<usize> {
        series_len
            .checked_sub(self.differencing_loss())
            .filter(|count| *count > 0)
    }
}

fn add_parameter_operator(
    kind: ParameterKind,
    factor: ArimaFactor,
    order: u8,
    period: usize,
    parameter_slots: &mut Vec<ParameterSlot>,
    operators: &mut Vec<OperatorSpec>,
) {
    if order == 0 {
        return;
    }

    let start = parameter_slots.len();
    let mut lags = Vec::with_capacity(usize::from(order));
    for factor_lag in 1..=usize::from(order) {
        let lag = factor_lag * period;
        lags.push(lag);
        parameter_slots.push(ParameterSlot::new(kind, factor, period, lag));
    }
    operators.push(OperatorSpec::parameters(
        start..parameter_slots.len(),
        lags,
        factor,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regarima::ParameterKind;
    use crate::{ArimaModel, ArimaOrder};

    #[test]
    fn model_stores_parameters_in_build_62_order() {
        let model = ArimaModel::new(
            ArimaOrder::new(1, 0, 2).unwrap(),
            ArimaOrder::new(1, 0, 1).unwrap(),
        )
        .unwrap();
        let expanded = ExpandedModel::from_spec(model, Frequency::Monthly);

        assert_eq!(
            expanded.parameter_slots(),
            [
                ParameterSlot::new(ParameterKind::Ar, ArimaFactor::Regular, 1, 1),
                ParameterSlot::new(ParameterKind::Ar, ArimaFactor::Seasonal, 12, 12),
                ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Regular, 1, 1),
                ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Regular, 1, 2),
                ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Seasonal, 12, 12),
            ]
        );
    }

    #[test]
    fn airline_layout_keeps_regular_and_seasonal_factors_separate() {
        let model = ArimaModel::new(
            ArimaOrder::new(0, 1, 1).unwrap(),
            ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        let monthly = ExpandedModel::from_spec(model, Frequency::Monthly);
        let quarterly = ExpandedModel::from_spec(model, Frequency::Quarterly);

        assert_eq!(monthly.differencing_operators()[0].coefficients(&[]), [1.0]);
        assert_eq!(monthly.differencing_operators()[0].lags(), [1]);
        assert_eq!(monthly.differencing_operators()[1].coefficients(&[]), [1.0]);
        assert_eq!(monthly.differencing_operators()[1].lags(), [12]);
        assert_eq!(monthly.ma_operators()[0].lags(), [1]);
        assert_eq!(monthly.ma_operators()[1].lags(), [12]);
        assert_eq!(quarterly.differencing_operators()[1].lags(), [4]);
        assert_eq!(quarterly.ma_operators()[1].lags(), [4]);
    }

    #[test]
    fn first_inside_ma_factor_names_the_first_factor_with_an_inside_root() {
        let order = ArimaOrder::new(0, 0, 1).unwrap();
        let model = ArimaModel::new(order, order).unwrap();
        let expanded = ExpandedModel::from_spec(model, Frequency::Monthly);
        for (parameters, expected) in [
            ([0.5, 0.5], None),
            ([1.5, 0.5], Some(ArimaFactor::Regular)),
            ([0.5, 1.5], Some(ArimaFactor::Seasonal)),
            ([1.5, 1.5], Some(ArimaFactor::Regular)),
        ] {
            assert_eq!(
                expanded.first_inside_ma_factor(&parameters).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn a_third_regular_and_second_seasonal_difference_are_two_operators() {
        for (frequency, seasonal_period, expected_loss) in
            [(Frequency::Monthly, 12, 27), (Frequency::Quarterly, 4, 11)]
        {
            let actual = ExpandedModel::from_spec(
                ResolvedArimaModel::new([1, 3, 1], [1, 2, 1]).unwrap(),
                frequency,
            );
            let operators = actual.differencing_operators();
            assert_eq!(operators.len(), 2);
            assert_eq!(operators[0].coefficients(&[]), [3.0, -3.0, 1.0]);
            assert_eq!(operators[0].lags(), [1, 2, 3]);
            assert_eq!(operators[1].coefficients(&[]), [2.0, -1.0]);
            assert_eq!(operators[1].lags(), [seasonal_period, 2 * seasonal_period]);
            assert_eq!(actual.differencing_loss(), expected_loss);
            assert_eq!(
                actual.parameter_slots(),
                [
                    ParameterSlot::new(ParameterKind::Ar, ArimaFactor::Regular, 1, 1),
                    ParameterSlot::new(
                        ParameterKind::Ar,
                        ArimaFactor::Seasonal,
                        seasonal_period,
                        seasonal_period
                    ),
                    ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Regular, 1, 1),
                    ParameterSlot::new(
                        ParameterKind::Ma,
                        ArimaFactor::Seasonal,
                        seasonal_period,
                        seasonal_period
                    ),
                ]
            );
        }
    }
}
