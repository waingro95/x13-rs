use crate::api::spec::{
    MAX_REGULAR_AR_ORDER, MAX_REGULAR_MA_ORDER, MAX_SEASONAL_AR_ORDER, MAX_SEASONAL_MA_ORDER,
    validate_order,
};
use crate::api::{ArimaModel, ArimaOrder, Error};
use crate::regarima::ArimaFactor;

/// One factor, `(p d q)` or `(P D Q)`, of a fitted [`ResolvedArimaModel`].
///
/// It is read only and cannot be used to request a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolvedArimaOrder {
    p: u8,
    d: u8,
    q: u8,
}

/// Displays as `(p d q)`, such as `(0 1 1)`.
impl std::fmt::Display for ResolvedArimaOrder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "({} {} {})", self.p, self.d, self.q)
    }
}

impl ResolvedArimaOrder {
    /// Autoregressive order.
    #[must_use]
    pub const fn p(self) -> u8 {
        self.p
    }
    /// Differencing order.
    #[must_use]
    pub const fn d(self) -> u8 {
        self.d
    }
    /// Moving-average order.
    #[must_use]
    pub const fn q(self) -> u8 {
        self.q
    }
}

/// The fitted ARIMA model, from
/// [`ModelResult::model`](crate::ModelResult::model).
///
/// It differs from a requested [`ArimaModel`] when the automatic search
/// revises its choice. When a fitted AR root is close to one, the search
/// replaces that AR term with a difference, which can take the differencing
/// orders past what [`ArimaOrder::new`] and [`ArimaModel::new`] accept, to
/// at most 3 regular and 2 seasonal differences. A fitted model therefore
/// has its own type and cannot be passed back as a request directly. A model
/// within the request limits converts in both directions without loss:
///
/// ```
/// let order = x13::ArimaOrder::new(1, 1, 1).unwrap();
/// let requested = x13::ArimaModel::new(order, order).unwrap();
/// let resolved = x13::regarima::ResolvedArimaModel::from(requested);
/// assert_eq!(x13::ArimaModel::try_from(resolved), Ok(requested));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolvedArimaModel {
    regular: ResolvedArimaOrder,
    seasonal: ResolvedArimaOrder,
}

/// Displays as `(p d q)(P D Q)`, such as `(0 1 1)(0 1 1)`.
impl std::fmt::Display for ResolvedArimaModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}{}", self.regular, self.seasonal)
    }
}

impl ResolvedArimaModel {
    pub(crate) fn new(regular: [u8; 3], seasonal: [u8; 3]) -> Result<Self, Error> {
        validate_order(
            ArimaFactor::Regular,
            regular[0],
            regular[1],
            regular[2],
            MAX_REGULAR_AR_ORDER,
            3,
            MAX_REGULAR_MA_ORDER,
        )?;
        validate_order(
            ArimaFactor::Seasonal,
            seasonal[0],
            seasonal[1],
            seasonal[2],
            MAX_SEASONAL_AR_ORDER,
            2,
            MAX_SEASONAL_MA_ORDER,
        )?;
        Ok(Self {
            regular: ResolvedArimaOrder {
                p: regular[0],
                d: regular[1],
                q: regular[2],
            },
            seasonal: ResolvedArimaOrder {
                p: seasonal[0],
                d: seasonal[1],
                q: seasonal[2],
            },
        })
    }

    pub(crate) const fn from_explicit(model: ArimaModel) -> Self {
        Self {
            regular: ResolvedArimaOrder {
                p: model.regular().p(),
                d: model.regular().d(),
                q: model.regular().q(),
            },
            seasonal: ResolvedArimaOrder {
                p: model.seasonal().p(),
                d: model.seasonal().d(),
                q: model.seasonal().q(),
            },
        }
    }

    /// Nonseasonal factor.
    #[must_use]
    pub const fn regular(self) -> ResolvedArimaOrder {
        self.regular
    }
    /// Seasonal factor.
    #[must_use]
    pub const fn seasonal(self) -> ResolvedArimaOrder {
        self.seasonal
    }
}

impl From<ArimaModel> for ResolvedArimaModel {
    fn from(model: ArimaModel) -> Self {
        Self::from_explicit(model)
    }
}

impl TryFrom<ResolvedArimaModel> for ArimaModel {
    type Error = Error;

    fn try_from(model: ResolvedArimaModel) -> Result<Self, Self::Error> {
        let regular = ArimaOrder::new(model.regular.p, model.regular.d, model.regular.q)?;
        let seasonal = ArimaOrder::new(model.seasonal.p, model.seasonal.d, model.seasonal.q)?;
        ArimaModel::new(regular, seasonal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ArimaOrderTerm;
    use crate::error::{InvalidSpec, SpecProblem};

    #[test]
    fn accepts_the_differencing_orders_root_repair_can_reach() {
        let model = ResolvedArimaModel::new([1, 3, 1], [1, 2, 1])
            .expect("root repair can reach d = 3 and D = 2");
        assert_eq!(model.regular().d(), 3);
        assert_eq!(model.seasonal().d(), 2);
        assert!(matches!(
            ArimaModel::try_from(model),
            Err(Error::InvalidSpec(InvalidSpec(SpecProblem::ArimaOrder {
                factor: ArimaFactor::Regular,
                term: ArimaOrderTerm::Differencing,
                value: 3,
                maximum: 2,
            })))
        ));
    }

    #[test]
    fn rejects_orders_outside_the_fitted_model_limits() {
        for (regular, seasonal, factor, term, value, maximum) in [
            (
                [4, 0, 0],
                [0, 0, 0],
                ArimaFactor::Regular,
                ArimaOrderTerm::Autoregressive,
                4,
                3,
            ),
            (
                [0, 4, 0],
                [0, 0, 0],
                ArimaFactor::Regular,
                ArimaOrderTerm::Differencing,
                4,
                3,
            ),
            (
                [0, 0, 4],
                [0, 0, 0],
                ArimaFactor::Regular,
                ArimaOrderTerm::MovingAverage,
                4,
                3,
            ),
            (
                [0, 0, 0],
                [3, 0, 0],
                ArimaFactor::Seasonal,
                ArimaOrderTerm::Autoregressive,
                3,
                2,
            ),
            (
                [0, 0, 0],
                [0, 3, 0],
                ArimaFactor::Seasonal,
                ArimaOrderTerm::Differencing,
                3,
                2,
            ),
            (
                [0, 0, 0],
                [0, 0, 3],
                ArimaFactor::Seasonal,
                ArimaOrderTerm::MovingAverage,
                3,
                2,
            ),
        ] {
            assert_eq!(
                ResolvedArimaModel::new(regular, seasonal),
                Err(Error::InvalidSpec(InvalidSpec(SpecProblem::ArimaOrder {
                    factor,
                    term,
                    value,
                    maximum
                })))
            );
        }
    }

    #[test]
    fn conversion_to_a_request_names_the_factor_out_of_range() {
        let seasonal_only = ResolvedArimaModel::new([1, 2, 1], [1, 2, 1]).unwrap();
        assert_eq!(
            ArimaModel::try_from(seasonal_only),
            Err(Error::InvalidSpec(InvalidSpec(SpecProblem::ArimaOrder {
                factor: ArimaFactor::Seasonal,
                term: ArimaOrderTerm::Differencing,
                value: 2,
                maximum: 1,
            })))
        );
    }

    #[test]
    fn every_requestable_model_round_trips() {
        let mut checked = 0;
        for p in 0..=3 {
            for d in 0..=2 {
                for q in 0..=3 {
                    for seasonal_p in 0..=2 {
                        for seasonal_d in 0..=1 {
                            for seasonal_q in 0..=2 {
                                let regular = ArimaOrder::new(p, d, q).unwrap();
                                let seasonal =
                                    ArimaOrder::new(seasonal_p, seasonal_d, seasonal_q).unwrap();
                                let requested = ArimaModel::new(regular, seasonal).unwrap();
                                let resolved = ResolvedArimaModel::from(requested);
                                assert_eq!(resolved.regular().p(), p);
                                assert_eq!(resolved.regular().d(), d);
                                assert_eq!(resolved.regular().q(), q);
                                assert_eq!(resolved.seasonal().p(), seasonal_p);
                                assert_eq!(resolved.seasonal().d(), seasonal_d);
                                assert_eq!(resolved.seasonal().q(), seasonal_q);
                                assert_eq!(ArimaModel::try_from(resolved), Ok(requested));
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 864);
    }
}
