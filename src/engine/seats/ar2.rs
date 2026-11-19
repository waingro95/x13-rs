//! The AR(2)/AR(3) family's SEATS run on a regARIMA fit.
use super::pipeline::SummaryQSource;
use super::strategy::{Check, FittedPreconditions};
use crate::arima::EstimatedRegArima;
use crate::arima::model::ParameterSlot;
use crate::regarima::{ArimaFactor, ParameterKind};
use crate::regression::model::{RegressionModel, RegressorKind};
use crate::seats::ar2::transform::Ar2Coefficients;
use crate::seats::ar2::{Ar2Run, compose_ar2};
use crate::seats::effects::FittedEffectsInput;
use crate::seats::failure::{Stage, usable};
use crate::seats::pipeline::RunInput;
use crate::seats::trading_day::TradingDayPresence;
use crate::time::TimeAxis;
use crate::{Error, Frequency};

/// Whether the fitted model is `(2 1 0)(0 1 1)` or `(3 1 0)(0 1 1)` and the
/// ARMA workspace holds it.
fn ar2_orders(fit: &EstimatedRegArima) -> bool {
    let orders = fit.spec.model();
    let (regular, seasonal) = (orders.regular(), orders.seasonal());
    [2, 3].contains(&regular.p())
        && (
            regular.d(),
            regular.q(),
            seasonal.p(),
            seasonal.d(),
            seasonal.q(),
        ) == (1, 0, 0, 1, 1)
        && fit.arma_workspace.model() == orders
}

/// The parameter slots of an AR(`p`) with seasonal MA: the regular AR lags,
/// then the seasonal MA.
fn ar2_slots(frequency: Frequency, p: usize) -> Vec<ParameterSlot> {
    let s = usize::from(frequency.periods_per_year());
    (1..=p)
        .map(|lag| ParameterSlot::new(ParameterKind::Ar, ArimaFactor::Regular, 1, lag))
        .chain([ParameterSlot::new(
            ParameterKind::Ma,
            ArimaFactor::Seasonal,
            s,
            s,
        )])
        .collect()
}

fn compose_fitted_ar2(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    effects: Option<&FittedEffectsInput>,
    summary_q: SummaryQSource,
) -> Result<Ar2Run, Error> {
    let fitted = FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    };
    debug_assert!(ar2_orders(fit));
    let p = usize::from(fit.spec.model().regular().p());
    debug_assert!(fitted.holds(&[
        Check::Span,
        Check::Layout(&ar2_slots(axis.frequency(), p)),
        Check::FiniteEstimates { parameters: p + 1 },
        Check::Residuals,
        Check::Likelihood,
    ]));
    let Some(parameters) = Ar2Coefficients::from_parameters(&fit.parameters) else {
        return Err(Error::internal(
            "the fit does not carry the model's parameters",
        ));
    };
    // ansub9:1055 derives NPATD from configured flow/length and supplied TD
    // columns, even when their fitted coefficients are zero.
    let presence = configured_trading_day_presence(
        fit.spec.calendar(),
        fit.regression_model.user_regressors(),
    );
    debug_assert!(regression_matches_spec(fitted));
    for (index, &value) in fit.parameters.iter().enumerate() {
        usable("fitted model coefficient", Some(index), value)
            .map_err(|failure| failure.at(Stage::Import))?;
    }
    let input = RunInput {
        original,
        transformed: &fit.transformed,
        transform: fit.spec.transform().into(),
        requested_mean: super::strategy::requested_mean(fit),
        summary_q: summary_q.resolve(fit)?,
    };
    // Only a run with prepared fitted effects carries the configured trading
    // day into SIGEX.
    let presence = if effects.is_some() {
        presence
    } else {
        TradingDayPresence::Absent
    };
    Ok(compose_ar2(axis, parameters, presence, input, effects)?)
}

/// Whether fitted effects are prepared exactly when the regression needs
/// them, and the fitted regression holds the configured regressors once
/// each, in the fitted order, with only declared outliers and missing values
/// on observed dates besides them, finite coefficients and finite columns
/// that span the series.
fn regression_matches_spec(fitted: FittedPreconditions<'_>) -> bool {
    let FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    } = fitted;
    let noncalendar_effects = super::strategy::has_noncalendar_effects(&fit.regressor_kinds);
    if effects.is_none() && noncalendar_effects
        || effects.is_some() && fit.spec.calendar().is_none() && !noncalendar_effects
    {
        return false;
    }
    let n = original.len();
    let Ok(mut base) = RegressionModel::from_spec(fit.spec, axis, n) else {
        return false;
    };
    if base
        .add_manual_outliers(
            fit.regression_model.manual_outliers(),
            axis,
            n,
            fit.spec.resolved_temporary_change_rate(axis.frequency()),
        )
        .is_err()
        || base
            .add_user_regressors(fit.regression_model.user_regressors(), axis, n)
            .is_err()
    {
        return false;
    }
    let base_kinds = base.kinds().collect::<Vec<_>>();
    let observed = |period| {
        axis.periods(n)
            .is_ok_and(|mut dates| dates.any(|date| date == period))
    };
    // automd appends Constant after calendar/MV columns. Keep the fitted
    // numerical order; validate configured identities, not a cold-start prefix.
    fit.regressor_kinds.len() >= base_kinds.len()
        && base_kinds.iter().all(|kind| {
            fit.regressor_kinds
                .iter()
                .filter(|fitted| *fitted == kind)
                .count()
                == 1
        })
        && fit.regressor_kinds.iter().all(|kind| match kind {
            kind if base_kinds.contains(kind) => true,
            RegressorKind::MissingValue { period, frequency } => {
                *frequency == axis.frequency() && observed(*period)
            }
            RegressorKind::Outlier {
                kind,
                period,
                frequency,
            } => {
                *frequency == axis.frequency()
                    && fit
                        .spec
                        .outliers()
                        .is_some_and(|request| request.types().contains(*kind))
                    && observed(*period)
            }
            _ => false,
        })
        && fit.regression_model.kinds().collect::<Vec<_>>() == fit.regressor_kinds
        && fit.regression_coefficients.len() == fitted.regression_count()
        && fit.regression_coefficients.iter().all(|v| v.is_finite())
        && fit
            .regression_model
            .columns()
            .all(|(_, c)| c.len() == n && c.iter().all(|v| v.is_finite()))
}

/// Whether SIGEX keeps a stochastic trading-day allocation: a flow
/// trading-day or length calendar is configured, or a trading-day regressor
/// is supplied.
pub(super) fn configured_trading_day_presence(
    calendar: Option<crate::regarima::CalendarSpec>,
    user_regressors: &[crate::regarima::UserRegressor],
) -> TradingDayPresence {
    let configured = calendar.is_some_and(|calendar| {
        matches!(
            calendar.basis(),
            Some(
                crate::regarima::CalendarBasis::FlowTradingDay(_)
                    | crate::regarima::CalendarBasis::Length(_)
            )
        )
    });
    // chkadj.f includes PRGUTD in Nflwtd; ansub9 carries that presence into
    // NPATD, which SIGEX uses to retain stochastic trading-day allocation.
    let supplied = user_regressors
        .iter()
        .any(|regressor| regressor.kind() == crate::regarima::UserRegressorType::TradingDay);
    TradingDayPresence::from_configured_basis(configured || supplied)
}

pub(super) struct Strategy;

impl super::strategy::FamilyStrategy for Strategy {
    fn compose(
        input: super::strategy::FamilyInput<'_>,
    ) -> Result<super::strategy::Components, crate::Error> {
        let super::strategy::FamilyInput {
            original,
            axis,
            fit,
            effects,
            summary_q,
        } = input;
        let observed = compose_fitted_ar2(original, axis, fit, effects, summary_q)?;
        let p = observed.model.coefficients.regular.as_slice().len();
        let passes = super::strategy::PassSummary::new(
            vec![false; p + 1],
            observed.passes.iter().map(|pass| {
                (
                    pass.decision == crate::seats::airline::RetryDecision::RemoveMeanAndRepeat,
                    pass.refit.as_ref().map(|refit| refit.bound_mask.clone()),
                )
            }),
        );
        let diagnostics = crate::seats::Diagnostics {
            mean_requested: observed.requested_mean.into(),
            mean_included: observed.mean.requested.into(),
            model: reported_model(&observed.model)?,
            refitted: passes.refit_occurred,
            mean_temporarily_removed: passes.temporary_mean_removal,
            coefficients_at_bound: passes.bound_fixed,
        };
        Ok(super::strategy::Components {
            saved: observed.saved,
            diagnostics,
            transitory: observed.transitory,
        })
    }
}

/// The public form of an AR2/AR3 family model: `(p 1 0)(0 1 1)`.
pub(super) fn reported_model(
    model: &crate::seats::ar2::canonical::ResolvedAr2,
) -> Result<crate::seats::Model, crate::Error> {
    let regular = model.coefficients.regular;
    crate::seats::Model::new(
        model.frequency,
        [regular.degree(), 1, 0],
        [0, 1, 1],
        &[regular.as_slice(), &[model.coefficients.seasonal_ma]].concat(),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn supplied_trading_day_controls_stochastic_allocation_presence() {
        use super::configured_trading_day_presence;
        use crate::Series;
        use crate::regarima::{
            CalendarBasis, CalendarSpec, LengthEffect, UserRegressor, UserRegressorType,
        };
        use crate::seats::trading_day::TradingDayPresence::{Absent, Present};
        use crate::time::TimeAxis;

        let axis = TimeAxis::monthly(2000, 1).unwrap();
        assert_eq!(configured_trading_day_presence(None, &[]), Absent);
        assert_eq!(
            configured_trading_day_presence(
                Some(CalendarSpec::new().with_basis(CalendarBasis::Length(LengthEffect::LeapYear))),
                &[],
            ),
            Present,
        );
        for (kind, expected) in [
            (UserRegressorType::TradingDay, Present),
            (UserRegressorType::Holiday, Absent),
            (UserRegressorType::User, Absent),
        ] {
            // chkadj counts the declared group even when its values are zero.
            let supplied = UserRegressor::new(kind, Series::new(axis.start(), &[0.0])).unwrap();
            assert_eq!(configured_trading_day_presence(None, &[supplied]), expected);
        }
    }

    #[test]
    fn supplied_only_effects_compose_with_ar2_seats() {
        use crate::regarima::{RegressionVariable, UserRegressor, UserRegressorType};
        use crate::time::TimeAxis;
        use crate::{ArimaModel, ArimaOrder, RegArimaSpec, Series, Spec};

        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let supplied = (0..396)
            .map(|i| ((i * 13 % 31) as f64 - 15.0) / 15.0)
            .collect::<Vec<_>>();
        let values = (0..360)
            .map(|i| {
                100.0
                    + i as f64 * 0.2
                    + 12.0 * (i as f64 * std::f64::consts::TAU / 12.0).sin()
                    + ((i * 37 % 101) as f64 - 50.0) * 0.1
                    + 3.0 * supplied[i]
            })
            .collect::<Vec<_>>();
        let regressor = UserRegressor::new(
            UserRegressorType::User,
            Series::new(axis.start(), &supplied),
        )
        .unwrap();
        let request = RegArimaSpec::new(
            ArimaModel::new(
                ArimaOrder::new(2, 1, 0).unwrap(),
                ArimaOrder::new(0, 1, 1).unwrap(),
            )
            .unwrap(),
        )
        .with_user_regressors(&[regressor]);
        let adjusted =
            crate::adjust(Series::new(axis.start(), &values), &Spec::seats(request)).unwrap();
        assert_eq!(adjusted.seasonally_adjusted().len(), values.len());
        let model = adjusted.model().unwrap();
        assert_eq!(model.regression_coefficients().len(), 1);
        assert_eq!(
            model.regression_coefficients()[0].variable(),
            RegressionVariable::User {
                index: 0,
                kind: UserRegressorType::User
            },
        );
    }
}
