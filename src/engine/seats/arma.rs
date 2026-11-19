//! The ARMA family's SEATS run on a regARIMA fit.

use super::strategy::{Check, ConstantValues, FittedPreconditions};
use crate::api::{SeatsFamily, selected_family};
use crate::arima::EstimatedRegArima;
use crate::arima::model::ParameterSlot;
use crate::regarima::{ArimaFactor, ParameterKind, ResolvedArimaModel};
use crate::seats::arma::model::{
    ArOrder, ArmaInput, ArmaLayout, ImportedArma, LowOrder, import_arma,
};
use crate::seats::arma::residual::ArmaPolynomials;
use crate::seats::arma::{ArmaOutcome, ArmaRun, ArmaRunInput, compose_arma};
use crate::seats::effects::FittedEffectsInput;
use crate::time::TimeAxis;
use crate::{Error, Frequency, seats};

fn prepare_fitted_arma<const N: usize>(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    effects: Option<&FittedEffectsInput>,
) -> Result<ImportedArma<N>, Error> {
    let fitted = FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    };
    debug_assert!(fitted.holds(&[Check::Span]));
    let Some(layout) = fitted_layout(fit.spec.model()) else {
        return Err(Error::internal("the fitted orders name no ARMA layout"));
    };
    debug_assert!(fitted.holds(&[
        Check::Layout(&layout_slots(layout, axis.frequency())),
        Check::NoEffects,
        Check::Constant(ConstantValues::CoefficientsAndColumns),
        Check::FiniteEstimates { parameters: N },
        Check::Residuals,
        Check::Likelihood,
    ]));
    let Ok(parameters) = <[f64; N]>::try_from(fit.parameters.as_slice()) else {
        return Err(Error::internal(
            "the fit does not carry the model's parameters",
        ));
    };
    Ok(import_arma(&ArmaInput::with_layout(
        axis.frequency(),
        parameters,
        layout,
    )?)?)
}

/// The ARMA layout the fitted orders name, if the family runs it.
fn fitted_layout(orders: ResolvedArimaModel) -> Option<ArmaLayout> {
    let (regular, seasonal) = (orders.regular(), orders.seasonal());
    let ar = ArOrder::new(regular.p())?;
    let layout = if let Some(SeatsFamily::IntegratedArma(_)) = selected_family(orders) {
        ArmaLayout::Integrated {
            ar,
            ma: LowOrder::new(regular.q())?,
            seasonal_ar: seasonal.p() == 1,
            ordinary_difference: LowOrder::new(regular.d())?,
            seasonal_difference: seasonal.d() == 1,
            seasonal_ma: seasonal.q() == 1,
        }
    } else {
        match (
            regular.d(),
            regular.q(),
            (seasonal.p(), seasonal.d(), seasonal.q()),
        ) {
            (1, 1, (0, 0, 0)) => ArmaLayout::Nonseasonal(ar),
            (1, 2, (0, 0, 0)) => ArmaLayout::NonseasonalMa2(ar),
            (1, 2, (0, 1, 1)) if ar == ArOrder::Zero => ArmaLayout::SeasonalMa2,
            (1, 2, (0, 1, 1)) => ArmaLayout::RegularSeasonalMa2(ar),
            (1, 1, (1, 1, 0)) if ar == ArOrder::Zero => ArmaLayout::SeasonalAr,
            (1, 1, (0, 0, 1)) => ArmaLayout::RegularSeasonalUndifferenced(ar),
            (1, 1, (0, 1, 1)) => ArmaLayout::RegularSeasonal(ar),
            _ => return None,
        }
    };
    layout.is_supported().then_some(layout)
}

/// The layout's free parameters in coordinate order: the regular AR lags,
/// the seasonal AR, the regular MA lags, then the seasonal MA.
fn layout_slots(layout: ArmaLayout, frequency: Frequency) -> Vec<ParameterSlot> {
    let period = usize::from(frequency.periods_per_year());
    let regular = |kind, lags| {
        (1..=lags).map(move |lag| ParameterSlot::new(kind, ArimaFactor::Regular, 1, lag))
    };
    let seasonal = |kind, present: bool| {
        present.then(|| ParameterSlot::new(kind, ArimaFactor::Seasonal, period, period))
    };
    regular(ParameterKind::Ar, layout.regular_ar_coordinates())
        .chain(seasonal(ParameterKind::Ar, layout.has_seasonal_ar()))
        .chain(regular(ParameterKind::Ma, layout.regular_ma_count()))
        .chain(seasonal(ParameterKind::Ma, layout.has_seasonal_ma()))
        .collect()
}

pub(super) struct Strategy<const N: usize = 3>;

impl<const N: usize> super::strategy::FamilyStrategy for Strategy<N> {
    fn compose(
        input: super::strategy::FamilyInput<'_>,
    ) -> Result<super::strategy::Components, Error> {
        let super::strategy::FamilyInput {
            original,
            axis,
            fit,
            effects,
            summary_q,
        } = input;
        let imported = prepare_fitted_arma::<N>(original, axis, fit, effects)?;
        let input = ArmaRunInput {
            summary_q: summary_q.resolve(fit)?,
            imported: &imported,
            initial: crate::seats::airline::Init::Imported2,
            trading_day: super::ar2::configured_trading_day_presence(
                fit.spec.calendar(),
                fit.regression_model.user_regressors(),
            ),
            original,
            transformed: &fit.transformed,
            transform: fit.spec.transform().into(),
            requested_mean: super::strategy::requested_mean(fit),
            carried_lags: None,
        };
        let layout = imported.original.layout;
        let regular_p = layout.regular_ar_coordinates();
        if layout.orders() == crate::seats::Orders::new([0, 0, 1], [0, 0, 0]) {
            // ANSUB1 ChangeModel removes a pure nonseasonal MA(1) before
            // differencing and enters INIT=0; the original regARIMA fit stays.
            let noise = import_arma(&ArmaInput::with_layout(
                axis.frequency(),
                [],
                ArmaLayout::Integrated {
                    ar: ArOrder::Zero,
                    ma: LowOrder::Zero,
                    seasonal_ar: false,
                    ordinary_difference: LowOrder::Zero,
                    seasonal_difference: false,
                    seasonal_ma: false,
                },
            )?)?;
            return compose_prepared(
                ArmaRunInput {
                    imported: &noise,
                    initial: crate::seats::airline::Init::Refit0,
                    summary_q: input.summary_q,
                    trading_day: input.trading_day,
                    original: input.original,
                    transformed: input.transformed,
                    transform: input.transform,
                    requested_mean: input.requested_mean,
                    carried_lags: None,
                },
                effects,
                axis,
            );
        }
        // ANALTS calls ChangeModel before the first differencing pass. Its
        // positive BP/BQ cancellation removes BQ and enters STAVAL/SEARCH
        // with INIT=0. This changes only the SEATS model, never the regARIMA
        // fit or the already prepared regression effects.
        if layout.has_seasonal_ar()
            && !layout.seasonal_difference()
            && layout.has_seasonal_ma()
            && imported.parameters[regular_p] > 0.0
            && imported.parameters[N - 1] > imported.parameters[regular_p]
        {
            return compose_without_seasonal_ma(input, effects, axis);
        }
        crate::seats::arma::model::degenerate_model(&imported)?;
        compose_prepared(input, effects, axis)
    }
}

fn compose_without_seasonal_ma<const N: usize>(
    input: ArmaRunInput<'_, N>,
    effects: Option<&FittedEffectsInput>,
    axis: TimeAxis,
) -> Result<super::strategy::Components, Error> {
    let Some(layout) = input.imported.original.layout.without_seasonal_ma() else {
        return Err(Error::internal(
            "the model has no form without its seasonal MA",
        ));
    };
    match layout.coordinates() {
        1 => compose_reduced::<N, 1>(layout, input, effects, axis),
        2 => compose_reduced::<N, 2>(layout, input, effects, axis),
        3 => compose_reduced::<N, 3>(layout, input, effects, axis),
        4 => compose_reduced::<N, 4>(layout, input, effects, axis),
        5 => compose_reduced::<N, 5>(layout, input, effects, axis),
        _ => Err(Error::internal(
            "the model without its seasonal MA has no coefficients",
        )),
    }
}

fn compose_reduced<const N: usize, const M: usize>(
    layout: ArmaLayout,
    input: ArmaRunInput<'_, N>,
    effects: Option<&FittedEffectsInput>,
    axis: TimeAxis,
) -> Result<super::strategy::Components, Error> {
    let imported = import_arma(&ArmaInput::with_layout(
        axis.frequency(),
        std::array::from_fn(|i| input.imported.original.parameters[i]),
        layout,
    )?)?;
    compose_prepared::<M>(
        ArmaRunInput {
            imported: &imported,
            initial: crate::seats::airline::Init::Refit0,
            summary_q: input.summary_q,
            trading_day: input.trading_day,
            original: input.original,
            transformed: input.transformed,
            transform: input.transform,
            requested_mean: input.requested_mean,
            carried_lags: input.carried_lags,
        },
        effects,
        axis,
    )
}

fn compose_prepared<const N: usize>(
    input: ArmaRunInput<'_, N>,
    effects: Option<&FittedEffectsInput>,
    axis: TimeAxis,
) -> Result<super::strategy::Components, Error> {
    let observed = match compose_arma(input, effects)? {
        ArmaOutcome::Completed(observed) => *observed,
        ArmaOutcome::ChangeModelRestart {
            lags,
            requested_mean,
        } => {
            // ANALTS goto 10 after the post-SEARCH ChangeModel: the same BQ
            // removal as the pre-loop call, entered with INIT=0 and loop 10's
            // persistent lag dimensions. Its own SEARCH sees BQ=0, so it
            // cannot request this restart again.
            return compose_without_seasonal_ma(
                ArmaRunInput {
                    requested_mean,
                    carried_lags: Some(lags),
                    ..input
                },
                effects,
                axis,
            );
        }
    };
    Ok(super::strategy::Components {
        diagnostics: diagnostics(&observed, axis.frequency())?,
        saved: observed.saved,
        transitory: observed.transitory,
    })
}

/// The final model and bound-fixed mask the diagnostics report for these
/// coefficients.
fn reported_model<const N: usize>(
    coefficients: &ArmaPolynomials<N>,
    mask: [bool; N],
    frequency: Frequency,
) -> Result<(seats::Model, Vec<bool>), crate::Error> {
    let crate::seats::Orders {
        p,
        d,
        q,
        bp,
        bd,
        bq,
    } = coefficients.layout.orders();
    let model = seats::Model::new(frequency, [p, d, q], [bp, bd, bq], &coefficients.parameters)?;
    Ok((model, mask.to_vec()))
}

/// The final model is backward CALCFX's after FCAST, else (pure MA) the
/// forward fit's.
fn diagnostics<const N: usize>(
    observed: &ArmaRun<N>,
    frequency: Frequency,
) -> Result<crate::seats::Diagnostics, crate::Error> {
    let summary = super::strategy::PassSummary::new(
        [false; N],
        observed.passes.iter().map(|pass| {
            (
                pass.decision == crate::seats::airline::RetryDecision::RemoveMeanAndRepeat,
                pass.refit.as_ref().map(|search| search.bound_mask),
            )
        }),
    );
    let (model, coefficients_at_bound) = reported_model(
        observed.model.coefficients(),
        summary.bound_fixed,
        frequency,
    )?;
    Ok(crate::seats::Diagnostics {
        mean_requested: observed.requested_mean.into(),
        mean_included: observed.mean.requested.into(),
        model,
        refitted: summary.refit_occurred,
        mean_temporarily_removed: summary.temporary_mean_removal,
        coefficients_at_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::{fitted_layout, reported_model};
    use crate::api::{ArmaCoordinates, SeatsStrategy, selected_family};
    use crate::seats::arma::model::{ArOrder, ArmaLayout, LowOrder};
    use crate::seats::arma::residual::ArmaPolynomials;

    /// The reported orders, after checking that the model keeps every
    /// coordinate and the mask has one flag per coefficient.
    fn orders<const N: usize>(layout: ArmaLayout) -> String {
        let coefficients = ArmaPolynomials {
            layout,
            parameters: std::array::from_fn::<_, N, _>(|i| (i + 1) as f64),
            phist: Vec::new(),
            thstar: Vec::new(),
            previous: None,
        };
        let (model, mask) =
            reported_model(&coefficients, [true; N], crate::Frequency::Monthly).unwrap();
        assert_eq!(model.coefficients(), coefficients.parameters);
        assert_eq!(mask, [true; N]);
        model.orders().to_string()
    }

    /// Every `[p, d, q]` up to `limits`.
    fn orders_up_to([p, d, q]: [u8; 3]) -> impl Iterator<Item = [u8; 3]> {
        (0..=p).flat_map(move |p| (0..=d).flat_map(move |d| (0..=q).map(move |q| [p, d, q])))
    }

    /// Exactly the shared ARMA families name a layout, with the fit's orders
    /// and the coordinate count their strategy was chosen for.
    #[test]
    fn every_arma_family_names_its_layout() {
        let mut named = 0;
        for regular in orders_up_to([6, 3, 4]) {
            for seasonal in orders_up_to([2, 2, 2]) {
                let Ok(model) = crate::regarima::ResolvedArimaModel::new(regular, seasonal) else {
                    continue;
                };
                let count = selected_family(model).and_then(|family| match family.strategy() {
                    SeatsStrategy::Arma(count) => Some(count),
                    _ => None,
                });
                let layout = fitted_layout(model);
                assert_eq!(layout.is_some(), count.is_some(), "{model}");
                let Some(layout) = layout else { continue };
                named += 1;
                let [p, d, q] = regular;
                let [bp, bd, bq] = seasonal;
                assert_eq!(layout.orders().to_array(), [p, d, q, bp, bd, bq]);
                let coordinates = u8::try_from(layout.coordinates()).unwrap();
                assert_eq!(ArmaCoordinates::new(coordinates), count, "{model}");
            }
        }
        assert!(named > 0);
    }

    /// The orders each layout and coordinate count reports.
    #[test]
    fn every_layout_reports_its_orders() {
        use ArmaLayout::*;
        let integrated =
            |ar, ma, seasonal_ar, ordinary_difference, seasonal_difference, seasonal_ma| {
                Integrated {
                    ar,
                    ma,
                    seasonal_ar,
                    ordinary_difference,
                    seasonal_difference,
                    seasonal_ma,
                }
            };
        let cases = [
            (orders::<2>(SeasonalAr), "(0 1 1)(1 1 0)"),
            (orders::<2>(Nonseasonal(ArOrder::One)), "(1 1 1)(0 0 0)"),
            (orders::<3>(Nonseasonal(ArOrder::Two)), "(2 1 1)(0 0 0)"),
            (orders::<4>(Nonseasonal(ArOrder::Three)), "(3 1 1)(0 0 0)"),
            (orders::<2>(NonseasonalMa2(ArOrder::Zero)), "(0 1 2)(0 0 0)"),
            (orders::<3>(NonseasonalMa2(ArOrder::One)), "(1 1 2)(0 0 0)"),
            (orders::<4>(NonseasonalMa2(ArOrder::Two)), "(2 1 2)(0 0 0)"),
            (orders::<3>(SeasonalMa2), "(0 1 2)(0 1 1)"),
            (
                orders::<4>(RegularSeasonalMa2(ArOrder::One)),
                "(1 1 2)(0 1 1)",
            ),
            (
                orders::<5>(RegularSeasonalMa2(ArOrder::Two)),
                "(2 1 2)(0 1 1)",
            ),
            (orders::<3>(RegularSeasonal(ArOrder::One)), "(1 1 1)(0 1 1)"),
            (orders::<4>(RegularSeasonal(ArOrder::Two)), "(2 1 1)(0 1 1)"),
            (
                orders::<5>(RegularSeasonal(ArOrder::Three)),
                "(3 1 1)(0 1 1)",
            ),
            (
                orders::<3>(RegularSeasonalUndifferenced(ArOrder::One)),
                "(1 1 1)(0 0 1)",
            ),
            (
                orders::<4>(RegularSeasonalUndifferenced(ArOrder::Two)),
                "(2 1 1)(0 0 1)",
            ),
            (
                orders::<5>(RegularSeasonalUndifferenced(ArOrder::Three)),
                "(3 1 1)(0 0 1)",
            ),
            (
                orders::<0>(integrated(
                    ArOrder::Zero,
                    LowOrder::Zero,
                    false,
                    LowOrder::One,
                    false,
                    false,
                )),
                "(0 1 0)(0 0 0)",
            ),
            (
                orders::<3>(integrated(
                    ArOrder::One,
                    LowOrder::Zero,
                    true,
                    LowOrder::One,
                    false,
                    true,
                )),
                "(1 1 0)(1 0 1)",
            ),
            (
                orders::<6>(integrated(
                    ArOrder::Three,
                    LowOrder::One,
                    true,
                    LowOrder::Two,
                    true,
                    true,
                )),
                "(3 2 1)(1 1 1)",
            ),
        ];
        for (actual, expected) in cases {
            assert_eq!(actual, expected);
        }
    }
}
