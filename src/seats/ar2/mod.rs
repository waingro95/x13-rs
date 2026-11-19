//! The AR(2)/AR(3) family, `(2 1 0)(0 1 1)` and `(3 1 0)(0 1 1)`, on the
//! shared ANALTS run loop.
//!
//! The model is two or three regular AR coefficients and a seasonal MA
//! coefficient. A pass computes CALCFX's residuals for the imported
//! coefficients or refits them with SEARCH from STAVAL's starting values; the
//! completed pass runs FCAST, SPECTRUM's canonical decomposition with the AR
//! roots from RPQ, the ESTBUR extraction and DETCOMP's final components. AR
//! roots allocated to the transitory component give the run a transitory
//! series. A refit after which CHMODEL would lower the AR order ends the run
//! with an error.

pub(crate) mod canonical;
pub(crate) mod refit;
pub(crate) mod residual;
pub(crate) mod transform;

use super::decomposition::root_allocation::RegularArCoefficients;
use super::effects::FittedEffectsInput;
use super::extraction::{
    ExtractedComponents, FinalWithTransitory, extract_ar2, final_identity_with_transitory,
};
use super::failure::{CanonicalComponent, Error, Failure, ModelChange, Stage};
use super::log_transform::final_log_with_transitory;
use super::mean::{MeanCorrection, MeanFamily, RequestedMean, mean_correction};
use super::pipeline::{
    Composed, DifferencedPass, Family, FittedPass, PassShape, Restoration, RunInput,
    SeatsTransform, run_family,
};
use super::residual::{ResidualResult, finite};
use super::retry::Init;
use super::tails::{
    DifferenceOrder, ForwardTailState, MeanForcing, SeriesTails, TailInputs, bphist_ordered,
    finish_tails_numeric, resolve_horizons, single_lag_bj_ma,
};
use super::trading_day::{EffectiveTradingDay, TradingDayPresence};
use crate::Frequency;
use crate::time::TimeAxis;
use canonical::ResolvedAr2;
use refit::Ar2RefitResult;
use transform::{
    Ar2Coefficients, starting_parameters, trans1_coefficients, transform_regular_start,
    transform_seasonal_start,
};

/// A completed AR(2)/AR(3) run.
pub(crate) type Ar2Run = Composed<ResolvedAr2, Ar2RefitResult>;

/// CALCFX's residuals of `model` on the differenced series `wd`.
fn resolved_residuals(model: ResolvedAr2, wd: &[f64]) -> Result<ResidualResult, Error> {
    let u = residual::ar2_prefilter(model.coefficients.regular, wd)?;
    super::seasonal_ma::residual::seasonal_constrained_residuals(
        usize::from(model.frequency.periods_per_year()),
        model.coefficients.seasonal_ma,
        &u,
    )
}

/// SEARCH's fixed mask: ANALTS (analts.f:2175–2186) fixes the first P-1
/// coordinates when all of their coefficients are zero within CEPS.
fn refit_mask<const N: usize>(regular: RegularArCoefficients) -> [bool; N] {
    let regular = regular.as_slice();
    let lower = &regular[..regular.len() - 1];
    let fix = lower.iter().all(|v| v.abs() <= 1e-13);
    std::array::from_fn(|coordinate| fix && coordinate < lower.len())
}

/// The orders CHMODEL proposes after SEARCH fixed `fixed` coordinates and
/// left the regular AR coordinates at `regular`, if any.
fn chmodel_orders(regular: RegularArCoefficients, fixed: usize) -> Option<super::Orders> {
    if fixed == 0 {
        return None;
    }
    // Default seas=1, P=2/3, D=1: only the first encountered eligible positive
    // unit root changes orders, because its D++ prevents a second reduction.
    // Believed unreachable: with XL=.99, SEARCH (ansub1.f:169-205) fixes
    // bounded AR coordinates only at -UR or .99, so Build 62's +UR test is
    // dead unless a free coordinate lands on +1. Kept as in Build 62.
    regular
        .as_slice()
        .iter()
        .any(|coordinate| (coordinate - 1.0).abs() < 1e-12)
        .then(|| super::Orders::new([regular.degree() - 1, 2, 0], [0, 1, 1]))
}

/// The error for CHMODEL's proposal to re-estimate the refit `model` with
/// the `proposed` orders.
fn model_change(model: &ResolvedAr2, proposed: super::Orders) -> Error {
    let regular = model.coefficients.regular;
    Failure::ModelChange(ModelChange {
        current: super::Orders::new([regular.degree(), 1, 0], [0, 1, 1]),
        proposed,
    })
    .at(Stage::Refit)
}

/// Runs SEATS on the regARIMA fit's `parameters` over `axis`, restoring
/// `calendar`'s fitted effects when given. `presence` is the configured
/// trading day, which SIGEX resolves against the model of the last pass.
pub(crate) fn compose_ar2(
    axis: TimeAxis,
    parameters: Ar2Coefficients,
    presence: TradingDayPresence,
    input: RunInput<'_>,
    calendar: Option<&FittedEffectsInput>,
) -> Result<Ar2Run, Error> {
    debug_assert!(calendar.is_none_or(|calendar| calendar.axis() == axis));
    let family = Ar2 {
        axis,
        parameters,
        presence,
    };
    run_family(&family, input, calendar)
}

struct Ar2 {
    axis: TimeAxis,
    parameters: Ar2Coefficients,
    presence: TradingDayPresence,
}

impl Ar2 {
    /// The regular AR degree P.
    fn degree(&self) -> usize {
        self.parameters.regular.as_slice().len()
    }
}

/// The decomposition: ESTBUR's components, the innovation variance of the
/// transitory component and the trading day SIGEX resolved.
struct Ar2Decomposition {
    components: ExtractedComponents,
    cycle_variance: f64,
    trading_day: EffectiveTradingDay,
}

impl Family for Ar2 {
    type Start = ResolvedAr2;
    type Model = ResolvedAr2;
    type Refit = Ar2RefitResult;
    type Tails = SeriesTails;
    type Decomposition = Ar2Decomposition;
    type Failure = Error;

    const TMU_BEFORE_LOG: bool = false;
    const DIFFERENCES_LOGGED_OZ: bool = false;

    fn frequency(&self) -> Frequency {
        self.axis.frequency()
    }

    fn initial_start(&self) -> Result<ResolvedAr2, Error> {
        ResolvedAr2::imported(self.frequency(), self.parameters)
    }

    fn pass_shape(
        &self,
        _model: &ResolvedAr2,
        _observed_len: usize,
        _requested: RequestedMean,
    ) -> Result<PassShape, Error> {
        Ok(PassShape {
            differences: DifferenceOrder::First,
            free_parameters: self.degree() + 1,
        })
    }

    fn fit_pass(
        &self,
        model: &mut ResolvedAr2,
        init: Init,
        pass: &DifferencedPass,
        _requested: RequestedMean,
    ) -> Result<FittedPass<ResolvedAr2, Ar2RefitResult>, Error> {
        let frequency = self.frequency();
        let p = self.degree();
        let (residuals, refit) = if init == Init::Imported2 {
            // ANALTS calls TRANS1 before its INIT branch, but INIT=2 never
            // reads the transformed x, and after the import snap its Phi2
            // division can be 0/0. Only the in-place PHI mutation reaches
            // CALCFX.
            model.coefficients.regular = trans1_coefficients(model.coefficients.regular);
            (resolved_residuals(*model, &pass.wd)?, None)
        } else {
            let start = starting_parameters(&pass.auto.acf, frequency, p)?;
            let regular = transform_regular_start(start.regular)?;
            let ma = transform_seasonal_start(start.seasonal_ma)?;
            let search = match regular.coordinates {
                RegularArCoefficients::Two([a, b]) => refit::search(
                    frequency,
                    &pass.wd,
                    [a, b, ma],
                    refit_mask(regular.coefficients),
                ),
                RegularArCoefficients::Three([a, b, c]) => refit::search(
                    frequency,
                    &pass.wd,
                    [a, b, c, ma],
                    refit_mask(regular.coefficients),
                ),
            }?;
            *model = ResolvedAr2::from_refit(frequency, search.coefficients);
            if let Some(proposed) = chmodel_orders(search.coordinates.regular, search.fixed) {
                return Err(model_change(model, proposed));
            }
            let normalized_residuals = search
                .scaled_residuals
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    finite(
                        value / search.determinant_scale,
                        "normalized optimizer residual",
                        Some(index),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let residuals = ResidualResult {
                scaled_residuals: search.scaled_residuals.clone(),
                normalized_residuals,
                determinant_scale: search.determinant_scale,
                objective: search.objective,
            };
            (residuals, Some(search))
        };
        Ok(FittedPass {
            model: *model,
            residuals,
            refit,
        })
    }

    fn correct_mean(
        &self,
        model: &ResolvedAr2,
        a: &mut [f64],
        requested: RequestedMean,
        wm: f64,
    ) -> Result<MeanCorrection, crate::seats::Error> {
        mean_correction(
            a,
            requested,
            wm,
            MeanFamily::Ar2(model.coefficients.regular),
        )
    }

    fn tails(
        &self,
        model: &mut ResolvedAr2,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<SeriesTails, Error> {
        finish_ar2_tails(*model, series, wd, state)
    }

    fn decompose(
        &self,
        model: &ResolvedAr2,
        series: &[f64],
        tails: &SeriesTails,
    ) -> Result<Ar2Decomposition, Error> {
        // SIGEX resolves the configured trading day against the model of the
        // last pass, not the imported one.
        let preparation = model.prepare(self.presence, None, DifferenceOrder::First)?;
        if preparation.model.irregular_candidate < 0.0 {
            return Err(Failure::NegativeVariance {
                component: CanonicalComponent::Irregular,
                variance: preparation.model.irregular_candidate,
            }
            .at(Stage::CanonicalDecomposition));
        }
        let cycle_variance = preparation.model.cycle.as_ref().map_or(0.0, |c| c.variance);
        Ok(Ar2Decomposition {
            components: extract_ar2(&preparation, series, tails)?,
            cycle_variance,
            trading_day: preparation.trading_day,
        })
    }

    fn final_components(
        &self,
        decomposition: Ar2Decomposition,
        restoration: Restoration<'_>,
        _transform: SeatsTransform,
    ) -> Result<FinalWithTransitory, Error> {
        let Ar2Decomposition {
            components,
            cycle_variance,
            trading_day,
        } = decomposition;
        Ok(match restoration {
            Restoration::FittedEffects(effects) => super::effects::final_with_effects_ar2(
                effects,
                &components,
                cycle_variance,
                trading_day,
            )?,
            Restoration::Log(log) => final_log_with_transitory(log, &components, cycle_variance)?,
            Restoration::Identity(original) => {
                final_identity_with_transitory(original, &components, cycle_variance)?
            }
        })
    }
}

/// FCAST for AR(2)/AR(3) with seasonal MA; the regular AR forces the mean.
/// A failed backward CALCFX is reported at the tails stage.
fn finish_ar2_tails(
    model: ResolvedAr2,
    series: &[f64],
    wd: &[f64],
    state: ForwardTailState,
) -> Result<SeriesTails, Error> {
    let frequency = model.frequency;
    let s = usize::from(frequency.periods_per_year());
    let horizons = resolve_horizons(frequency, series.len());
    let bj_ma = single_lag_bj_ma(s, model.coefficients.seasonal_ma);
    let inputs = TailInputs {
        bj_ma: bj_ma.clone(),
        horizons,
        bphist: bphist_ordered(
            s,
            &model.coefficients.regular.polynomial(),
            DifferenceOrder::First,
        ),
        forcing_ar: MeanForcing::Regular(model.coefficients.regular),
        order: DifferenceOrder::First,
        working_horizon: horizons
            .fh
            .max(2 * s + model.coefficients.regular.as_slice().len() + 1),
    };
    finish_tails_numeric(frequency, series, wd, state, inputs, |back| {
        resolved_residuals(model, back)
            .map(|residuals| (residuals, bj_ma, ()))
            .map_err(|error| error.restage(Stage::Tails))
    })
    .map(|core| core.tails)
}

#[cfg(test)]
mod tests {
    use super::{RegularArCoefficients, chmodel_orders, model_change, refit_mask};
    use crate::Frequency;
    use crate::seats::Orders;
    use crate::seats::ar2::canonical::ResolvedAr2;
    use crate::seats::ar2::transform::Ar2Coefficients;
    use crate::seats::failure::{Failure, ModelChange, Stage};

    #[test]
    fn refit_mask_fixes_lower_coefficients_only_when_all_are_zero() {
        use RegularArCoefficients::{Three, Two};
        let two = |regular| refit_mask::<3>(regular);
        let three = |regular| refit_mask::<4>(regular);
        assert_eq!(two(Two([0.0, 0.5])), [true, false, false]);
        assert_eq!(two(Two([0.1, 0.5])), [false; 3]);
        assert_eq!(three(Three([0.0, 1e-14, 0.5])), [true, true, false, false]);
        assert_eq!(three(Three([0.0, 0.1, 0.5])), [false; 4]);
    }

    #[test]
    fn chmodel_lowers_the_ar_degree_at_a_unit_coordinate_once_search_fixed_one() {
        use RegularArCoefficients::{Three, Two};
        assert_eq!(
            chmodel_orders(Two([0.1, 1.0]), 1),
            Some(Orders::new([1, 2, 0], [0, 1, 1]))
        );
        assert_eq!(
            chmodel_orders(Three([1.0, 0.2, 0.3]), 1),
            Some(Orders::new([2, 2, 0], [0, 1, 1]))
        );
        assert_eq!(chmodel_orders(Two([0.1, 1.0]), 0), None);
        assert_eq!(chmodel_orders(Two([0.1, 0.2]), 1), None);
    }

    #[test]
    fn a_model_change_carries_the_refit_orders_and_the_proposed_orders() {
        let model = ResolvedAr2::from_refit(
            Frequency::Monthly,
            Ar2Coefficients {
                regular: RegularArCoefficients::Two([0.5, -0.25]),
                seasonal_ma: 0.75,
            },
        );
        let proposed = Orders::new([1, 2, 0], [0, 1, 1]);
        assert_eq!(
            model_change(&model, proposed),
            Failure::ModelChange(ModelChange {
                current: Orders::new([2, 1, 0], [0, 1, 1]),
                proposed,
            })
            .at(Stage::Refit)
        );
    }
}
