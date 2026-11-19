//! The ARMA family: the models SEATS decomposes that no other family
//! covers. They have a regular AR of up to three lags, a regular MA of up to
//! two, one seasonal AR and MA lag and up to two regular differences;
//! `model::ArmaLayout` lists them.
//!
//! `compose_arma` runs ANALTS's loop (`run_family`) for one model with N
//! coefficients. A pass differences the series, then either evaluates CALCFX
//! at the imported coefficients (INIT=2) or starts from STAVAL and refits
//! with SEARCH (INIT=0). FCAST extends the series and SIGEX decomposes it.
//! ANALTS gives this family three ends the others lack:
//! - a model change (CHMODEL or ChangeModel after a pass's fit), which is
//!   [`Failure::ModelChange`], or a negative canonical irregular or cycle in
//!   the decomposition, which is [`Failure::NegativeVariance`];
//! - a ChangeModel restart after SEARCH, which the engine re-enters with the
//!   smaller model, the pass's lag dimensions and its mean;
//! - a pure MA model (P+D+BP+BD=0), which ends after the forward
//!   diagnostics without FCAST or a decomposition (ANALTS 5018).
//!
//! The restart leaves the loop as `Interrupt::Outcome`, and `compose_arma`
//! hands it back as `ArmaOutcome`. The pure MA model completes: its tails
//! and decomposition carry only WM, and `final_components` builds the saved
//! components from WM and the effects.
//!
//! Each pass fits from the import alone, so nothing carries between passes
//! and the model is the pass's fit (`ArmaFit`): the coordinates CALCFX
//! evaluated, forward CALCFX's coefficients, controls and state, and RPQ's
//! view of the regular AR group. FCAST's backward CALCFX resumes that state
//! and adds its coefficients, which SIGEX and the engine read.
//!
//! `residual` is CALCFX with TRANSC, `refit` STAVAL and the SEARCH run, and
//! `model` the layouts, the import (NMLMDL and TRANS0) and ANALTS's
//! degenerate-model test.

pub(crate) mod model;
pub(crate) mod refit;
pub(crate) mod residual;

use super::ar1_seasonal_ma::model::{Ar1RealRoot, ResolvedAr1SeasonalMa, ar1_seasonal_ma_geometry};
use super::ar2::canonical::{CanonicalArma, ResolvedAr2};
use super::autocorrelation::AutoLagState;
use super::decomposition::adjusted::{check_show_comp, factor_adjusted};
use super::decomposition::root_allocation::RegularArCoefficients;
use super::effects::FittedEffectsInput;
use super::extraction::{ExtractedComponents, FinalComponents, FinalWithTransitory, extract_arma};
use super::failure::{CanonicalComponent, Error, Failure, ModelChange, Stage, usable};
use super::mean::{
    MeanCorrection, MeanFamily, RequestedMean, mean_correction, residual_mean_state,
};
use super::pipeline::{
    Composed, DifferencedPass, Family, FittedPass, PassShape, Restoration, RunInput,
    SeatsTransform, run_family,
};
use super::residual::ResidualResult;
use super::retry::{Init, RetryState};
use super::tails::{
    DifferenceOrder, ForwardTailState, MeanForcing, TailInputs, bphist_ordered,
    finish_tails_numeric, resolve_horizons,
};
use super::trading_day::{EffectiveTradingDay, TradingDayPresence};
use crate::Frequency;
use model::{ArOrder, ArmaLayout, ImportedArma, RegularArGroup, arma_degrees_of_freedom};
use refit::{ArmaRefitResult, search_arma};
use residual::{ArmaCalcContinuation, ArmaCalcControls, ArmaObjective, ArmaPolynomials};

/// One model and the series ANALTS's loop runs it on.
#[derive(Clone, Copy)]
pub(crate) struct ArmaRunInput<'a, const N: usize = 3> {
    pub(crate) summary_q: crate::seats::pipeline::SummaryQ,
    pub imported: &'a ImportedArma<N>,
    pub initial: Init,
    pub trading_day: TradingDayPresence,
    pub original: &'a [f64],
    pub transformed: &'a [f64],
    pub transform: SeatsTransform,
    pub requested_mean: RequestedMean,
    /// ANALTS M/IQM carried over a post-SEARCH ChangeModel `goto 10`; `None`
    /// enters loop 10 with the defaults.
    pub carried_lags: Option<AutoLagState>,
}

/// A completed ARMA run.
pub(crate) type ArmaRun<const N: usize> = Composed<ArmaFit<N>, ArmaRefitResult<N>>;

#[derive(Debug)]
pub(crate) enum ArmaOutcome<const N: usize = 3> {
    /// The decomposition, or a pure MA model's saved components.
    Completed(Box<ArmaRun<N>>),
    /// analts.f:2258–2265: ChangeModel after SEARCH dropped BQ and set INIT=0.
    /// The engine re-enters loop 10 with the smaller model, these lag
    /// dimensions and this mean; the regARIMA fit is unchanged.
    ChangeModelRestart {
        lags: AutoLagState,
        requested_mean: RequestedMean,
    },
}

/// A pass's forward CALCFX.
#[derive(Debug)]
struct ForwardCalc<const N: usize> {
    /// The pass's requested mean.
    requested: RequestedMean,
    /// The coordinates CALCFX evaluated: the import after TRANS1, or
    /// SEARCH's result.
    coordinates: [f64; N],
    /// Forward CALCFX's coefficients.
    coefficients: ArmaPolynomials<N>,
    controls: ArmaCalcControls,
    /// Forward CALCFX's state, which backward CALCFX resumes.
    continuation: ArmaCalcContinuation<N>,
}

/// The forward fit's regular AR group as the decomposition reads it.
#[derive(Debug, Clone, Copy)]
enum ForwardAr {
    Empty,
    /// RPQ's root of a single coefficient.
    Single(Ar1RealRoot),
    Polynomial(RegularArCoefficients),
}

/// One pass's fit.
#[derive(Debug)]
pub(crate) struct ArmaFit<const N: usize> {
    forward: ForwardCalc<N>,
    regular_ar: ForwardAr,
    /// Backward CALCFX's coefficients, once FCAST ran.
    backward: Option<ArmaPolynomials<N>>,
}

impl<const N: usize> ArmaFit<N> {
    /// The coefficients the engine reports: backward CALCFX's after FCAST,
    /// else the forward fit's.
    pub(crate) fn coefficients(&self) -> &ArmaPolynomials<N> {
        self.backward.as_ref().unwrap_or(&self.forward.coefficients)
    }
}

/// What ANALTS does with a model it will not decompose as fitted.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ArmaRouteKind {
    /// ANSUB1 CHMODEL, after SEARCH fixed a coordinate at a unit root, with
    /// the smaller orders it would re-estimate.
    Chmodel(super::Orders),
    /// ANSUB1 ChangeModel's BP/BQ cancellation, which drops BQ, with the
    /// smaller orders it would re-estimate.
    ChangeModel(super::Orders),
    /// SPECTRUM found a canonical component of negative `variance`.
    InvalidDecomposition {
        component: CanonicalComponent,
        variance: f64,
    },
}

/// A model change or an invalid decomposition. A CHMODEL or ChangeModel
/// route carries the smaller model it would re-estimate in its `kind`; an
/// invalid decomposition carries the negative variance.
#[derive(Debug)]
pub(crate) struct ArmaModelRoute {
    pub kind: ArmaRouteKind,
    pub current_orders: super::Orders,
}

impl ArmaModelRoute {
    /// The error that ends the run: a model change at [`Stage::Refit`], or a
    /// negative canonical variance at [`Stage::CanonicalDecomposition`].
    fn into_error(self) -> Error {
        match self.kind {
            ArmaRouteKind::Chmodel(proposed) | ArmaRouteKind::ChangeModel(proposed) => {
                Failure::ModelChange(ModelChange {
                    current: self.current_orders,
                    proposed,
                })
                .at(Stage::Refit)
            }
            ArmaRouteKind::InvalidDecomposition {
                component,
                variance,
            } => Failure::NegativeVariance {
                component,
                variance,
            }
            .at(Stage::CanonicalDecomposition),
        }
    }
}

/// FCAST's tails, with the coefficients forward and backward CALCFX left.
#[derive(Debug)]
pub(crate) struct ArmaRunTails<const N: usize = 3> {
    pub base: super::tails::SeriesTails,
    pub forward_coefficients: ArmaPolynomials<N>,
    pub backward_bj_ma: Vec<f64>,
    pub backward_coefficients: ArmaPolynomials<N>,
}

/// Runs ANALTS's loop on one ARMA model, restoring `calendar`'s fitted
/// effects when given.
pub(crate) fn compose_arma<const N: usize>(
    input: ArmaRunInput<'_, N>,
    calendar: Option<&FittedEffectsInput>,
) -> Result<ArmaOutcome<N>, Error> {
    let family = Arma {
        imported: input.imported,
        initial: input.initial,
        trading_day: input.trading_day,
        carried_lags: input.carried_lags,
        original: input.original,
        transformed: input.transformed,
    };
    let run = RunInput {
        original: input.original,
        transformed: input.transformed,
        transform: input.transform,
        requested_mean: input.requested_mean,
        summary_q: input.summary_q,
    };
    match run_family(&family, run, calendar) {
        Ok(completed) => Ok(ArmaOutcome::Completed(Box::new(completed))),
        Err(Interrupt::Outcome(outcome)) => Ok(outcome),
        Err(Interrupt::Failed(failure)) => Err(failure),
    }
}

/// How a step ends the run early: a failure, or an outcome the engine
/// resolves.
enum Interrupt<const N: usize> {
    Failed(Error),
    Outcome(ArmaOutcome<N>),
}

impl<const N: usize> From<Error> for Interrupt<N> {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

/// FCAST's tails, or only WM for a pure MA model.
enum ArmaTails<const N: usize> {
    PureMa { wm: f64 },
    Fcast(Box<ArmaRunTails<N>>),
}

/// The extracted components, or only WM for a pure MA model.
enum ArmaDecomposition {
    PureMa {
        wm: f64,
    },
    Components {
        raw: ExtractedComponents,
        cycle_variance: f64,
        trading_day: EffectiveTradingDay,
        has_seasonal_component: bool,
    },
}

/// One ARMA model on ANALTS's loop.
struct Arma<'a, const N: usize> {
    imported: &'a ImportedArma<N>,
    initial: Init,
    trading_day: TradingDayPresence,
    carried_lags: Option<AutoLagState>,
    original: &'a [f64],
    transformed: &'a [f64],
}

impl<const N: usize> Family for Arma<'_, N> {
    type Start = ();
    type Model = ArmaFit<N>;
    type Refit = ArmaRefitResult<N>;
    type Tails = ArmaTails<N>;
    type Decomposition = ArmaDecomposition;
    type Failure = Interrupt<N>;

    const TMU_BEFORE_LOG: bool = false;
    const DIFFERENCES_LOGGED_OZ: bool = false;

    fn frequency(&self) -> Frequency {
        self.imported.original.frequency
    }

    fn validate(&self, input: &RunInput<'_>) -> Result<(), Interrupt<N>> {
        arma_degrees_of_freedom::<N>(
            self.frequency(),
            input.original.len(),
            input.requested_mean,
            self.layout(),
        )?;
        Ok(())
    }

    fn initial_start(&self) -> Result<(), Interrupt<N>> {
        Ok(())
    }

    fn initial_retry(&self, requested: RequestedMean) -> RetryState {
        match self.initial {
            Init::Imported2 => RetryState::new(requested),
            Init::Refit0 => RetryState::refit(requested),
        }
    }

    fn initial_lags(&self) -> AutoLagState {
        self.carried_lags
            .unwrap_or_else(|| AutoLagState::new(self.frequency()))
    }

    fn pass_shape(
        &self,
        _start: &(),
        observed_len: usize,
        requested: RequestedMean,
    ) -> Result<PassShape, Interrupt<N>> {
        let layout = self.layout();
        // A retry can add the mean, so each pass repeats `validate`'s check.
        arma_degrees_of_freedom::<N>(self.frequency(), observed_len, requested, layout)?;
        Ok(PassShape {
            differences: layout.differences(),
            free_parameters: N,
        })
    }

    fn fit_pass(
        &self,
        _start: &mut (),
        init: Init,
        pass: &DifferencedPass,
        requested: RequestedMean,
    ) -> Result<FittedPass<ArmaFit<N>, ArmaRefitResult<N>>, Interrupt<N>> {
        let (forward, residuals, search) = match init {
            Init::Imported2 => {
                let (forward, residuals) = self.evaluate_imported(&pass.wd, requested)?;
                (forward, residuals, None)
            }
            Init::Refit0 => {
                let (forward, residuals, search) = self.search(pass, requested)?;
                (forward, residuals, Some(search))
            }
        };
        if let Some(route) = forward_model_policy(init, search.as_ref(), &forward) {
            if matches!(route.kind, ArmaRouteKind::ChangeModel(_)) && init == Init::Refit0 {
                // analts.f:2260–2264: ChangeModel (2) runs only in the SEARCH
                // branch; a positive result jumps to loop 10's end. M/IQM and
                // IMEAN persist, and NORETRY stays set from the reset.
                return Err(Interrupt::Outcome(ArmaOutcome::ChangeModelRestart {
                    lags: pass.lags,
                    requested_mean: requested,
                }));
            }
            return Err(Interrupt::Failed(route.into_error()));
        }
        let regular_ar = self.forward_ar(&forward.coefficients)?;
        Ok(FittedPass {
            model: ArmaFit {
                forward,
                regular_ar,
                backward: None,
            },
            residuals,
            refit: search,
        })
    }

    fn correct_mean(
        &self,
        model: &ArmaFit<N>,
        a: &mut [f64],
        requested: RequestedMean,
        wm: f64,
    ) -> Result<MeanCorrection, crate::seats::Error> {
        residual_mean(
            a,
            requested,
            self.frequency(),
            wm,
            self.layout(),
            &model.forward.coefficients.parameters,
        )
    }

    fn tails(
        &self,
        model: &mut ArmaFit<N>,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<ArmaTails<N>, Interrupt<N>> {
        let orders = self.layout().orders();
        if orders.p + orders.d + orders.bp + orders.bd == 0 {
            // ANALTS 5018 branches after forward diagnostics, before FCAST,
            // backward CALCFX and SIGEX. It has no stochastic decomposition.
            return Ok(ArmaTails::PureMa { wm: state.wm });
        }
        let tails = finish_arma_tails(self.frequency(), series, wd, model, state)?;
        Ok(ArmaTails::Fcast(Box::new(tails)))
    }

    fn decompose(
        &self,
        fit: &ArmaFit<N>,
        series: &[f64],
        tails: &ArmaTails<N>,
    ) -> Result<ArmaDecomposition, Interrupt<N>> {
        let tails = match tails {
            ArmaTails::PureMa { wm } => return Ok(ArmaDecomposition::PureMa { wm: *wm }),
            ArmaTails::Fcast(tails) => tails,
        };
        let frequency = self.frequency();
        let layout = self.layout();
        let orders = layout.orders();
        let policy = |route: ArmaModelRoute| Interrupt::Failed(route.into_error());
        let (adjusted, effective_td) = if layout == ArmaLayout::SeasonalAr {
            // BP is not in RPQ's regular-AR root list. SIGEX:477–555
            // preallocates bphi from the live coefficients after backward
            // CALCFX, with no trading day.
            let parameters = &tails.backward_coefficients.parameters;
            let (model, td) = CanonicalArma {
                frequency,
                roots: &[],
                seasonal_ar: Some(parameters[0]),
                regular_ma: &parameters[1..2],
                seasonal_ma: None,
                differences: DifferenceOrder::First,
            }
            .prepare(TradingDayPresence::Absent)?;
            if let Some(route) = irregular_policy(model.irregular_candidate, orders) {
                return Err(policy(route));
            }
            (model, td)
        } else if layout == ArmaLayout::SeasonalMa2 {
            // Regular MA(2) with seasonal MA, with no trading day.
            let parameters = &tails.backward_coefficients.parameters;
            let (model, td) = CanonicalArma {
                frequency,
                roots: &[],
                seasonal_ar: None,
                regular_ma: &parameters[..2],
                seasonal_ma: Some(parameters[2]),
                differences: DifferenceOrder::First,
            }
            .prepare(TradingDayPresence::Absent)?;
            if let Some(route) = irregular_policy(model.irregular_candidate, orders) {
                return Err(policy(route));
            }
            (model, td)
        } else {
            let general_canonical = matches!(
                layout,
                ArmaLayout::RegularSeasonalMa2(_)
                    | ArmaLayout::NonseasonalMa2(_)
                    | ArmaLayout::Integrated { .. }
            ) || matches!(layout, ArmaLayout::Nonseasonal(ar) if ar != ArOrder::One);
            match fit.regular_ar {
                ForwardAr::Polynomial(regular) if !general_canonical => {
                    // RPQ's AR allocation belongs to the forward fit; SIGEX sees the
                    // MA coefficients refreshed by backward CALCFX.
                    let ma = &tails.backward_coefficients.parameters;
                    let preparation = ResolvedAr2::from_refit(
                        frequency,
                        super::ar2::transform::Ar2Coefficients {
                            regular,
                            seasonal_ma: ma[N - 1],
                        },
                    )
                    .prepare(
                        self.trading_day,
                        Some(ma[N - 2]),
                        layout.differences(),
                    )?;
                    if let Some(route) =
                        irregular_policy(preparation.model.irregular_candidate, orders)
                    {
                        return Err(policy(route));
                    }
                    (preparation.model, preparation.trading_day)
                }
                ForwardAr::Single(root) if !general_canonical => {
                    let geometry =
                        model::arma_geometry(frequency, &root, &tails.backward_coefficients)?;
                    let prefix = super::decomposition::rational::lower_degree_rational_prefix(
                        &geometry.inputs,
                    )?;
                    let minima =
                        super::decomposition::minima::spectral_minima_prefix(frequency, prefix);
                    if let Some(route) = irregular_policy(minima.irregular_candidate, orders) {
                        return Err(policy(route));
                    }
                    let canonical = super::decomposition::canonical::prepare_canonical(minima);
                    (
                        factor_adjusted(
                            canonical,
                            geometry.inputs.clone(),
                            EffectiveTradingDay::Ordinary,
                        )?,
                        EffectiveTradingDay::Ordinary,
                    )
                }
                regular_ar => {
                    let p = layout.ar_coordinates();
                    let regular_p = layout.regular_ar_coordinates();
                    let roots = match regular_ar {
                        ForwardAr::Empty => Vec::new(),
                        ForwardAr::Single(root) => {
                            vec![super::decomposition::root_allocation::ArRoot {
                                real: root.real,
                                imaginary: root.imaginary,
                                modulus: root.modulus,
                                angle: root.argument_degrees,
                                period: root.period,
                            }]
                        }
                        ForwardAr::Polynomial(regular) => {
                            super::decomposition::root_allocation::regular_ar_roots(regular)?
                        }
                    };
                    let ma = &tails.backward_coefficients.parameters;
                    let q = layout.regular_ma_count();
                    let (model, td) = CanonicalArma {
                        frequency,
                        roots: &roots,
                        seasonal_ar: layout.has_seasonal_ar().then(|| ma[regular_p]),
                        regular_ma: &ma[p..p + q],
                        seasonal_ma: layout.has_seasonal_ma().then(|| ma[p + q]),
                        differences: layout.differences(),
                    }
                    .prepare(self.trading_day)?;
                    if let Some(route) = irregular_policy(model.irregular_candidate, orders) {
                        return Err(policy(route));
                    }
                    (model, td)
                }
            }
        };
        let cycle_variance = adjusted.cycle.as_ref().map_or(0.0, |c| c.variance);
        if let Some(route) = cycle_policy(cycle_variance, orders) {
            return Err(policy(route));
        }
        check_show_comp(&adjusted)?;
        let has_seasonal_component = adjusted.inputs.seasonal.combined.len() > 1;
        let raw = extract_arma(
            frequency,
            &adjusted.inputs,
            &adjusted.factors.preparation,
            series,
            tails,
            fit.forward.requested,
        )?;
        Ok(ArmaDecomposition::Components {
            raw,
            cycle_variance,
            trading_day: effective_td,
            has_seasonal_component,
        })
    }

    fn final_components(
        &self,
        decomposition: ArmaDecomposition,
        restoration: Restoration<'_>,
        transform: SeatsTransform,
    ) -> Result<FinalWithTransitory, Interrupt<N>> {
        let (raw, cycle_variance, trading_day, has_seasonal_component) = match decomposition {
            ArmaDecomposition::PureMa { wm } => {
                let effects = match restoration {
                    Restoration::FittedEffects(calendar) => Some(calendar),
                    Restoration::Log(_) | Restoration::Identity(_) => None,
                };
                let (saved, transitory) =
                    pure_ma_saved(self.original, self.transformed, effects, transform, wm)?;
                return Ok(FinalWithTransitory { saved, transitory });
            }
            ArmaDecomposition::Components {
                raw,
                cycle_variance,
                trading_day,
                has_seasonal_component,
            } => (raw, cycle_variance, trading_day, has_seasonal_component),
        };
        let mut saved = match restoration {
            Restoration::FittedEffects(calendar) => {
                let ar1_shaped = N == 3
                    && !matches!(
                        self.layout(),
                        ArmaLayout::Nonseasonal(_)
                            | ArmaLayout::NonseasonalMa2(_)
                            | ArmaLayout::Integrated { .. }
                    );
                if ar1_shaped {
                    super::effects::final_with_effects_and_transitory(
                        calendar,
                        &raw,
                        cycle_variance,
                    )
                } else {
                    super::effects::final_with_effects_ar2(
                        calendar,
                        &raw,
                        cycle_variance,
                        trading_day,
                    )
                }?
            }
            Restoration::Log(log) => {
                super::log_transform::final_log_with_transitory(log, &raw, cycle_variance)?
            }
            Restoration::Identity(original) => {
                super::extraction::final_identity_with_transitory(original, &raw, cycle_variance)?
            }
        };
        if !has_seasonal_component {
            restoration.without_seasonal_roots(&mut saved.saved, transform);
        }
        Ok(saved)
    }
}

impl<const N: usize> Arma<'_, N> {
    fn layout(&self) -> ArmaLayout {
        self.imported.original.layout
    }

    /// INIT=2: CALCFX at TRANS1's imported coordinates.
    fn evaluate_imported(
        &self,
        wd: &[f64],
        requested: RequestedMean,
    ) -> Result<(ForwardCalc<N>, ResidualResult), Error> {
        let layout = self.layout();
        let controls = ArmaCalcControls {
            init: Init::Imported2,
            ifn: 0,
        };
        let coordinates = trans1_imported(self.imported.parameters, layout);
        let mut objective = ArmaObjective::new(self.frequency(), wd, layout);
        let residuals = objective.evaluate(coordinates, controls)?;
        let forward = ForwardCalc {
            requested,
            coordinates,
            coefficients: objective.coefficients(),
            controls,
            continuation: objective.into_continuation(),
        };
        Ok((forward, residuals))
    }

    /// INIT=0: STAVAL's starts, then SEARCH.
    fn search(
        &self,
        pass: &DifferencedPass,
        requested: RequestedMean,
    ) -> Result<(ForwardCalc<N>, ResidualResult, ArmaRefitResult<N>), Error> {
        let layout = self.layout();
        let (starts, mask) = self.starts(&pass.auto.acf)?;
        let (search, continuation) = search_arma(self.frequency(), &pass.wd, starts, mask, layout)?;
        let residuals = ResidualResult {
            scaled_residuals: search.scaled_residuals.clone(),
            normalized_residuals: search
                .scaled_residuals
                .iter()
                .map(|v| v / search.determinant_scale)
                .collect(),
            objective: search.objective,
            determinant_scale: search.determinant_scale,
        };
        let controls = ArmaCalcControls {
            init: Init::Refit0,
            ifn: search.evaluations,
        };
        let forward = ForwardCalc {
            requested,
            coordinates: search.parameters,
            coefficients: search.coefficients.clone(),
            controls,
            continuation,
        };
        Ok((forward, residuals, search))
    }

    /// STAVAL's starting coordinates and SEARCH's fixed-coordinate mask.
    fn starts(&self, acf: &[f64; 36]) -> Result<([f64; N], [bool; N]), Error> {
        use super::refit::start::contract_start;
        use refit::{
            ma1_start, ma2_start, seasonal_ar_coordinate, seasonal_ar_start, seasonal_ma_start,
            starting_parameters,
        };
        let frequency = self.frequency();
        let layout = self.layout();
        let mut starts = [0.0; N];
        let mut mask = [false; N];
        if matches!(layout, ArmaLayout::Integrated { .. }) {
            let p = layout.ar_coordinates();
            let regular_p = layout.regular_ar_coordinates();
            let q = layout.regular_ma_count();
            let acf = if regular_p == 0 {
                *acf
            } else {
                let start = starting_parameters(acf, frequency, layout.has_seasonal_ma(), q)?;
                fill_ar_start(
                    layout.regular_ar(),
                    start.parameters[0],
                    &mut starts,
                    &mut mask,
                )?;
                start.conditioned_acf
            };
            if layout.has_seasonal_ar() {
                // STAVAL reads the autocorrelations with the regular AR
                // removed; it removes it from lag Mq only when BQ is present.
                starts[regular_p] = seasonal_ar_coordinate(&acf, frequency)?;
            }
            if q == 2 {
                fill_ma2_start(
                    ma2_start(&acf, frequency, layout.has_seasonal_ma())?,
                    &mut starts[p..],
                    &mut mask[p..],
                );
            } else if q == 1 {
                starts[p..].copy_from_slice(
                    &ma1_start(&acf, frequency, layout.has_seasonal_ma())?[..N - p],
                );
            } else if layout.has_seasonal_ma() {
                starts[p] = seasonal_ma_start(&acf, frequency)?;
            }
        } else if layout == ArmaLayout::SeasonalAr {
            starts.copy_from_slice(&seasonal_ar_start(acf, frequency)?);
        } else if matches!(
            layout,
            ArmaLayout::RegularSeasonalMa2(_) | ArmaLayout::NonseasonalMa2(_)
        ) {
            let p = layout.ar_coordinates();
            let acf = if p == 0 {
                *acf
            } else {
                let start = starting_parameters(acf, frequency, layout.has_seasonal_ma(), 2)?;
                fill_ar_start(
                    layout.regular_ar(),
                    start.parameters[0],
                    &mut starts,
                    &mut mask,
                )?;
                start.conditioned_acf
            };
            fill_ma2_start(
                ma2_start(&acf, frequency, layout.has_seasonal_ma())?,
                &mut starts[p..],
                &mut mask[p..],
            );
        } else if layout == ArmaLayout::SeasonalMa2 {
            fill_ma2_start(ma2_start(acf, frequency, true)?, &mut starts, &mut mask);
        } else if matches!(layout, ArmaLayout::Nonseasonal(_)) {
            let start = starting_parameters(acf, frequency, false, 1)?;
            let p = layout.ar_coordinates();
            fill_ar_start(
                layout.regular_ar(),
                start.parameters[0],
                &mut starts,
                &mut mask,
            )?;
            starts[p] = contract_start(start.parameters[1]);
        } else {
            let start = starting_parameters(acf, frequency, true, 1)?.parameters;
            if N == 3 {
                starts.copy_from_slice(&start.map(contract_start));
            } else {
                // STAVAL halves successive Phi values, but removes only
                // Phi1 from the autocorrelations, for P=3 too.
                let p = N - 2;
                let raw = if p == 2 {
                    super::decomposition::root_allocation::RegularArCoefficients::Two([
                        start[0],
                        0.5 * start[0],
                    ])
                } else {
                    super::decomposition::root_allocation::RegularArCoefficients::Three([
                        start[0],
                        0.5 * start[0],
                        0.5 * (0.5 * start[0]),
                    ])
                };
                let ar = super::ar2::transform::transform_regular_start(raw)?;
                starts[..p].copy_from_slice(ar.coordinates.as_slice());
                starts[p] = contract_start(start[1]);
                starts[p + 1] = contract_start(start[2]);
                if raw.as_slice()[..p - 1].iter().all(|v| v.abs() <= 1e-13) {
                    mask[..p - 1].fill(true);
                }
            }
        }
        Ok((starts, mask))
    }

    /// RPQ's view of the forward fit's regular AR group: it takes a single
    /// coefficient's root here, and a polynomial's in the decomposition.
    fn forward_ar(&self, forward: &ArmaPolynomials<N>) -> Result<ForwardAr, Error> {
        Ok(match self.layout().regular_ar_group(&forward.parameters) {
            RegularArGroup::Empty => ForwardAr::Empty,
            RegularArGroup::Single(_) => ForwardAr::Single(self.forward_root(forward)?),
            RegularArGroup::Polynomial(regular) => ForwardAr::Polynomial(regular),
        })
    }

    /// RPQ's root of the forward fit's single regular AR coefficient.
    fn forward_root(&self, forward: &ArmaPolynomials<N>) -> Result<Ar1RealRoot, Error> {
        let layout = self.layout();
        let seasonal_ma = if layout.has_seasonal_ma() {
            forward.parameters[layout.ar_coordinates() + layout.regular_ma_count()]
        } else {
            0.0
        };
        let model = ResolvedAr1SeasonalMa::from_refit(
            self.frequency(),
            [forward.parameters[0], seasonal_ma],
            layout.seasonal_difference(),
        )?;
        Ok(ar1_seasonal_ma_geometry(&model)?.root)
    }
}

/// ANALTS 5018: the saved components of a pure MA model. The trend is WM
/// with the level shifts, and the rest of the series is irregular.
fn pure_ma_saved(
    original: &[f64],
    transformed: &[f64],
    effects: Option<&FittedEffectsInput>,
    transform: SeatsTransform,
    wm: f64,
) -> Result<(FinalComponents, Option<Vec<f64>>), Error> {
    let log = transform == SeatsTransform::Log;
    let neutral = if log { 1.0 } else { 0.0 };
    let mut saved = FinalComponents {
        seasonal: if log {
            vec![1.0 / 100.0; original.len()]
        } else {
            Vec::new()
        },
        seasonally_adjusted: Vec::with_capacity(original.len()),
        trend: Vec::with_capacity(original.len()),
        irregular: Vec::with_capacity(original.len()),
    };
    for (i, &value) in transformed.iter().enumerate() {
        let tram = effects.map_or_else(|| if log { value.exp() } else { value }, |e| e.tram()[i]);
        let paeast = effects.map_or(neutral, |e| e.paeast()[i]);
        let patd = effects.map_or(neutral, |e| e.patd()[i]);
        let paoutr = effects.map_or(neutral, |e| e.paoutr()[i]);
        // ANALTS 5018 restores LS into trend and calendar into the adjustment
        // factor; AO/TC stay in irregular. PAREG and PAOUS are neutral here.
        let (trend, adjusted, irregular) = if log {
            let trend = wm.exp() * paoutr * 1.0 * 1.0;
            let adjustment = paeast * patd * 1.0 * 1.0 * 100.0;
            let adjusted = tram / (adjustment / 100.0);
            // Unlike SIGEX, ANALTS supplies unscaled seasStoch=1 and irtemp.
            // seatad still divides each by 100 when saving these components.
            let irregular = (tram / (adjustment / 100.0) / trend) / 100.0;
            (trend, adjusted, irregular)
        } else {
            // The neutral PAREG and PAOUS terms. One `+ 0.0` keeps their
            // effect: it turns a -0.0 sum into +0.0; a second adds nothing.
            let trend = wm + paoutr + 0.0;
            let adjustment = paeast + patd + 0.0;
            let adjusted = tram - adjustment;
            let irregular = tram - adjustment - trend - 0.0;
            (trend, adjusted, irregular)
        };
        let finite = |quantity, value| {
            usable(quantity, Some(i), value).map_err(|failure| failure.at(Stage::Mean))
        };
        saved.trend.push(finite("pure MA saved trend", trend)?);
        saved
            .seasonally_adjusted
            .push(finite("pure MA saved adjusted", adjusted)?);
        saved
            .irregular
            .push(finite("pure MA saved irregular", irregular)?);
    }
    // USRENTRY scans nz+fh. The only trend effect that varies is LS; every
    // LS date is in the series, so its future value equals its last one.
    if !super::extraction::component_present(&saved.trend) {
        saved.trend.clear();
    }
    // PAREG5 is neutral: zero in identity mode, 100 before seatad in log mode.
    let transitory = log.then(|| vec![1.0; original.len()]);
    Ok((saved, transitory))
}

/// Fills the leading regular AR coordinates of `starts` for an AR(`order`).
fn fill_ar_start(
    order: ArOrder,
    phi: f64,
    starts: &mut [f64],
    mask: &mut [bool],
) -> Result<(), Error> {
    // STAVAL halves higher coefficients; TRANS1 transforms the group.
    let raw = match order {
        ArOrder::Zero => return Ok(()),
        ArOrder::One => {
            starts[0] = super::refit::start::contract_start(phi);
            return Ok(());
        }
        ArOrder::Two => RegularArCoefficients::Two([phi, 0.5 * phi]),
        ArOrder::Three => RegularArCoefficients::Three([phi, 0.5 * phi, 0.5 * (0.5 * phi)]),
    };
    let coordinates = super::ar2::transform::transform_regular_start(raw)?.coordinates;
    let coordinates = coordinates.as_slice();
    starts[..coordinates.len()].copy_from_slice(coordinates);
    let lower = coordinates.len() - 1;
    if raw.as_slice()[..lower].iter().all(|v| v.abs() <= 1e-13) {
        mask[..lower].fill(true);
    }
    Ok(())
}

/// ANALTS loop 15 calls TRANS1 on PHI, BPHI, TH and BTH before its INIT
/// branch. INIT=2 never reads TRANS1's x, but the in-place PHI(2)/PHI(3) snap
/// reaches CALCFX and the decomposition, as in AR2: an imported unit PHI(2)
/// becomes .9999999. The TH(2) snap cannot fire after the MA(2) import clips
/// TH(2) to .99, and one-coefficient groups (P=1, Q=1, BPHI, BTH) have no
/// in-place snap.
fn trans1_imported<const N: usize>(mut parameters: [f64; N], layout: ArmaLayout) -> [f64; N] {
    if let RegularArGroup::Polynomial(regular) = layout.regular_ar_group(&parameters) {
        let regular = super::ar2::transform::trans1_coefficients(regular);
        parameters[..regular.as_slice().len()].copy_from_slice(regular.as_slice());
    }
    parameters
}

/// Places STAVAL's MA(2) start at the first regular-MA coordinate. ANALTS
/// 5005–5006 fixes Th(1) for SEARCH when STAVAL left it within CEPS of zero
/// (Build 62 keeps exactly 0.0 where a free search would move it), as it
/// fixes the lower AR coefficients for P>1.
fn fill_ma2_start(start: refit::Ma2Start, coordinates: &mut [f64], mask: &mut [bool]) {
    let n = coordinates.len();
    coordinates.copy_from_slice(&start.coordinates[..n]);
    if start.fixes_first {
        mask[0] = true;
    }
}

/// SPECTRUM:521-547 NOADMISS0's invalid-irregular alternate return.
fn irregular_policy(variance: f64, orders: super::Orders) -> Option<ArmaModelRoute> {
    (variance < 0.0).then_some(ArmaModelRoute {
        kind: ArmaRouteKind::InvalidDecomposition {
            component: CanonicalComponent::Irregular,
            variance,
        },
        current_orders: orders,
    })
}

/// DecompSpectrum:1713-1738 NOADMISS0 NoDecompOut; SPECTRUM:482-489's
/// alternate return.
fn cycle_policy(variance: f64, orders: super::Orders) -> Option<ArmaModelRoute> {
    (variance < 0.0).then_some(ArmaModelRoute {
        kind: ArmaRouteKind::InvalidDecomposition {
            component: CanonicalComponent::Cycle,
            variance,
        },
        current_orders: orders,
    })
}

/// The mean of the residuals `a` and WM. A model without regular MA takes
/// the mean out of the residuals and into WM (ANALTS:2417–2483).
fn residual_mean<const N: usize>(
    a: &mut [f64],
    request: RequestedMean,
    frequency: Frequency,
    wm: f64,
    layout: ArmaLayout,
    parameters: &[f64; N],
) -> Result<MeanCorrection, Error> {
    let regular_p = layout.regular_ar_coordinates();
    if layout.regular_ma_count() == 0 {
        let seasonal_ar = layout.has_seasonal_ar().then(|| {
            (
                usize::from(frequency.periods_per_year()),
                parameters[regular_p],
            )
        });
        return mean_correction(
            a,
            request,
            wm,
            MeanFamily::Regular {
                ar: &parameters[..regular_p],
                seasonal_ar,
                seasonal_ma: layout.has_seasonal_ma(),
            },
        );
    }
    Ok(MeanCorrection {
        statistics: residual_mean_state(a, request)?,
        wm,
    })
}

/// ANSUB1's CHMODEL and ChangeModel checks on a pass's fit.
fn forward_model_policy<const N: usize>(
    init: Init,
    search: Option<&ArmaRefitResult<N>>,
    forward: &ForwardCalc<N>,
) -> Option<ArmaModelRoute> {
    let layout = forward.coefficients.layout;
    let current_orders = layout.orders();
    let regular_p = layout.regular_ar_coordinates();
    let fixed_by_search = init == Init::Refit0 && search.is_some_and(|r| r.fixed > 0);
    if layout.has_seasonal_ar() && !layout.seasonal_difference() {
        if layout.has_seasonal_ma() {
            let bphi = forward.coefficients.parameters[regular_p];
            let bth = forward.coefficients.parameters[N - 1];
            // ANSUB1 ChangeModel:3661–3671, independent of STATSEAS: a
            // positive BPHI below BTH cancels against it, and ChangeModel
            // drops BQ and refits. ANALTS runs CHMODEL first, but no pass
            // reaches both: CHMODEL below needs |BPHI|=1, and this needs
            // |BPHI|<|BTH|<=.99 (XMAX).
            if bphi.abs() < bth.abs() && bphi > 0.0 && bth > 0.0 {
                return Some(ArmaModelRoute {
                    kind: ArmaRouteKind::ChangeModel(super::Orders {
                        bq: 0,
                        ..current_orders
                    }),
                    current_orders,
                });
            }
        }
        // ANSUB1 CHMODEL:3193–3213: BP-- BD++.
        if fixed_by_search && (forward.coordinates[regular_p] - 1.0).abs() < 1e-10 {
            return Some(ArmaModelRoute {
                kind: ArmaRouteKind::Chmodel(super::Orders {
                    bp: 0,
                    bd: 1,
                    ..current_orders
                }),
                current_orders,
            });
        }
    }
    if layout == ArmaLayout::SeasonalAr {
        // CHMODEL's seasonal-unit branch requires BD<1. ChangeModel needs
        // POSBPHI=1 for BD=1/BP=1, but the default is POSBPHI=0 (ansub9.f).
        return None;
    }
    if layout.differences().ordinary() < 2 && fixed_by_search {
        // Default seas=1: CHMODEL's regular positive-unit-root branch
        // (CHMODEL:3158-3188, P-- D++).
        // With XL=.99, SEARCH (ansub1.f:169-205) fixes an AR coordinate only
        // at -UR or .99, so this +UR test can pass only if a free coordinate
        // lands on +1. Kept as in Build 62.
        for coordinate in 0..regular_p {
            if (forward.coordinates[coordinate] - 1.0).abs() < 1e-12 {
                return Some(ArmaModelRoute {
                    kind: ArmaRouteKind::Chmodel(super::Orders {
                        p: current_orders.p - 1,
                        d: current_orders.d + 1,
                        ..current_orders
                    }),
                    current_orders,
                });
            }
        }
    }
    // ANSUB1 ChangeModel:3660-3689 cannot rewrite BP=0. Its D=0/P>0
    // branches need STATSEAS=1 (default 0). The pure nonseasonal-MA
    // branch requires P=BD=0 and is outside the D=0 layouts the family runs.
    None
}

/// FCAST's forecasts and backcasts of `series` from the forward fit.
/// Backward CALCFX runs on the reversed differenced series from forward
/// CALCFX's state, and its coefficients go into `fit`.
fn finish_arma_tails<const N: usize>(
    frequency: Frequency,
    series: &[f64],
    wd: &[f64],
    fit: &mut ArmaFit<N>,
    state: ForwardTailState,
) -> Result<ArmaRunTails<N>, Error> {
    let s = usize::from(frequency.periods_per_year());
    let forward = &fit.forward;
    let ar = &forward.coefficients.phist;
    let layout = forward.coefficients.layout;
    let polynomial = std::iter::once(1.0)
        .chain(ar.iter().map(|v| -v))
        .collect::<Vec<_>>();
    let horizons = resolve_horizons(frequency, series.len());
    let inputs = TailInputs {
        bj_ma: forward.coefficients.thstar.clone(),
        horizons,
        bphist: bphist_ordered(s, &polynomial, layout.differences()),
        forcing_ar: if layout.has_seasonal_ar() && layout.regular_ar_coordinates() > 0 {
            MeanForcing::Expanded(ar.clone())
        } else if layout.has_seasonal_ar() {
            MeanForcing::One(ar[s - 1])
        } else if ar.is_empty() {
            MeanForcing::None
        } else if layout.ar_coordinates() == 1 {
            MeanForcing::One(ar[0])
        } else if layout.ar_coordinates() == 2 {
            MeanForcing::Two([ar[0], ar[1]])
        } else {
            MeanForcing::Regular(
                super::decomposition::root_allocation::RegularArCoefficients::Three([
                    ar[0], ar[1], ar[2],
                ]),
            )
        },
        order: layout.differences(),
        working_horizon: horizons.fhi.max(
            ar.len()
                + layout.differences().ordinary()
                + usize::from(layout.differences().seasonal()) * s
                + forward.coefficients.thstar.len(),
        ),
    };
    let core = finish_tails_numeric::<Error, _>(frequency, series, wd, state, inputs, |back| {
        let mut objective = ArmaObjective::resume(frequency, back, forward.continuation.clone());
        let residuals = objective.evaluate(forward.coordinates, forward.controls)?;
        let coefficients = objective.coefficients();
        let ma = coefficients.thstar.clone();
        Ok((residuals, ma, coefficients))
    })?;
    let tails = ArmaRunTails {
        base: core.tails,
        forward_coefficients: forward.coefficients.clone(),
        backward_bj_ma: core.backward_bj_ma,
        backward_coefficients: core.backward,
    };
    fit.backward = Some(tails.backward_coefficients.clone());
    Ok(tails)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::arma::model::LowOrder;

    #[test]
    fn a_model_without_regular_ma_takes_the_mean_out_of_its_residuals() {
        // ANALTS:2417–2483: Q=0 has a defined numerator, 1 for BQ=0
        // and 0 for BQ=1. The former gives WM=3+1000/(1-.25-.125)=1603.
        for (seasonal_ma, expected_wm) in [(false, 1603.0), (true, 3.0)] {
            let mut a = [999.0, 1001.0];
            let layout = ArmaLayout::Integrated {
                ar: ArOrder::Two,
                ma: LowOrder::Zero,
                seasonal_ar: false,
                ordinary_difference: LowOrder::One,
                seasonal_difference: false,
                seasonal_ma,
            };
            let correction = if seasonal_ma {
                residual_mean(
                    &mut a,
                    RequestedMean::Yes,
                    crate::Frequency::Quarterly,
                    3.0,
                    layout,
                    &[0.25, 0.125, 0.3],
                )
            } else {
                residual_mean(
                    &mut a,
                    RequestedMean::Yes,
                    crate::Frequency::Quarterly,
                    3.0,
                    layout,
                    &[0.25, 0.125],
                )
            }
            .expect("Q=0 correction is defined");
            assert_eq!(a, [-1.0, 1.0]);
            assert_eq!(correction.wm, expected_wm);
            // The statistics are those of the residuals before the correction.
            assert_eq!(correction.statistics.mean, 1000.0);
            assert!(correction.statistics.t > 100.0);
        }
    }

    #[test]
    fn a_negative_canonical_variance_ends_the_run_at_the_decomposition() {
        let orders = crate::seats::Orders::new([1, 1, 1], [0, 1, 1]);
        let route = cycle_policy(-0.25, orders).expect("a negative cycle variance");
        assert_eq!(
            route.into_error(),
            Failure::NegativeVariance {
                component: CanonicalComponent::Cycle,
                variance: -0.25,
            }
            .at(Stage::CanonicalDecomposition)
        );
        assert!(irregular_policy(0.0, orders).is_none());
    }

    #[test]
    fn a_model_change_ends_the_run_at_the_refit() {
        let current = crate::seats::Orders::new([1, 1, 1], [1, 0, 1]);
        let proposed = crate::seats::Orders::new([1, 1, 1], [1, 0, 0]);
        let route = ArmaModelRoute {
            kind: ArmaRouteKind::ChangeModel(proposed),
            current_orders: current,
        };
        assert_eq!(
            route.into_error(),
            Failure::ModelChange(ModelChange { current, proposed }).at(Stage::Refit)
        );
    }
}
