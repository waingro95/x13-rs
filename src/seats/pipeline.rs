//! The ANALTS run every SEATS family shares: the retry passes and the
//! decomposition of the completed pass, around per-family hooks.
//!
//! Every SEATS family runs the same ANALTS steps around its own model.
//! [`run_family`] is those steps:
//!
//! 1. Input checks: the family's own (`validate`).
//! 2. TMU and the log preparation. TMU is the mean of the original series; the
//!    retry decision reads it. The family's `TMU_BEFORE_LOG` keeps the order
//!    of the two, which decides the error an input that fails both reports.
//!    The initial model comes after the log preparation.
//! 3. The retry passes. Each pass rebuilds WD from the unchanged series:
//!    difference → mean removal (WM) → AUTO → INIT=2 residuals or SEARCH →
//!    residual mean correction → AUTO on the residuals → retry decision.
//!    `RetryState::decide` ends the loop: it completes by the third pass.
//! 4. The completed pass: AUTO on the squared residuals, FCAST tails, the
//!    family's decomposition (root allocation, spectral minima, canonical and
//!    adjusted components, ESTBUR extraction), then the saved components,
//!    restored through the fitted effects or the log.
//!
//! Every step reads one series: with fitted effects, the series ADJREG leaves
//! for the extraction; else the re-logged OZ of a log fit; else the
//! transformed series. AR2 differences the transformed series even under a
//! log (`DIFFERENCES_LOGGED_OZ`).
//!
//! # Family hooks
//!
//! A family implements [`Family`]: its model, failure and decomposition
//! types, the constants above, and the hooks. Each family keeps Build 62's
//! arithmetic, its check order and its error variants; `run_family` holds only
//! the steps every family runs the same way, and a step that differs is a
//! hook.
//!
//! - `validate`: its check of the input, which only the ARMA family has;
//! - `initial_start`: what the first pass fits from, usually the imported
//!   model; AR2's import can fail;
//! - `initial_retry` and `initial_lags`: the retry state and AUTO lags the
//!   first pass starts from; the ARMA family can start at INIT=0, with lags
//!   kept from the run it restarts;
//! - `pass_shape`: the difference operator and the free-parameter count that
//!   sizes AUTO's lags;
//! - `fit_pass`: INIT=2 residuals or SEARCH, which replaces what the next pass
//!   fits from; the pass's fit is the model the later hooks read;
//! - `correct_mean`: the residual mean step and the WM the tails use;
//! - `tails`: FCAST, whose backward CALCFX may refresh the model (ARMA);
//! - `decompose`: root allocation through ESTBUR extraction;
//! - `final_components`: the saved components under a restoration.
//!
//! The engine reads the returned [`Composed`]. A pass record holds what it
//! reads: the retry decision and the SEARCH result. A family whose run can
//! end early without failing (the ARMA family, which may restart with another
//! model) carries that outcome in its `Failure` type and unwraps it in its
//! entry point.

use super::autocorrelation::{AutoCallDimensions, AutoLagState, AutoResult, autocorrelations};
use super::effects::FittedEffectsInput;
use super::extraction::{FinalComponents, FinalWithTransitory};
use super::log_transform::{LogInput, prepare_transformed_log_input};
use super::mean::{
    MeanCorrection, MeanDiagnosticStage, RequestedMean, ResidualMean, checked, prepare_mean_wd,
};
use super::residual::ResidualResult;
use super::retry::{Init, RetryDecision, RetryState};
use super::tails::{DifferenceOrder, ForwardTailState, difference_ordered};
use crate::Frequency;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeatsTransform {
    Identity,
    Log,
}

/// Whether the regARIMA summary gave this run a Ljung-Box Q statistic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SummaryQ {
    Absent,
    Present { blqt: f64 },
}

impl SummaryQ {
    /// The statistic, or zero when the summary has none.
    pub(crate) fn blqt(self) -> f64 {
        match self {
            Self::Absent => 0.0,
            Self::Present { blqt } => blqt,
        }
    }
}

/// The series and regARIMA results a family's run starts from.
pub(crate) struct RunInput<'a> {
    pub(crate) original: &'a [f64],
    pub(crate) transformed: &'a [f64],
    pub(crate) transform: SeatsTransform,
    pub(crate) requested_mean: RequestedMean,
    pub(crate) summary_q: SummaryQ,
}

/// What the saved components are restored through.
#[derive(Clone, Copy)]
pub(super) enum Restoration<'a> {
    /// ADJREG's fitted effects, which DETCOMP puts back.
    FittedEffects(&'a FittedEffectsInput),
    /// The log, with DETCOMP's bias factor of one.
    Log(&'a LogInput),
    /// Nothing: the components add up to the original series.
    Identity(&'a [f64]),
}

impl Restoration<'_> {
    /// DETCOMP (ansub4:1971–2093/2838–2957) for a model without seasonal
    /// roots: without trend or irregular outliers, final SA is TRAM, even with
    /// TD/Easter; presence is a regressor count, not a test of effect size.
    /// USRENTRY(IFUNC=1201), ansub9:45–53, marks an identity fit's all-zero
    /// additive seasonal absent; a log fit saves ones instead.
    pub(super) fn without_seasonal_roots(
        self,
        saved: &mut FinalComponents,
        transform: SeatsTransform,
    ) {
        if let Self::FittedEffects(effects) = self
            && !effects.requires_final_adjustment()
        {
            saved.seasonally_adjusted = effects.tram().to_vec();
        }
        if transform == SeatsTransform::Identity {
            saved.seasonal.clear();
        }
    }
}

/// One pass's differencing: the operator and the free-parameter count ANALTS
/// sizes AUTO's lags with.
pub(super) struct PassShape {
    pub(super) differences: DifferenceOrder,
    pub(super) free_parameters: usize,
}

/// WD after the mean removal and AUTO, its mean WM, AUTO's result and the
/// lag counts the pass entered with.
pub(super) struct DifferencedPass {
    pub(super) wd: Vec<f64>,
    pub(super) wm: f64,
    pub(super) auto: AutoResult,
    pub(super) lags: AutoLagState,
}

/// A pass's fit and its residuals, plus SEARCH's result when the pass refit.
pub(super) struct FittedPass<M, R> {
    pub(super) model: M,
    pub(super) residuals: ResidualResult,
    pub(super) refit: Option<R>,
}

/// One pass's record: SEARCH's result when the pass refit, and the retry
/// decision that followed it.
#[derive(Debug)]
pub(crate) struct Pass<R> {
    pub(crate) refit: Option<R>,
    pub(crate) decision: RetryDecision,
}

/// A completed run: the final model and residual mean, the saved components
/// and each pass's record.
#[derive(Debug)]
pub(crate) struct Composed<M, R> {
    pub(crate) model: M,
    pub(crate) requested_mean: RequestedMean,
    pub(crate) mean: ResidualMean,
    pub(crate) saved: FinalComponents,
    pub(crate) transitory: Option<Vec<f64>>,
    pub(crate) passes: Vec<Pass<R>>,
}

/// A SEATS family's part of the ANALTS run; see the module docs.
pub(super) trait Family {
    /// What a pass fits from: the imported model, then the previous pass's
    /// result.
    type Start;
    /// A pass's fit, which the later hooks read and the run returns.
    type Model;
    /// SEARCH's result, kept on the pass record.
    type Refit;
    /// What `tails` hands `decompose`.
    type Tails;
    /// What `decompose` hands `final_components`.
    type Decomposition;
    type Failure: From<super::Error>;

    /// Whether TMU runs before the log preparation.
    const TMU_BEFORE_LOG: bool;
    /// Whether a log fit without fitted effects differences the re-logged OZ;
    /// otherwise it differences the transformed series.
    const DIFFERENCES_LOGGED_OZ: bool = true;

    fn frequency(&self) -> Frequency;

    /// The family's check of the input before the first pass.
    fn validate(&self, _input: &RunInput<'_>) -> Result<(), Self::Failure> {
        Ok(())
    }
    /// What the first pass fits from, resolved between the log preparation
    /// and a late TMU.
    fn initial_start(&self) -> Result<Self::Start, Self::Failure>;
    /// The retry state the first pass starts from: INIT=2.
    fn initial_retry(&self, requested: RequestedMean) -> RetryState {
        RetryState::new(requested)
    }
    /// AUTO's lag counts on entry to the first pass: the defaults.
    fn initial_lags(&self) -> AutoLagState {
        AutoLagState::new(self.frequency())
    }
    fn pass_shape(
        &self,
        start: &Self::Start,
        observed_len: usize,
        requested: RequestedMean,
    ) -> Result<PassShape, Self::Failure>;
    /// INIT=2 residuals of `start`, or SEARCH, which replaces `start`.
    fn fit_pass(
        &self,
        start: &mut Self::Start,
        init: Init,
        pass: &DifferencedPass,
        requested: RequestedMean,
    ) -> Result<FittedPass<Self::Model, Self::Refit>, Self::Failure>;
    /// The residual mean statistics of `a`, which it may correct in place,
    /// and the WM the tails use.
    fn correct_mean(
        &self,
        model: &Self::Model,
        a: &mut [f64],
        requested: RequestedMean,
        wm: f64,
    ) -> Result<MeanCorrection, super::Error>;
    /// FCAST, whose backward CALCFX may refresh `model`.
    fn tails(
        &self,
        model: &mut Self::Model,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<Self::Tails, Self::Failure>;
    fn decompose(
        &self,
        model: &Self::Model,
        series: &[f64],
        tails: &Self::Tails,
    ) -> Result<Self::Decomposition, Self::Failure>;
    fn final_components(
        &self,
        decomposition: Self::Decomposition,
        restoration: Restoration<'_>,
        transform: SeatsTransform,
    ) -> Result<FinalWithTransitory, Self::Failure>;
}

/// Runs `family`'s ANALTS steps over `input`, restoring the fitted `effects`
/// when given.
pub(super) fn run_family<F: Family>(
    family: &F,
    input: RunInput<'_>,
    effects: Option<&FittedEffectsInput>,
) -> Result<Composed<F::Model, F::Refit>, F::Failure> {
    let frequency = family.frequency();
    // With fitted effects the transformed series is no longer the transform
    // of the original; building the effects checked both.
    debug_assert!(effects.is_some() || transforms_original(&input));
    debug_assert!(length_in_range(family.frequency(), input.original.len()));
    family.validate(&input)?;
    let early_tmu = if F::TMU_BEFORE_LOG {
        Some(original_tmu(input.original)?)
    } else {
        None
    };
    let log = match (effects, input.transform) {
        (None, SeatsTransform::Log) => Some(prepare_transformed_log_input(
            frequency,
            input.original,
            input.transformed,
        )?),
        _ => None,
    };
    let mut start = family.initial_start()?;
    let tmu = match early_tmu {
        Some(tmu) => tmu,
        None => original_tmu(input.original)?,
    };
    let series = match (effects, &log) {
        (Some(effects), _) => effects.extraction_response(),
        (None, Some(log)) if F::DIFFERENCES_LOGGED_OZ => log.logged(),
        (None, _) => input.transformed,
    };
    let blqt = input.summary_q.blqt();
    let mut retry = family.initial_retry(input.requested_mean);
    let mut lags = family.initial_lags();
    let mut passes = Vec::new();
    // `decide` completes by the third pass or fails, so the loop ends.
    loop {
        let requested = retry.current_mean();
        let init = retry.init();
        let shape = family.pass_shape(&start, series.len(), requested)?;
        // Each pass rebuilds WD from the unchanged series.
        let pass = difference_pass(series, frequency, requested, &shape, &mut lags)?;
        let FittedPass {
            mut model,
            residuals,
            refit,
        } = family.fit_pass(&mut start, init, &pass, requested)?;
        let mut a = residuals.normalized_residuals.clone();
        let correction = family.correct_mean(&model, &mut a, requested, pass.wm)?;
        let na = a.len();
        let residual_auto = autocorrelations(
            &mut a,
            MeanDiagnosticStage::ForwardResidual,
            AutoCallDimensions {
                m: lags.m,
                iq: lags.iq,
                nw: na,
            },
        )?;
        let decision = retry.decide(residual_auto.qstat, tmu, blqt);
        passes.push(Pass { refit, decision });
        if decision != RetryDecision::Complete {
            continue;
        }
        squared_auto(&a, pass.wd.len(), lags)?;
        let mean = correction.statistics;
        let tails = family.tails(
            &mut model,
            series,
            &pass.wd,
            ForwardTailState {
                wm: correction.wm,
                requested_mean: mean.requested,
                residuals: a,
            },
        )?;
        let decomposition = family.decompose(&model, series, &tails)?;
        let restoration = match (effects, &log) {
            (Some(effects), _) => Restoration::FittedEffects(effects),
            (None, Some(log)) => Restoration::Log(log),
            (None, None) => Restoration::Identity(input.original),
        };
        let saved = family.final_components(decomposition, restoration, input.transform)?;
        return Ok(Composed {
            model,
            requested_mean: input.requested_mean,
            mean,
            saved: saved.saved,
            transitory: saved.transitory,
            passes,
        });
    }
}

/// Whether `transformed` is the transform of `original`, bit for bit, and
/// both are finite.
fn transforms_original(input: &RunInput<'_>) -> bool {
    input.transformed.len() == input.original.len()
        && input
            .original
            .iter()
            .zip(input.transformed)
            .all(|(&original, &transformed)| {
                original.is_finite()
                    && transformed.is_finite()
                    && transformed.to_bits()
                        == match input.transform {
                            SeatsTransform::Identity => original,
                            SeatsTransform::Log => original.ln(),
                        }
                        .to_bits()
            })
}

/// Whether `observed` is within CheckLen's length range: 36 to 780 monthly
/// observations, 16 to 328 quarterly.
fn length_in_range(frequency: Frequency, observed: usize) -> bool {
    let (minimum, maximum) = match frequency {
        Frequency::Monthly => (36, 780),
        Frequency::Quarterly => (16, 328),
    };
    (minimum..=maximum).contains(&observed)
}

/// ANALTS's TMU, the mean of the original series.
fn original_tmu(original: &[f64]) -> Result<f64, super::Error> {
    let count = original.len();
    let mut tmu = 0.0;
    for &value in original {
        tmu = checked(tmu + value, "original mean sum")?;
    }
    checked(tmu / count as f64, "original mean")
}

/// Fits the AUTO lags to the pass, then differences `series`, removes the
/// requested mean and runs AUTO on the result.
pub(super) fn difference_pass(
    series: &[f64],
    frequency: Frequency,
    requested: RequestedMean,
    shape: &PassShape,
    lags: &mut AutoLagState,
) -> Result<DifferencedPass, super::Error> {
    lags.enter_difference_pass(
        frequency,
        series.len(),
        requested,
        shape.free_parameters,
        shape.differences,
    )?;
    let mut wd = difference_ordered(
        series,
        usize::from(frequency.periods_per_year()),
        shape.differences,
    )?;
    let wm = prepare_mean_wd(&mut wd, requested)?;
    let nw = wd.len();
    let auto = autocorrelations(
        &mut wd,
        MeanDiagnosticStage::Differenced,
        AutoCallDimensions {
            m: lags.m,
            iq: 0,
            nw,
        },
    )?;
    Ok(DifferencedPass {
        wd,
        wm,
        auto,
        lags: *lags,
    })
}

/// ANALTS 2728: AUTO on the squares of the completed pass's residuals `a`,
/// with a Q statistic over the differenced length `nw`. Build 62 only prints
/// its results, but a run fails where the call does.
fn squared_auto(a: &[f64], nw: usize, lags: AutoLagState) -> Result<(), super::Error> {
    let mut squares = a.iter().map(|value| value * value).collect::<Vec<_>>();
    autocorrelations(
        &mut squares,
        MeanDiagnosticStage::SquaredResidual,
        AutoCallDimensions {
            m: lags.m,
            iq: lags.iq,
            nw,
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::{Failure, Stage};

    #[test]
    fn squared_residuals_without_variation_fail_the_completed_pass() {
        let a = (0..80)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        assert_eq!(
            squared_auto(&a, 75, AutoLagState::new(Frequency::Quarterly)),
            Err(Failure::Unusable {
                quantity: "energy of the squared residuals",
                index: None,
                value: 0.0,
            }
            .at(Stage::Autocorrelation))
        );
    }

    #[test]
    fn squared_q_needs_a_differenced_series_longer_than_its_lags() {
        for (frequency, n, iq) in [
            (Frequency::Monthly, 180, 24),
            (Frequency::Quarterly, 80, 16),
        ] {
            let a = (0..n).map(|i| (i as f64 * 0.43).sin()).collect::<Vec<_>>();
            let lags = AutoLagState::new(frequency);
            for nw in [0, iq] {
                assert_eq!(
                    squared_auto(&a, nw, lags),
                    Err(Failure::AutocorrelationSample { count: nw }.at(Stage::Mean))
                );
            }
            assert_eq!(squared_auto(&a, iq + 1, lags), Ok(()));
        }
    }
}
