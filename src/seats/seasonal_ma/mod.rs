//! The single-MA family, `(0 1 0)(0 1 1)` and `(0 1 1)(0 0 0)`, on the shared
//! ANALTS run loop.
//!
//! The model is one MA coefficient, seasonal or regular ([`SingleMaShape`]).
//! A pass computes CALCFX's residuals for the imported coefficient or refits
//! it with the one-coordinate SEARCH; the completed pass runs FCAST, the
//! ESTBUR extraction and DETCOMP's final components. The nonseasonal shape
//! has no seasonal component.

pub(crate) mod refit;
pub(crate) mod residual;

use super::effects::FittedEffectsInput;
use super::extraction::{
    ExtractedComponents, FinalWithTransitory, extract_seasonal_ma, final_identity,
};
use super::import::{MA_BOUND, trans0_degree_one};
use super::log_transform::final_log;
use super::mean::{
    MeanCorrection, MeanFamily, RequestedMean, mean_correction, residual_mean_state,
};
use super::pipeline::{
    Composed, DifferencedPass, Family, FittedPass, PassShape, Restoration, RunInput,
    SeatsTransform, run_family,
};
use super::refit::one_coordinate::SeasonalMaRefit;
use super::retry::Init;
use super::tails::{
    CoreTails, ForwardTailState, SeriesTails, TailInputs, finish_ma_only_tails, resolve_horizons,
    single_lag_bj_ma,
};
use crate::Frequency;
use crate::seats::failure::{Error, Stage, usable};
use refit::refit_seasonal_ma;
use residual::single_ma_residuals;

/// Which of the family's two models a run fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SingleMaShape {
    /// `(0 1 0)(0 1 1)`: the MA coefficient is at the seasonal lag.
    Seasonal,
    /// `(0 1 1)(0 0 0)`: the MA coefficient is at lag one, and the series is
    /// differenced once.
    Nonseasonal,
}

impl SingleMaShape {
    /// The lag of the MA coefficient.
    pub(super) fn lag(self, frequency: Frequency) -> usize {
        match self {
            Self::Seasonal => usize::from(frequency.periods_per_year()),
            Self::Nonseasonal => 1,
        }
    }
    pub(super) fn differences(self) -> super::tails::DifferenceOrder {
        match self {
            Self::Seasonal => super::tails::DifferenceOrder::First,
            Self::Nonseasonal => super::tails::DifferenceOrder::NonseasonalFirst,
        }
    }
}

/// The regARIMA estimate of a single-MA model: its MA coefficient, finite.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeasonalMaInput {
    frequency: Frequency,
    seasonal_ma: f64,
    shape: SingleMaShape,
}

impl SeasonalMaInput {
    /// The seasonal shape, `(0 1 0)(0 1 1)`.
    pub(crate) fn new(frequency: Frequency, seasonal_ma: f64) -> Result<Self, Error> {
        usable("fitted model coefficient", Some(0), seasonal_ma)
            .map_err(|failure| failure.at(Stage::Import))?;
        Ok(Self {
            frequency,
            seasonal_ma,
            shape: SingleMaShape::Seasonal,
        })
    }

    /// The nonseasonal shape, `(0 1 1)(0 0 0)`.
    pub(crate) fn nonseasonal(frequency: Frequency, ma: f64) -> Result<Self, Error> {
        let mut input = Self::new(frequency, ma)?;
        input.shape = SingleMaShape::Nonseasonal;
        Ok(input)
    }
}

/// A single-MA model in the SEATS convention: the MA coefficient of the
/// imported model, or of a later refit, at the lag `shape` gives.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSeasonalMa {
    pub(super) frequency: Frequency,
    pub(super) seasonal_ma: f64,
    pub(crate) shape: SingleMaShape,
}

impl ResolvedSeasonalMa {
    pub(crate) fn seasonal_ma(&self) -> f64 {
        self.seasonal_ma
    }
}

pub(crate) fn import_seasonal_ma(input: &SeasonalMaInput) -> ResolvedSeasonalMa {
    // NMLMDL subtracts the coefficient from zero and sends its group through
    // degree-one TRANS0/TRANS2; ANALTS reverses the sign.
    let bth = trans0_degree_one(0.0 - input.seasonal_ma, MA_BOUND);
    ResolvedSeasonalMa {
        frequency: input.frequency,
        seasonal_ma: -bth,
        shape: input.shape,
    }
}

/// A completed single-MA run.
pub(crate) type SeasonalMaRun = Composed<ResolvedSeasonalMa, SeasonalMaRefit>;

/// Runs SEATS on the imported single-MA model, restoring `calendar`'s fitted
/// effects when given.
pub(crate) fn compose_seasonal_ma(
    imported: &ResolvedSeasonalMa,
    input: RunInput<'_>,
    calendar: Option<&FittedEffectsInput>,
) -> Result<SeasonalMaRun, Error> {
    run_family(&SingleMa(imported), input, calendar)
}

struct SingleMa<'a>(&'a ResolvedSeasonalMa);

impl Family for SingleMa<'_> {
    type Start = ResolvedSeasonalMa;
    type Model = ResolvedSeasonalMa;
    type Refit = SeasonalMaRefit;
    type Tails = SeriesTails;
    type Decomposition = ExtractedComponents;
    type Failure = Error;

    const TMU_BEFORE_LOG: bool = true;

    fn frequency(&self) -> Frequency {
        self.0.frequency
    }

    fn initial_start(&self) -> Result<ResolvedSeasonalMa, Error> {
        Ok(self.0.clone())
    }

    fn pass_shape(
        &self,
        model: &ResolvedSeasonalMa,
        _observed_len: usize,
        _requested: RequestedMean,
    ) -> Result<PassShape, Error> {
        Ok(PassShape {
            differences: model.shape.differences(),
            free_parameters: 1,
        })
    }

    fn fit_pass(
        &self,
        model: &mut ResolvedSeasonalMa,
        init: Init,
        pass: &DifferencedPass,
        _requested: RequestedMean,
    ) -> Result<FittedPass<ResolvedSeasonalMa, SeasonalMaRefit>, Error> {
        let (residuals, refit) = if init == Init::Imported2 {
            let residuals =
                single_ma_residuals(model.frequency, model.shape, model.seasonal_ma, &pass.wd)?;
            (residuals, None)
        } else {
            let refit = refit_seasonal_ma(model.frequency, model.shape, &pass.wd, &pass.auto)?;
            model.seasonal_ma = refit.ma[0];
            (refit.residuals.clone(), Some(refit))
        };
        Ok(FittedPass {
            model: model.clone(),
            residuals,
            refit,
        })
    }

    fn correct_mean(
        &self,
        model: &ResolvedSeasonalMa,
        a: &mut [f64],
        requested: RequestedMean,
        wm: f64,
    ) -> Result<MeanCorrection, Error> {
        match model.shape {
            SingleMaShape::Seasonal => mean_correction(a, requested, wm, MeanFamily::SeasonalMa),
            SingleMaShape::Nonseasonal => Ok(MeanCorrection {
                statistics: residual_mean_state(a, requested)?,
                wm,
            }),
        }
    }

    fn tails(
        &self,
        model: &mut ResolvedSeasonalMa,
        series: &[f64],
        wd: &[f64],
        state: ForwardTailState,
    ) -> Result<SeriesTails, Error> {
        Ok(finish_seasonal_ma_tails(model, series, wd, state)?.tails)
    }

    fn decompose(
        &self,
        model: &ResolvedSeasonalMa,
        series: &[f64],
        tails: &SeriesTails,
    ) -> Result<ExtractedComponents, Error> {
        extract_seasonal_ma(model, series, tails)
    }

    fn final_components(
        &self,
        raw: ExtractedComponents,
        restoration: Restoration<'_>,
        transform: SeatsTransform,
    ) -> Result<FinalWithTransitory, Error> {
        let mut saved = match restoration {
            Restoration::FittedEffects(calendar) => {
                super::effects::final_with_effects(calendar, &raw)?
            }
            Restoration::Log(log) => final_log(log, &raw)?,
            Restoration::Identity(original) => final_identity(original, &raw)?,
        };
        if self.0.shape == SingleMaShape::Nonseasonal {
            restoration.without_seasonal_roots(&mut saved, transform);
        }
        Ok(FinalWithTransitory {
            saved,
            transitory: None,
        })
    }
}

/// FCAST for the single MA, with FHI at least two years.
pub(super) fn finish_seasonal_ma_tails(
    model: &ResolvedSeasonalMa,
    observed: &[f64],
    wd: &[f64],
    state: ForwardTailState,
) -> Result<CoreTails, Error> {
    let period = usize::from(model.frequency.periods_per_year());
    let bj_ma = single_lag_bj_ma(model.shape.lag(model.frequency), model.seasonal_ma);
    let mut horizons = resolve_horizons(model.frequency, observed.len());
    horizons.fhi = horizons.fh.max(2 * period);
    let inputs = TailInputs::ma_only(period, bj_ma, horizons, model.shape.differences());
    finish_ma_only_tails(
        model.frequency,
        observed,
        wd,
        state,
        inputs,
        |backward_wd| {
            single_ma_residuals(model.frequency, model.shape, model.seasonal_ma, backward_wd)
        },
    )
}
