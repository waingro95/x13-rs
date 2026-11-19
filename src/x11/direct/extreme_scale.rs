//! Which extreme scale a pass uses: the robust `sdxtrm.f` estimators and
//! the `calendarsigma` choice.

use std::ops::Range;

use super::extension_geometry;
use super::extreme_weights::{in_scale_sample, irregular_standard_deviation};
use crate::error::{
    NumericalDetail, X11ExtremeScaleUnavailableReason, X11ExtremeStage, X11ExtremeWeightingPass,
};
use crate::time::{AxisSpan, TimeAxis};
use crate::x11::robust;
use crate::{Error, x11};

/// Robust extreme-scale estimation (`sdxtrm.f`) at one X-11 stage.
#[derive(Clone, Copy)]
pub(super) struct RobustContext {
    pub(super) estimator: x11::ExtremeEstimator,
    pub(super) axis: TimeAxis,
    /// One-based `sdxtrm.f` index of `axis`'s first value. Build 62 numbers its
    /// arrays from period 1 of the requested start year, and the tau
    /// estimators divide by the index one stride past the sample.
    pub(super) fortran_start: usize,
    /// The stage an error names.
    pub(super) stage: X11ExtremeStage,
}

/// One extreme-scale sample: the inclusive endpoints and stride of its window
/// (indices into the series), the weighting pass, the center, and the sigma
/// group when the scale is grouped.
#[derive(Clone, Copy)]
pub(super) struct ScaleSample {
    pub(super) first: usize,
    pub(super) last: usize,
    pub(super) stride: usize,
    pub(super) pass: X11ExtremeWeightingPass,
    pub(super) center: f64,
    pub(super) group: Option<x11::SigmaGroup>,
}

impl RobustContext {
    /// The context for a robust `estimator`; `None` for the standard
    /// deviation.
    pub(super) fn new(
        estimator: x11::ExtremeEstimator,
        requested_axis: TimeAxis,
        axis: TimeAxis,
    ) -> Result<Option<Self>, Error> {
        if estimator == x11::ExtremeEstimator::StandardDeviation {
            return Ok(None);
        }
        Self::required(estimator, requested_axis, axis).map(Some)
    }

    /// The context for any `estimator`. A pseudo-additive run takes the
    /// standard deviation through it too, for its empty-sample error.
    pub(super) fn required(
        estimator: x11::ExtremeEstimator,
        requested_axis: TimeAxis,
        axis: TimeAxis,
    ) -> Result<Self, Error> {
        debug_assert_eq!(requested_axis.frequency(), axis.frequency());
        let period = i64::from(axis.frequency().periods_per_year());
        let years = i64::from(axis.start().year()) - i64::from(requested_axis.start().year());
        let fortran_start = years
            .checked_mul(period)
            .and_then(|value| value.checked_add(i64::from(axis.start().subperiod())))
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(extension_geometry)?;
        Ok(Self {
            estimator,
            axis,
            fortran_start,
            stage: X11ExtremeStage::InitialSeasonalIrregular,
        })
    }

    /// The same context at `stage`.
    pub(super) fn at(self, stage: X11ExtremeStage) -> Self {
        Self { stage, ..self }
    }

    /// The scale of `irregular` over `indices`, less the values `sample.pass`
    /// leaves out.
    pub(super) fn scale(
        self,
        irregular: &[f64],
        weights: &[f64],
        indices: impl IntoIterator<Item = usize>,
        sample: ScaleSample,
    ) -> Result<f64, Error> {
        let ScaleSample {
            first,
            last,
            stride,
            pass,
            center,
            group,
        } = sample;
        let fortran_first = self
            .fortran_start
            .checked_add(first)
            .ok_or_else(extension_geometry)?;
        let fortran_last = self
            .fortran_start
            .checked_add(last)
            .ok_or_else(extension_geometry)?;
        let post_loop = robust::post_loop_index(fortran_first, fortran_last, stride)
            .ok_or_else(extension_geometry)?;
        let first = self.axis.period_at(first)?;
        let last = self.axis.period_at(last)?;
        let estimator = match self.estimator {
            x11::ExtremeEstimator::StandardDeviation => None,
            x11::ExtremeEstimator::MedianAbsoluteDeviation => Some(robust::RobustEstimator::Mad),
            x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation => {
                Some(robust::RobustEstimator::Tau)
            }
            x11::ExtremeEstimator::LogMedianAbsoluteDeviation => {
                Some(robust::RobustEstimator::LogMad)
            }
            x11::ExtremeEstimator::TauAdjustedLogMedianAbsoluteDeviation => {
                Some(robust::RobustEstimator::LogTau)
            }
        };
        let result = if let Some(estimator) = estimator {
            robust::scale(
                indices.into_iter().filter_map(|index| {
                    in_scale_sample(pass, weights[index]).then_some(irregular[index])
                }),
                center,
                estimator,
                post_loop,
            )
        } else {
            let scale = irregular_standard_deviation(irregular, weights, indices, pass, center);
            if scale.is_finite() {
                Ok(scale)
            } else {
                // An empty reweighting sample makes `sdxtrm.f` take
                // sqrt(0/0). Build 62 raises IEEE_INVALID and carries on with
                // a NaN scale; this returns a typed error instead.
                Err(robust::RobustScaleFailure::NonFiniteScale)
            }
        };
        result.map_err(|failure| {
            Error::from(NumericalDetail::X11ExtremeScale {
                estimator: self.estimator,
                reason: match failure {
                    robust::RobustScaleFailure::EmptySample => {
                        X11ExtremeScaleUnavailableReason::EmptySample
                    }
                    robust::RobustScaleFailure::ZeroTauBase => {
                        X11ExtremeScaleUnavailableReason::ZeroTauBase
                    }
                    robust::RobustScaleFailure::NonFiniteDeviation => {
                        X11ExtremeScaleUnavailableReason::NonFiniteDeviation
                    }
                    robust::RobustScaleFailure::NonPositiveLogInput => {
                        X11ExtremeScaleUnavailableReason::NonPositiveLogInput
                    }
                    robust::RobustScaleFailure::NonFiniteScale => {
                        X11ExtremeScaleUnavailableReason::NonFiniteScale
                    }
                },
                stage: self.stage,
                pass,
                first,
                last,
                group,
            })
        })
    }
}

/// The extreme scale in use: the requested one, or the one an automatic
/// request resolved to.
#[derive(Clone, Copy)]
pub(super) enum OperativeScale {
    RollingFiveYear,
    ByCalendarPeriod,
    Grouped(crate::x11::SigmaGroups),
}

/// How one irregular's extremes are weighted: the scale and, for a robust
/// estimator, its context.
#[derive(Clone, Copy)]
pub(super) struct ExtremeWeightPolicy {
    pub(super) scale: OperativeScale,
    pub(super) robust: Option<RobustContext>,
}

impl ExtremeWeightPolicy {
    /// The same scale estimated through `robust`.
    pub(super) fn with_context(mut self, robust: Option<RobustContext>) -> Self {
        self.robust = robust;
        self
    }
}

impl From<crate::x11::ExtremeScale> for ExtremeWeightPolicy {
    fn from(request: crate::x11::ExtremeScale) -> Self {
        let scale = match request {
            crate::x11::ExtremeScale::RollingFiveYear | crate::x11::ExtremeScale::Automatic => {
                OperativeScale::RollingFiveYear
            }
            crate::x11::ExtremeScale::ByCalendarPeriod => OperativeScale::ByCalendarPeriod,
            crate::x11::ExtremeScale::Grouped(groups) => OperativeScale::Grouped(groups),
        };
        Self {
            scale,
            robust: None,
        }
    }
}

/// The extreme scale in force through the B and C passes, and the automatic
/// (`calendarsigma=signif`) choices that set it.
pub(super) struct ExtremeScaleState {
    /// The legacy Easter state machine, saved on pass entry; `None` when the
    /// test's raw choice applies directly.
    sigma_entry: Option<SigmaState>,
    /// The state after the latest test; equals `sigma_entry` before the first.
    pub(super) sigma: Option<SigmaState>,
    pub(super) request: x11::Options,
    operative: OperativeScale,
    pub(super) initial_si: Option<crate::x11::AutomaticExtremeScale>,
    preliminary_irregular: Option<crate::x11::AutomaticExtremeScale>,
}

impl ExtremeScaleState {
    /// The requested scale, with an automatic one rolling until its first
    /// test.
    pub(super) fn new(request: x11::Options) -> Self {
        Self {
            request,
            sigma_entry: None,
            sigma: None,
            operative: ExtremeWeightPolicy::from(request.extreme_scale()).scale,
            initial_si: None,
            preliminary_irregular: None,
        }
    }

    /// Run both tests through the `entsch.f` state machine from `entry`.
    pub(super) fn with_sigma_entry(self, entry: SigmaState) -> Self {
        Self {
            sigma_entry: Some(entry),
            sigma: Some(entry),
            ..self
        }
    }

    /// Make one of pass B's two calendar-sigma tests on `irregular` over
    /// `span` (`initial` for the test of the initial SI's irregular) and set
    /// the scale the weightings after it use. Only an automatic scale tests.
    pub(super) fn resolve(
        &mut self,
        irregular: &[f64],
        span: Range<usize>,
        period: usize,
        mode: x11::Mode,
        initial: bool,
        dates: AxisSpan,
    ) -> Result<(), Error> {
        if self.request.extreme_scale() != crate::x11::ExtremeScale::Automatic {
            return Ok(());
        }
        // The test's critical-value table covers monthly and quarterly spans
        // of two years or more.
        debug_assert!(
            matches!(period, 4 | 12) && span.end <= irregular.len() && span.len() >= 2 * period
        );
        let raw = if self.sigma_entry == Some(SigmaState::Rolling) {
            // From the rolling state either result weights by calendar period,
            // so an unusable statistic is not an error.
            crate::x11::calendar_sigma::select(irregular, span, period, mode)
        } else {
            crate::x11::calendar_sigma::select_checked(irregular, span, period, mode, dates)?
        };
        let choice = if let Some(entry) = self.sigma_entry {
            let sigma =
                entry.after_test(raw == crate::x11::AutomaticExtremeScale::ByCalendarPeriod);
            self.sigma = Some(sigma);
            sigma.operative()
        } else {
            raw
        };
        self.operative = match choice {
            crate::x11::AutomaticExtremeScale::RollingFiveYear => OperativeScale::RollingFiveYear,
            crate::x11::AutomaticExtremeScale::ByCalendarPeriod => OperativeScale::ByCalendarPeriod,
        };
        if initial {
            self.initial_si = Some(choice);
        } else {
            self.preliminary_irregular = Some(choice);
        }
        Ok(())
    }

    /// The scale in force, without an estimator context.
    pub(super) fn weight_policy(&self) -> ExtremeWeightPolicy {
        ExtremeWeightPolicy {
            scale: self.operative,
            robust: None,
        }
    }

    /// The automatic choices, once both B-pass tests have run. `resolve` makes
    /// them only for an automatic scale.
    pub(super) fn selection(&self) -> Option<crate::x11::CalendarSigmaSelection> {
        self.initial_si.zip(self.preliminary_irregular).map(
            |(initial_si, preliminary_irregular)| {
                crate::x11::CalendarSigmaSelection::new(initial_si, preliminary_irregular)
            },
        )
    }
}

/// `entsch.f`'s `(Kersa, Ksdev)` pair under `calendarsigma=signif`, which the
/// legacy Easter prepass hands to the final run. Build 62 enters at `(0, 2)`,
/// and from there `entsch.f` reaches only these four pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum SigmaState {
    /// `(0, 2)`: the entry pair, also reached when a rolling state's test
    /// finds no calendar effect.
    #[default]
    Undecided,
    /// `(0, 0)`: rolling five-year scale.
    Rolling,
    /// `(1, 1)`: one scale per calendar period.
    ByCalendarPeriod,
    /// `(1, 2)`: a rolling state's test found a calendar effect; every later
    /// test keeps this pair.
    Latched,
}

impl SigmaState {
    /// `entsch.f` dispatches on the pass-entry pair (`k = Kersa + Ksdev + 1`),
    /// not on an earlier test's result, so both B-pass tests start from the
    /// state saved on entry. Arms `k = 2` and `k = 5` need entry pairs
    /// `(1, 0)` or `(0, 4)`, which `signif` never reaches.
    pub(super) fn after_test(self, by_calendar_period: bool) -> Self {
        match (self, by_calendar_period) {
            // k = 3: (Iv, Iv).
            (Self::Undecided | Self::ByCalendarPeriod, false) => Self::Rolling,
            (Self::Undecided | Self::ByCalendarPeriod, true) => Self::ByCalendarPeriod,
            // k = 1: (Iv, 2).
            (Self::Rolling, false) => Self::Undecided,
            (Self::Rolling, true) => Self::Latched,
            // k = 4: (1, 2).
            (Self::Latched, _) => Self::Latched,
        }
    }

    /// `Ksdev = 0` keeps the rolling scale; any other pair weights by
    /// calendar period.
    fn operative(self) -> crate::x11::AutomaticExtremeScale {
        if self == Self::Rolling {
            crate::x11::AutomaticExtremeScale::RollingFiveYear
        } else {
            crate::x11::AutomaticExtremeScale::ByCalendarPeriod
        }
    }

    /// `x11pt3.f`'s `Ksdev.gt.1`: rebuild the final SI from the C extremes.
    pub(super) fn retains_c_extremes(self) -> bool {
        matches!(self, Self::Undecided | Self::Latched)
    }
}
