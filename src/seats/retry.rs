//! ANALTS's retry of a pass: the Q test on the residuals, the mean it may
//! drop and the INIT option each pass runs under.
//!
//! ANALTS first computes the residuals of the imported coefficients (INIT=2)
//! and tests their Ljung-Box Q. A Q above 50 and above 1.5 times BLQT, the Q
//! of the regARIMA summary, fails the test. ANALTS then either drops a mean
//! whose series averages near zero and tests again, or has SEARCH re-estimate
//! the coefficients (INIT=0). Once the test passes or the model is refit,
//! ANALTS runs one more pass without the test, and that pass completes the
//! run. A run so takes two or three passes, or one when it starts at INIT=0.

use super::mean::RequestedMean;

/// SEATS's INIT option for a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Init {
    /// INIT=2: the pass keeps the imported coefficients and computes their
    /// residuals.
    Imported2,
    /// INIT=0: SEARCH estimates the coefficients from STAVAL's starting
    /// values.
    Refit0,
}

/// What ANALTS does after a pass's Q test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetryDecision {
    /// The test failed with a mean ANALTS takes for negligible: repeat the
    /// pass without the mean and test again.
    RemoveMeanAndRepeat,
    /// The test failed: repeat the pass with INIT=0. `restored_mean` says a
    /// mean removed for the previous pass is back.
    ResetToRefit { restored_mean: bool },
    /// The test passed: repeat the pass as it is, without the test.
    KeepImportedAndRepeat,
    /// The pass ran without the test and completes the run.
    Complete,
}

/// ANALTS's retry flags over the passes of one run.
#[derive(Debug, Clone)]
pub(super) struct RetryState {
    init: Init,
    /// The next pass runs without the Q test.
    noretry: bool,
    /// The current pass runs without a mean the run asked for.
    nprova: bool,
    saved_mean: RequestedMean,
    current_mean: RequestedMean,
    pass_index: usize,
    complete: bool,
}

impl RetryState {
    /// The state of a run that starts at INIT=0. ANALTS 2649 enters the Q
    /// test only under INIT=2, so the first pass completes the run.
    pub(super) fn refit(request: RequestedMean) -> Self {
        Self {
            init: Init::Refit0,
            noretry: true,
            ..Self::new(request)
        }
    }

    /// The state of a run that starts at INIT=2 with the Q test on.
    pub(super) fn new(request: RequestedMean) -> Self {
        Self {
            init: Init::Imported2,
            noretry: false,
            nprova: false,
            saved_mean: request,
            current_mean: request,
            pass_index: 0,
            complete: false,
        }
    }

    /// The INIT option of the pass about to run.
    pub(super) fn init(&self) -> Init {
        self.init
    }

    /// Whether the pass about to run estimates a mean.
    pub(super) fn current_mean(&self) -> RequestedMean {
        self.current_mean
    }

    /// Decides what follows a pass whose residuals have the Ljung-Box
    /// statistic `q`, given TMU and the regARIMA summary's BLQT.
    ///
    /// The run calls it once per pass until it returns
    /// [`RetryDecision::Complete`], which it does by the third pass. The
    /// three statistics are finite: AUTO, TMU and the regARIMA summary
    /// checked them.
    pub(super) fn decide(&mut self, q: f64, tmu: f64, blqt: f64) -> RetryDecision {
        debug_assert!(
            !self.complete
                && self.pass_index <= 2
                && q.is_finite()
                && tmu.is_finite()
                && blqt.is_finite()
                && (self.init == Init::Imported2 || self.noretry),
            "retry decision outside ANALTS's sequence after {} passes",
            self.pass_index
        );
        if self.noretry {
            self.complete = true;
            return RetryDecision::Complete;
        }
        self.pass_index += 1;
        // ANALTS 3137–3167: both tests are strict, and every branch returns
        // to label 10 for another pass.
        if q > 1.5_f64 * blqt && q > f64::from(50) {
            if self.current_mean == RequestedMean::Yes && tmu.abs() < 1.90_f64 && !self.nprova {
                self.saved_mean = self.current_mean;
                self.current_mean = RequestedMean::No;
                self.nprova = true;
                RetryDecision::RemoveMeanAndRepeat
            } else {
                self.init = Init::Refit0;
                let restored_mean = self.nprova;
                if self.nprova {
                    self.nprova = false;
                    self.current_mean = self.saved_mean;
                }
                self.noretry = true;
                RetryDecision::ResetToRefit { restored_mean }
            }
        } else {
            self.noretry = true;
            RetryDecision::KeepImportedAndRepeat
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_must_exceed_both_fifty_and_one_and_a_half_blqt_to_refit() {
        for (q, blqt, refit) in [
            (50.0, 0.0, false),
            (50.0_f64.next_up(), 0.0, true),
            (75.0, 50.0, false),
            (75.0_f64.next_up(), 50.0, true),
            (50.0, 100.0, false),
        ] {
            let mut state = RetryState::new(RequestedMean::No);
            let (decision, init) = if refit {
                (
                    RetryDecision::ResetToRefit {
                        restored_mean: false,
                    },
                    Init::Refit0,
                )
            } else {
                (RetryDecision::KeepImportedAndRepeat, Init::Imported2)
            };
            assert_eq!(state.decide(q, 100.0, blqt), decision);
            assert_eq!(state.init(), init);
            assert_eq!(state.decide(q, 100.0, blqt), RetryDecision::Complete);
        }
    }

    #[test]
    fn a_mean_is_dropped_only_below_a_tmu_of_1_90_in_magnitude() {
        for tmu in [1.90, -1.90, 1.90_f64.next_up(), -1.90_f64.next_up()] {
            let mut state = RetryState::new(RequestedMean::Yes);
            assert_eq!(
                state.decide(60.0, tmu, 0.0),
                RetryDecision::ResetToRefit {
                    restored_mean: false
                }
            );
            assert_eq!(state.current_mean(), RequestedMean::Yes);
        }
        for tmu in [0.0, 1.90_f64.next_down(), -1.90_f64.next_down()] {
            let mut state = RetryState::new(RequestedMean::Yes);
            assert_eq!(
                state.decide(60.0, tmu, 0.0),
                RetryDecision::RemoveMeanAndRepeat
            );
            assert_eq!(state.current_mean(), RequestedMean::No);
            assert_eq!(state.init(), Init::Imported2);
        }
    }

    #[test]
    fn a_dropped_mean_returns_only_with_a_refit() {
        for (second_q, decision, mean, init) in [
            (
                60.0,
                RetryDecision::ResetToRefit {
                    restored_mean: true,
                },
                RequestedMean::Yes,
                Init::Refit0,
            ),
            (
                50.0,
                RetryDecision::KeepImportedAndRepeat,
                RequestedMean::No,
                Init::Imported2,
            ),
        ] {
            let mut state = RetryState::new(RequestedMean::Yes);
            assert_eq!(
                state.decide(60.0, 0.0, 0.0),
                RetryDecision::RemoveMeanAndRepeat
            );
            assert_eq!(state.decide(second_q, 0.0, 0.0), decision);
            assert_eq!(state.current_mean(), mean);
            assert_eq!(state.init(), init);
            assert_eq!(state.decide(60.0, 0.0, 0.0), RetryDecision::Complete);
        }
    }

    #[test]
    fn a_run_starting_at_refit_completes_with_its_first_pass() {
        let mut state = RetryState::refit(RequestedMean::Yes);
        assert_eq!(state.init(), Init::Refit0);
        assert_eq!(state.decide(60.0, 0.0, 0.0), RetryDecision::Complete);
        assert_eq!(state.current_mean(), RequestedMean::Yes);
    }
}
