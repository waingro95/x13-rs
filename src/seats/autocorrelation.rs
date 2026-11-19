//! AUTO (`ansub1.f`): the autocorrelations of a series and their Ljung-Box
//! statistic, with the lag counts ANALTS resolves for each pass.
//!
//! ANALTS calls AUTO three times in a pass. The autocorrelations of the
//! differenced series give STAVAL its starting values, the Q statistic of the
//! residuals decides a retry, and the squared residuals of the completed pass
//! are checked the same way. AUTO takes the differenced series as it is and
//! centres the residuals and their squares around their own mean, which it
//! adds back afterwards.
//!
//! AUTO's standard errors and seasonal statistic feed only Build 62's printed
//! tables and are not computed.

use super::failure::{Error, Failure, Stage};
use super::mean::{MeanDiagnosticStage, RequestedMean, checked, ordered_mean};
use super::tails::DifferenceOrder;
use crate::Frequency;

/// AUTO's lag counts: `m` autocorrelations, of which the Q statistic sums the
/// first `iq`. ANALTS resolves the default IQ before its retry loop, and both
/// counts persist over the passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AutoLagState {
    pub(crate) m: usize,
    pub(crate) iq: usize,
}

impl AutoLagState {
    /// ANALTS's defaults: 36 lags, and a Q over two years of them.
    pub(crate) fn new(frequency: Frequency) -> Self {
        Self {
            m: 36,
            iq: match frequency {
                Frequency::Monthly => 24,
                Frequency::Quarterly => 16,
            },
        }
    }

    /// ANALTS 1837–1854: on entry to a pass, fits the lag counts to the
    /// differenced length of a short series.
    pub(super) fn enter_difference_pass(
        &mut self,
        frequency: Frequency,
        observed_len: usize,
        mean: RequestedMean,
        free_parameters: usize,
        order: DifferenceOrder,
    ) -> Result<(), Error> {
        let fail = || {
            Failure::AutocorrelationSample {
                count: observed_len,
            }
            .at(Stage::Mean)
        };
        let period = usize::from(frequency.periods_per_year());
        let nw = observed_len
            .checked_sub(usize::from(order.seasonal()) * period + order.ordinary())
            .ok_or_else(fail)?;
        if self.m >= nw {
            self.m = nw;
            if self.iq >= self.m {
                self.iq = self.m.checked_sub(2).ok_or_else(fail)?;
            }
            if self.iq < free_parameters + usize::from(mean == RequestedMean::Yes) {
                self.iq = self.m;
            }
        }
        if self.m < period || self.iq == 0 || nw == 0 || self.m > 36 {
            return Err(fail());
        }
        if self.iq > self.m {
            self.iq = self.m;
        }
        Ok(())
    }
}

/// The sizes of one AUTO call: `m` lags, a Q statistic over the first `iq`
/// of them (none when zero) and `nw`, the sample size that statistic uses.
#[derive(Debug, Clone, Copy)]
pub(super) struct AutoCallDimensions {
    pub(super) m: usize,
    pub(super) iq: usize,
    pub(super) nw: usize,
}

impl AutoCallDimensions {
    fn validate(self, n: usize) -> Result<(), Error> {
        // AUTO does not need M <= N: its lag loop sums nothing at the lags
        // from N on.
        debug_assert!(
            n != 0 && (1..=36).contains(&self.m) && self.iq <= self.m,
            "AUTO dimensions n={n}, m={}, iq={}",
            self.m,
            self.iq
        );
        if self.iq != 0 && (self.nw <= self.iq || self.nw > i32::MAX as usize) {
            return Err(Failure::AutocorrelationSample { count: self.nw }.at(Stage::Mean));
        }
        Ok(())
    }
}

/// What a run reads of AUTO's results.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AutoResult {
    /// M, the number of autocorrelations computed.
    pub(crate) active_lags: usize,
    /// The autocorrelations at lags `1..=active_lags`; the rest are zero.
    pub(crate) acf: [f64; 36],
    /// The Ljung-Box Q over the first IQ lags, or -1 when IQ is zero.
    pub(crate) qstat: f64,
}

/// AUTO on `values`, which it centres and restores unless `stage` is the
/// differenced series.
pub(super) fn autocorrelations(
    values: &mut [f64],
    stage: MeanDiagnosticStage,
    dimensions: AutoCallDimensions,
) -> Result<AutoResult, Error> {
    let n = values.len();
    dimensions.validate(n)?;
    let AutoCallDimensions { m, iq, nw } = dimensions;
    let (mean, energy) = center(values, stage)?;
    let c0 = checked(energy / n as f64, "lag-zero autocovariance")?;
    let mut acf = [0.0; 36];
    for (lag, slot) in (1..=m).zip(&mut acf) {
        let mut covariance = 0.0;
        for (&value, &lagged) in values.iter().skip(lag).zip(values.iter()) {
            covariance = value.mul_add(lagged, covariance);
        }
        covariance /= n as f64;
        *slot = checked(covariance / c0, "autocorrelation")?;
    }
    // QSTAT stays -1 when AUTO computes no statistic.
    let qstat = if iq == 0 {
        -1.0
    } else {
        let mut qstat = 0.0;
        for lag in 1..=iq {
            qstat += acf[lag - 1] * acf[lag - 1] / (nw - lag) as f64;
        }
        checked(qstat * nw as f64 * (nw + 2) as f64, "Ljung-Box Q")?
    };
    restore(values, stage, mean)?;
    Ok(AutoResult {
        active_lags: m,
        acf,
        qstat,
    })
}

/// Centres the residuals or their squares around their mean and returns that
/// mean with the sum of squares. The differenced series keeps its values and
/// a mean of zero. A sum of squares that is not finite, or is zero by DPEQ's
/// test, is [`Unusable`](Failure::Unusable) at [`Stage::Autocorrelation`].
fn center(values: &mut [f64], stage: MeanDiagnosticStage) -> Result<(f64, f64), Error> {
    debug_assert!(!values.is_empty(), "AUTO on an empty series");
    let mean = if stage == MeanDiagnosticStage::Differenced {
        0.0
    } else {
        ordered_mean(values, "autocorrelation mean")?
    };
    if stage != MeanDiagnosticStage::Differenced {
        for value in values.iter_mut() {
            *value = checked(*value - mean, "autocorrelation centering")?;
        }
    }
    let unusable = |energy| {
        Failure::Unusable {
            quantity: stage.label(),
            index: None,
            value: energy,
        }
        .at(Stage::Autocorrelation)
    };
    let mut energy = 0.0;
    for &value in values.iter() {
        energy = value.mul_add(value, energy);
        if !energy.is_finite() {
            return Err(unusable(energy));
        }
    }
    // DPEQ (`dpeq.f`) calls the sum zero when it is strictly below 3.834e-20.
    if energy.abs() < 3.834e-20 {
        return Err(unusable(energy));
    }
    Ok((mean, energy))
}

/// Adds the mean [`center`] removed back to `values`.
fn restore(values: &mut [f64], stage: MeanDiagnosticStage, mean: f64) -> Result<(), Error> {
    if stage != MeanDiagnosticStage::Differenced {
        for value in values {
            *value = checked(*value + mean, "autocorrelation mean restore")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
