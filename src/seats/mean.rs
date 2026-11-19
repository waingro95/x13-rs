//! ANALTS's mean steps: the mean it removes from the differenced series and
//! the residual mean test with its correction.
//!
//! A pass that estimates a mean subtracts WM, the mean of the differenced
//! series, before AUTO and CALCFX. After CALCFX, ANALTS computes the mean and
//! variance of the residuals (DMEAN, DVAR) and the mean's t-value. When the
//! pass estimates a mean and the t-value is above TA=100, it subtracts the
//! residual mean from the residuals and adds it to WM, scaled by the ratio of
//! the MA and AR polynomials at one, which CONV builds.
//!
//! [`mean_correction`] is that step for the models without a regular MA. With
//! a regular MA, Build 62's CONV call reads a coefficient ANALTS never set, so
//! [`residual_mean_state`] reports the entered correction as an error.

use super::failure::{Error, Failure, Stage, usable};

/// ANALTS's default TA: the residual mean is corrected when its t-value is
/// above this.
const RESIDUAL_MEAN_TA: f64 = 100.0;

/// Whether a pass estimates the mean of the differenced series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestedMean {
    No,
    Yes,
}

/// The residual mean and its t-value (ANALTS's `rmean` and `rtval`), with
/// the mean request of the pass they belong to.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResidualMean {
    pub(crate) requested: RequestedMean,
    pub(crate) mean: f64,
    pub(crate) t: f64,
}

/// Which of ANALTS's three AUTO calls runs: on the differenced series, the
/// residuals or their squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeanDiagnosticStage {
    Differenced,
    ForwardResidual,
    SquaredResidual,
}

impl MeanDiagnosticStage {
    /// The sum of squares AUTO computes in this call.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Differenced => "energy of the differenced series",
            Self::ForwardResidual => "energy of the residuals",
            Self::SquaredResidual => "energy of the squared residuals",
        }
    }
}

/// `value` when it is finite, else [`Unusable`](Failure::Unusable) at
/// [`Stage::Mean`].
pub(super) fn checked(value: f64, quantity: &'static str) -> Result<f64, Error> {
    usable(quantity, None, value).map_err(|failure| failure.at(Stage::Mean))
}

/// DMEAN: the mean of `values`, summed in order.
pub(super) fn ordered_mean(values: &[f64], quantity: &'static str) -> Result<f64, Error> {
    let count = values.len();
    if count == 0 {
        return Err(Failure::Unusable {
            quantity,
            index: None,
            value: f64::NAN,
        }
        .at(Stage::Mean));
    }
    let mut sum = 0.0;
    for &value in values {
        sum = checked(sum + value, quantity)?;
    }
    checked(sum / count as f64, quantity)
}

/// Removes the mean of the differenced series when the pass estimates one,
/// and returns WM: that mean, or zero when it stays in.
pub(crate) fn prepare_mean_wd(wd: &mut [f64], requested: RequestedMean) -> Result<f64, Error> {
    // ANALTS 1956–1975 sums the full sample under either request.
    let wm = ordered_mean(wd, "differenced mean")?;
    if requested == RequestedMean::No {
        return Ok(0.0);
    }
    for value in wd {
        *value = checked(*value - wm, "centered differenced series")?;
    }
    Ok(wm)
}

/// Refuses the correction ANALTS enters above TA for a model with a regular
/// MA: at ANALTS 2417 its CONV call reads `bth1(Mq+1)`, which was never set.
/// The test is signed, so a large negative t-value passes.
fn guard_correction(count: usize, t: f64) -> Result<(), Error> {
    if t > RESIDUAL_MEAN_TA {
        return Err(Failure::UndefinedResidualMeanCorrection {
            residual_count: count,
            t,
        }
        .at(Stage::Mean));
    }
    Ok(())
}

/// The standard error of the residual mean. ANALTS raises to the power 0.5,
/// which does not always round as `sqrt` does.
fn residual_rstd(variance: f64, count: usize) -> Result<f64, Error> {
    checked(
        (variance / count as f64).powf(0.5),
        "residual standard deviation",
    )
}

/// What ANALTS's residual mean step leaves a pass: the statistics it tested
/// and the WM FCAST uses, which includes the correction when one was made.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeanCorrection {
    pub(crate) statistics: ResidualMean,
    pub(crate) wm: f64,
}

/// The models without a regular MA, whose residual mean ANALTS corrects
/// through CONV.
#[derive(Debug, Clone, Copy)]
pub(super) enum MeanFamily<'a> {
    /// P=Bp=0 with one seasonal MA: denominator one.
    SeasonalMa,
    /// AR1 with a seasonal MA.
    Ar1(f64),
    /// AR1 with no MA at all, so Q=Bq=0.
    NonseasonalAr1(f64),
    Ar2(super::decomposition::root_allocation::RegularArCoefficients),
    /// The generic ARMA family with no regular MA.
    Regular {
        ar: &'a [f64],
        seasonal_ar: Option<(usize, f64)>,
        seasonal_ma: bool,
    },
}

impl MeanFamily<'_> {
    /// The AR polynomial at one: CONV's sum over the AR factors.
    fn denominator(&self) -> f64 {
        match *self {
            Self::SeasonalMa => 1.0,
            Self::Ar1(phi) | Self::NonseasonalAr1(phi) => 1.0 - phi,
            Self::Ar2(ar) => ar.as_slice().iter().fold(1.0, |sum, value| sum - value),
            Self::Regular {
                ar,
                seasonal_ar: None,
                ..
            } => ar.iter().fold(1.0, |sum, value| sum - value),
            Self::Regular {
                ar,
                seasonal_ar: Some((s, bphi)),
                ..
            } => {
                // ANALTS 2434 writes BPHI1(i*Mq), not BPHI1(i*Mq+1). The
                // sum takes CONV's product in its order.
                let phi = std::iter::once(1.0)
                    .chain(ar.iter().map(|v| -v))
                    .collect::<Vec<_>>();
                let mut bphi1 = vec![0.0; s + 1];
                bphi1[0] = 1.0;
                bphi1[s - 1] = -bphi;
                super::decomposition::polynomial::convolve(&phi, &bphi1)
                    .iter()
                    .fold(0.0, |sum, v| sum + v)
            }
        }
    }

    /// The MA polynomial at one. With Q=0 and Bq=1, CONV's second input has
    /// length one and its first coefficient is zero. With Q=Bq=0, ANALTS
    /// keeps bth1(1)=1, so CONV's sum is one.
    fn numerator(&self) -> f64 {
        match self {
            Self::NonseasonalAr1(_)
            | Self::Regular {
                seasonal_ma: false, ..
            } => 1.0,
            Self::SeasonalMa | Self::Ar1(_) | Self::Ar2(_) | Self::Regular { .. } => 0.0,
        }
    }
}

/// ANALTS's residual mean statistics of `a` and, above TA, the correction of
/// `a` and WM for a model without a regular MA.
pub(super) fn mean_correction(
    a: &mut [f64],
    requested: RequestedMean,
    wm: f64,
    family: MeanFamily<'_>,
) -> Result<MeanCorrection, Error> {
    let statistics = residual_statistics(a, requested)?;
    correct_mean(a, statistics, wm, family)
}

fn correct_mean(
    a: &mut [f64],
    statistics: ResidualMean,
    wm: f64,
    family: MeanFamily<'_>,
) -> Result<MeanCorrection, Error> {
    let requested = statistics.requested;
    if !(requested == RequestedMean::Yes && statistics.t > RESIDUAL_MEAN_TA) {
        return Ok(MeanCorrection { statistics, wm });
    }
    for value in a.iter_mut() {
        *value = checked(*value - statistics.mean, "corrected residual")?;
    }
    let denominator = checked(family.denominator(), "mean denominator")?;
    let ratio = checked(family.numerator() / denominator, "mean correction ratio")?;
    let wm = checked(ratio.mul_add(statistics.mean, wm), "corrected mean")?;
    // ANALTS recomputes the statistics of the corrected residuals. Nothing
    // reads them, but a run fails where they do.
    residual_statistics(a, requested)?;
    Ok(MeanCorrection { statistics, wm })
}

/// ANALTS's residual mean statistics of `a` for a model with a regular MA,
/// which has no defined correction.
pub(crate) fn residual_mean_state(
    a: &[f64],
    requested: RequestedMean,
) -> Result<ResidualMean, Error> {
    let statistics = residual_statistics(a, requested)?;
    if requested == RequestedMean::Yes {
        guard_correction(a.len(), statistics.t)?;
    }
    Ok(statistics)
}

/// DVAR: the variance about the mean, clamped at zero.
fn dvar(values: &[f64]) -> Result<f64, Error> {
    let count = values.len();
    // DVAR calls DMEAN itself and squares the result, then sums the squares
    // in order. gfortran fuses each square and add of that sum.
    let dmean = ordered_mean(values, "variance mean")?;
    let dmean2 = checked(dmean * dmean, "variance mean square")?;
    let mut ym2 = 0.0;
    for &value in values {
        ym2 = checked(value.mul_add(value, ym2), "variance square sum")?;
    }
    ym2 = checked(ym2 / count as f64, "variance second moment")?;
    let variance = checked(ym2 - dmean2, "variance")?;
    Ok(if variance < 0.0 { 0.0 } else { variance })
}

fn residual_statistics(a: &[f64], requested: RequestedMean) -> Result<ResidualMean, Error> {
    let count = a.len();
    let mean = ordered_mean(a, "residual mean")?;
    let rstd = residual_rstd(dvar(a)?, count)?;
    if rstd == 0.0 {
        return Err(Failure::Unusable {
            quantity: "residual standard deviation",
            index: None,
            value: rstd,
        }
        .at(Stage::Mean));
    }
    let t = checked(mean / rstd, "residual t")?;
    Ok(ResidualMean { requested, mean, t })
}

#[cfg(test)]
mod tests;
