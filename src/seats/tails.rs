//! FCAST: the forecasts and backcasts that extend the series for ESTBUR.
//!
//! ANALTS runs FCAST twice on the completed pass. The forward run extends the
//! series from the pass's residuals. The backward run extends the reversed
//! series, with residuals from a new CALCFX solve on the reversed differenced
//! series. [`finish_tails_numeric`] is that body for every family: a family
//! supplies its Box-Jenkins MA, its differenced AR polynomial (`bphist`), its
//! mean forcing and its backward CALCFX.
//!
//! The module also holds the differencing ANALTS applies before each pass
//! ([`difference_ordered`]).

use super::failure::{Error, Stage, usable};
use super::mean::RequestedMean;
use super::residual::{MAXIMUM_OBSERVATIONS, ResidualResult};
use crate::Frequency;

/// The tails with what the backward CALCFX produced.
pub(super) struct CoreTails<C = ()> {
    pub(super) tails: SeriesTails,
    pub(super) backward_bj_ma: Vec<f64>,
    /// What the family's backward CALCFX handed back.
    pub(super) backward: C,
}

/// The series extended in both directions, with what FCAST extended it by.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SeriesTails {
    pub(crate) frequency: Frequency,
    pub(crate) observed_len: usize,
    /// FHI, the number of forecasts ESTBUR reads.
    pub(crate) horizon: usize,
    /// The series followed by its forecasts.
    pub(crate) forward_extended: Vec<f64>,
    /// The reversed series followed by its backcasts.
    pub(crate) backward_extended: Vec<f64>,
    /// The mean forcing of the forward and the backward recurrence.
    pub(crate) zaf: f64,
    pub(crate) zab: f64,
    /// The forward Box-Jenkins MA.
    pub(crate) bj_ma: Vec<f64>,
    /// The differenced AR polynomial, in FCAST's recurrence signs.
    pub(crate) bphist: Vec<f64>,
}

/// FCAST's horizons: `fh` forecasts requested, `fhi` computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Horizons {
    pub(crate) fh: usize,
    pub(crate) fhi: usize,
}

/// The horizons of a series of `observed_len` observations, which holds at
/// least two years and two observations and at most SEATS's maximum.
pub(crate) fn resolve_horizons(frequency: Frequency, observed_len: usize) -> Horizons {
    let period = usize::from(frequency.periods_per_year());
    debug_assert!((2 * period + 2..=MAXIMUM_OBSERVATIONS).contains(&observed_len));
    // editor.f requests 3*s; ANALTS CheckLen then clamps at n-2 on >= n-1.
    let mut fh = 3 * period;
    if fh >= observed_len - 1 {
        fh = observed_len - 2;
    }
    // ANALTS FCAST extension floor.
    let fhi = fh.max(2 * period + 2);
    Horizons { fh, fhi }
}

/// `value` when it is finite, else [`Unusable`](super::Failure::Unusable) at
/// [`Stage::Tails`].
fn finite(value: f64, quantity: &'static str, index: Option<usize>) -> Result<f64, Error> {
    usable(quantity, index, value).map_err(|failure| failure.at(Stage::Tails))
}

/// The differences a family takes: the seasonal one or not, and up to two
/// ordinary ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DifferenceOrder {
    /// No differencing.
    Stationary,
    /// The seasonal difference alone.
    SeasonalOnly,
    /// One ordinary and the seasonal difference.
    First,
    /// Two ordinary and the seasonal difference.
    Second,
    /// One ordinary difference.
    NonseasonalFirst,
    /// Two ordinary differences.
    NonseasonalSecond,
}

impl DifferenceOrder {
    pub(super) fn ordinary(self) -> usize {
        match self {
            Self::Stationary | Self::SeasonalOnly => 0,
            Self::First | Self::NonseasonalFirst => 1,
            Self::Second | Self::NonseasonalSecond => 2,
        }
    }
    /// `(-1)^(D+BD)`, the sign ANALTS gives the reversed differenced series.
    pub(super) fn backward_sign(self) -> f64 {
        match self {
            Self::Stationary | Self::First | Self::NonseasonalSecond => 1.0,
            Self::SeasonalOnly | Self::Second | Self::NonseasonalFirst => -1.0,
        }
    }
    pub(super) fn seasonal(self) -> bool {
        matches!(self, Self::SeasonalOnly | Self::First | Self::Second)
    }
}

/// Differences `observed`, which is longer than the differences reach, as
/// ANALTS does before each pass.
pub(super) fn difference_ordered(
    observed: &[f64],
    period: usize,
    order: DifferenceOrder,
) -> Result<Vec<f64>, Error> {
    // ANALTS: observations are copied to WD, then differenced in-place by
    // seasonal lag before separate ordinary differences.
    let seasonal_lag = if order.seasonal() { period } else { 0 };
    debug_assert!(observed.len() > seasonal_lag + order.ordinary());
    let mut wd = observed.to_vec();
    let mut nw = wd.len() - seasonal_lag;
    if order.seasonal() {
        for j in 0..nw {
            wd[j] = finite(
                wd[j + period] - wd[j],
                "seasonally differenced series",
                Some(j),
            )?;
        }
    }
    for _ in 0..order.ordinary() {
        nw -= 1;
        for j in 0..nw {
            wd[j] = finite(wd[j + 1] - wd[j], "regularly differenced series", Some(j))?;
        }
    }
    wd.truncate(nw);
    Ok(wd)
}

/// The AR polynomial `ar` times the differences of `order`, without its
/// constant and negated: the coefficients FCAST's recurrence applies to the
/// latest values.
pub(super) fn bphist_ordered(period: usize, ar: &[f64], order: DifferenceOrder) -> Vec<f64> {
    // FCAST multiplies the ordinary factors in first and the seasonal factor
    // second, each in place from the top coefficient down.
    let mut b = vec![0.0; period + ar.len() + order.ordinary()];
    b[..ar.len()].copy_from_slice(ar);
    let mut len = ar.len();
    for _ in 0..order.ordinary() {
        for k in (1..=len).rev() {
            b[k] -= b[k - 1];
        }
        len += 1;
    }
    if order.seasonal() {
        for k in (period..len + period).rev() {
            b[k] -= b[k - period];
        }
        len += period;
    }
    b[1..len].iter().map(|value| -value).collect()
}

/// FCAST's recurrence: appends `horizon` forecasts to `extended`.
fn forecast(
    mut extended: Vec<f64>,
    residuals: &[f64],
    ma: &[f64],
    bp: &[f64],
    horizon: usize,
    za: f64,
    direction: &'static str,
) -> Result<Vec<f64>, Error> {
    let mut a = Vec::with_capacity(residuals.len() + horizon);
    a.extend_from_slice(residuals);
    extended.reserve(horizon);
    for step in 1..=horizon {
        let mut sz = za;
        for (&coefficient, &residual) in ma.iter().zip(a[a.len() - ma.len()..].iter().rev()) {
            sz = (-coefficient).mul_add(residual, sz);
        }
        let recent = &extended[extended.len() - bp.len()..];
        for (&coefficient, &value) in bp.iter().zip(recent.iter().rev()) {
            sz = coefficient.mul_add(value, sz);
        }
        a.push(0.0); // FCAST writes +0 to the new residual position.
        extended.push(finite(sz, direction, Some(step))?);
    }
    Ok(extended)
}

/// What the completed pass hands the forward FCAST.
pub(super) struct ForwardTailState {
    /// WM, the mean of the differenced series.
    pub(super) wm: f64,
    /// Whether the pass removed that mean.
    pub(super) requested_mean: RequestedMean,
    /// The residuals after ANALTS's residual-mean correction.
    pub(super) residuals: Vec<f64>,
}

/// The airline's BJ-sign MA: `θ` at lag 1, `Θ` at lag s and `-θΘ` at
/// lag s+1.
pub(super) fn airline_bj_ma(period: usize, ma: [f64; 2]) -> Vec<f64> {
    let mut bj_ma = vec![0.0; period + 1];
    bj_ma[0] = ma[0];
    bj_ma[period - 1] = ma[1];
    bj_ma[period] = -ma[0] * ma[1];
    bj_ma
}

/// A single MA coefficient at `lag`.
pub(super) fn single_lag_bj_ma(lag: usize, coefficient: f64) -> Vec<f64> {
    let mut bj_ma = vec![0.0; lag];
    bj_ma[lag - 1] = coefficient;
    bj_ma
}

/// AR1's MA: the seasonal MA where the model has one, else none.
pub(super) fn ar1_bj_ma(model: &super::ar1_seasonal_ma::model::ResolvedAr1SeasonalMa) -> Vec<f64> {
    if model.has_seasonal() {
        let period = usize::from(model.frequency().periods_per_year());
        single_lag_bj_ma(period, model.parameters()[1])
    } else {
        Vec::new()
    }
}

/// The shared FCAST body for the MA-only families (airline, single MA and
/// second difference), whose backward CALCFX reuses the forward BJ MA. A
/// failed backward CALCFX is reported at the tails stage.
pub(super) fn finish_ma_only_tails(
    frequency: Frequency,
    observed: &[f64],
    wd: &[f64],
    state: ForwardTailState,
    inputs: TailInputs,
    backward: impl FnOnce(&[f64]) -> Result<ResidualResult, Error>,
) -> Result<CoreTails, Error> {
    let backward_ma = inputs.bj_ma.clone();
    finish_tails_numeric(frequency, observed, wd, state, inputs, |back| {
        backward(back)
            .map(|residuals| (residuals, backward_ma, ()))
            .map_err(|error| error.restage(Stage::Tails))
    })
}

/// One family's FCAST inputs: its BJ MA and AR history, the mean forcing,
/// the differencing and the horizons.
pub(super) struct TailInputs {
    pub(super) bj_ma: Vec<f64>,
    pub(super) horizons: Horizons,
    pub(super) bphist: Vec<f64>,
    pub(super) forcing_ar: MeanForcing,
    pub(super) order: DifferenceOrder,
    pub(super) working_horizon: usize,
}

impl TailInputs {
    /// An MA-only family: unit AR, no mean forcing, FHI working values.
    pub(super) fn ma_only(
        period: usize,
        bj_ma: Vec<f64>,
        horizons: Horizons,
        order: DifferenceOrder,
    ) -> Self {
        Self {
            bj_ma,
            horizons,
            bphist: bphist_ordered(period, &[1.0], order),
            forcing_ar: MeanForcing::None,
            order,
            working_horizon: horizons.fhi,
        }
    }
}

/// The AR polynomial that scales WM into FCAST's constant.
pub(super) enum MeanForcing {
    None,
    One(f64),
    Two([f64; 2]),
    Regular(super::decomposition::root_allocation::RegularArCoefficients),
    Expanded(Vec<f64>),
}

/// The shared FCAST body. Each family's error `E` carries both FCAST's own
/// [`Error`] and whatever its backward CALCFX callback refuses with.
pub(super) fn finish_tails_numeric<E: From<Error>, C>(
    frequency: Frequency,
    observed: &[f64],
    wd: &[f64],
    state: ForwardTailState,
    inputs: TailInputs,
    backward: impl FnOnce(&[f64]) -> Result<(ResidualResult, Vec<f64>, C), E>,
) -> Result<CoreTails<C>, E> {
    let TailInputs {
        bj_ma,
        horizons,
        bphist,
        forcing_ar,
        order,
        working_horizon,
    } = inputs;
    let ForwardTailState {
        wm,
        requested_mean,
        residuals,
    } = state;

    let zaf = if requested_mean == RequestedMean::Yes {
        match forcing_ar {
            MeanForcing::One(phi) => (1.0 - phi) * wm,
            MeanForcing::None => wm,
            MeanForcing::Two(ar) => ((1.0 - ar[0]) - ar[1]) * wm,
            MeanForcing::Regular(ar) => {
                ar.as_slice().iter().fold(1.0, |sum, value| sum - value) * wm
            }
            MeanForcing::Expanded(ar) => ar.iter().fold(1.0, |sum, value| sum - value) * wm,
        }
    } else {
        0.0
    };
    let forward_extended = forecast(
        observed.to_vec(),
        &residuals,
        &bj_ma,
        &bphist,
        working_horizon,
        zaf,
        "forward tail",
    )?;

    // analts.f:2792–2805 reverses WD with the sign kd=(-1)^(D+BD) and Z
    // without it. The reversed WD keeps the forward mean removal; its
    // residuals come from a new CALCFX solve.
    let backward_wd: Vec<_> = wd
        .iter()
        .rev()
        .map(|value| value * order.backward_sign())
        .collect();
    let (backward_residuals, backward_bj_ma, backward) = backward(&backward_wd)?;
    let zab = zaf * order.backward_sign();
    let backward_observed: Vec<_> = observed.iter().rev().copied().collect();
    let backward_extended = forecast(
        backward_observed,
        &backward_residuals.normalized_residuals,
        &backward_bj_ma,
        &bphist,
        working_horizon,
        zab,
        "backward tail",
    )?;

    Ok(CoreTails {
        backward,
        backward_bj_ma,
        tails: SeriesTails {
            frequency,
            observed_len: observed.len(),
            horizon: horizons.fhi,
            forward_extended,
            backward_extended,
            zaf,
            zab,
            bj_ma,
            bphist,
        },
    })
}

#[cfg(test)]
mod tests;
