//! SIGEX's polynomials (`sigex.f`): the aggregate moving average (Thstr0)
//! and the trend, seasonal and cycle autoregressive factors (Chins, Psins)
//! with the products the spectral decomposition divides by, and CONV
//! (`ansub2.f`), the polynomial product that builds them.

use crate::seats::airline::model::ResolvedAirline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaFamily {
    Airline,
    SeasonalMa,
}

impl MaFamily {
    pub(crate) fn ma_coefficients(self, period: usize) -> usize {
        match self {
            Self::Airline => period + 2,
            Self::SeasonalMa => period + 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenominatorFactors {
    pub(crate) stationary: Vec<f64>,
    pub(crate) nonstationary: Vec<f64>,
    pub(crate) combined: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpectralInputs {
    pub(crate) theta: Vec<f64>,
    pub(crate) seasonal_theta: Vec<f64>,
    pub(crate) aggregate_ma: Vec<f64>,
    pub(crate) trend: DenominatorFactors,
    pub(crate) seasonal: DenominatorFactors,
    pub(crate) cycle: DenominatorFactors,
    pub(crate) trend_cycle: Vec<f64>,
    pub(crate) seasonal_cycle: Vec<f64>,
    pub(crate) trend_seasonal: Vec<f64>,
    pub(crate) adjustment_stationary: Vec<f64>,
    pub(crate) adjustment_nonstationary: Vec<f64>,
    pub(crate) aggregate_denominator: Vec<f64>,
}

/// SIGEX's polynomials for the airline model.
pub(crate) fn spectral_inputs(model: &ResolvedAirline) -> SpectralInputs {
    let period = usize::from(model.frequency().periods_per_year());
    let [ma, seasonal_ma] = model.ma_coefficients();
    // analts.f:2854–2879 expands the imported SEATS coefficients at true lags.
    let theta = vec![1.0, -ma];
    spectral_inputs_parts(period, theta, seasonal_ma)
}

pub(crate) fn seasonal_ma_spectral_inputs(
    model: &crate::seats::seasonal_ma::ResolvedSeasonalMa,
) -> SpectralInputs {
    if model.shape == crate::seats::seasonal_ma::SingleMaShape::Nonseasonal {
        return spectral_inputs_ordered(
            usize::from(model.frequency.periods_per_year()),
            vec![1.0, -model.seasonal_ma],
            0.0,
            model.shape.differences(),
        );
    }
    spectral_inputs_parts(
        usize::from(model.frequency.periods_per_year()),
        vec![1.0],
        model.seasonal_ma,
    )
}

fn spectral_inputs_parts(period: usize, theta: Vec<f64>, seasonal_ma: f64) -> SpectralInputs {
    spectral_inputs_ordered(
        period,
        theta,
        seasonal_ma,
        crate::seats::tails::DifferenceOrder::First,
    )
}

pub(crate) fn spectral_inputs_ordered(
    period: usize,
    theta: Vec<f64>,
    seasonal_ma: f64,
    order: crate::seats::tails::DifferenceOrder,
) -> SpectralInputs {
    let seasonal_lag = if order.seasonal() { period } else { 0 };
    let mut seasonal_theta = vec![0.0; seasonal_lag + 1];
    if order.seasonal() {
        seasonal_theta[period] = -seasonal_ma;
    }
    seasonal_theta[0] = 1.0;
    // sigex.f:447–454, Thstr0: nonseasonal first, seasonal second.
    let aggregate_ma = convolve(&theta, &seasonal_theta);

    // sigex.f:477–492, Chins: explicit ordinary order plus BD=1.
    // Descending updates preserve the previous order's lower coefficients.
    let dplusd = order.ordinary() + usize::from(order.seasonal());
    let mut trend_nonstationary = vec![1.0];
    for i in 1..=dplusd {
        trend_nonstationary.push(0.0);
        for k in (1..=i).rev() {
            trend_nonstationary[k] -= trend_nonstationary[k - 1];
        }
    }
    let trend_stationary = vec![1.0];

    // sigex.f:505–523, Psins: bd = 1 convolves the seasonal sum into identity.
    let mut seasonal_nonstationary = vec![1.0];
    let seasonal_stationary = vec![1.0];
    let seasonal_sum = vec![1.0; period];
    if order.seasonal() {
        seasonal_nonstationary = convolve(&seasonal_sum, &seasonal_nonstationary);
    }

    // sigex.f:539–555. No seasonal AR group; F1RST (sigsub.f:60) returns
    // immediately for p = 0, retaining these factors without any root work.
    let cycle_stationary = vec![1.0];
    let cycle_nonstationary = vec![1.0];

    // sigex.f:582–622: retain each operand order in the non-TD branch.
    let trend_combined = convolve(&trend_stationary, &trend_nonstationary);
    let seasonal_combined = convolve(&seasonal_stationary, &seasonal_nonstationary);
    let cycle_combined = convolve(&cycle_stationary, &cycle_nonstationary);
    let trend_cycle = convolve(&trend_combined, &cycle_combined);
    let seasonal_cycle = convolve(&seasonal_combined, &cycle_combined);
    let trend_seasonal = convolve(&trend_combined, &seasonal_combined);
    let aggregate_denominator = convolve(&seasonal_combined, &trend_cycle);
    let adjustment_stationary = convolve(&cycle_stationary, &trend_stationary);
    let adjustment_nonstationary = convolve(&cycle_nonstationary, &trend_nonstationary);

    SpectralInputs {
        theta,
        seasonal_theta,
        aggregate_ma,
        trend: DenominatorFactors {
            stationary: trend_stationary,
            nonstationary: trend_nonstationary,
            combined: trend_combined,
        },
        seasonal: DenominatorFactors {
            stationary: seasonal_stationary,
            nonstationary: seasonal_nonstationary,
            combined: seasonal_combined,
        },
        cycle: DenominatorFactors {
            stationary: cycle_stationary,
            nonstationary: cycle_nonstationary,
            combined: cycle_combined,
        },
        trend_cycle,
        seasonal_cycle,
        trend_seasonal,
        adjustment_stationary,
        adjustment_nonstationary,
        aggregate_denominator,
    }
}

/// CONV (`ansub2.f`): the product of two polynomials, each term fused into
/// its coefficient with the left operand in the outer loop.
pub(crate) fn convolve(left: &[f64], right: &[f64]) -> Vec<f64> {
    debug_assert!(!left.is_empty() && !right.is_empty());
    let mut output = vec![0.0; left.len() + right.len() - 1];
    for (i, &left_value) in left.iter().enumerate() {
        for (j, &right_value) in right.iter().enumerate() {
            output[i + j] = left_value.mul_add(right_value, output[i + j]);
        }
    }
    output
}

#[cfg(test)]
mod tests;
