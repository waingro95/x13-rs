//! Build 62's autoregressive spectrum estimate and the seasonal and
//! trading-day peak counts read from it (`spgrh`, `sicp2`, `crosco`,
//! `ispeak`).

use super::BUILD_62_PI;

/// Counts the seasonal and trading-day peaks of a monthly spectrum
/// (`ispeak`), from the regular grid's decibels and the expanded grid's
/// decibels and frequencies.
///
/// Returns `(seasonal, trading_day)`; the trading-day count is zero unless
/// `td` is set.
pub(crate) fn peak_counts(
    regular: &[f64],
    expanded: &[f64],
    frequencies: &[f64],
    td: bool,
) -> (usize, usize) {
    let mut sorted = regular.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[30]; // mkmdsx: the middle of 61 sorted values
    let limit = (sorted[60] - sorted[0]) * (6.0 / 52.0);
    let count = |seasonal: bool, candidates: &[(usize, usize, usize)]| {
        candidates
            .iter()
            .filter(|&&(peak, lower, upper)| {
                if expanded[peak] <= median {
                    return false;
                }
                let start = lower + 1;
                let end = if seasonal { peak - 1 } else { upper - 1 };
                if end > start {
                    let left = frequencies[peak] - 0.002;
                    let right = frequencies[peak] + 0.002;
                    for k in start..=end {
                        if k != peak
                            && (frequencies[k] < left || frequencies[k] > right)
                            && expanded[k] > expanded[peak]
                        {
                            return false;
                        }
                    }
                }
                let threshold = expanded[peak] - limit;
                expanded[lower] < threshold && expanded[upper] < threshold
            })
            .count()
    };
    // ispeak excludes the fifth monthly candidate (53,52,55).
    (
        count(
            true,
            &[(10, 9, 11), (20, 19, 21), (30, 29, 31), (40, 39, 42)],
        ),
        if td {
            count(false, &[(43, 41, 45), (56, 54, 58)])
        } else {
            0
        },
    )
}

/// The step of the spectrum calculation that produced a nonfinite value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpectralArithmeticStage {
    MeanSum,
    Mean,
    Demeaning,
    CovarianceSum,
    Covariance,
    InitialVariance,
    InitialAic,
    InitialLagCovariance,
    VarianceRatio,
    Reflection,
    ReflectionSquare,
    InnovationVariance,
    Aic,
    ArCoefficient,
    NextLagCovariance,
    CovarianceRecurrence,
    CosineSum,
    SineSum,
    PowerDenominator,
    SpectralPower,
    Decibels,
}

/// The frequency grid of a spectrum. `Regular` has 61 points, six of them
/// overwritten by the two trading-day frequencies and their neighbours;
/// `Expanded` inserts those six instead, for 67 points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpectralGrid {
    Regular,
    Expanded,
}

/// An autoregressive spectrum in decibels on a frequency grid, with the
/// order of the autoregression behind it.
#[derive(Debug, PartialEq)]
pub(crate) struct ArSpectrum {
    pub(crate) order: usize,
    pub(crate) frequencies: Vec<f64>,
    pub(crate) decibels: Vec<f64>,
}

/// Why no spectrum could be estimated.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum SpectralUnavailable {
    #[error("spectral input has {length} values; at least {required} are required")]
    InsufficientInput { length: usize, required: usize },
    #[error("nonfinite spectral input at index {index}")]
    NonfiniteInput { index: usize },
    #[error("spectral covariance is zero under Build 62 dpeq")]
    ZeroCovariance,
    #[error(
        "spectral variance collapsed after order {completed_order}, before required order {required_order}"
    )]
    EarlyVarianceCollapse {
        completed_order: usize,
        required_order: usize,
    },
    #[error("nonpositive spectral innovation variance at order {order}")]
    NonpositiveInnovationVariance { order: usize },
    #[error("nonfinite spectral arithmetic at {stage:?}")]
    NonfiniteArithmetic { stage: SpectralArithmeticStage },
}

/// A fitted autoregression: its coefficients and innovation variance.
#[derive(Debug)]
struct ArFit {
    coefficients: Vec<f64>,
    variance: f64,
}

/// Estimates the spectrum of `values` on `grid` from an autoregression of
/// order 30, or of `values.len() - 1` for a shorter series (`spgrh`).
pub(crate) fn estimate(
    values: &[f64],
    grid: SpectralGrid,
) -> Result<ArSpectrum, SpectralUnavailable> {
    // Finiteness first: a lone nonfinite value is reported as such rather
    // than as too short.
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SpectralUnavailable::NonfiniteInput { index });
        }
    }
    if values.len() < 2 {
        return Err(SpectralUnavailable::InsufficientInput {
            length: values.len(),
            required: 2,
        });
    }
    let frequencies = frequencies(grid);
    let covariances = autocovariances(values, frequencies.len())?;
    if super::build_62_equal(covariances[0], 0.0) {
        return Err(SpectralUnavailable::ZeroCovariance);
    }
    let order = 30.min(values.len() - 1);
    let fitted = fit_autocovariances(&covariances[..=order], values.len())?;
    let mut powers = Vec::with_capacity(frequencies.len());
    // `spgrh` sums the cosines and the sines in separate loops and
    // multiplies each angle out left to right.
    for &frequency in &frequencies {
        let mut cosine = 1.0;
        for (index, &coefficient) in fitted.coefficients.iter().enumerate() {
            let angle = (2 * (index + 1)) as f64 * BUILD_62_PI * frequency;
            cosine = checked(
                cosine + coefficient * angle.cos(),
                SpectralArithmeticStage::CosineSum,
            )?;
        }
        let mut sine = 0.0;
        for (index, &coefficient) in fitted.coefficients.iter().enumerate() {
            let angle = (2 * (index + 1)) as f64 * BUILD_62_PI * frequency;
            sine = checked(
                sine + coefficient * angle.sin(),
                SpectralArithmeticStage::SineSum,
            )?;
        }
        // gfortran fuses only the cosine square in `spgrh`.
        let denominator = checked(
            cosine.mul_add(cosine, sine * sine),
            SpectralArithmeticStage::PowerDenominator,
        )?;
        powers.push(checked(
            fitted.variance / denominator,
            SpectralArithmeticStage::SpectralPower,
        )?);
    }
    let mut decibels = Vec::with_capacity(powers.len());
    for power in powers {
        decibels.push(checked(
            10.0 * power.abs().log10(),
            SpectralArithmeticStage::Decibels,
        )?);
    }
    Ok(ArSpectrum {
        order,
        frequencies,
        decibels,
    })
}

fn checked(value: f64, stage: SpectralArithmeticStage) -> Result<f64, SpectralUnavailable> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SpectralUnavailable::NonfiniteArithmetic { stage })
    }
}

/// The grid's frequencies, in cycles per period.
fn frequencies(grid: SpectralGrid) -> Vec<f64> {
    // mkpeak/mkfreq: default monthly peakwidth=1, altfreq=no, prsfreq=no.
    let mut regular: Vec<_> = (0..61).map(|i| i as f64 / 120.0).collect();
    let step = regular[1];
    if grid == SpectralGrid::Regular {
        for (center, frequency) in [(42, 0.3482), (52, 0.432)] {
            regular[center - 1] = frequency - step;
            regular[center] = frequency;
            regular[center + 1] = frequency + step;
        }
        regular
    } else {
        // Ascending inserts land at their final slots of the 67-point grid.
        for ((lower, center, upper), frequency) in [((41, 43, 45), 0.3482), ((54, 56, 58), 0.432)] {
            regular.insert(lower, frequency - step);
            regular.insert(center, frequency);
            regular.insert(upper, frequency + step);
        }
        regular
    }
}

/// Autocovariances of the demeaned series at lags zero onward (`smeadl`,
/// `crosco`).
fn autocovariances(
    values: &[f64],
    frequency_count: usize,
) -> Result<Vec<f64>, SpectralUnavailable> {
    // smeadl -> sumf: forward sum, then division, then forward subtraction.
    let mut sum = 0.0;
    for &value in values {
        sum = checked(sum + value, SpectralArithmeticStage::MeanSum)?;
    }
    let mean = checked(sum / values.len() as f64, SpectralArithmeticStage::Mean)?;
    let mut centered = Vec::with_capacity(values.len());
    for &value in values {
        centered.push(checked(value - mean, SpectralArithmeticStage::Demeaning)?);
    }
    let inverse_length = 1.0 / values.len() as f64;
    let mut covariances = Vec::with_capacity(frequency_count.min(values.len()));
    // crosco: each covariance has denominator N, not N-lag.
    for lag in 0..frequency_count.min(values.len()) {
        let mut sum = 0.0;
        for (&lagged, &value) in centered[lag..].iter().zip(&centered) {
            // gfortran fuses each term of this sum in `crosco`.
            sum = checked(
                lagged.mul_add(value, sum),
                SpectralArithmeticStage::CovarianceSum,
            )?;
        }
        covariances.push(checked(
            sum * inverse_length,
            SpectralArithmeticStage::Covariance,
        )?);
    }
    Ok(covariances)
}

/// The Levinson-Durbin recursion up to the highest order the covariances
/// allow (`sicp2`).
fn fit_autocovariances(covariances: &[f64], length: usize) -> Result<ArFit, SpectralUnavailable> {
    let order = covariances.len() - 1;
    let mut variance = checked(covariances[0], SpectralArithmeticStage::InitialVariance)?;
    if variance <= 0.0 {
        return Err(SpectralUnavailable::NonpositiveInnovationVariance { order: 0 });
    }
    let n = length as f64;
    // sicp2 tracks the AIC optimum, but label10 then overwrites it with the
    // maximum order, so only the AIC's nonfinite checks still matter.
    checked(n * variance.ln(), SpectralArithmeticStage::InitialAic)?;
    let mut a = vec![0.0_f64; order];
    let mut b = vec![0.0_f64; order];
    let mut next_covariance = checked(
        covariances[1],
        SpectralArithmeticStage::InitialLagCovariance,
    )?;
    for m in 1..=order {
        let ratio = checked(
            variance / covariances[0],
            SpectralArithmeticStage::VarianceRatio,
        )?;
        if ratio < 0.00001 {
            return Err(SpectralUnavailable::EarlyVarianceCollapse {
                completed_order: m - 1,
                required_order: order,
            });
        }
        let reflection = checked(
            next_covariance / variance,
            SpectralArithmeticStage::Reflection,
        )?;
        a[m - 1] = reflection;
        checked(
            reflection * reflection,
            SpectralArithmeticStage::ReflectionSquare,
        )?;
        // gfortran fuses `1 - d*d` in `sicp2`, then multiplies by the variance.
        variance = checked(
            (-reflection).mul_add(reflection, 1.0) * variance,
            SpectralArithmeticStage::InnovationVariance,
        )?;
        if variance <= 0.0 {
            return Err(SpectralUnavailable::NonpositiveInnovationVariance { order: m });
        }
        // gfortran fuses the AIC's multiply-add in `sicp2`.
        checked(
            variance.ln().mul_add(n, 2.0 * m as f64),
            SpectralArithmeticStage::Aic,
        )?;
        // gfortran fuses `a(i) - b(i)*d` in `sicp2`.
        for (coefficient, &reversed) in a[..m - 1].iter_mut().zip(&b) {
            *coefficient = checked(
                (-reversed).mul_add(reflection, *coefficient),
                SpectralArithmeticStage::ArCoefficient,
            )?;
        }
        for (reversed, &coefficient) in b[..m].iter_mut().zip(a[..m].iter().rev()) {
            *reversed = coefficient;
        }
        if m != order {
            next_covariance = checked(
                covariances[m + 1],
                SpectralArithmeticStage::NextLagCovariance,
            )?;
            for (&reversed, &covariance) in b[..m].iter().zip(&covariances[1..]) {
                // gfortran fuses `se - b(i)*Cyy(i+1)` in `sicp2`.
                next_covariance = checked(
                    (-reversed).mul_add(covariance, next_covariance),
                    SpectralArithmeticStage::CovarianceRecurrence,
                )?;
            }
        }
    }
    Ok(ArFit {
        coefficients: a.into_iter().map(|coefficient| -coefficient).collect(),
        variance,
    })
}

#[cfg(test)]
mod tests;
