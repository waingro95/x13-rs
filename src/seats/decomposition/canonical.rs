//! DecompSpectrum's canonical components (`spectrum.f`): each component's
//! numerator with its spectral minimum removed, and the harmonics of the
//! filter that extracts the component.

use super::harmonic::multiply_harmonics;
use super::minima::SpectralMinima;

/// One canonical component.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalComponent {
    /// The numerator harmonics after the minimum is removed.
    pub(super) numerator: Vec<f64>,
    /// The filter numerator by lag: the numerator times the denominators of
    /// the other components. Lags past the product stay positive zeros.
    pub(crate) filter_lags: [f64; 32],
}

/// The canonical components with the minima they were shifted by.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalPreparation {
    pub(super) minima: SpectralMinima,
    pub(crate) trend: CanonicalComponent,
    pub(crate) seasonal: CanonicalComponent,
    pub(crate) cycle: Option<CanonicalComponent>,
}

/// `raw - minimum * denominator`, padded with positive zeros to the longer of
/// the two. gfortran fuses each product into its subtraction.
fn shift_numerator(raw: &[f64], denominator: &[f64], minimum: f64) -> Vec<f64> {
    let mut shifted = raw.to_vec();
    shifted.resize(raw.len().max(denominator.len()), 0.0);
    for (index, value) in shifted.iter_mut().enumerate() {
        let coefficient = denominator.get(index).copied().unwrap_or(0.0);
        *value = (-minimum).mul_add(coefficient, *value);
    }
    shifted
}

/// Harmonics as lag coefficients: lag zero is copied and every other
/// amplitude is halved.
fn lag_coefficients(cosine: &[f64]) -> [f64; 32] {
    debug_assert!((1..=32).contains(&cosine.len()));
    let mut lags = [0.0; 32];
    lags[0] = cosine[0];
    for (lag, &amplitude) in lags[1..cosine.len()].iter_mut().zip(&cosine[1..]) {
        *lag = 0.5 * amplitude;
    }
    lags
}

/// DecompSpectrum's canonical components and their filters.
pub(crate) fn prepare_canonical(minima: SpectralMinima) -> CanonicalPreparation {
    let harmonics = &minima.harmonics;
    let trend_numerator = shift_numerator(&minima.trend_numerator, &harmonics.trend, minima.enot);
    // DecompSpectrum multiplies by the cycle even when it is the constant 1.
    let trend_with_cycle = multiply_harmonics(&trend_numerator, &harmonics.cycle);
    let trend_filter = multiply_harmonics(&trend_with_cycle, &harmonics.seasonal);
    let trend = CanonicalComponent {
        numerator: trend_numerator,
        filter_lags: lag_coefficients(&trend_filter),
    };

    let seasonal = if harmonics.seasonal.len() == 1 {
        // DecompSpectrum skips an absent seasonal component and keeps its
        // zero filter.
        CanonicalComponent {
            numerator: vec![0.0],
            filter_lags: [0.0; 32],
        }
    } else {
        let seasonal_numerator = shift_numerator(
            &minima.seasonal_numerator,
            &harmonics.seasonal,
            minima.estar,
        );
        let seasonal_with_cycle = multiply_harmonics(&seasonal_numerator, &harmonics.cycle);
        let seasonal_filter = multiply_harmonics(&seasonal_with_cycle, &harmonics.trend);
        CanonicalComponent {
            numerator: seasonal_numerator,
            filter_lags: lag_coefficients(&seasonal_filter),
        }
    };

    let cycle = minima.cycle.as_ref().map(|cycle| {
        let numerator = shift_numerator(&cycle.numerator, &harmonics.cycle, cycle.minimum);
        let with_seasonal = multiply_harmonics(&numerator, &harmonics.seasonal);
        let filter = multiply_harmonics(&with_seasonal, &harmonics.trend);
        CanonicalComponent {
            numerator,
            filter_lags: lag_coefficients(&filter),
        }
    });
    CanonicalPreparation {
        minima,
        trend,
        seasonal,
        cycle,
    }
}

/// The shifted numerators ShowInvalDecomp writes for a model whose irregular
/// variance is negative. It builds no filters, so they stay zero here;
/// nothing extracts a component from this preparation.
pub(crate) fn prepare_rejected_ar2(minima: SpectralMinima) -> CanonicalPreparation {
    let harmonics = &minima.harmonics;
    let component = |numerator| CanonicalComponent {
        numerator,
        filter_lags: [0.0; 32],
    };
    let trend = component(shift_numerator(
        &minima.trend_numerator,
        &harmonics.trend,
        minima.enot,
    ));
    let seasonal = component(shift_numerator(
        &minima.seasonal_numerator,
        &harmonics.seasonal,
        minima.estar,
    ));
    let cycle = minima.cycle.as_ref().map(|cycle| {
        component(shift_numerator(
            &cycle.numerator,
            &harmonics.cycle,
            cycle.minimum,
        ))
    });
    CanonicalPreparation {
        minima,
        trend,
        seasonal,
        cycle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_fuses_the_product_and_pads_with_positive_zeros() {
        let e = f64::EPSILON;
        let shifted = shift_numerator(&[1.0], &[1.0 - e, -0.0, 2.0], 1.0 + e);
        assert_eq!(shifted[0].to_bits(), (e * e).to_bits());
        assert_eq!((1.0 - (1.0 + e) * (1.0 - e)).to_bits(), 0.0_f64.to_bits());
        assert_eq!(shifted[1].to_bits(), 0.0_f64.to_bits());
        assert_eq!(shifted[2], -2.0 * (1.0 + e));
        assert_eq!(shift_numerator(&[3.0, 4.0], &[1.0], 2.0), [1.0, 4.0]);
    }

    #[test]
    fn lags_copy_lag_zero_and_halve_the_rest() {
        let lags = lag_coefficients(&[3.0, 4.0, -6.0]);
        assert_eq!(lags[..3], [3.0, 2.0, -3.0]);
        assert!(lags[3..].iter().all(|lag| lag.to_bits() == 0));
    }
}
