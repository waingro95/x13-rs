//! The cosine (harmonic) form of SIGEX's polynomials: CONJ and MULTFN
//! (`ansub2.f`), and the numerator and denominator harmonics SPECTRUM
//! (`spectrum.f`) builds from them.

use super::polynomial::{SpectralInputs, convolve};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Harmonics {
    pub(crate) numerator: Vec<f64>,
    pub(crate) trend: Vec<f64>,
    pub(crate) cycle: Vec<f64>,
    pub(crate) seasonal: Vec<f64>,
    pub(crate) trend_cycle: Vec<f64>,
    pub(crate) denominator: Vec<f64>,
}

/// CONJ (`ansub2.f`): the product `left(B) · right(F)` with both lag signs
/// folded into one harmonic slot. gfortran fuses each term into its slot.
pub(super) fn conjugate(left: &[f64], right: &[f64]) -> Vec<f64> {
    debug_assert!(!left.is_empty() && !right.is_empty());
    let mut output = vec![0.0; left.len().max(right.len())];
    for (i, &left_value) in left.iter().enumerate() {
        for (j, &right_value) in right.iter().enumerate() {
            let lag = i.abs_diff(j);
            output[lag] = left_value.mul_add(right_value, output[lag]);
        }
    }
    output
}

/// MULTFN (`ansub2.f`): the product of two harmonic polynomials.
pub(super) fn multiply_harmonics(left: &[f64], right: &[f64]) -> Vec<f64> {
    if left.is_empty() || right.is_empty() {
        // MULTFN with mplus1=0 (or nplus1=0): CONV and CONJ only clear their
        // outputs, so C holds lplus1=mplus1+nplus1-1 zeros. An empty factor
        // is the zero polynomial here, never the identity.
        return vec![0.0; (left.len() + right.len()).saturating_sub(1)];
    }
    // MULTFN clears its whole scratch, including the tail beyond CONJ's
    // max(input counts), before calling CONV then CONJ.
    let mut folded = vec![0.0; left.len() + right.len() - 1];
    let mut output = convolve(left, right);
    let conjugated = conjugate(left, right);
    folded[..conjugated.len()].copy_from_slice(&conjugated);
    for (value, folded_value) in output.iter_mut().zip(folded) {
        // The addition and the halving round separately. Build 62 has its
        // small-coefficient clamp commented out.
        *value = (*value + folded_value) / 2.0;
    }
    output
}

/// SPECTRUM's four CONJ calls, then its two MULTFN calls.
pub(crate) fn build_harmonics(inputs: &SpectralInputs) -> Harmonics {
    let numerator = conjugate(&inputs.aggregate_ma, &inputs.aggregate_ma);
    let trend = conjugate(&inputs.trend.combined, &inputs.trend.combined);
    let cycle = conjugate(&inputs.cycle.combined, &inputs.cycle.combined);
    let seasonal = conjugate(&inputs.seasonal.combined, &inputs.seasonal.combined);
    let trend_cycle = multiply_harmonics(&trend, &cycle);
    let denominator = multiply_harmonics(&trend_cycle, &seasonal);
    Harmonics {
        numerator,
        trend,
        cycle,
        seasonal,
        trend_cycle,
        denominator,
    }
}

#[cfg(test)]
mod tests;
