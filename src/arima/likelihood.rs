//! The log likelihood of a fit and the model-selection statistics
//! `prlkhd.f` derives from it.

use std::iter;

use crate::numerics::{BUILD_62_PI, build_62_equal};
use crate::{Error, Transform};

/// The log likelihood on the transformed scale, and the criteria on the
/// original one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LikelihoodStatistics {
    pub(crate) log_likelihood: f64,
    pub(crate) aic: f64,
    pub(crate) aicc: Option<f64>,
    pub(crate) hannan_quinn: f64,
    pub(crate) bic: f64,
}

/// The likelihood of an exact fit, or of one conditional on its first AR
/// observations, for which Build 62 has no selection criteria.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FitLikelihood {
    Exact(LikelihoodStatistics),
    ConditionalAr {
        log_likelihood: f64,
        transformation_adjustment: f64,
    },
}

impl FitLikelihood {
    pub(crate) fn exact_statistics(self) -> Option<LikelihoodStatistics> {
        match self {
            Self::Exact(statistics) => Some(statistics),
            Self::ConditionalAr { .. } => None,
        }
    }
}

/// The log Jacobian of the log transform and the prior adjustment, which
/// puts the likelihood on the scale of the original series.
pub(crate) fn transformation_adjustment(
    observations: &[f64],
    prior_adjustments: &[f64],
    conditioning_loss: usize,
    transform: Transform,
) -> Result<f64, Error> {
    if transform == Transform::None {
        return Ok(0.0);
    }

    if !prior_adjustments.is_empty() && prior_adjustments.len() != observations.len() {
        return Err(Error::internal(
            "prior adjustment length differs from the series",
        ));
    }
    // An empty prior is the identity.
    let priors = prior_adjustments
        .iter()
        .copied()
        .chain(iter::repeat(1.0))
        .skip(conditioning_loss);
    let mut adjustment = 0.0;
    for (&observation, prior) in observations[conditioning_loss..].iter().zip(priors) {
        // `prlkhd.f` tests the values before dividing and skips a zero under
        // `dpeq`'s tolerance.
        if build_62_equal(observation, 0.0) || build_62_equal(prior, 0.0) {
            continue;
        }
        let adjusted = observation / prior;
        // `prlkhd.f` raises to `Lam - 1` with `Lam = 0`: a power, which need
        // not round as a reciprocal does.
        let jacobian = adjusted.abs().powf(-1.0) / prior;
        adjustment += jacobian.ln();
    }
    Ok(adjustment)
}

/// The Jacobian `prlkhd.f` computes when `Adjmod` is 2, as `editor.f` sets it
/// for additive X-11: this loop reads no prior and, unlike
/// [`transformation_adjustment`], skips no value.
pub(crate) fn additive_transformation_adjustment(
    observations: &[f64],
    conditioning_loss: usize,
    transform: Transform,
) -> f64 {
    if transform == Transform::None {
        return 0.0;
    }
    observations[conditioning_loss..]
        .iter()
        .fold(0.0, |adjustment, observation| {
            adjustment + observation.abs().powf(-1.0).ln()
        })
}

/// The statistics of a fit with this innovation variance.
pub(crate) fn likelihood_statistics(
    effective_observations: usize,
    estimated_parameter_count: usize,
    log_determinant_correction: f64,
    innovation_variance: f64,
    transformation_adjustment: f64,
) -> LikelihoodStatistics {
    let log_likelihood = gaussian_log_likelihood(
        effective_observations,
        log_determinant_correction,
        innovation_variance,
    );
    statistics_from_log_likelihood(
        effective_observations,
        estimated_parameter_count,
        log_likelihood,
        transformation_adjustment,
    )
}

/// AIC, AICC, Hannan-Quinn and BIC for a log likelihood, each with the
/// transformation adjustment added. AICC needs more than one observation
/// beyond the parameter count.
pub(crate) fn statistics_from_log_likelihood(
    effective_observations: usize,
    estimated_parameter_count: usize,
    log_likelihood: f64,
    transformation_adjustment: f64,
) -> LikelihoodStatistics {
    let observations = effective_observations as f64;
    let parameters = estimated_parameter_count as f64;
    let adjusted = log_likelihood + transformation_adjustment;
    let log_observations = observations.ln();

    LikelihoodStatistics {
        log_likelihood,
        aic: -2.0 * (adjusted - parameters),
        aicc: (effective_observations > estimated_parameter_count + 1).then(|| {
            -2.0 * (adjusted - observations * parameters / (observations - (parameters + 1.0)))
        }),
        hannan_quinn: -2.0 * (-parameters).mul_add(log_observations.ln(), adjusted),
        bic: (-2.0_f64).mul_add(adjusted, parameters * log_observations),
    }
}

/// The Gaussian log likelihood at the maximum-likelihood innovation variance.
pub(crate) fn gaussian_log_likelihood(
    effective_observations: usize,
    log_determinant_correction: f64,
    innovation_variance: f64,
) -> f64 {
    let observations = effective_observations as f64;
    -(log_determinant_correction
        + observations * ((2.0 * BUILD_62_PI * innovation_variance).ln() + 1.0))
        / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prior_of_another_span_fails_only_where_the_jacobian_reads_it() {
        let (observations, prior) = ([1.0, 2.0], [1.0]);
        assert_eq!(
            transformation_adjustment(&observations, &prior, 0, Transform::None),
            Ok(0.0)
        );
        assert!(matches!(
            transformation_adjustment(&observations, &prior, 0, Transform::Log),
            Err(Error::Internal(_))
        ));
    }

    #[test]
    fn a_tiny_observation_leaves_the_jacobian() {
        let without =
            transformation_adjustment(&[2.0, 4.0], &[1.5, 0.5], 0, Transform::Log).unwrap();
        for tiny in [0.0, 1e-25, -1e-25, 3.8e-20] {
            let actual =
                transformation_adjustment(&[2.0, tiny, 4.0], &[1.5, 1.25, 0.5], 0, Transform::Log)
                    .unwrap();
            assert_eq!(actual.to_bits(), without.to_bits());
        }
    }

    #[test]
    fn a_tiny_prior_leaves_the_jacobian() {
        let without =
            transformation_adjustment(&[2.0, 4.0], &[1.5, 0.5], 0, Transform::Log).unwrap();
        for tiny in [0.0, 1e-25, -1e-25, 3.8e-20] {
            let actual =
                transformation_adjustment(&[2.0, 3.0, 4.0], &[1.5, tiny, 0.5], 0, Transform::Log)
                    .unwrap();
            assert_eq!(actual.to_bits(), without.to_bits());
        }
    }

    #[test]
    fn values_at_the_tolerance_keep_their_ordinary_term() {
        // dpeq's test is strict, so its tolerance itself is not zero.
        for (observation, prior) in [(2.0, 1.5), (4.0, 0.5), (3.834e-20, 1.0), (1.0, 3.834e-20)] {
            let adjusted: f64 = observation / prior;
            let expected = (adjusted.abs().powf(-1.0) / prior).ln();
            let actual =
                transformation_adjustment(&[observation], &[prior], 0, Transform::Log).unwrap();
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn additive_jacobian_keeps_a_tiny_observation() {
        let term = |value: f64| value.powf(-1.0).ln();
        let observations = [2.0, 1e-25, 4.0];
        assert_eq!(
            additive_transformation_adjustment(&observations, 0, Transform::Log).to_bits(),
            (0.0 + term(2.0) + term(1e-25) + term(4.0)).to_bits()
        );
        assert_eq!(
            additive_transformation_adjustment(&observations, 2, Transform::Log).to_bits(),
            (0.0 + term(4.0)).to_bits()
        );
        assert_eq!(
            additive_transformation_adjustment(&observations, 0, Transform::None),
            0.0
        );
    }

    #[test]
    fn additive_jacobian_matches_the_identity_prior_on_ordinary_values() {
        let observations = [3.1, 7.9, 0.13];
        assert_eq!(
            additive_transformation_adjustment(&observations, 1, Transform::Log).to_bits(),
            transformation_adjustment(&observations, &[], 1, Transform::Log)
                .unwrap()
                .to_bits()
        );
    }

    #[test]
    fn hannan_quinn_fuses_the_build_62_multiply_subtract() {
        let effective_observations = 35;
        let estimated_parameter_count = 10;
        let target_adjusted_likelihood = 12.684_529_660_527_513;
        let log_likelihood =
            -(effective_observations as f64 * ((2.0 * BUILD_62_PI).ln() + 1.0)) / 2.0;
        let transformation_adjustment = target_adjusted_likelihood - log_likelihood;

        let actual = likelihood_statistics(
            effective_observations,
            estimated_parameter_count,
            0.0,
            1.0,
            transformation_adjustment,
        );
        let adjusted = actual.log_likelihood + transformation_adjustment;
        let expected = -2.0
            * (-(estimated_parameter_count as f64))
                .mul_add((effective_observations as f64).ln().ln(), adjusted);

        assert_eq!(actual.hannan_quinn.to_bits(), expected.to_bits());
    }

    #[test]
    fn aicc_needs_more_than_one_observation_beyond_the_parameter_count() {
        assert_eq!(likelihood_statistics(7, 7, 0.0, 1.0, 0.0).aicc, None);
        assert_eq!(likelihood_statistics(8, 7, 0.0, 1.0, 0.0).aicc, None);
        assert!(likelihood_statistics(9, 7, 0.0, 1.0, 0.0).aicc.is_some());
    }

    #[test]
    fn the_criteria_add_the_transformation_adjustment_to_the_log_likelihood() {
        let statistics = statistics_from_log_likelihood(35, 4, 17.25, -8.0);
        assert_eq!(statistics.log_likelihood.to_bits(), 17.25_f64.to_bits());
        assert_eq!(statistics.aic.to_bits(), (-10.5_f64).to_bits());
        assert_eq!(
            statistics.aicc.unwrap().to_bits(),
            (-2.0_f64 * (9.25 - 35.0 * 4.0 / 30.0)).to_bits()
        );
    }
}
