//! The original series a fit reports its likelihood for, and what the fit
//! needs to know about it besides the transformed response.

use crate::error::NumericalDetail;
use std::iter;

use crate::{Error, Transform};

/// The original series behind a fit's response: its values, the X-11 Easter
/// factors or likelihood prior that scale them, its missing observations and
/// the user regressors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FitContext<'a> {
    raw: &'a [f64],
    observed_h1: Option<&'a [f64]>,
    likelihood_prior: Option<&'a [f64]>,
    missing: &'a [usize],
    user_regressors: &'a [crate::regarima::UserRegressor],
    additive_adjustment: bool,
}

impl<'a> From<&'a [f64]> for FitContext<'a> {
    fn from(raw: &'a [f64]) -> Self {
        Self::unadjusted(raw)
    }
}
impl<'a> From<&'a Vec<f64>> for FitContext<'a> {
    fn from(raw: &'a Vec<f64>) -> Self {
        Self::unadjusted(raw)
    }
}
impl<'a, const N: usize> From<&'a [f64; N]> for FitContext<'a> {
    fn from(raw: &'a [f64; N]) -> Self {
        Self::unadjusted(raw)
    }
}

impl<'a> FitContext<'a> {
    /// A series with no adjustment, missing observation or user regressor.
    pub(crate) fn unadjusted(raw: &'a [f64]) -> Self {
        Self {
            raw,
            observed_h1: None,
            likelihood_prior: None,
            missing: &[],
            user_regressors: &[],
            additive_adjustment: false,
        }
    }

    /// A series with the X-11 Easter factors (table H1) removed from the
    /// response, which `prlkhd.f` multiplies into the prior.
    pub(crate) fn with_legacy_easter(raw: &'a [f64], h1: &'a [f64]) -> Result<Self, Error> {
        if raw.len() != h1.len() {
            return Err(Error::internal(
                "X-11 Easter factor count differs from the series",
            ));
        }
        if h1.iter().any(|value| !value.is_finite()) {
            return Err(Error::numerical(
                crate::error::NumericalOperation::ExactLikelihood,
            ));
        }
        if let Some(index) = raw.iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteValue { index });
        }
        Ok(Self {
            raw,
            observed_h1: Some(h1),
            likelihood_prior: None,
            missing: &[],
            user_regressors: &[],
            additive_adjustment: false,
        })
    }

    /// Sets the prior `trnaic.f` uses for the likelihood of a model span.
    pub(crate) fn with_likelihood_prior(mut self, prior: &'a [f64]) -> Result<Self, Error> {
        if !prior.is_empty() && prior.len() != self.raw.len() {
            return Err(Error::internal(
                "likelihood prior length differs from the series",
            ));
        }
        self.likelihood_prior = Some(prior);
        Ok(self)
    }

    pub(crate) fn with_missing(mut self, missing: &'a [usize]) -> Self {
        self.missing = missing;
        self
    }

    pub(crate) fn missing(self) -> &'a [usize] {
        self.missing
    }

    pub(crate) fn with_user_regressors(
        mut self,
        values: &'a [crate::regarima::UserRegressor],
    ) -> Self {
        self.user_regressors = values;
        self
    }

    pub(crate) fn user_regressors(self) -> &'a [crate::regarima::UserRegressor] {
        self.user_regressors
    }

    /// Whether the fit serves an additive X-11 adjustment, which selects
    /// `prlkhd.f`'s `Adjmod` 2 Jacobian.
    pub(crate) fn with_additive_adjustment(mut self, additive: bool) -> Self {
        self.additive_adjustment = additive;
        self
    }

    pub(crate) fn raw(self) -> &'a [f64] {
        self.raw
    }

    /// The log Jacobian that puts the likelihood on the original scale.
    /// `prior` is the fit's own prior adjustment, used unless
    /// [`Self::with_likelihood_prior`] set another.
    pub(crate) fn transformation_adjustment(
        self,
        prior: &[f64],
        conditioning_loss: usize,
        transform: Transform,
    ) -> Result<f64, Error> {
        if self.additive_adjustment {
            return Ok(super::likelihood::additive_transformation_adjustment(
                self.raw,
                conditioning_loss,
                transform,
            ));
        }
        let prior = self.likelihood_prior.unwrap_or(prior);
        let Some(h1) = self.observed_h1 else {
            return super::likelihood::transformation_adjustment(
                self.raw,
                prior,
                conditioning_loss,
                transform,
            );
        };
        if conditioning_loss > self.raw.len()
            || (!prior.is_empty() && prior.len() != self.raw.len())
        {
            return Err(Error::internal(
                "likelihood prior or conditioning loss does not fit the series",
            ));
        }
        // An empty prior is the identity.
        let priors = prior.iter().copied().chain(iter::repeat(1.0));
        let observed = self.raw.iter().zip(h1).zip(priors).enumerate();
        let mut adjustment = 0.0;
        for (index, ((&observation, &h1_factor), mut jacobian)) in observed.skip(conditioning_loss)
        {
            // `prlkhd.f` tests for zero before the H1 product and any logarithm.
            if crate::numerics::build_62_equal(observation, 0.0)
                || crate::numerics::build_62_equal(jacobian, 0.0)
            {
                continue;
            }
            jacobian *= h1_factor;
            let unavailable = || Error::from(NumericalDetail::LikelihoodPriorUnavailable { index });
            if !jacobian.is_finite() || jacobian == 0.0 {
                return Err(unavailable());
            }
            if transform == Transform::Log {
                let adjusted = observation / jacobian;
                if !adjusted.is_finite() || adjusted == 0.0 {
                    return Err(unavailable());
                }
                jacobian = adjusted.abs().powf(-1.0) / jacobian;
            }
            if !jacobian.is_finite() || jacobian <= 0.0 {
                return Err(unavailable());
            }
            adjustment += jacobian.ln();
            if !adjustment.is_finite() {
                return Err(unavailable());
            }
        }
        Ok(adjustment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_likelihood_prior_replaces_the_fits_own_prior() {
        let original_axis = crate::time::TimeAxis::monthly(2000, 2).unwrap();
        let model_axis = crate::time::TimeAxis::monthly(2001, 2).unwrap();
        let prior = |axis| {
            crate::regression::calendar::evaluate_fixed_length(
                crate::regarima::LengthEffect::LeapYear,
                axis,
                1,
            )
            .unwrap()
        };
        let supplied = prior(original_axis);
        let model_aligned = prior(model_axis);
        assert_ne!(supplied, model_aligned);
        let mut distinguishes_prior = false;
        for step in 0..64 {
            let raw = [0.8 + f64::from(step) / 100.0];
            let selected = FitContext::unadjusted(&raw)
                .with_likelihood_prior(&supplied)
                .unwrap()
                .transformation_adjustment(&model_aligned, 0, Transform::Log)
                .unwrap();
            let expected = super::super::likelihood::transformation_adjustment(
                &raw,
                &supplied,
                0,
                Transform::Log,
            )
            .unwrap();
            let with_fit_prior = FitContext::unadjusted(&raw)
                .transformation_adjustment(&model_aligned, 0, Transform::Log)
                .unwrap();
            assert_eq!(selected.to_bits(), expected.to_bits());
            distinguishes_prior |= selected.to_bits() != with_fit_prior.to_bits();
        }
        // The chain-rule expression can round differently even though both
        // choices simplify algebraically to -log(raw).
        assert!(distinguishes_prior);
    }

    #[test]
    fn an_additive_adjustment_uses_the_jacobian_without_a_prior() {
        let raw = [2.0, 1e-25, 4.0];
        let prior = [1.5, 1.25, 0.5];
        let actual = |additive| {
            FitContext::unadjusted(&raw)
                .with_additive_adjustment(additive)
                .transformation_adjustment(&prior, 0, Transform::Log)
                .unwrap()
        };
        let additive =
            super::super::likelihood::additive_transformation_adjustment(&raw, 0, Transform::Log);
        let guarded =
            super::super::likelihood::transformation_adjustment(&raw, &prior, 0, Transform::Log)
                .unwrap();
        assert_eq!(actual(true).to_bits(), additive.to_bits());
        assert_eq!(actual(false).to_bits(), guarded.to_bits());
        assert_ne!(additive.to_bits(), guarded.to_bits());
    }

    #[test]
    fn easter_factors_multiply_the_prior_after_the_conditioning_rows() {
        let context =
            FitContext::with_legacy_easter(&[10.0, 20.0, 30.0], &[0.5, 2.0, 3.0]).unwrap();
        assert_eq!(
            context
                .transformation_adjustment(&[2.0, 2.0, 3.0], 1, Transform::None)
                .unwrap(),
            4.0_f64.ln() + 9.0_f64.ln()
        );
        assert_eq!(
            context
                .transformation_adjustment(&[2.0, 2.0, 3.0], 2, Transform::None)
                .unwrap(),
            9.0_f64.ln()
        );
    }

    #[test]
    fn a_zero_observation_or_prior_skips_its_easter_factor() {
        let context =
            FitContext::with_legacy_easter(&[0.0, 10.0, 20.0], &[-1.0, -2.0, 2.0]).unwrap();
        assert_eq!(
            context
                .transformation_adjustment(&[1.0, 0.0, 0.5], 0, Transform::Log)
                .unwrap(),
            0.05_f64.ln()
        );
    }

    #[test]
    fn a_nonpositive_or_nonfinite_jacobian_is_an_error() {
        for factor in [0.0, -1.0, f64::MAX] {
            let factors = [factor];
            let context = FitContext::with_legacy_easter(&[10.0], &factors).unwrap();
            assert!(
                context
                    .transformation_adjustment(&[2.0], 0, Transform::None)
                    .is_err()
            );
        }
    }

    #[test]
    fn the_log_jacobian_with_easter_factors_keeps_the_operations_of_prlkhd() {
        let context = FitContext::with_legacy_easter(&[1.3], &[0.9]).unwrap();
        let actual = context
            .transformation_adjustment(&[1.1], 0, Transform::Log)
            .unwrap();
        // Simplifying the expression to -ln(1.3) moves this result by two ulps.
        assert_eq!(actual.to_bits(), 0xbfd0_ca93_7be1_b9dd);
        assert_ne!(actual.to_bits(), (-1.3_f64.ln()).to_bits());
    }
}
