/// Whether a fit maximizes the exact likelihood or the one conditional on
/// its first AR observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LikelihoodMode {
    Exact,
    ConditionalAr,
}

/// The likelihood, tolerances and iteration limits of one `rgarma.f` fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EstimationPolicy {
    mode: LikelihoodMode,
    overall_likelihood_tolerance: f64,
    initial_nonlinear_likelihood_tolerance: f64,
    final_nonlinear_likelihood_tolerance: f64,
    maximum_overall_iterations: usize,
    maximum_nonlinear_iterations: usize,
}

impl EstimationPolicy {
    /// The `estimate` spec's defaults, for a model that is given or chosen.
    pub(super) const EXPLICIT: Self = Self {
        mode: LikelihoodMode::Exact,
        overall_likelihood_tolerance: 1e-5,
        initial_nonlinear_likelihood_tolerance: 1e-3,
        final_nonlinear_likelihood_tolerance: 1e-5,
        maximum_overall_iterations: 1500,
        maximum_nonlinear_iterations: 40,
    };
    /// The looser tolerance and 200 iterations of the fits that identify
    /// the differencing orders.
    pub(super) const AUTOMATIC_IDENTIFICATION: Self = Self {
        mode: LikelihoodMode::Exact,
        overall_likelihood_tolerance: 1e-3,
        initial_nonlinear_likelihood_tolerance: 0.1,
        final_nonlinear_likelihood_tolerance: 1e-3,
        maximum_overall_iterations: 200,
        maximum_nonlinear_iterations: 40,
    };
    /// The same tolerance with 1500 iterations, for the fits that compare
    /// ARMA orders.
    pub(super) const AUTOMATIC_SEARCH: Self = Self {
        maximum_overall_iterations: 1500,
        ..Self::AUTOMATIC_IDENTIFICATION
    };
    /// The conditional-likelihood retry of an identification fit.
    pub(super) const CONDITIONAL_IDENTIFICATION: Self = Self {
        mode: LikelihoodMode::ConditionalAr,
        ..Self::AUTOMATIC_SEARCH
    };

    pub(super) fn mode(self) -> LikelihoodMode {
        self.mode
    }

    /// The likelihood tolerance as `rgarma.f` applies it to the deviance.
    pub(super) fn overall_deviance_tolerance(self, effective: f64) -> f64 {
        2.0 / effective * self.overall_likelihood_tolerance
    }

    /// The tolerance of one nonlinear step. With regressors the first two
    /// overall iterations use a looser one; without them the single step
    /// uses the overall tolerance.
    pub(super) fn nonlinear_deviance_tolerance(
        self,
        effective: f64,
        has_regression: bool,
        overall_iteration: usize,
    ) -> f64 {
        if !has_regression {
            return self.overall_deviance_tolerance(effective);
        }
        let tolerance = if overall_iteration <= 2 {
            self.initial_nonlinear_likelihood_tolerance
        } else {
            self.final_nonlinear_likelihood_tolerance
        };
        2.0 / effective * tolerance
    }

    /// The cumulative iteration count at which the next nonlinear step
    /// stops: 40 more with regressors, the overall limit without.
    pub(super) fn nonlinear_iteration_limit(
        self,
        has_regression: bool,
        cumulative: usize,
    ) -> usize {
        if has_regression {
            cumulative + self.maximum_nonlinear_iterations
        } else {
            self.maximum_overall_iterations
        }
    }

    pub(super) fn maximum_overall_iterations(self) -> usize {
        self.maximum_overall_iterations
    }
}

#[cfg(test)]
mod tests;
