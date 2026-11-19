//! The refit `automd.f` runs each time it changes the model.

use super::warnings::push_reported_fit;
use crate::api::ResolvedRegArimaSpec;
use crate::arima::{
    ArmaStart, AutomaticFit, AutomaticInput, EstimatedRegArima, FitContext, fit_automatic_input,
};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{Error, Warning};

/// One fit of the transformed series `response`.
pub(super) struct Refit<'a> {
    pub(super) response: &'a [f64],
    pub(super) original: FitContext<'a>,
    pub(super) axis: TimeAxis,
    pub(super) spec: ResolvedRegArimaSpec,
    pub(super) regression: RegressionModel,
    pub(super) start: ArmaStart,
    /// `stpitr.f`'s convergence flag as the previous fit left it.
    pub(super) carried_convergence: bool,
}

impl Refit<'_> {
    pub(super) fn fit(self) -> Result<AutomaticFit, Error> {
        fit_automatic_input(
            AutomaticInput::Transformed(self.response),
            self.original,
            self.axis,
            self.spec,
            self.regression,
            self.start,
            self.carried_convergence,
        )
    }

    /// Fits, reports the fit's warnings and fails unless it converged.
    pub(super) fn fit_converged(
        self,
        warnings: &mut Vec<Warning>,
    ) -> Result<EstimatedRegArima, Error> {
        let fit = self.fit()?;
        push_reported_fit(warnings, &fit);
        let iterations = fit.nonlinear_iterations();
        fit.into_converged()
            .map_err(|_| Error::EstimationDidNotConverge { iterations })
    }
}
