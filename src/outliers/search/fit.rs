//! The kinds of fit the outlier search loop runs on.

use crate::Warning;
use crate::arima::{AutomaticFit, EstimatedRegArima, FitReport, FittedNumericalView};

/// A fit the outlier search can test candidates against.
pub(super) trait SearchFit {
    /// The numbers the candidate and deletion tests read.
    fn numerical_view(&self) -> FittedNumericalView<'_>;
    /// The fit's warnings, which the search collects and replaces.
    fn warnings_mut(&mut self) -> &mut Vec<Warning>;
    /// What decides which of the fit's warnings an automatic search reports.
    fn report(&self) -> FitReport;
}

impl SearchFit for EstimatedRegArima {
    fn numerical_view(&self) -> FittedNumericalView<'_> {
        self.numerical_view()
    }

    fn warnings_mut(&mut self) -> &mut Vec<Warning> {
        &mut self.warnings
    }

    fn report(&self) -> FitReport {
        self.report()
    }
}

impl SearchFit for AutomaticFit {
    fn numerical_view(&self) -> FittedNumericalView<'_> {
        self.numerical_view()
    }

    fn warnings_mut(&mut self) -> &mut Vec<Warning> {
        self.warnings_mut()
    }

    fn report(&self) -> FitReport {
        self.report()
    }
}

/// An automatic fit whose search residuals may start with a saved prefix in
/// place of the fit's own.
pub(super) struct ResidualFit {
    pub(super) fit: AutomaticFit,
    pub(super) residual_override: Option<Vec<f64>>,
}

impl ResidualFit {
    /// Overwrites the leading residuals of `fit` with `prefix`, where one is
    /// given.
    pub(super) fn new(fit: AutomaticFit, prefix: Option<&[f64]>) -> Self {
        let residual_override = prefix.map(|prefix| {
            let mut residuals = fit.numerical_view().residuals.to_vec();
            let overlap = residuals.len().min(prefix.len());
            residuals[..overlap].copy_from_slice(&prefix[..overlap]);
            residuals
        });
        Self {
            fit,
            residual_override,
        }
    }
}

impl SearchFit for ResidualFit {
    fn numerical_view(&self) -> FittedNumericalView<'_> {
        let mut view = self.fit.numerical_view();
        if let Some(residuals) = &self.residual_override {
            view.residuals = residuals;
        }
        view
    }

    fn warnings_mut(&mut self) -> &mut Vec<Warning> {
        self.fit.warnings_mut()
    }

    fn report(&self) -> FitReport {
        self.fit.report()
    }
}
