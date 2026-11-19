mod covariance;
mod estimation;
pub(crate) mod exact;
mod fit_context;
mod forecast;
pub(crate) use fit_context::FitContext;
mod likelihood;
pub(crate) use likelihood::statistics_from_log_likelihood;
pub(crate) mod model;
mod operators;
pub(crate) use operators::polynomials;
pub(crate) mod roots;

pub(crate) use estimation::ArmaCovarianceDiagnostic;
pub(crate) use estimation::automatic::{
    AutomaticFit, AutomaticInput, fit_automatic, fit_automatic_input,
};
pub(crate) use estimation::identification::{
    MeanFit, MeanFitBudget, fit_local_mean, fit_prepared_identification,
};
pub(crate) use estimation::search::{
    PreparedSearchInput, fit_prepared_candidate, refit_prepared_selected,
};
pub(crate) use estimation::uncertainty_is_available;
pub(crate) use estimation::{
    ArmaStart, EstimatedRegArima, FittedNumericalView, TransformCandidateFit, estimate,
    fit_transform_candidate,
};
// Tests elsewhere in the crate build fits with these statuses.
#[cfg(test)]
pub(crate) use estimation::{FitDiagnostic, FitStatus};
pub(crate) use estimation::{FitReport, append_dynamic_fit_warnings, append_reported_fit_warnings};
pub(crate) use forecast::{BackcastPoint, ExtensionsResult, ForecastPoint, generate_extensions};
