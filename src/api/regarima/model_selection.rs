/// Requested ARIMA model: a fixed model or the automatic search.
///
/// The fitted orders are in [`ModelResult::model`](crate::ModelResult::model).
///
/// ```
/// let requested = x13::RegArimaSpec::automatic_model();
/// assert_eq!(requested.model(), x13::regarima::ModelSelection::Automatic);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSelection {
    /// Use the specified model without searching for alternatives.
    Explicit(crate::ArimaModel),
    /// The automatic model search with its default settings, which also decides
    /// the constant term. It has no tuning controls, and does not turn on an
    /// automatic transform or outlier search.
    Automatic,
}

/// Displays an explicit model as `(p d q)(P D Q)`, otherwise `automatic`.
impl std::fmt::Display for ModelSelection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Explicit(model) => model.fmt(formatter),
            Self::Automatic => formatter.write_str("automatic"),
        }
    }
}

/// Requested constant term. Whether the fitted model has one is in
/// [`ModelResult::includes_constant`](crate::ModelResult::includes_constant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstantSelection {
    /// Include a constant term.
    Included,
    /// Exclude the constant term.
    Excluded,
    /// Let the default automatic model search select the constant.
    Automatic,
}

display_phrases!(ConstantSelection {
    Included => "included",
    Excluded => "excluded",
    Automatic => "automatic",
});

/// Outcome of the automatic model search, from
/// [`Adjustment::model_selection`](crate::Adjustment::model_selection).
///
/// The final model, which may be revised after the search, is in
/// [`ModelResult::model`](crate::ModelResult::model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelSelectionResult {
    /// The search ran.
    Selected {
        /// The first model the search identified, before any later revision.
        first_model: crate::ArimaModel,
        /// Whether the search's test found the mean significant. The final
        /// model need not include a constant term.
        mean_significant: bool,
    },
    /// The search was skipped because the series is constant (its maximum and
    /// minimum differ by less than 3.834e-20, the reference program's equality
    /// threshold), so no model was fitted.
    ConstantSeries {
        /// The transform in effect.
        transform: crate::Transform,
    },
}
