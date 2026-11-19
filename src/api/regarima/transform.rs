use crate::api::{Error, Transform};
use crate::error::SpecProblem;
use crate::x11;

/// Requested transform: fixed, or the automatic choice.
///
/// The transform used is in
/// [`ModelResult::transform`](crate::ModelResult::transform).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformSelection {
    /// Use the selected transform without comparing alternatives.
    Explicit(Transform),
    /// Compare no transformation and logarithms by AICC.
    Automatic(AutoTransformOptions),
}

impl TransformSelection {
    /// The automatic choice with the default settings
    /// ([`AutoTransformOptions::new`]).
    #[must_use]
    pub const fn automatic() -> Self {
        Self::Automatic(AutoTransformOptions::new())
    }
}

impl From<Transform> for TransformSelection {
    fn from(transform: Transform) -> Self {
        Self::Explicit(transform)
    }
}

impl From<AutoTransformOptions> for TransformSelection {
    fn from(options: AutoTransformOptions) -> Self {
        Self::Automatic(options)
    }
}

/// Settings for the automatic choice between no transform and logs.
///
/// Both models are fitted and compared by AICC: logs win when the log fit's
/// AICC plus [`aic_difference`](AutoTransformOptions::aic_difference) is
/// below the no-transform AICC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoTransformOptions {
    aic_difference: f64,
}

impl Default for AutoTransformOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoTransformOptions {
    /// The X-13ARIMA-SEATS default for monthly and quarterly series: an AICC
    /// difference of −2, so logs win unless they fit worse by 2 or more.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            aic_difference: -2.0,
        }
    }

    /// Sets a finite AICC difference; negative values favor logarithms.
    pub fn with_aic_difference(mut self, difference: f64) -> Result<Self, Error> {
        if !difference.is_finite() {
            return Err(Error::from(SpecProblem::TransformAicDifference));
        }
        self.aic_difference = difference;
        Ok(self)
    }

    /// Offset added to log AICC before the strict comparison against no-log AICC.
    #[must_use]
    pub const fn aic_difference(self) -> f64 {
        self.aic_difference
    }
}

/// Requested X-11 mode, which may depend on a not-yet-selected transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11ModeSelection {
    /// An explicitly chosen decomposition mode.
    Explicit(x11::Mode),
    /// Additive for no transformation, multiplicative for logarithms.
    FromTransform,
}

/// Outcome of an automatic transformation request, with the scores compared.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TransformSelectionResult {
    /// Both candidate fits produced AICC values and were compared.
    Compared {
        /// Selected transform.
        selected: Transform,
        /// AICC of the fit without a transform.
        no_transform_aicc: f64,
        /// AICC of the log fit.
        log_aicc: f64,
        /// Offset added to the log AICC before the comparison.
        aic_difference: f64,
    },
    /// The comparison was skipped and no transform used, because an
    /// observation is zero or negative.
    NonPositiveInput {
        /// Zero-based index of the first nonpositive observation.
        index: usize,
    },
}

impl TransformSelectionResult {
    /// The transform selected: [`Transform::None`] when the comparison was
    /// skipped.
    #[must_use]
    pub const fn selected(&self) -> Transform {
        match self {
            Self::Compared { selected, .. } => *selected,
            Self::NonPositiveInput { .. } => Transform::None,
        }
    }
}
