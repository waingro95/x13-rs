//! Failures of the SEATS kernels: what went wrong, and the step it went wrong
//! in.

use std::fmt;

use super::Orders;

/// SEATS step at which a failure occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Preparing the regARIMA fitted effects for SEATS.
    FittedEffectsPreparation,
    /// Restoring the fitted effects to the SEATS components.
    FittedEffectsRestoration,
    /// Importing the regARIMA model.
    Import,
    /// Mean correction and its retry passes.
    Mean,
    /// Residual autocorrelation diagnostics.
    Autocorrelation,
    /// Re-estimating the model.
    Refit,
    /// Computing the model residuals.
    Residual,
    /// Forecasting and backcasting the series tails.
    Tails,
    /// Canonical decomposition of the model.
    CanonicalDecomposition,
    /// Extracting the components.
    Extraction,
    /// Bias correction of the components.
    BiasCorrection,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FittedEffectsPreparation => "fitted-effects preparation",
            Self::FittedEffectsRestoration => "fitted-effects restoration",
            Self::Import => "model import",
            Self::Mean => "mean correction",
            Self::Autocorrelation => "autocorrelation diagnostics",
            Self::Refit => "model refit",
            Self::Residual => "residual calculation",
            Self::Tails => "tail forecasting",
            Self::CanonicalDecomposition => "canonical decomposition",
            Self::Extraction => "component extraction",
            Self::BiasCorrection => "bias correction",
        })
    }
}

/// A failure and the step it happened in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Error {
    pub(crate) stage: Stage,
    pub(crate) failure: Failure,
}

impl Error {
    /// The same failure, reported at `stage`.
    pub(crate) fn restage(self, stage: Stage) -> Self {
        Self { stage, ..self }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SEATS failed during {}: {}", self.stage, self.failure)
    }
}

impl std::error::Error for Error {}

/// What failed in a SEATS step, with the values involved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Failure {
    /// The canonical decomposition gives a component a negative variance.
    NegativeVariance {
        component: CanonicalComponent,
        variance: f64,
    },
    /// A component's innovation variance, in units of the model's, is outside
    /// `[0, 1]`.
    InnovationVariance {
        component: Component,
        direction: VarianceDirection,
        value: f64,
    },
    /// The spectral numerator of the seasonally adjusted component is not
    /// positive at lag zero.
    UnusableNumerator { zero_lag: f64 },
    /// The imported AR coefficients equal the MA coefficients.
    DegenerateModel {
        seasonal: bool,
        ar: Vec<f64>,
        ma: Vec<f64>,
    },
    /// Too few values for the autocorrelation diagnostics.
    AutocorrelationSample { count: usize },
    /// Differencing and the model's parameters leave no residual degrees of
    /// freedom.
    NoDegreesOfFreedom { observed: usize, consumed: usize },
    /// The regARIMA estimation did not converge, so the Q statistic SEATS
    /// reads is undefined.
    EstimationNotConverged,
    /// Build 62 reads a residual mean coefficient it never set.
    UndefinedResidualMeanCorrection { residual_count: usize, t: f64 },
    /// Build 62's search reads an iteration count it never set.
    UnwrittenSearchCount { objective: f64 },
    /// Build 62's search reads a column scale it never set.
    UnwrittenSearchScale { gradient: [f64; 2] },
    /// A linear system has no usable pivot.
    SingularSystem { row: usize },
    /// The root solver did not find every root.
    RootSolverStalled { degree: usize },
    /// The roots of a polynomial are not usable.
    UnusableRoots {
        step: &'static str,
        index: Option<usize>,
    },
    /// A factor needs more coefficients than SEATS holds.
    FactorCapacity {
        factor: &'static str,
        count: usize,
        maximum: usize,
    },
    /// Build 62 would re-estimate a smaller model.
    ModelChange(ModelChange),
    /// A value is nonfinite, or otherwise outside what the step can use.
    Unusable {
        quantity: &'static str,
        index: Option<usize>,
        value: f64,
    },
}

impl Failure {
    /// This failure, at `stage`.
    pub(crate) fn at(self, stage: Stage) -> Error {
        Error {
            stage,
            failure: self,
        }
    }
}

/// `Ok(value)` when `value` is finite, else [`Failure::Unusable`] naming it.
pub(crate) fn usable(
    quantity: &'static str,
    index: Option<usize>,
    value: f64,
) -> Result<f64, Failure> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Failure::Unusable {
            quantity,
            index,
            value,
        })
    }
}

/// Build 62 would re-estimate the `current` model with the `proposed` orders
/// (CHMODEL, ChangeModel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelChange {
    pub(crate) current: Orders,
    pub(crate) proposed: Orders,
}

/// A component of the canonical decomposition that can have a negative
/// variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CanonicalComponent {
    Irregular,
    Cycle,
}

/// A component whose innovation variance SHOWCOMP checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    Trend,
    Seasonal,
    Transitory,
    SeasonallyAdjusted,
}

/// Which side of `[0, 1]` an innovation variance falls on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VarianceDirection {
    BelowZero,
    AboveOne,
}

/// A value as `{:?}` prints it: the shortest round trip, NaN and infinities
/// included.
struct Value(f64);

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

struct Values<'a>(&'a [f64]);

impl fmt::Display for Values<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, &value) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", Value(value))?;
        }
        f.write_str("]")
    }
}

/// Orders as `(p d q)(P D Q)`.
struct ModelOrders(Orders);

impl fmt::Display for ModelOrders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [p, d, q, bp, bd, bq] = self.0.to_array();
        write!(f, "({p} {d} {q})({bp} {bd} {bq})")
    }
}

const NO_EXTRACTION: &str = "Build 62 stops here without signal extraction";

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeVariance {
                component,
                variance,
            } => write!(
                f,
                "the canonical decomposition gives the {} component a negative variance ({}); \
                 {NO_EXTRACTION}",
                match component {
                    CanonicalComponent::Irregular => "irregular",
                    CanonicalComponent::Cycle => "cycle",
                },
                Value(*variance)
            ),
            Self::InnovationVariance {
                component,
                direction,
                value,
            } => write!(
                f,
                "the {} innovation variance {} is {}, in units of the model's innovation \
                 variance; {NO_EXTRACTION}",
                match component {
                    Component::Trend => "trend",
                    Component::Seasonal => "seasonal",
                    Component::Transitory => "transitory",
                    Component::SeasonallyAdjusted => "seasonally adjusted",
                },
                Value(*value),
                match direction {
                    VarianceDirection::BelowZero => "below zero",
                    VarianceDirection::AboveOne => "above one",
                }
            ),
            Self::UnusableNumerator { zero_lag } => write!(
                f,
                "the spectral numerator of the seasonally adjusted component is not positive at \
                 lag zero ({})",
                Value(*zero_lag)
            ),
            Self::DegenerateModel { seasonal, ar, ma } => write!(
                f,
                "imported {} AR coefficients {} equal the MA coefficients {}, a degenerate \
                 model; Build 62 performs no signal extraction",
                if *seasonal { "seasonal" } else { "regular" },
                Values(ar),
                Values(ma)
            ),
            Self::AutocorrelationSample { count } => write!(
                f,
                "{count} values are too few for the autocorrelation diagnostics"
            ),
            Self::NoDegreesOfFreedom { observed, consumed } => write!(
                f,
                "{observed} observations leave no residual degrees of freedom: differencing and \
                 the model's parameters use {consumed}"
            ),
            Self::EstimationNotConverged => f.write_str(
                "the regARIMA estimation did not converge, so the residual Q statistic SEATS \
                 reads is undefined",
            ),
            Self::UndefinedResidualMeanCorrection { residual_count, t } => write!(
                f,
                "the residual mean correction is undefined for {residual_count} residuals \
                 (t = {}); Build 62 reads a coefficient it never set",
                Value(*t)
            ),
            Self::UnwrittenSearchCount { objective } => write!(
                f,
                "the first search objective {} is below 1e-9; Build 62 then reads an iteration \
                 count it never set",
                Value(*objective)
            ),
            Self::UnwrittenSearchScale { gradient } => write!(
                f,
                "the first search gradient {} is too small; Build 62 then reads a column scale \
                 it never set",
                Values(gradient)
            ),
            Self::SingularSystem { row } => {
                write!(f, "the linear system is singular at row {row}")
            }
            Self::RootSolverStalled { degree } => write!(
                f,
                "the root solver did not find every root of a degree-{degree} polynomial"
            ),
            Self::UnusableRoots { step, index } => {
                write!(f, "the polynomial roots are not usable ({step}")?;
                if let Some(index) = index {
                    write!(f, ", root {index}")?;
                }
                f.write_str(")")
            }
            Self::FactorCapacity {
                factor,
                count,
                maximum,
            } => write!(
                f,
                "the {factor} factor needs {count} coefficients; SEATS holds at most {maximum}"
            ),
            Self::ModelChange(change) => write!(
                f,
                "SEATS would re-estimate the model {} as {}, as Build 62 does when a fitted AR \
                 root reaches the unit circle or a seasonal AR factor cancels the seasonal MA",
                ModelOrders(change.current),
                ModelOrders(change.proposed)
            ),
            Self::Unusable {
                quantity,
                index,
                value,
            } => {
                f.write_str(quantity)?;
                if let Some(index) = index {
                    write!(f, " at index {index}")?;
                }
                write!(f, " is not usable: {}", Value(*value))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_failure_says_what_failed() {
        let change = ModelChange {
            current: Orders::new([2, 1, 0], [0, 1, 1]),
            proposed: Orders::new([1, 2, 0], [0, 1, 1]),
        };
        for (failure, text) in [
            (
                Failure::NegativeVariance {
                    component: CanonicalComponent::Cycle,
                    variance: -0.25,
                },
                "the canonical decomposition gives the cycle component a negative variance \
                 (-0.25); Build 62 stops here without signal extraction",
            ),
            (
                Failure::InnovationVariance {
                    component: Component::SeasonallyAdjusted,
                    direction: VarianceDirection::AboveOne,
                    value: 1.5,
                },
                "the seasonally adjusted innovation variance 1.5 is above one, in units of the \
                 model's innovation variance; Build 62 stops here without signal extraction",
            ),
            (
                Failure::UnusableNumerator { zero_lag: -0.0 },
                "the spectral numerator of the seasonally adjusted component is not positive at \
                 lag zero (-0.0)",
            ),
            (
                Failure::DegenerateModel {
                    seasonal: true,
                    ar: vec![0.5],
                    ma: vec![0.5],
                },
                "imported seasonal AR coefficients [0.5] equal the MA coefficients [0.5], a \
                 degenerate model; Build 62 performs no signal extraction",
            ),
            (
                Failure::AutocorrelationSample { count: 16 },
                "16 values are too few for the autocorrelation diagnostics",
            ),
            (
                Failure::NoDegreesOfFreedom {
                    observed: 16,
                    consumed: 17,
                },
                "16 observations leave no residual degrees of freedom: differencing and the \
                 model's parameters use 17",
            ),
            (
                Failure::EstimationNotConverged,
                "the regARIMA estimation did not converge, so the residual Q statistic SEATS \
                 reads is undefined",
            ),
            (
                Failure::UndefinedResidualMeanCorrection {
                    residual_count: 40,
                    t: 2.5,
                },
                "the residual mean correction is undefined for 40 residuals (t = 2.5); Build 62 \
                 reads a coefficient it never set",
            ),
            (
                Failure::UnwrittenSearchCount { objective: 1e-10 },
                "the first search objective 1e-10 is below 1e-9; Build 62 then reads an \
                 iteration count it never set",
            ),
            (
                Failure::UnwrittenSearchScale {
                    gradient: [0.0, 1e-12],
                },
                "the first search gradient [0.0, 1e-12] is too small; Build 62 then reads a \
                 column scale it never set",
            ),
            (
                Failure::SingularSystem { row: 3 },
                "the linear system is singular at row 3",
            ),
            (
                Failure::RootSolverStalled { degree: 12 },
                "the root solver did not find every root of a degree-12 polynomial",
            ),
            (
                Failure::UnusableRoots {
                    step: "grouping",
                    index: Some(2),
                },
                "the polynomial roots are not usable (grouping, root 2)",
            ),
            (
                Failure::FactorCapacity {
                    factor: "stationary trend",
                    count: 9,
                    maximum: 8,
                },
                "the stationary trend factor needs 9 coefficients; SEATS holds at most 8",
            ),
            (
                Failure::ModelChange(change),
                "SEATS would re-estimate the model (2 1 0)(0 1 1) as (1 2 0)(0 1 1), as Build 62 \
                 does when a fitted AR root reaches the unit circle or a seasonal AR factor \
                 cancels the seasonal MA",
            ),
            (
                Failure::Unusable {
                    quantity: "search objective",
                    index: None,
                    value: f64::NAN,
                },
                "search objective is not usable: NaN",
            ),
            (
                Failure::Unusable {
                    quantity: "regARIMA residual",
                    index: Some(4),
                    value: f64::INFINITY,
                },
                "regARIMA residual at index 4 is not usable: inf",
            ),
        ] {
            assert_eq!(failure.to_string(), text);
        }
    }

    #[test]
    fn restage_keeps_the_failure() {
        let failure = Failure::SingularSystem { row: 1 };
        let error = failure.clone().at(Stage::Residual).restage(Stage::Tails);
        assert_eq!(
            error,
            Error {
                stage: Stage::Tails,
                failure
            }
        );
    }

    #[test]
    fn usable_names_a_nonfinite_value() {
        assert_eq!(usable("variance", None, 2.0), Ok(2.0));
        assert_eq!(
            usable("variance", Some(1), f64::NEG_INFINITY),
            Err(Failure::Unusable {
                quantity: "variance",
                index: Some(1),
                value: f64::NEG_INFINITY,
            })
        );
    }
}
