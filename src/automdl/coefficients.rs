//! The final coefficient check: `chkurt.f`'s near-unit-root veto and
//! `tstmd2.f`'s choice of an insignificant ARMA coefficient to drop.

use super::errors::estimation_failure;
use crate::Error;
use crate::arima::roots::{DetailedRoots, detailed_roots};
use crate::error::NumericalOperation;
use crate::regarima::ResolvedArimaModel;

/// The variances of the ARMA parameters: `innovation_variance * diagonal[i]`
/// for parameter `i`.
pub(crate) struct ArmaVariances<'a> {
    pub(crate) innovation_variance: f64,
    pub(crate) diagonal: &'a [f64],
}

pub(crate) struct CoefficientCheck {
    /// How many operators end in an insignificant coefficient, counted from
    /// the one with the smallest |t| on.
    pub(crate) insignificant_count: u8,
    /// The model without the least significant of them.
    pub(crate) reduced_model: Option<ResolvedArimaModel>,
}

/// The four ARMA operators in the order `chkurt.f` and `tstmd2.f` walk them,
/// which is also their order in the parameter vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    RegularAr,
    SeasonalAr,
    RegularMa,
    SeasonalMa,
}

impl Operator {
    const ORDER: [Self; 4] = [
        Self::RegularAr,
        Self::SeasonalAr,
        Self::RegularMa,
        Self::SeasonalMa,
    ];

    fn order(self, model: ResolvedArimaModel) -> usize {
        usize::from(match self {
            Self::RegularAr => model.regular().p(),
            Self::SeasonalAr => model.seasonal().p(),
            Self::RegularMa => model.regular().q(),
            Self::SeasonalMa => model.seasonal().q(),
        })
    }

    /// The model with this operator's order lowered by one.
    fn reduced(self, model: ResolvedArimaModel) -> Result<ResolvedArimaModel, Error> {
        let regular = model.regular();
        let seasonal = model.seasonal();
        let mut regular = [regular.p(), regular.d(), regular.q()];
        let mut seasonal = [seasonal.p(), seasonal.d(), seasonal.q()];
        match self {
            Self::RegularAr => regular[0] -= 1,
            Self::SeasonalAr => seasonal[0] -= 1,
            Self::RegularMa => regular[2] -= 1,
            Self::SeasonalMa => seasonal[2] -= 1,
        }
        ResolvedArimaModel::new(regular, seasonal)
    }
}

/// Checks the last coefficient of each ARMA operator of `model`.
/// `variances` is `None` when the ARMA covariance matrix is singular, which
/// is an error once the roots have been computed.
pub(crate) fn check_coefficients(
    model: ResolvedArimaModel,
    observed_count: usize,
    parameters: &[f64],
    variances: Option<ArmaVariances<'_>>,
) -> Result<CoefficientCheck, Error> {
    let mut end = 0;
    let spans = Operator::ORDER.map(|operator| {
        let start = end;
        end += operator.order(model);
        start..end
    });
    debug_assert_eq!(parameters.len(), end);

    // `chkurt.f` walks every operator's roots before `armats.f` looks at the
    // covariance.
    let mut near_unit = [false; 4];
    for (span, near_unit) in spans.iter().zip(&mut near_unit) {
        if !span.is_empty() {
            let roots =
                detailed_roots(&parameters[span.clone()]).map_err(|_| estimation_failure())?;
            *near_unit = has_near_unit_root(&roots);
        }
    }

    let Some(variances) = variances else {
        return Err(Error::numerical(NumericalOperation::Covariance));
    };
    debug_assert_eq!(variances.diagonal.len(), parameters.len());
    let t_values = parameters
        .iter()
        .zip(variances.diagonal)
        .map(|(&parameter, &diagonal)| t_value(parameter, variances.innovation_variance, diagonal))
        .collect::<Vec<_>>();

    // `tstmd2.f` checks only the last coefficient of each nonempty operator.
    // It flags each insignificant one, and a new smallest |t| clears the
    // flags of the operators before it: the count is the flags from the
    // selected operator on. A flagged |t| is below 1, so the first one always
    // selects.
    let cutoff = if observed_count <= 150 { 0.15 } else { 0.1 };
    let mut smallest = 999.0; // `DNOTST` in `notset.prm`.
    let mut selected = None;
    let mut insignificant_count = 0;
    for ((operator, span), near_unit) in Operator::ORDER.into_iter().zip(&spans).zip(near_unit) {
        if span.is_empty() || near_unit {
            continue;
        }
        let last = span.end - 1;
        let magnitude = t_values[last].abs();
        if magnitude < 1.0 && parameters[last].abs() < cutoff {
            if smallest > magnitude {
                smallest = magnitude;
                selected = Some(operator);
                insignificant_count = 1;
            } else {
                insignificant_count += 1;
            }
        }
    }
    // A near-unit root in any operator vetoes the reduction, and a model's
    // only parameter is never dropped.
    if parameters.len() == 1 || near_unit.contains(&true) {
        insignificant_count = 0;
    }
    let reduced_model = match selected {
        Some(operator) if insignificant_count > 0 => Some(operator.reduced(model)?),
        _ => None,
    };
    Ok(CoefficientCheck {
        insignificant_count,
        reduced_model,
    })
}

/// `armats.f`'s t-value: the variance is scaled before the square root.
fn t_value(parameter: f64, innovation_variance: f64, diagonal: f64) -> f64 {
    parameter / (innovation_variance * diagonal).sqrt()
}

/// Whether `chkurt.f` counts a root as near the unit circle: a modulus of at
/// most 1/0.95.
fn has_near_unit_root(details: &DetailedRoots) -> bool {
    let limit = 1.0 / 0.95;
    details.roots.iter().any(|root| root.modulus <= limit)
}

#[cfg(test)]
mod tests;
