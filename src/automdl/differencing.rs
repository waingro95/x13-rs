//! Differencing identification: Build 62's `iddiff.f` rounds, which fit
//! working models to the differenced series and read their leading
//! coefficients to add regular or seasonal differences.

use crate::arima::exact::{DifferenceError, difference_series};
use crate::arima::model::ExpandedModel;
use crate::arima::{
    MeanFit, MeanFitBudget, append_reported_fit_warnings, fit_local_mean,
    fit_prepared_identification,
};
use crate::automdl::hannan_rissanen::{
    HrEstimate, HrFailure, HrFailureReason, HrOrders, estimate_prepared,
};
use crate::numerics::packed::packed_inverse;
use crate::numerics::{BUILD_62_MACHINE_EPSILON, build_62_equal};
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, Error, Warning};

use super::errors::estimation_failure;
use super::roots::{ByFactor, RootDecision, check_ar_roots};

/// A first-round AR root counts as a unit root within this modulus.
const UNIT_LIMIT: f64 = 1.0 / 0.96;

/// The differencing orders `iddiff.f` chose and its mean test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DifferencingIdentification {
    pub(crate) regular: u8,
    pub(crate) seasonal: u8,
    pub(crate) mean_significant: bool,
}

/// The cancellation limit, which each later round lowers and which a second
/// identification of the same series inherits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DifferencingState {
    cancellation_limit: f64,
}

impl Default for DifferencingState {
    fn default() -> Self {
        Self {
            cancellation_limit: 0.1,
        }
    }
}

fn working_model(round: u8, differences: ByFactor<u8>) -> Result<ArimaModel, Error> {
    let (regular_ar, regular_ma, seasonal_ar, seasonal_ma) = if round == 1 {
        (2, 0, 1, 0)
    } else {
        (1, 1, 1, 1)
    };
    ArimaModel::new(
        ArimaOrder::new(regular_ar, differences.regular, regular_ma)?,
        ArimaOrder::new(seasonal_ar, differences.seasonal, seasonal_ma)?,
    )
}

fn centered_differences(
    values: &[f64],
    model: &ExpandedModel,
) -> Result<Vec<f64>, DifferenceError> {
    let mut result = difference_series(values, model)?;
    let mut total = 0.0;
    for &value in &result {
        total += value;
    }
    let average = total / result.len() as f64;
    for value in &mut result {
        *value -= average;
    }
    Ok(result)
}

/// A round's working-model parameters before any exact fit.
struct RoundStart {
    parameters: Vec<f64>,
    /// The Hannan-Rissanen regression was singular, so the round fits the
    /// model exactly from these values.
    needs_fit: bool,
}

/// The Hannan-Rissanen estimate, or the values `iddiff.f` starts an exact fit
/// from when a regression of the estimate was singular: 0.1 throughout if
/// it was the first, the first-stage estimate if it was the MA refit.
fn round_start(
    estimate: Result<HrEstimate, HrFailure>,
    parameter_count: usize,
) -> Result<RoundStart, HrFailure> {
    match estimate {
        Ok(estimate) => Ok(RoundStart {
            parameters: estimate.parameters,
            needs_fit: false,
        }),
        Err(HrFailure::Initial(HrFailureReason::SingularRegression { .. })) => Ok(RoundStart {
            parameters: vec![0.1; parameter_count],
            needs_fit: true,
        }),
        Err(HrFailure::MaRefit {
            first_stage,
            reason: HrFailureReason::SingularRegression { .. },
        }) => Ok(RoundStart {
            parameters: first_stage.parameters,
            needs_fit: true,
        }),
        Err(error) => Err(error),
    }
}

/// The mean test of a series no difference was taken of: the t statistic of
/// its mean, skipping values equal to -999 as `iddiff.f` does.
fn no_difference_mean(values: &[f64]) -> bool {
    let mut sum = 0.0;
    let mut squares = 0.0;
    let mut included = 0.0;
    for &value in values {
        if !build_62_equal(value, -999.0) {
            sum += value;
            squares += value * value;
            included += 1.0;
        }
    }
    let average = if included > 0.0 {
        sum / included
    } else {
        -999.0
    };
    let rms = if included > 0.0 {
        (squares / included).sqrt()
    } else {
        -999.0
    };
    let statistic = (values.len() as f64).sqrt() * average / rms;
    let threshold = match values.len() {
        0..=80 => 1.96,
        81..=200 => 2.0,
        _ => 2.55,
    };
    mean_is_significant(statistic, threshold)
}

/// `iddiff.f` drops the mean only for a statistic below the threshold, which
/// a NaN is not.
fn mean_is_significant(statistic: f64, threshold: f64) -> bool {
    statistic.abs() >= threshold || statistic.is_nan()
}

/// `iddiff.f` always runs a second round, then keeps going through round 7
/// while the last one signalled anything.
fn continue_after_round(next_round: u8, signal: RoundSignal) -> bool {
    (signal != RoundSignal::Settled && next_round <= 7) || next_round == 2
}

fn fitted_mean_threshold(count: usize) -> f64 {
    match count {
        0..=80 => 1.96,
        81..=155 => 1.98,
        156..=230 => 2.1,
        231..=350 => 2.3,
        _ => 2.5,
    }
}

fn fitted_mean_statistic(coefficient: f64, factor: &[f64], variance: f64) -> f64 {
    if variance <= 2.0 * BUILD_62_MACHINE_EPSILON {
        return 0.0;
    }
    let mut inverse = factor.to_vec();
    packed_inverse(&mut inverse, 1);
    let standard_error = inverse[0].sqrt() * variance.sqrt();
    if standard_error > 0.0 {
        coefficient / standard_error
    } else {
        0.0
    }
}

fn fitted_mean_is_significant(
    fit: MeanFit,
    observation_count: usize,
    warnings: &mut Vec<Warning>,
) -> bool {
    append_reported_fit_warnings(warnings, &fit.warnings, fit.report);
    let statistic =
        fitted_mean_statistic(fit.coefficient, &fit.packed_factor, fit.innovation_variance);
    mean_is_significant(statistic, fitted_mean_threshold(observation_count))
}

/// What a round's coefficients asked of the differencing orders: the
/// combinations of `iddiff.f`'s `icon` and `irm1` that occur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundSignal {
    /// `icon = 0`: no new difference.
    Settled,
    /// `icon = 1`, `irm1` false: one new difference.
    Differenced,
    /// `icon = 1`, `irm1` true: an explosive AR coefficient. The next round
    /// refits from default starting values.
    Explosive,
    /// `icon = 2`: a regular and a seasonal difference, of which one is kept.
    BothDifferenced,
}

impl RoundSignal {
    /// `icon = icon + 1`, clearing a pending `irm1` as `iddiff.f` does
    /// wherever it adds a difference after an explosive coefficient.
    fn add_difference(self) -> Self {
        match self {
            Self::Settled | Self::Explosive => Self::Differenced,
            // Only the seasonal step adds after another difference, and
            // nothing adds after it: `icon` never passes 2.
            Self::Differenced | Self::BothDifferenced => Self::BothDifferenced,
        }
    }
}

/// The leading coefficient of each working-model factor, which `iddiff.f`
/// reads after every later round.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LeadingCoefficients {
    regular_ar: f64,
    seasonal_ar: f64,
    regular_ma: f64,
    seasonal_ma: f64,
}

/// Where each factor's coefficients sit in a working model's parameter
/// vector: regular AR, seasonal AR, regular MA, then seasonal MA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParameterLayout {
    regular_ar: usize,
    seasonal_ar: usize,
    regular_ma: usize,
}

impl ParameterLayout {
    fn new(model: ArimaModel) -> Self {
        Self {
            regular_ar: usize::from(model.regular().p()),
            seasonal_ar: usize::from(model.seasonal().p()),
            regular_ma: usize::from(model.regular().q()),
        }
    }

    /// The number of AR coefficients, which is where the MA ones start.
    fn ar_count(self) -> usize {
        self.regular_ar + self.seasonal_ar
    }

    fn ar_roots(self, parameters: &[f64]) -> Result<RootDecision, Error> {
        check_ar_roots(
            &parameters[..self.regular_ar],
            &parameters[self.regular_ar..self.ar_count()],
            UNIT_LIMIT,
        )
        .map_err(|_| estimation_failure())
    }

    fn leading(self, parameters: &[f64]) -> LeadingCoefficients {
        LeadingCoefficients {
            regular_ar: parameters[0],
            seasonal_ar: parameters[self.regular_ar],
            regular_ma: parameters[self.ar_count()],
            seasonal_ma: parameters[self.ar_count() + self.regular_ma],
        }
    }
}

/// One later round's result, handed to the next round.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RoundOutcome {
    orders: ByFactor<u8>,
    signal: RoundSignal,
    first_maxima: ByFactor<f64>,
    cancellation_limit: f64,
}

/// `iddiff.f`'s `din`: how close to 1 the regular AR coefficient must be to
/// count as a unit root. The band widens unless the round's exact fit left
/// an estimation or covariance diagnostic (`Armaer` nonzero).
///
/// The constants keep Build 62's arithmetic (`1.005 - 0.88`, widened by
/// `0.88 - 0.869`): folding them into literals would move a boundary by an ulp.
fn regular_unit_band(fit_diagnostic: bool) -> f64 {
    let band = 1.005 - 0.88;
    if fit_diagnostic {
        band
    } else {
        band + (0.88 - 0.869)
    }
}

/// Adds the difference of the factor with the larger maximum; ties go
/// seasonal.
fn difference_larger_maximum(orders: &mut ByFactor<u8>, maxima: ByFactor<f64>) {
    if maxima.regular > maxima.seasonal {
        orders.regular += 1;
    } else {
        orders.seasonal += 1;
    }
}

/// One round after the first: reads the leading coefficients of the
/// `(1 d 1)(1 D 1)` working model against the unit bands, the first round's
/// root maxima and the cancellation limit.
fn later_round(
    round: u8,
    mut orders: ByFactor<u8>,
    leading: LeadingCoefficients,
    mut first_maxima: ByFactor<f64>,
    cancellation_limit: f64,
    fit_diagnostic: bool,
) -> RoundOutcome {
    let LeadingCoefficients {
        regular_ar,
        seasonal_ar,
        regular_ma,
        seasonal_ma,
    } = leading;
    let mut signal = RoundSignal::Settled;
    // Each later round lowers the cancellation limit by 0.002, as a subtraction.
    let cancellation_limit = cancellation_limit - 0.002;
    if (1.0 - regular_ar).abs() <= regular_unit_band(fit_diagnostic) {
        if regular_ar > 1.02 {
            signal = RoundSignal::Explosive;
        } else if (regular_ar - regular_ma).abs() > cancellation_limit {
            signal = signal.add_difference();
            orders.regular += 1;
        }
    } else if regular_ar.abs() > 1.12 {
        signal = RoundSignal::Explosive;
    }
    if (1.0 - seasonal_ar).abs() <= 0.19 {
        if seasonal_ar > 1.02 {
            signal = RoundSignal::Explosive;
        } else if (seasonal_ar - seasonal_ma).abs() > cancellation_limit && orders.seasonal == 0 {
            signal = signal.add_difference();
            orders.seasonal += 1;
        }
    } else if seasonal_ar.abs() > 1.12 {
        signal = RoundSignal::Explosive;
    }
    let cancellation = (regular_ar - regular_ma).abs() <= cancellation_limit
        || (seasonal_ar - seasonal_ma).abs() <= cancellation_limit;
    if cancellation
        && round == 2
        && orders.regular == 0
        && orders.seasonal == 0
        && (first_maxima.regular >= 0.9 || first_maxima.seasonal >= 0.9)
    {
        signal = signal.add_difference();
        difference_larger_maximum(&mut orders, first_maxima);
    }
    let lchks = (1.0 - seasonal_ar).abs() <= 0.16;
    // iddiff.f assigns lchks2 only under `IF(lchks)`, and it starts true, so
    // it is always true: `<= 0.16` implies `<= 0.17`.
    let lchks2 = if lchks {
        (1.0 - seasonal_ar).abs() <= 0.17
    } else {
        true
    };
    if round == 2
        && orders.regular == 0
        && orders.seasonal == 0
        && (((1.0 - regular_ar).abs() <= 0.15 || lchks)
            && (first_maxima.regular >= 0.9 || first_maxima.seasonal >= 0.88)
            || ((1.0 - regular_ar).abs() <= 0.16 || lchks2)
                && (first_maxima.regular >= 0.91 || first_maxima.seasonal >= 0.89))
    {
        signal = signal.add_difference();
        difference_larger_maximum(&mut orders, first_maxima);
    }
    // This late near-unit difference leaves `icon` alone.
    if round >= 2
        && orders.regular == 0
        && orders.seasonal == 0
        && ((1.0 - regular_ar).abs() <= 0.15 || (1.0 - seasonal_ar).abs() <= 0.25)
    {
        first_maxima = ByFactor::new(regular_ar, seasonal_ar);
        difference_larger_maximum(&mut orders, first_maxima);
    }
    if signal == RoundSignal::BothDifferenced {
        orders.regular -= 1;
        orders.seasonal -= 1;
        if fit_diagnostic {
            if regular_ar.abs() > seasonal_ar.abs() {
                orders.regular += 1;
            } else {
                orders.seasonal += 1;
            }
        } else if first_maxima.regular > first_maxima.seasonal {
            // `iddiff.f` tests the larger maximum against -9999, though a
            // factor without a maximum holds -999.
            if first_maxima.regular > -9999.0 {
                orders.regular += 1;
            }
        } else if first_maxima.seasonal > -9999.0 {
            orders.seasonal += 1;
        }
    }
    RoundOutcome {
        orders,
        signal,
        first_maxima,
        cancellation_limit,
    }
}

impl DifferencingState {
    /// Chooses the differencing orders of `values`, the series with its
    /// regression effects removed, and tests its mean. The fits' warnings go
    /// to `warnings`.
    pub(crate) fn identify_prepared(
        &mut self,
        values: &[f64],
        axis: TimeAxis,
        warnings: &mut Vec<Warning>,
    ) -> Result<DifferencingIdentification, Error> {
        let mut differences = ByFactor::new(0, 0);
        let mut round = 1;
        let mut previous_signal = RoundSignal::Settled;
        let mut first_maxima = ByFactor::new(0.0, 0.0);
        let mut budget = MeanFitBudget::Exact1500;
        let last_model = loop {
            let model = working_model(round, differences)?;
            let expanded = ExpandedModel::from_spec(model, axis.frequency());
            let mut prepared =
                centered_differences(values, &expanded).map_err(|_| estimation_failure())?;
            let hr_orders = HrOrders::new(
                model.regular().p(),
                model.seasonal().p(),
                model.regular().q(),
                model.seasonal().q(),
            );
            let estimate = estimate_prepared(&prepared, axis.frequency(), hr_orders);
            prepared.clear();
            let start = round_start(estimate, expanded.parameter_slots().len())
                .map_err(|_| estimation_failure())?;
            let mut parameters = start.parameters;
            let mut needs_fit = start.needs_fit;
            let layout = ParameterLayout::new(model);
            let mut root_counts = ByFactor::new(0, 0);
            if round == 1 && !needs_fit {
                let decision = layout.ar_roots(&parameters)?;
                root_counts = decision.unit_counts;
                first_maxima = decision.maxima;
                if !decision.all_invertible
                    && (layout.regular_ar > 1
                        || (layout.regular_ar == 1 && parameters[0].abs() > 1.02)
                        || (layout.seasonal_ar == 1 && parameters[layout.regular_ar].abs() > 1.02))
                {
                    needs_fit = true;
                    parameters[..layout.ar_count()].fill(0.1);
                }
            } else if needs_fit {
                parameters[..layout.ar_count()].fill(0.1);
                if round > 1 {
                    parameters[layout.ar_count()..].fill(0.2);
                }
            }
            if previous_signal == RoundSignal::Explosive {
                needs_fit = true;
                parameters.fill(0.1);
            }
            if differences.regular == 0 && differences.seasonal == 0 && round > 2 {
                needs_fit = true;
            }
            let mut fit_diagnostic = false;
            if needs_fit {
                let fit = fit_prepared_identification(values, axis, model, parameters)?;
                // The mean test runs under the iteration limit this fit
                // leaves: 200, or 1500 after a conditional retry.
                budget = if fit.retried_conditional {
                    MeanFitBudget::Exact1500
                } else {
                    MeanFitBudget::Exact200
                };
                parameters = fit.parameters;
                fit_diagnostic = fit.has_diagnostic;
                append_reported_fit_warnings(warnings, &fit.warnings, fit.report);
                if round == 1 {
                    let decision = layout.ar_roots(&parameters)?;
                    root_counts = decision.unit_counts;
                    first_maxima = decision.maxima;
                }
            }

            let mut signal = RoundSignal::Settled;
            if round == 1 {
                differences.regular += root_counts.regular;
                differences.seasonal += root_counts.seasonal;
            } else {
                let outcome = later_round(
                    round,
                    differences,
                    layout.leading(&parameters),
                    first_maxima,
                    self.cancellation_limit,
                    fit_diagnostic,
                );
                differences = outcome.orders;
                signal = outcome.signal;
                first_maxima = outcome.first_maxima;
                self.cancellation_limit = outcome.cancellation_limit;
            }
            round += 1;
            if differences.regular == 3 {
                differences.regular = 2;
                signal = RoundSignal::Settled;
            }
            if differences.seasonal == 2 {
                differences.seasonal = 1;
                signal = RoundSignal::Settled;
            }
            if differences.regular > 2 {
                differences.regular = 2;
                break model;
            }
            if differences.seasonal > 1 {
                differences.seasonal = 1;
                break model;
            }
            // The caps clear `icon` but not `irm1`. A capped round goes on
            // only if it was round 1, which never sets `irm1`.
            if continue_after_round(round, signal) {
                previous_signal = signal;
                continue;
            }
            break model;
        };
        let total_lag = usize::from(differences.regular)
            + usize::from(differences.seasonal) * usize::from(axis.frequency().periods_per_year());
        let mean_significant = if total_lag == 0 {
            no_difference_mean(values)
        } else {
            let fit = fit_local_mean(values, axis, last_model, budget)?;
            fitted_mean_is_significant(fit, values.len(), warnings)
        };
        Ok(DifferencingIdentification {
            regular: differences.regular,
            seasonal: differences.seasonal,
            mean_significant,
        })
    }
}

#[cfg(test)]
mod tests;
