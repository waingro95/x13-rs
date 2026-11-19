//! The automatic outlier search (`idotlr.f`): the candidate t-tests
//! (`ttest.f`), forward addition and backward deletion (`deltst.f`).

use crate::api::ResolvedRegArimaSpec;
use crate::arima::exact::{FilterOptions, exact_filter};
use crate::arima::model::ExpandedModel;
use crate::arima::{ArmaStart, EstimatedRegArima, FitContext, FittedNumericalView, estimate};
use crate::error::NumericalDetail;
use crate::numerics::BUILD_62_MIN_POSITIVE;
use crate::numerics::RowMajorMatrix;
use crate::numerics::blas1::dot;
use crate::numerics::build_62_equal;
use crate::numerics::packed::{packed_column, packed_diagonal, packed_inverse, packed_len};
use crate::outliers::candidates::{Candidate, candidate_is_eligible, candidates};
use crate::outliers::critical::{OutlierCutoffs, critical_value, robust_scale};
use crate::outliers::{outlier_failure, outlier_failure_from};
use crate::regarima::{OutlierMethod, OutlierSpec, OutlierType, OutlierTypes};
use crate::regression::design::RegressionMatrix;
use crate::regression::gls::OutlierTestState;
use crate::regression::model::{RegressionModel, RegressorKind};
use crate::time::TimeAxis;
use crate::{Error, Period};

mod automatic;
mod fit;
mod warnings;
pub(crate) use automatic::{
    ClearedRegressionFit, search_automatic, search_automatic_cleared_input,
    search_automatic_finalized_input, search_automatic_residuals,
};
use fit::SearchFit;
use warnings::SearchWarningState;

/// One `ttest.f` result before division by the robust residual scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CandidateStatistic {
    pub(crate) kind: OutlierType,
    /// This candidate's position in the `kinds` given to
    /// [`candidate_statistics`], which its sort does not preserve.
    pub(crate) column: usize,
    pub(crate) proportional: f64,
    pub(crate) singular: bool,
}

/// Build 62 `ttest.f`: tests one date's filtered candidate columns, given in
/// AO, LS, TC order, and returns them by descending magnitude.
///
/// `filtered_candidates` is row-major: one row per observation, one column per
/// entry of `kinds`.
pub(crate) fn candidate_statistics(
    state: OutlierTestState<'_>,
    kinds: &[OutlierType],
    filtered_candidates: &[f64],
) -> Vec<CandidateStatistic> {
    debug_assert!(!kinds.is_empty() && kinds.len() <= 3);
    let matrix = state.filtered_matrix().as_matrix();
    let column_count = matrix.column_count();
    debug_assert_eq!(filtered_candidates.len(), matrix.row_count() * kinds.len());

    let factor = state.packed_factor();
    let regression_count = column_count - 1;
    let response_factor = &factor[packed_column(regression_count)];
    let underflow_cutoff = BUILD_62_MIN_POSITIVE.sqrt();
    let candidate_rows = filtered_candidates.chunks_exact(kinds.len());

    let mut candidate_norms = vec![0.0; kinds.len()];
    for row in candidate_rows.clone() {
        for (norm, value) in candidate_norms.iter_mut().zip(row) {
            if value.abs() > underflow_cutoff {
                *norm = value.mul_add(*value, *norm);
            }
        }
    }

    // `ttest.f` loops column → row → candidate. Each (candidate, column)
    // cross product is its own accumulator, fed rows in order, so walking the
    // rows outermost runs the same `mul_add`s on each one in the same order.
    let mut cross_products = vec![vec![0.0; column_count]; kinds.len()];
    for (matrix_row, candidate_row) in matrix.rows().zip(candidate_rows) {
        for (products, candidate_value) in cross_products.iter_mut().zip(candidate_row) {
            for (product, matrix_value) in products.iter_mut().zip(matrix_row) {
                *product = matrix_value.mul_add(*candidate_value, *product);
            }
        }
    }

    // Forward-solve each candidate's regressor cross products through the
    // packed factor, column by column, folding each solved term into the
    // response product and the candidate norm.
    let mut response_products = cross_products
        .iter()
        .map(|products| products[regression_count])
        .collect::<Vec<_>>();
    for column in 0..regression_count {
        let (above_diagonal, diagonal) = factor[packed_column(column)].split_at(column);
        let diagonal = diagonal[0];
        for ((products, response_product), norm) in cross_products
            .iter_mut()
            .zip(&mut response_products)
            .zip(&mut candidate_norms)
        {
            let (solved_columns, current) = products.split_at_mut(column);
            let solved = (current[0] - dot(above_diagonal, solved_columns)) / diagonal;
            current[0] = solved;
            *response_product = (-response_factor[column]).mul_add(solved, *response_product);
            *norm = (-solved).mul_add(solved, *norm);
        }
    }

    let mut statistics = kinds
        .iter()
        .zip(response_products.iter().zip(&candidate_norms))
        .enumerate()
        .map(|(column, (&kind, (&response_product, &norm)))| {
            let singular = norm <= 0.0;
            CandidateStatistic {
                kind,
                column,
                proportional: if singular {
                    0.0
                } else {
                    response_product / norm.sqrt()
                },
                singular,
            }
        })
        .collect::<Vec<_>>();
    sort_by_descending_magnitude(&mut statistics);
    statistics
}

/// `ttest.f`'s insertion sort by descending `|proportional|`.
///
/// Its strict comparison keeps exact ties in the caller's AO/LS/TC order, and
/// a NaN statistic, false against everything, ends every insertion that
/// meets it. A library sort would order ties or NaN differently, so the loop
/// stays.
fn sort_by_descending_magnitude(statistics: &mut [CandidateStatistic]) {
    for index in 1..statistics.len() {
        let current = statistics[index];
        let mut target = index;
        while target > 0 && statistics[target - 1].proportional.abs() < current.proportional.abs() {
            statistics[target] = statistics[target - 1];
            target -= 1;
        }
        statistics[target] = current;
    }
}

/// Build 62's `sign(propt**2, propt)` starting coefficient.
pub(crate) fn initial_candidate_coefficient(proportional: f64) -> f64 {
    let squared = proportional * proportional;
    squared.copysign(proportional)
}

/// An outlier the search added to the regression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SelectedOutlier {
    pub(crate) kind: OutlierType,
    /// Zero-based observation of `period` in the model span.
    pub(crate) index: usize,
    pub(crate) period: Period,
    /// [`initial_candidate_coefficient`] of the statistic that selected it.
    pub(crate) initial_coefficient: f64,
}

/// The last fit of a search, its regression model and the outliers in it,
/// ordered by date and then type.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OutlierSearchResult<F = EstimatedRegArima> {
    pub(crate) fit: F,
    pub(crate) regression_model: RegressionModel,
    pub(crate) selected: Vec<SelectedOutlier>,
}

/// Whether the forward pass still tests a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateStatus {
    Eligible,
    /// Outside the search span, at an endpoint or missing-value date, or
    /// singular against the regression.
    Excluded,
    Selected,
}

/// Where a search starts: from starting values for its first fit, or from a
/// fit and the outliers already in its regression.
enum SearchInitial<F = EstimatedRegArima> {
    Explicit(ArmaStart),
    Automatic {
        fit: Box<F>,
        selected: Vec<SelectedOutlier>,
    },
}

/// What stays fixed while one outlier search refits.
#[derive(Clone, Copy)]
struct SearchProblem {
    observation_count: usize,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    outlier_spec: OutlierSpec,
}

/// A candidate whose statistic exceeds its cutoff.
#[derive(Debug, Clone, Copy)]
struct QualifyingCandidate {
    generated_index: usize,
    proportional: f64,
}

/// Fits `series` and searches it for outliers under `spec`'s outlier
/// settings, starting the first fit from `start`.
pub(crate) fn search(
    series: &[f64],
    original_series: FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
) -> Result<OutlierSearchResult, Error> {
    let outlier_spec = spec.outliers().ok_or(Error::internal(
        "outlier search ran without an outlier spec",
    ))?;
    let problem = SearchProblem {
        observation_count: series.len(),
        axis,
        spec,
        outlier_spec,
    };
    search_with_initial_fit(
        problem,
        regression_model,
        SearchInitial::Explicit(start),
        |regression, start| estimate(series, original_series, axis, spec, regression, start),
    )
}

/// Build 62 `idotlr.f`: add qualifying candidates until none qualifies, then
/// delete the weakest selected outlier while it falls below its cutoff,
/// calling `refit` after every change.
///
/// The explicit search and automdl's entry points share this loop. Each
/// passes a `refit` closure that fits its series with a given regression model
/// and starting values. `idotlr.f` also prints a per-pass table and retests
/// "almost" outliers against cutoffs lowered by `Cvrduc`; neither changes the
/// selection, so neither is computed.
fn search_with_initial_fit<F: SearchFit>(
    problem: SearchProblem,
    mut regression_model: RegressionModel,
    initial: SearchInitial<F>,
    mut refit: impl FnMut(RegressionModel, ArmaStart) -> Result<F, Error>,
) -> Result<OutlierSearchResult<F>, Error> {
    let SearchProblem {
        observation_count,
        axis,
        spec,
        outlier_spec,
    } = problem;
    let automatic_mode = matches!(initial, SearchInitial::Automatic { .. });
    let search_span = outlier_spec.resolved_span(axis, observation_count)?;
    let expanded = ExpandedModel::from_spec(spec.model(), axis.frequency());
    // Build 62's one `Tcalfa` (makotl.f), whichever policy drives this search.
    let temporary_change_rate = spec.resolved_temporary_change_rate(axis.frequency());
    let generated = candidates(
        outlier_spec.types(),
        axis,
        observation_count,
        temporary_change_rate,
    )?;
    let mut statuses = generated
        .iter()
        .map(|candidate| {
            // idotlr excludes AO, LS and TC candidates at every MV date.
            if regression_model.kinds().any(|kind| {
                matches!(kind, RegressorKind::MissingValue { period, .. }
                    if period == candidate.period)
            }) {
                return CandidateStatus::Excluded;
            }
            if let SearchInitial::Automatic { selected, .. } = &initial
                && selected.iter().any(|outlier| {
                    outlier.kind == candidate.kind && outlier.period == candidate.period
                })
            {
                return CandidateStatus::Selected;
            }
            // idotlr.f restricts candidate dates, while endpoint exclusions
            // and candidate columns remain relative to the whole model span.
            if search_span.contains(&candidate.index)
                && candidate_is_eligible(
                    outlier_spec.types(),
                    candidate.kind,
                    candidate.index,
                    observation_count,
                )
            {
                CandidateStatus::Eligible
            } else {
                CandidateStatus::Excluded
            }
        })
        .collect::<Vec<_>>();
    let cutoffs = resolve_cutoffs(outlier_spec, axis, observation_count)?;
    let (mut fit, mut selected) = match initial {
        SearchInitial::Explicit(start) => (refit(regression_model.clone(), start)?, Vec::new()),
        SearchInitial::Automatic { fit, selected } => (*fit, selected),
    };
    let mut warning_history = Vec::new();
    let mut warning_state = if automatic_mode {
        SearchWarningState::AutomaticIncoming
    } else {
        SearchWarningState::Explicit
    };

    loop {
        let qualifying = test_candidates(
            fit.numerical_view(),
            &regression_model,
            &expanded,
            &generated,
            &mut statuses,
            &cutoffs,
            observation_count,
        )?;
        let additions = match outlier_spec.method() {
            OutlierMethod::AddOne => largest_qualifying(&qualifying).into_iter().collect(),
            OutlierMethod::AddAll => qualifying,
        };
        if additions.is_empty() {
            break;
        }

        regression_model.ensure_additional_capacity(additions.len())?;
        for addition in additions {
            let candidate = &generated[addition.generated_index];
            let initial_coefficient = initial_candidate_coefficient(addition.proportional);
            statuses[addition.generated_index] = CandidateStatus::Selected;
            insert_selected(
                &mut selected,
                SelectedOutlier {
                    kind: candidate.kind,
                    index: candidate.index,
                    period: candidate.period,
                    initial_coefficient,
                },
            );
            regression_model.insert_outlier(
                candidate.kind,
                candidate.period,
                axis.frequency(),
                candidate.values.clone(),
                temporary_change_rate,
            )?;
        }
        let start = fit.numerical_view().refit_start();
        warning_state.before_refit(&mut warning_history, &mut fit);
        fit = refit(regression_model.clone(), start)?;
    }

    let effective_observations =
        expanded
            .effective_observations(observation_count)
            .ok_or(Error::internal(
                "outlier search series is shorter than the differencing",
            ))?;
    while let Some(deletion) =
        deletion_candidate(fit.numerical_view(), &cutoffs, effective_observations)?
    {
        let removed = regression_model.remove_outlier(deletion.kind, deletion.period);
        debug_assert!(removed, "the fit's outliers are this model's columns");
        selected
            .retain(|outlier| outlier.kind != deletion.kind || outlier.period != deletion.period);
        let start = fit.numerical_view().refit_start();
        warning_state.before_refit(&mut warning_history, &mut fit);
        fit = refit(regression_model.clone(), start)?;
    }

    warning_state.finish(&mut warning_history, &mut fit);
    *fit.warnings_mut() = warning_history;
    Ok(OutlierSearchResult {
        fit,
        regression_model,
        selected,
    })
}

/// `idotlr.f`'s AddOne pick: the largest `|proportional|`, keeping the
/// earliest date on a tie.
fn largest_qualifying(qualifying: &[QualifyingCandidate]) -> Option<QualifyingCandidate> {
    qualifying.iter().copied().reduce(|largest, candidate| {
        if largest.proportional.abs() >= candidate.proportional.abs() {
            largest
        } else {
            candidate
        }
    })
}

/// One forward pass: tests every eligible candidate against `fit`, date by
/// date, and returns each date's largest statistic that exceeds its cutoff
/// times the robust residual scale. Singular candidates become excluded.
fn test_candidates(
    fit: FittedNumericalView<'_>,
    regression_model: &RegressionModel,
    expanded: &ExpandedModel,
    generated: &[Candidate],
    statuses: &mut [CandidateStatus],
    cutoffs: &OutlierCutoffs,
    observation_count: usize,
) -> Result<Vec<QualifyingCandidate>, Error> {
    let effective_observations =
        expanded
            .effective_observations(observation_count)
            .ok_or(Error::internal(
                "outlier search series is shorter than the differencing",
            ))?;
    let effective_start = fit.residuals.len() - effective_observations;
    let scale = robust_scale(&fit.residuals[effective_start..])?;

    let matrix = regression_model.matrix(fit.transformed)?;
    let mut workspace = fit.filter_workspace.clone();
    let filtered = exact_filter(
        matrix.into_matrix(),
        expanded,
        fit.parameters,
        &mut workspace,
        FilterOptions::OUTLIER_TEST,
    )
    .map_err(outlier_failure_from)?;
    let filtered_matrix = RegressionMatrix::from(filtered.into_matrix());
    let state = OutlierTestState::new(&filtered_matrix, fit.outlier_test_factor);
    let mut qualifying = Vec::new();
    let mut candidate_indices = Vec::with_capacity(3);
    let mut kinds = Vec::with_capacity(3);
    let mut candidate_values = Vec::with_capacity(observation_count * 3);
    let mut next_candidate = 0;
    debug_assert!(
        generated
            .windows(2)
            .all(|pair| pair[0].index <= pair[1].index)
    );

    for index in 0..observation_count {
        candidate_indices.clear();
        kinds.clear();
        // IDOTLR tests dates in order, and generated candidates are already
        // date-major AO/LS/TC. Walk them once, in that order, rather than
        // scanning every date's candidates again.
        while next_candidate < generated.len() && generated[next_candidate].index <= index {
            if generated[next_candidate].index == index
                && statuses[next_candidate] == CandidateStatus::Eligible
            {
                candidate_indices.push(next_candidate);
                kinds.push(generated[next_candidate].kind);
            }
            next_candidate += 1;
        }
        if candidate_indices.is_empty() {
            continue;
        }
        candidate_values.clear();
        for row in 0..observation_count {
            candidate_values.extend(
                candidate_indices
                    .iter()
                    .map(|&candidate| generated[candidate].values[row]),
            );
        }
        let filtered_candidates = exact_filter(
            RowMajorMatrix::new(
                std::mem::take(&mut candidate_values),
                candidate_indices.len(),
            ),
            expanded,
            fit.parameters,
            &mut workspace,
            FilterOptions::OUTLIER_TEST,
        )
        .map_err(outlier_failure_from)?;
        let statistics =
            candidate_statistics(state, &kinds, filtered_candidates.matrix().as_slice());
        // Reuse the filtered buffer for the next date's candidate columns.
        candidate_values = filtered_candidates.into_matrix().into_values();

        for statistic in statistics.iter().filter(|statistic| statistic.singular) {
            statuses[candidate_indices[statistic.column]] = CandidateStatus::Excluded;
        }
        if let Some(statistic) = statistics.iter().find(|statistic| {
            !statistic.singular
                && cutoffs[statistic.kind]
                    .is_some_and(|cutoff| statistic.proportional.abs() > cutoff * scale)
        }) {
            qualifying.push(QualifyingCandidate {
                generated_index: candidate_indices[statistic.column],
                proportional: statistic.proportional,
            });
        }
    }

    Ok(qualifying)
}

#[derive(Debug, Clone, Copy)]
struct DeletionCandidate {
    kind: OutlierType,
    period: Period,
    t_value: f64,
}

/// Build 62 `deltst.f`: the outlier regressor to delete, if any. Each type's
/// smallest `|t|` is ranked, and the first below its cutoff is returned.
fn deletion_candidate(
    fit: FittedNumericalView<'_>,
    cutoffs: &OutlierCutoffs,
    effective_observations: usize,
) -> Result<Option<DeletionCandidate>, Error> {
    if !fit
        .regressor_kinds
        .iter()
        .any(|kind| matches!(kind, RegressorKind::Outlier { .. }))
    {
        return Ok(None);
    }

    // The fit's factor packs `[X:y]`: the regressor triangle, then the
    // response column with its diagonal last.
    let regression_count = fit.regressor_kinds.len();
    debug_assert_eq!(fit.regression_coefficients.len(), regression_count);
    debug_assert_eq!(
        fit.outlier_test_factor.len(),
        packed_len(regression_count + 1)
    );

    let response_diagonal = fit.outlier_test_factor[packed_diagonal(regression_count)];
    if build_62_equal(response_diagonal, 0.0) {
        return Err(Error::from(NumericalDetail::ZeroOutlierDeletionScale));
    }
    let rmse = response_diagonal / (effective_observations as f64).sqrt();
    let mut normal_inverse = fit.outlier_test_factor[..packed_len(regression_count)].to_vec();
    packed_inverse(&mut normal_inverse, regression_count);

    // Each type keeps its smallest `|t|`; a later outlier replaces an equal one.
    let mut minima: [Option<DeletionCandidate>; 3] = [None; 3];
    for (regressor_index, (kind, coefficient)) in fit
        .regressor_kinds
        .iter()
        .zip(fit.regression_coefficients)
        .enumerate()
    {
        let RegressorKind::Outlier { kind, period, .. } = *kind else {
            continue;
        };
        let t_value = coefficient / normal_inverse[packed_diagonal(regressor_index)].sqrt() / rmse;
        // Build 62's ranking below has no place for a nonfinite statistic.
        if !t_value.is_finite() {
            return Err(outlier_failure());
        }
        let candidate = DeletionCandidate {
            kind,
            period,
            t_value,
        };
        let slot = &mut minima[kind.build_62_index()];
        if slot.is_none_or(|current| candidate.t_value.abs() <= current.t_value.abs()) {
            *slot = Some(candidate);
        }
    }

    // Build 62 ranks the type minima with a strict insertion sort. The
    // t-values are finite here, so `total_cmp` on their magnitudes agrees with
    // it, and the stable sort keeps cross-type ties in AO, LS, TC order.
    let mut ranked = minima.into_iter().flatten().collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.t_value.abs().total_cmp(&right.t_value.abs()));
    Ok(ranked.into_iter().find(|candidate| {
        cutoffs[candidate.kind].is_some_and(|cutoff| candidate.t_value.abs() < cutoff)
    }))
}

/// The cutoff of each outlier type `outlier_spec` tests.
pub(crate) fn resolve_cutoffs(
    outlier_spec: OutlierSpec,
    axis: TimeAxis,
    observation_count: usize,
) -> Result<OutlierCutoffs, Error> {
    // arima.f feeds setcv the inclusive outlier test-span length, even when
    // estimation and the outlier regression columns cover a longer model span.
    let search_count = outlier_spec.resolved_span(axis, observation_count)?.len();
    let mut result = OutlierCutoffs::NONE;
    for kind in enabled_types(outlier_spec.types()) {
        result[kind] = Some(critical_value(
            outlier_spec.critical_values(),
            kind,
            search_count,
        )?);
    }
    Ok(result)
}

/// Inserts `outlier` at its place in date-then-type order.
fn insert_selected(selected: &mut Vec<SelectedOutlier>, outlier: SelectedOutlier) {
    let target = selected
        .iter()
        .position(|existing| {
            (outlier.period, outlier.kind.build_62_index())
                < (existing.period, existing.kind.build_62_index())
        })
        .unwrap_or(selected.len());
    selected.insert(target, outlier);
}

/// The zero-based observation at `period`, if `count` periods of `axis`
/// include it.
fn period_index(axis: TimeAxis, count: usize, period: Period) -> Option<usize> {
    axis.periods(count)
        .ok()?
        .position(|candidate| candidate == period)
}

fn enabled_types(types: OutlierTypes) -> impl Iterator<Item = OutlierType> {
    OutlierType::BUILD_62_ORDER
        .into_iter()
        .filter(move |kind| types.contains(*kind))
}

#[cfg(test)]
mod tests {
    use super::{
        OutlierCutoffs, QualifyingCandidate, candidate_statistics, deletion_candidate,
        initial_candidate_coefficient, largest_qualifying, search,
    };
    use crate::arima::{ArmaStart, estimate};
    use crate::regarima::{
        OutlierCriticalValue, OutlierCriticalValues, OutlierMethod, OutlierSpec, OutlierType,
        OutlierTypes, RegressionVariable,
    };
    use crate::regression::design::RegressionMatrix;
    use crate::regression::gls::{OutlierTestState, solve_filtered_regression};
    use crate::regression::model::{RegressionColumn, RegressionModel, RegressorKind};
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, RegArimaSpec};

    const fn shared(cutoff: f64) -> OutlierCutoffs {
        OutlierCutoffs::in_build_62_order([Some(cutoff); 3])
    }

    #[test]
    fn addone_picks_the_first_largest_magnitude_and_follows_nan_comparisons() {
        let picked = |values: &[f64]| {
            let qualifying = values
                .iter()
                .enumerate()
                .map(|(generated_index, &proportional)| QualifyingCandidate {
                    generated_index,
                    proportional,
                })
                .collect::<Vec<_>>();
            largest_qualifying(&qualifying).map(|candidate| candidate.generated_index)
        };
        assert_eq!(picked(&[]), None);
        assert_eq!(picked(&[2.0, -3.0, 3.0, 1.0]), Some(1));
        assert_eq!(picked(&[-0.0, 0.0]), Some(0));
        // `>=` against NaN is false either way, so a NaN is replaced by the
        // next candidate and replaces the current one.
        assert_eq!(picked(&[f64::NAN, 1.0]), Some(1));
        assert_eq!(picked(&[5.0, f64::NAN]), Some(1));
        assert_eq!(picked(&[5.0, f64::NAN, 1.0]), Some(2));
    }

    /// A fit whose factor, coefficients and regressors are replaced by the
    /// ones a deletion test should read.
    fn fit_with_deletion_state(
        factor: Vec<f64>,
        coefficients: Vec<f64>,
        kinds: Vec<RegressorKind>,
    ) -> crate::arima::EstimatedRegArima {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let model = ArimaModel::new(
            ArimaOrder::new(0, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap();
        let mut fit = estimate(
            &[1.0, 2.0, 3.0, 4.0],
            (&[1.0, 2.0, 3.0, 4.0]).into(),
            axis,
            RegArimaSpec::new(model).resolve_explicit().unwrap(),
            RegressionModel::new(),
            ArmaStart::Default,
        )
        .unwrap();
        fit.outlier_test_factor = factor;
        fit.regression_coefficients = coefficients;
        fit.regression_standard_errors = Some(Vec::new());
        fit.regressor_kinds = kinds;
        fit
    }

    fn outlier_kind(kind: OutlierType, period: crate::Period) -> RegressorKind {
        RegressorKind::Outlier {
            kind,
            period,
            frequency: TimeAxis::quarterly(2000, 1).unwrap().frequency(),
        }
    }

    #[test]
    fn deletion_statistics_come_from_the_factor_not_the_reported_standard_errors() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let fit = fit_with_deletion_state(
            vec![1.1, 0.2, 1.3, -0.4, 0.6, 1.7, 91.0, 92.0, 93.0, 13.7],
            vec![1.7, 1.7, 1.7],
            vec![
                outlier_kind(OutlierType::Additive, axis.period_at(1).unwrap()),
                outlier_kind(OutlierType::LevelShift, axis.period_at(2).unwrap()),
                outlier_kind(OutlierType::TemporaryChange, axis.period_at(3).unwrap()),
            ],
        );

        for (cutoffs, kind, expected_bits) in [
            (
                [Some(2.0), Some(2.0), Some(2.0)],
                OutlierType::Additive,
                0x3ff2_35bd_8df7_44b6,
            ),
            (
                [Some(1.0), Some(2.0), Some(2.0)],
                OutlierType::LevelShift,
                0x3ff5_5b71_636c_5793,
            ),
            (
                [Some(1.0), Some(1.0), Some(2.0)],
                OutlierType::TemporaryChange,
                0x3ffd_9dfa_a9f8_8e50,
            ),
        ] {
            let cutoffs = OutlierCutoffs::in_build_62_order(cutoffs);
            let actual = deletion_candidate(fit.numerical_view(), &cutoffs, 77)
                .unwrap()
                .unwrap();
            assert_eq!(actual.kind, kind);
            assert_eq!(actual.t_value.to_bits(), expected_bits);
        }

        assert_eq!(
            deletion_candidate(
                fit.numerical_view(),
                &OutlierCutoffs::in_build_62_order([
                    Some(f64::from_bits(0x3ff2_35bd_8df7_44b6)),
                    Some(f64::from_bits(0x3ff5_5b71_636c_5793)),
                    Some(f64::from_bits(0x3ffd_9dfa_a9f8_8e50)),
                ]),
                77,
            )
            .unwrap()
            .map(|candidate| candidate.kind),
            None
        );
    }

    #[test]
    fn deletion_reads_each_outlier_at_its_own_factor_position() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let fit = fit_with_deletion_state(
            vec![
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            vec![99.0, 1.0, 1.0, 1.0],
            vec![
                RegressorKind::Constant,
                outlier_kind(OutlierType::Additive, axis.period_at(1).unwrap()),
                outlier_kind(OutlierType::LevelShift, axis.period_at(2).unwrap()),
                outlier_kind(OutlierType::TemporaryChange, axis.period_at(3).unwrap()),
            ],
        );

        let actual = deletion_candidate(fit.numerical_view(), &shared(2.0), 1)
            .unwrap()
            .unwrap();

        assert_eq!(actual.kind, OutlierType::Additive);
        assert_eq!(actual.t_value, 1.0);
    }

    #[test]
    fn deletion_replaces_an_equal_type_minimum_but_keeps_type_order_on_cross_type_ties() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let tied_types = fit_with_deletion_state(
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![
                outlier_kind(OutlierType::Additive, axis.period_at(1).unwrap()),
                outlier_kind(OutlierType::LevelShift, axis.period_at(2).unwrap()),
                outlier_kind(OutlierType::TemporaryChange, axis.period_at(3).unwrap()),
            ],
        );
        assert_eq!(
            deletion_candidate(tied_types.numerical_view(), &shared(2.0), 1)
                .unwrap()
                .unwrap()
                .kind,
            OutlierType::Additive
        );

        let later_additive = fit_with_deletion_state(
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![
                outlier_kind(OutlierType::Additive, axis.period_at(1).unwrap()),
                outlier_kind(OutlierType::Additive, axis.period_at(2).unwrap()),
                outlier_kind(OutlierType::TemporaryChange, axis.period_at(3).unwrap()),
            ],
        );
        assert_eq!(
            deletion_candidate(later_additive.numerical_view(), &shared(2.0), 1)
                .unwrap()
                .unwrap()
                .period,
            axis.period_at(2).unwrap()
        );
    }

    #[test]
    fn deletion_checks_the_raw_response_diagonal_before_scaling() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        for raw_diagonal in [0.0, 1.0e-20] {
            let fit = fit_with_deletion_state(
                vec![1.0, 0.0, raw_diagonal],
                vec![1.0],
                vec![outlier_kind(OutlierType::Additive, axis.start())],
            );
            assert!(matches!(
                deletion_candidate(fit.numerical_view(), &shared(f64::INFINITY), 100),
                Err(crate::Error::Numerical(crate::error::NumericalFailure(
                    crate::error::NumericalDetail::ZeroOutlierDeletionScale
                )))
            ));
        }

        for raw_diagonal in [3.834e-20, 3.835e-20] {
            let fit = fit_with_deletion_state(
                vec![1.0, 0.0, raw_diagonal],
                vec![1.0],
                vec![outlier_kind(OutlierType::Additive, axis.start())],
            );
            assert!(
                deletion_candidate(fit.numerical_view(), &shared(f64::INFINITY), 100)
                    .unwrap()
                    .is_some()
            );
        }
    }

    #[test]
    fn deletion_ignores_a_zero_scale_when_no_outlier_is_selected() {
        let fit = fit_with_deletion_state(vec![0.0], Vec::new(), Vec::new());

        assert!(matches!(
            deletion_candidate(fit.numerical_view(), &shared(f64::INFINITY), 100),
            Ok(None)
        ));
    }

    #[test]
    fn deletion_rejects_a_nonfinite_t_value() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let kinds = vec![outlier_kind(OutlierType::Additive, axis.start())];
        for (factor, coefficient) in [
            (vec![1.0, 0.0, f64::NAN], 1.0),
            (vec![f64::NAN, 0.0, 1.0], 1.0),
            (vec![1.0, 0.0, 1.0], f64::NAN),
            (vec![1.0e154, 0.0, 1.0], f64::MAX),
        ] {
            let fit = fit_with_deletion_state(factor, vec![coefficient], kinds.clone());
            assert!(matches!(
                deletion_candidate(fit.numerical_view(), &shared(2.0), 1),
                Err(error) if error == crate::Error::numerical(crate::error::NumericalOperation::OutlierSearch)
            ));
        }
    }

    #[test]
    fn addall_rejects_additions_past_eighty_columns() {
        let observation_count = 120;
        let response = (0..observation_count)
            .map(|index| 100.0 + index as f64)
            .collect::<Vec<_>>();
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let model = ArimaModel::new(
            ArimaOrder::new(0, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap();
        let spec = RegArimaSpec::new(model).with_outliers(
            OutlierSpec::default()
                .with_types(OutlierTypes::ADDITIVE)
                .with_method(OutlierMethod::AddAll)
                .with_critical_values(OutlierCriticalValues::shared(
                    OutlierCriticalValue::new(0.1).unwrap(),
                )),
        );
        let mut regression = RegressionModel::new();
        for column in 0..50 {
            let mut values = vec![0.0; observation_count];
            values[column] = 1.0;
            regression
                .push(RegressionColumn::with_values(
                    RegressionVariable::Length(crate::regarima::LengthEffect::LeapYear),
                    values,
                ))
                .unwrap();
        }

        assert_eq!(
            search(
                &response,
                (&response).into(),
                axis,
                spec.resolve_explicit().unwrap(),
                regression,
                ArmaStart::Default,
            ),
            Err(crate::Error::TooManyRegressionVariables {
                actual: 120,
                maximum: 80,
            })
        );
    }

    #[test]
    fn exact_candidate_ties_keep_ao_ls_tc_order() {
        let matrix = RegressionMatrix::without_regressors(&[2.0, 3.0, 5.0]);
        let solution = solve_filtered_regression(&matrix).unwrap();
        let state = OutlierTestState::new(&matrix, solution.packed_factor());
        let statistics = candidate_statistics(
            state,
            &[OutlierType::Additive, OutlierType::TemporaryChange],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );

        assert_eq!(statistics[0].proportional, statistics[1].proportional);
        assert_eq!(
            statistics
                .iter()
                .map(|statistic| statistic.kind)
                .collect::<Vec<_>>(),
            [OutlierType::Additive, OutlierType::TemporaryChange]
        );
    }

    #[test]
    fn singular_candidates_return_exact_zero() {
        let matrix = RegressionMatrix::from_row_major(vec![1.0, 2.0, 0.0, 3.0, 0.0, 5.0], 3, 2);
        let solution = solve_filtered_regression(&matrix).unwrap();
        let state = OutlierTestState::new(&matrix, solution.packed_factor());

        let statistic = candidate_statistics(state, &[OutlierType::Additive], &[1.0, 0.0, 0.0])[0];

        assert!(statistic.singular);
        assert_eq!(statistic.proportional, 0.0);
    }

    fn statistic_summary(
        statistics: &[super::CandidateStatistic],
    ) -> Vec<(OutlierType, usize, u64, bool)> {
        statistics
            .iter()
            .map(|statistic| {
                (
                    statistic.kind,
                    statistic.column,
                    statistic.proportional.to_bits(),
                    statistic.singular,
                )
            })
            .collect()
    }

    #[test]
    fn candidate_statistics_solve_through_the_factor_and_sort_by_magnitude() {
        // Two regressors plus the response, three candidates whose
        // magnitudes reverse the AO/LS/TC order they are given in.
        let matrix = RegressionMatrix::from_row_major(
            vec![
                1.0, 0.5, 2.0, //
                1.0, -1.25, 3.5, //
                1.0, 2.0, -1.0, //
                1.0, 0.75, 4.25, //
                1.0, -0.5, 0.125, //
            ],
            5,
            3,
        );
        let solution = solve_filtered_regression(&matrix).unwrap();
        let state = OutlierTestState::new(&matrix, solution.packed_factor());
        let statistics = candidate_statistics(
            state,
            &[
                OutlierType::Additive,
                OutlierType::LevelShift,
                OutlierType::TemporaryChange,
            ],
            &[
                0.0, 0.0, 0.0, //
                0.1, 0.0, 1.0, //
                0.0, 1.0, 0.7, //
                0.0, 1.0, 0.49, //
                0.0, 1.0, 0.343, //
            ],
        );
        assert_eq!(
            statistic_summary(&statistics),
            [
                (OutlierType::LevelShift, 1, 0xbfed_7824_0a36_22f0, false),
                (OutlierType::Additive, 0, 0x3fe8_8e26_571b_9f75, false),
                (
                    OutlierType::TemporaryChange,
                    2,
                    0x3fc1_7985_935a_f377,
                    false
                ),
            ]
        );
    }

    #[test]
    fn candidate_statistics_with_signed_zeros_a_nan_and_no_rows() {
        // Signed zeros in the response and candidates.
        let matrix = RegressionMatrix::from_row_major(vec![1.0, -0.0, 1.0, 2.0, 1.0, -3.0], 3, 2);
        let factor = [2.0, 0.5, 1.5];
        let state = OutlierTestState::new(&matrix, &factor);
        let statistics = candidate_statistics(
            state,
            &[OutlierType::Additive, OutlierType::LevelShift],
            &[-0.0, 1.0, 0.0, -0.0, -0.0, 2.0],
        );
        assert_eq!(
            statistic_summary(&statistics),
            [
                (OutlierType::LevelShift, 1, 0xc010_4817_c38a_0fe0, false),
                (OutlierType::Additive, 0, 0x0, true),
            ]
        );

        // A NaN candidate value poisons that candidate's statistic but not
        // its neighbour's; the strict insertion comparison leaves it in place.
        let statistics = candidate_statistics(
            state,
            &[
                OutlierType::Additive,
                OutlierType::LevelShift,
                OutlierType::TemporaryChange,
            ],
            &[1.0, f64::NAN, 0.5, 0.0, 1.0, 0.25, 0.0, 1.0, 0.125],
        );
        assert_eq!(
            statistics
                .iter()
                .map(|statistic| (statistic.kind, statistic.proportional.is_nan()))
                .collect::<Vec<_>>(),
            [
                (OutlierType::Additive, false),
                (OutlierType::LevelShift, true),
                (OutlierType::TemporaryChange, false),
            ]
        );
        assert_eq!(
            statistics
                .iter()
                .filter(|statistic| !statistic.proportional.is_nan())
                .map(|statistic| (statistic.proportional.to_bits(), statistic.singular))
                .collect::<Vec<_>>(),
            [
                (0xbfd2_79a7_4590_331d, false),
                (0xbfd0_3a1a_294f_6094, false)
            ]
        );

        // No rows: every candidate is singular with an exact zero statistic.
        let empty = RegressionMatrix::from_row_major(Vec::new(), 0, 2);
        let factor = [1.0, 0.0, 1.0];
        let state = OutlierTestState::new(&empty, &factor);
        assert_eq!(
            statistic_summary(&candidate_statistics(
                state,
                &[OutlierType::Additive, OutlierType::TemporaryChange],
                &[],
            )),
            [
                (OutlierType::Additive, 0, 0x0, true),
                (OutlierType::TemporaryChange, 1, 0x0, true),
            ]
        );
    }

    #[test]
    fn deletion_ranks_signed_zero_t_values_as_ties_in_type_order() {
        // Regressors are LS, AO, TC, so each coefficient triple is in that
        // order; ranking still starts from Build 62's AO, LS, TC slots.
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let actual = [
            [-0.0, 0.0, 1.0],
            [0.0, -0.0, 1.0],
            [-0.0, 3.0, 0.0],
            [3.0, -1.0, -2.0],
        ]
        .map(|coefficients| {
            let fit = fit_with_deletion_state(
                vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                coefficients.to_vec(),
                vec![
                    outlier_kind(OutlierType::LevelShift, axis.period_at(2).unwrap()),
                    outlier_kind(OutlierType::Additive, axis.period_at(1).unwrap()),
                    outlier_kind(OutlierType::TemporaryChange, axis.period_at(3).unwrap()),
                ],
            );
            let candidate = deletion_candidate(fit.numerical_view(), &shared(4.0), 1)
                .unwrap()
                .unwrap();
            (candidate.kind, candidate.t_value.to_bits())
        });
        assert_eq!(
            actual,
            [
                (OutlierType::Additive, 0x0),
                (OutlierType::Additive, 0x8000_0000_0000_0000),
                (OutlierType::LevelShift, 0x8000_0000_0000_0000),
                (OutlierType::Additive, 0xbff0_0000_0000_0000),
            ]
        );
    }

    #[test]
    fn initial_coefficient_is_the_signed_square() {
        assert_eq!(
            initial_candidate_coefficient(-1.234_567_890_123_456_7).to_bits(),
            0xbff8_62f3_5e47_4f90,
        );
    }
}
