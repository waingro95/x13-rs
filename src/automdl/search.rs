//! The ARMA order search of `amdid.f`: a seasonal probe, a regular probe and
//! a seasonal revisit, each ranking its candidates by BIC2 in a best-five
//! table, then the parsimony pass and a refit of the chosen model.
//!
//! `stpitr.f`'s convergence flag passes from each fit to the next, the refit
//! included. Only the two probes must find a candidate; the revisit keeps
//! the ranking it inherits.

mod ranking;
use crate::error::NumericalDetail;
use ranking::{BestFive, RankedCandidate};

use super::differencing::DifferencingIdentification;
use crate::arima::{
    EstimatedRegArima, PreparedSearchInput, append_dynamic_fit_warnings, fit_prepared_candidate,
    refit_prepared_selected,
};
use crate::regarima::ArimaFactor;
use crate::{ArimaModel, ArimaOrder, Error, Warning};

/// The model the search chose and its refit.
#[derive(Debug)]
pub(crate) struct PreparedSearchResult {
    pub(crate) model: ArimaModel,
    /// The differencing identification's mean test.
    pub(crate) mean_significant: bool,
    pub(crate) fit: EstimatedRegArima,
}

fn model(p: u8, d: u8, q: u8, sp: u8, seasonal_d: u8, sq: u8) -> Result<ArimaModel, Error> {
    ArimaModel::new(
        ArimaOrder::new(p, d, q)?,
        ArimaOrder::new(sp, seasonal_d, sq)?,
    )
}

/// `amdid.f` stops when no candidate of a probe can be ranked.
fn no_candidate(factor: ArimaFactor) -> Error {
    Error::from(NumericalDetail::NoEstimableArmaCandidate { factor })
}

/// What the three stages share.
struct SearchState {
    ranking: BestFive,
    /// `stpitr.f`'s convergence flag as the last fit left it.
    converged: bool,
}

impl SearchState {
    /// Fits one candidate and ranks it if `amdid.f` would.
    fn try_candidate(
        &mut self,
        input: &PreparedSearchInput<'_>,
        model: ArimaModel,
        warnings: &mut Vec<Warning>,
    ) -> Result<(), Error> {
        let candidate = fit_prepared_candidate(input, model, &mut self.converged)?;
        if let Some(bic2) = candidate.bic2 {
            self.ranking.insert(RankedCandidate { model, bic2 });
        }
        warnings.extend(candidate.warnings);
        Ok(())
    }
}

/// The regular orders the revisit keeps: the regular probe's best, or its
/// second best when the best has no ARMA terms at all.
fn regular_orders_for_revisit(ranking: &BestFive, d: u8) -> Result<ArimaOrder, Error> {
    let best = ranking
        .best()
        .ok_or_else(|| no_candidate(ArimaFactor::Regular))?
        .model;
    let regular = best.regular();
    let seasonal = best.seasonal();
    if regular.p() + regular.q() + seasonal.p() + seasonal.q() == 0 {
        Ok(ranking.slots()[1]
            .as_ref()
            .map_or(ArimaOrder::new(0, d, 0)?, |entry| entry.model.regular()))
    } else {
        Ok(regular)
    }
}

/// Searches the ARMA orders of the series in `input` at the identified
/// differencing orders. The fits' warnings go to `warnings` in the order
/// Build 62 reports them.
pub(crate) fn search_prepared(
    input: &PreparedSearchInput<'_>,
    identification: &DifferencingIdentification,
    warnings: &mut Vec<Warning>,
) -> Result<PreparedSearchResult, Error> {
    let d = identification.regular;
    let seasonal_d = identification.seasonal;
    let mut state = SearchState {
        ranking: BestFive::default(),
        converged: true,
    };
    for sp in 0..=1 {
        for sq in 0..=1 {
            state.try_candidate(input, model(3, d, 0, sp, seasonal_d, sq)?, warnings)?;
        }
    }
    let seasonal_choice = state
        .ranking
        .best()
        .ok_or_else(|| no_candidate(ArimaFactor::Seasonal))?
        .model
        .seasonal();
    // The probe's AR(3) models exceed the default maximum regular order of
    // two, so none of them stays ranked.
    state.ranking.clear();

    for p in 0..=2 {
        for q in 0..=2 {
            let candidate = model(
                p,
                d,
                q,
                seasonal_choice.p(),
                seasonal_d,
                seasonal_choice.q(),
            )?;
            if !state.ranking.contains(candidate) {
                state.try_candidate(input, candidate, warnings)?;
            }
        }
    }
    let regular_choice = regular_orders_for_revisit(&state.ranking, d)?;

    let max_seasonal_ar = if seasonal_d == 1 { 0 } else { 1 };
    for sp in 0..=max_seasonal_ar {
        for sq in 0..=1 {
            let candidate = model(
                regular_choice.p(),
                d,
                regular_choice.q(),
                sp,
                seasonal_d,
                sq,
            )?;
            if !state.ranking.contains(candidate) {
                state.try_candidate(input, candidate, warnings)?;
            }
        }
    }
    let chosen = state
        .ranking
        .select(model(0, d, 0, 0, seasonal_d, 0)?)
        .ok_or_else(|| no_candidate(ArimaFactor::Regular))?;
    let fit = refit_prepared_selected(input, chosen, &mut state.converged)?;
    append_dynamic_fit_warnings(warnings, &fit.warnings);
    Ok(PreparedSearchResult {
        model: chosen,
        mean_significant: identification.mean_significant,
        fit,
    })
}

#[cfg(test)]
mod tests;
