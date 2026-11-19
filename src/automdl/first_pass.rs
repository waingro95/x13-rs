//! The first pass of `automd.f`: the airline baseline, the differencing
//! orders and the ARMA order search.

use super::baseline::{BaselineFit, initial_baseline};
use super::differencing::DifferencingState;
use super::preparation::{PreparedIdentification, prepare_identification};
use super::search::{PreparedSearchResult, search_prepared};
use crate::api::ResolvedRegArimaSpec;
use crate::arima::PreparedSearchInput;
use crate::time::TimeAxis;
use crate::{Error, Warning};

pub(crate) struct FirstPass {
    pub(crate) baseline: BaselineFit,
    pub(crate) prepared: PreparedIdentification,
    pub(crate) differencing_state: DifferencingState,
    pub(crate) search: PreparedSearchResult,
}

pub(crate) fn first_pass(
    series: &[f64],
    original: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    warnings: &mut Vec<Warning>,
) -> Result<FirstPass, Error> {
    let baseline = initial_baseline(series, original, axis, spec, warnings)?;
    let prepared = prepare_identification(&baseline.fit)?;
    let mut differencing_state = DifferencingState::default();
    let identification = differencing_state.identify_prepared(&prepared.values, axis, warnings)?;
    let input = PreparedSearchInput::new(
        &prepared.values,
        original,
        axis,
        spec.transform(),
        baseline.fit.regression_model.fixed_length(),
    )?;
    let search = search_prepared(&input, &identification, warnings)?;
    Ok(FirstPass {
        baseline,
        prepared,
        differencing_state,
        search,
    })
}
