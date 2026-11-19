//! SIGEX's canonical preparation, for the AR(2)/AR(3) family and for the
//! general ARMA family.
//!
//! Every model prepared here runs the same SIGEX sequence: allocate the AR
//! roots (any seasonal AR first), resolve stochastic trading day from that
//! allocation, build the spectral inputs, then take the rational prefix,
//! minima, canonical decomposition and ShowComp checks. `CanonicalArma` names
//! one model and runs that sequence once; `ResolvedAr2::prepare` adds the RPQ
//! roots of a resolved AR(2)/AR(3).

use crate::Frequency;
use crate::seats::decomposition::adjusted::Ar1AdjustedModel;
use crate::seats::decomposition::polynomial::SpectralInputs;
use crate::seats::decomposition::root_allocation::{
    ArRoot, RegularArCoefficients, allocate_mixed_ar, allocate_roots_with_differences,
    import_regular_ar, regular_ar_roots, spectral_inputs,
};
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::import::{MA_BOUND, trans0_degree_one};
use crate::seats::tails::DifferenceOrder;
use crate::seats::trading_day::{EffectiveTradingDay, TradingDayPresence};

/// The canonical model of one AR(2)/AR(3), with the coefficients and the
/// resolved trading day the extraction reads beside it.
#[derive(Debug)]
pub(crate) struct Ar2Preparation {
    pub frequency: Frequency,
    pub ar: RegularArCoefficients,
    pub seasonal_ma: f64,
    pub model: Ar1AdjustedModel,
    pub trading_day: EffectiveTradingDay,
}

/// The AR(2)/AR(3) model a pass works on: the regular AR coefficients and
/// the seasonal MA, as imported from regARIMA or as SEARCH left them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedAr2 {
    pub frequency: Frequency,
    pub coefficients: super::transform::Ar2Coefficients,
}

impl ResolvedAr2 {
    /// The model with the coefficients SEARCH left.
    pub(crate) fn from_refit(
        frequency: Frequency,
        coefficients: super::transform::Ar2Coefficients,
    ) -> Self {
        Self {
            frequency,
            coefficients,
        }
    }

    /// Imports regARIMA's `parameters`: the regular AR through TRANS0's
    /// bounds and the seasonal MA clipped to the invertibility bound.
    pub(crate) fn imported(
        frequency: Frequency,
        parameters: super::transform::Ar2Coefficients,
    ) -> Result<Self, Error> {
        for (index, value) in parameters
            .regular
            .as_slice()
            .iter()
            .copied()
            .chain([parameters.seasonal_ma])
            .enumerate()
        {
            finite(value, "canonical model coefficient", index)?;
        }
        let regular = import_regular_ar(parameters.regular)?;
        // NMLMDL transforms the MA group on its own, and ANALTS reverses its
        // sign.
        Ok(Self {
            frequency,
            coefficients: super::transform::Ar2Coefficients {
                regular,
                seasonal_ma: -trans0_degree_one(0.0 - parameters.seasonal_ma, MA_BOUND),
            },
        })
    }

    /// The canonical preparation of this AR(2)/AR(3) with seasonal MA, an
    /// optional regular MA(1) and the given differencing. The regular-AR
    /// roots come from RPQ on the resolved coefficients.
    pub(crate) fn prepare(
        self,
        presence: TradingDayPresence,
        regular_ma: Option<f64>,
        differences: DifferenceOrder,
    ) -> Result<Ar2Preparation, Error> {
        let Self {
            frequency,
            coefficients,
        } = self;
        let roots = regular_ar_roots(coefficients.regular)?;
        let (model, trading_day) = CanonicalArma {
            frequency,
            roots: &roots,
            seasonal_ar: None,
            regular_ma: regular_ma.as_slice(),
            seasonal_ma: Some(coefficients.seasonal_ma),
            differences,
        }
        .prepare(presence)?;
        Ok(Ar2Preparation {
            frequency,
            ar: coefficients.regular,
            seasonal_ma: coefficients.seasonal_ma,
            model,
            trading_day,
        })
    }
}

/// One model for SIGEX's canonical preparation: regular-AR roots from RPQ,
/// an optional seasonal AR (BP=1), the active regular MA coefficients, an
/// optional seasonal MA and the differencing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CanonicalArma<'a> {
    pub frequency: Frequency,
    pub roots: &'a [ArRoot],
    pub seasonal_ar: Option<f64>,
    pub regular_ma: &'a [f64],
    pub seasonal_ma: Option<f64>,
    pub differences: DifferenceOrder,
}

impl CanonicalArma<'_> {
    /// Allocation, trading-day resolution, spectral inputs, then the
    /// canonical model with its ShowComp checks.
    pub(crate) fn prepare(
        &self,
        presence: TradingDayPresence,
    ) -> Result<(Ar1AdjustedModel, EffectiveTradingDay), Error> {
        let Self {
            frequency,
            roots,
            seasonal_ar,
            regular_ma,
            seasonal_ma,
            differences,
        } = *self;
        let allocation = match seasonal_ar {
            Some(seasonal_ar) => allocate_mixed_ar(frequency, roots, seasonal_ar, differences)?,
            None => allocate_roots_with_differences(frequency, roots, differences)?,
        };
        let trading_day = presence.resolve(allocation.is_close_to_trading_day);
        let inputs = spectral_inputs(frequency, &allocation, regular_ma, seasonal_ma, trading_day)?;
        let model = canonical_model(inputs, frequency, trading_day, differences)?;
        Ok((model, trading_day))
    }
}

/// The rational prefix, minima, canonical (or rejected-irregular) model and
/// ShowComp's innovation-variance checks for one set of spectral inputs.
fn canonical_model(
    inputs: SpectralInputs,
    frequency: Frequency,
    trading_day: EffectiveTradingDay,
    differences: DifferenceOrder,
) -> Result<Ar1AdjustedModel, Error> {
    let prefix = crate::seats::decomposition::rational::lower_degree_rational_prefix(&inputs)?;
    let minima = crate::seats::decomposition::minima::regular_ar_spectral_minima(
        frequency,
        prefix,
        differences != DifferenceOrder::Stationary,
    );
    // The minima the canonical preparation reads: trend, seasonal,
    // irregular, then the cycle when there is one.
    for (index, value) in [minima.enot, minima.estar, minima.irregular_candidate]
        .into_iter()
        .chain(minima.cycle.as_ref().map(|cycle| cycle.minimum))
        .enumerate()
    {
        finite(value, "spectral minimum", index)?;
    }
    let model = if minima.irregular_candidate < 0.0 {
        crate::seats::decomposition::adjusted::factor_rejected_ar2_with_trading_day(
            crate::seats::decomposition::canonical::prepare_rejected_ar2(minima),
            inputs,
            trading_day,
        )?
    } else {
        crate::seats::decomposition::adjusted::factor_adjusted(
            crate::seats::decomposition::canonical::prepare_canonical(minima),
            inputs,
            trading_day,
        )?
    };
    // ShowComp's order, with the transitory component only when there is one.
    for (index, (component, variance)) in
        crate::seats::decomposition::adjusted::show_comp_variances(&model)
            .into_iter()
            .enumerate()
            .filter_map(|(index, value)| value.map(|value| (index, value)))
    {
        finite(variance, "innovation variance", index)?;
        if let Some(direction) =
            crate::seats::decomposition::adjusted::show_comp_direction(variance)
        {
            return Err(Failure::InnovationVariance {
                component,
                direction,
                value: variance,
            }
            .at(Stage::CanonicalDecomposition));
        }
    }
    Ok(model)
}

fn finite(value: f64, quantity: &'static str, index: usize) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Failure::Unusable {
            quantity,
            index: Some(index),
            value,
        }
        .at(Stage::CanonicalDecomposition))
    }
}
