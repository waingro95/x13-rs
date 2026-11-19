//! CALCFX (`ansub1.f`) for the ARMA family: the residuals of the differenced
//! series at SEARCH's coordinates.
//!
//! A call turns the coordinates into coefficients with TRANSC, one group at
//! a time (regular AR, seasonal AR, regular MA, seasonal MA) and only for a
//! group that changed since the last completed call. It filters the series
//! by the AR polynomial and solves for the MA residuals. After SEARCH's
//! first few calls, a call stops before the residuals when a regular AR
//! root and a regular MA root are within 0.1 of each other: the model then
//! has a common factor.

use super::model::{ArmaLayout, RegularArGroup};
use crate::Frequency;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::residual::{ResidualResult, ar_prefilter, finite};
use crate::seats::retry::Init;

/// The quantity coordinate `index` of `layout` is, and its index within its
/// group when the group has more than one lag. The coordinates are the
/// regular AR, the seasonal AR, the regular MA and the seasonal MA, in that
/// order.
pub(super) fn coordinate(layout: ArmaLayout, index: usize) -> (&'static str, Option<usize>) {
    let p = layout.ar_coordinates();
    let q = layout.regular_ma_count();
    if layout.has_seasonal_ar() && index == layout.regular_ar_coordinates() {
        ("seasonal AR coordinate", None)
    } else if index < p {
        ("regular AR coordinate", Some(index))
    } else if index < p + q {
        ("regular MA coordinate", Some(index - p))
    } else {
        ("seasonal MA coordinate", None)
    }
}

/// Whether `wd` is finite and implies an observed length in CheckLen's range
/// after `layout`'s differencing.
fn differenced_in_range(frequency: Frequency, wd: &[f64], layout: ArmaLayout) -> bool {
    let s = usize::from(frequency.periods_per_year());
    let observed =
        wd.len() + layout.differences().ordinary() + usize::from(layout.seasonal_difference()) * s;
    let range = if s == 12 { 36..=780 } else { 16..=328 };
    range.contains(&observed) && wd.iter().all(|value| value.is_finite())
}

/// CALCFX's controls. ANSUB1's IPR and OUT are always 0 here, so a common
/// factor always stops a call from SEARCH, and ICOMM, which only those set,
/// stays 0.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArmaCalcControls {
    /// ANALTS's INIT: INIT=2 takes the coordinates as coefficients without
    /// TRANSC and clears the roots.
    pub init: Init,
    /// SEARCH's evaluation count IFN; 0 clears the roots.
    pub ifn: usize,
}

/// The coefficients CALCFX holds between calls.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmaPolynomials<const N: usize = 3> {
    pub layout: ArmaLayout,
    /// The model's coefficients, in coordinate order.
    pub parameters: [f64; N],
    /// PHIST: the product of the regular and seasonal AR polynomials, lag 1
    /// first.
    pub phist: Vec<f64>,
    /// THSTAR: the product of the regular and seasonal MA polynomials.
    pub thstar: Vec<f64>,
    /// The coordinates of the last completed call, which the next call
    /// compares its own with.
    pub previous: Option<[f64; N]>,
}

/// CALCFX on one differenced series.
pub(crate) struct ArmaObjective<'a, const N: usize = 3> {
    frequency: Frequency,
    wd: &'a [f64],
    state: ArmaPolynomials<N>,
    /// The real roots of the regular AR and MA groups from their last
    /// TRANSC.
    regular_roots: Vec<f64>,
    ma_roots: Vec<f64>,
}

/// CALCFX's state between calls, without the series: forward CALCFX leaves
/// it and FCAST's backward CALCFX takes it up on the reversed series.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmaCalcContinuation<const N: usize = 3> {
    state: ArmaPolynomials<N>,
    regular_roots: Vec<f64>,
    ma_roots: Vec<f64>,
}

impl<'a, const N: usize> ArmaObjective<'a, N> {
    pub(crate) fn into_continuation(self) -> ArmaCalcContinuation<N> {
        ArmaCalcContinuation {
            state: self.state,
            regular_roots: self.regular_roots,
            ma_roots: self.ma_roots,
        }
    }

    /// CALCFX on `wd` with the state an earlier objective left.
    pub(crate) fn resume(
        frequency: Frequency,
        wd: &'a [f64],
        continuation: ArmaCalcContinuation<N>,
    ) -> Self {
        Self {
            frequency,
            wd,
            state: continuation.state,
            regular_roots: continuation.regular_roots,
            ma_roots: continuation.ma_roots,
        }
    }

    /// CALCFX on `wd` before its first call.
    pub(crate) fn new(frequency: Frequency, wd: &'a [f64], layout: ArmaLayout) -> Self {
        debug_assert_eq!(layout.coordinates(), N);
        let s = usize::from(frequency.periods_per_year());
        Self {
            frequency,
            wd,
            state: ArmaPolynomials {
                layout,
                parameters: [0.0; N],
                phist: vec![
                    0.0;
                    layout.regular_ar_coordinates()
                        + usize::from(layout.has_seasonal_ar()) * s
                ],
                thstar: vec![
                    0.0;
                    layout.regular_ma_count()
                        + if layout.has_seasonal_ma() { s } else { 0 }
                ],
                previous: None,
            },
            regular_roots: Vec::new(),
            ma_roots: Vec::new(),
        }
    }

    /// CALCFX run to completion: INIT=2's and FCAST's evaluations (JFAC=1),
    /// and SEARCH's first, whose IFN=0 leaves no roots to compare.
    pub(crate) fn evaluate(
        &mut self,
        x: [f64; N],
        controls: ArmaCalcControls,
    ) -> Result<ResidualResult, Error> {
        let conditioned = self.refresh(x, controls)?;
        self.residuals(x, conditioned)
    }

    /// CALCFX from SEARCH after its first call (JFAC=0). `None` means a
    /// common factor stopped it before its residual solve.
    pub(crate) fn evaluate_searching(
        &mut self,
        x: [f64; N],
        controls: ArmaCalcControls,
    ) -> Result<Option<ResidualResult>, Error> {
        let conditioned = self.refresh(x, controls)?;
        // CALCFX skips TRANSC for a group that did not change, so its root
        // arrays ap and aq hold each group's roots from its last TRANSC.
        // ansub1.f does not SAVE them; this keeps them with the
        // coefficients, across a stopped call too. INIT=2 and IFN=0 zero
        // the root counts, here the vectors' lengths, and with either empty
        // no pair can match. CALCFX also sets IFAC=1 here, which nothing
        // reads. With IPR=OUT=0 it never sets ICOMM, so its ICOMM=0
        // condition always holds.
        if controls.ifn > 2 * N + 3
            && self
                .regular_roots
                .iter()
                .any(|root| self.ma_roots.iter().any(|ma| (root - ma).abs() < 0.10))
        {
            return Ok(None);
        }
        self.residuals(x, conditioned).map(Some)
    }

    /// CALCFX up to the common-factor check: the input checks, the
    /// coefficients of the changed groups and the AR filter, whose output
    /// it returns.
    fn refresh(&mut self, x: [f64; N], controls: ArmaCalcControls) -> Result<Vec<f64>, Error> {
        let layout = self.state.layout;
        let p = layout.ar_coordinates();
        let regular_p = layout.regular_ar_coordinates();
        let q = layout.regular_ma_count();
        for (index, value) in x.into_iter().enumerate() {
            let polynomial_coefficient = controls.init == Init::Imported2
                && ((regular_p > 1 && index < regular_p) || (q > 1 && (p..p + q).contains(&index)));
            if !value.is_finite() || (!polynomial_coefficient && value.abs() > 1.0) {
                let (quantity, index) = coordinate(layout, index);
                return Err(Failure::Unusable {
                    quantity,
                    index,
                    value,
                }
                .at(Stage::Residual));
            }
        }
        let s = usize::from(self.frequency.periods_per_year());
        debug_assert!(differenced_in_range(self.frequency, self.wd, layout));
        let reset = controls.init == Init::Imported2 || controls.ifn == 0;
        if reset {
            self.regular_roots.clear();
            self.ma_roots.clear();
        }
        // On a reset CALCFX sets y = x + .01, so every group counts as
        // changed. Only a completed call saves its x as y.
        debug_assert!(reset || self.state.previous.is_some());
        let y = self
            .state
            .previous
            .filter(|_| !reset)
            .unwrap_or_else(|| x.map(|v| v + 0.01));
        let changed = std::array::from_fn::<_, N, _>(|i| (y[i] - x[i]).abs() >= 3.834e-20);
        if controls.init == Init::Imported2 {
            self.state.parameters = x;
        }
        if changed[..regular_p].iter().any(|&changed| changed) {
            let p = regular_p;
            if controls.init == Init::Refit0 {
                match layout.regular_ar_group(&x) {
                    RegularArGroup::Empty => {}
                    RegularArGroup::Single(phi) => {
                        self.state.parameters[0] = phi;
                        self.regular_roots = vec![phi];
                    }
                    RegularArGroup::Polynomial(coordinates) => {
                        let transformed =
                            crate::seats::ar2::transform::regular_coefficients(coordinates)
                                .map_err(|error| error.restage(Stage::Residual))?;
                        self.state.parameters[..p]
                            .copy_from_slice(transformed.coefficients.as_slice());
                        self.regular_roots =
                            if let crate::seats::decomposition::root_allocation::RegularArCoefficients::Three(c) =
                                transformed.coefficients
                            {
                                crate::seats::ar2::transform::cubic_detector_roots(c)
                                    .map_err(|error| error.restage(Stage::Residual))?
                            } else {
                                transformed.real_roots.map(Vec::from).unwrap_or_default()
                            };
                    }
                }
            }
        }
        if layout.has_seasonal_ar() && changed[regular_p] && controls.init == Init::Refit0 {
            self.state.parameters[regular_p] = x[regular_p];
        }
        if changed[..p].iter().any(|&changed| changed) {
            self.state.phist.fill(0.0);
            self.state.phist[..regular_p].copy_from_slice(&self.state.parameters[..regular_p]);
            if layout.has_seasonal_ar() {
                // CALCFX expands Phi*BPhi lag by lag, in this order.
                let bphi = self.state.parameters[regular_p];
                self.state.phist[s - 1] = bphi;
                for k in 0..regular_p {
                    self.state.phist[s + k] = -self.state.parameters[k] * bphi;
                }
            }
        }
        // CALCFX checks only each conditioned value, not the partial sums.
        let Ok(conditioned) = ar_prefilter(&self.state.phist, self.wd, |_, _, value| {
            Ok::<_, std::convert::Infallible>(value)
        });
        if let Some((index, &value)) = conditioned.iter().enumerate().find(|(_, v)| !v.is_finite())
        {
            return Err(Failure::Unusable {
                quantity: "AR prefilter",
                index: Some(index),
                value,
            }
            .at(Stage::Residual));
        }
        if changed[p..p + q].iter().any(|&changed| changed) && controls.init == Init::Refit0 {
            if q == 2 {
                // ANSUB1 TRANSC transforms an MA pair as it does an AR pair.
                // Its real roots are the ones the common-factor check reads.
                self.state.parameters[p] = x[p] * (1.0 - x[p + 1]);
                self.state.parameters[p + 1] = x[p + 1];
                let first = self.state.parameters[p];
                let discriminant = first.mul_add(first, 4.0 * x[p + 1]);
                self.ma_roots = if discriminant >= 0.0 {
                    let root = discriminant.sqrt();
                    vec![0.5 * (first + root), 0.5 * (first - root)]
                } else {
                    Vec::new()
                };
            } else {
                self.state.parameters[p] = x[p];
                self.ma_roots = vec![x[p]];
            }
        }
        if layout.has_seasonal_ma() && changed[p + q] && controls.init == Init::Refit0 {
            self.state.parameters[p + q] = x[p + q];
        }
        if changed[p..p + q].iter().any(|&changed| changed)
            || (layout.has_seasonal_ma() && changed[p + q])
        {
            self.state.thstar.fill(0.0);
            self.state.thstar[..q].copy_from_slice(&self.state.parameters[p..p + q]);
            if layout.has_seasonal_ma() {
                self.state.thstar[s - 1] = self.state.parameters[p + q];
                for k in 0..q {
                    self.state.thstar[s + k] =
                        -self.state.parameters[p + k] * self.state.parameters[p + q];
                }
            }
        }
        Ok(conditioned)
    }

    /// CALCFX past the common-factor check: the residual solve on
    /// `conditioned`, which saves `x` as the coordinates of the last
    /// completed call.
    fn residuals(&mut self, x: [f64; N], conditioned: Vec<f64>) -> Result<ResidualResult, Error> {
        let layout = self.state.layout;
        let s = usize::from(self.frequency.periods_per_year());
        let p = layout.ar_coordinates();
        let q = layout.regular_ma_count();
        let residuals = if q == 0 && !layout.has_seasonal_ma() {
            // CALCFX: Qstar=0 skips the initial-error solve. Detpri=1 and
            // the AR-conditioned values are the final residuals.
            let mut objective = 0.0;
            for (index, &value) in conditioned.iter().enumerate() {
                objective = finite(
                    value.mul_add(value, objective),
                    "pure AR objective",
                    Some(index),
                )?;
            }
            ResidualResult {
                scaled_residuals: conditioned.clone(),
                determinant_scale: 1.0,
                objective,
                normalized_residuals: conditioned,
            }
        } else if q == 0 {
            crate::seats::seasonal_ma::residual::seasonal_constrained_residuals(
                s,
                self.state.parameters[p],
                &conditioned,
            )?
        } else if !layout.has_seasonal_ma() && q == 1 {
            crate::seats::seasonal_ma::residual::regular_ma_residuals(
                self.state.parameters[p],
                &conditioned,
            )?
        } else {
            crate::seats::residual::ordered_ma_residuals(
                s,
                &self.state.parameters[p..p + q],
                layout
                    .has_seasonal_ma()
                    .then(|| self.state.parameters[p + q]),
                &conditioned,
            )?
        };
        self.state.previous = Some(x);
        Ok(residuals)
    }

    /// The coefficients the last call left.
    pub(crate) fn coefficients(&self) -> ArmaPolynomials<N> {
        self.state.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::arma::model::ArOrder;

    #[test]
    fn the_common_factor_check_keeps_the_roots_of_an_unchanged_group() {
        let wd = (0..64).map(|i| (i as f64 * 0.31).sin()).collect::<Vec<_>>();
        let controls = ArmaCalcControls {
            init: Init::Refit0,
            ifn: 0,
        };
        let mut objective = ArmaObjective::new(
            Frequency::Quarterly,
            &wd,
            ArmaLayout::RegularSeasonal(ArOrder::One),
        );
        objective.evaluate([0.5, 0.55, 0.3], controls).unwrap();
        let continuation = objective.into_continuation();
        let mut objective = ArmaObjective::resume(Frequency::Quarterly, &wd, continuation);
        let searched = ArmaCalcControls {
            ifn: 10,
            ..controls
        };
        // The third point changes only the AR group: its MA root is the
        // second point's, retained without TRANSC.
        for (x, roots_cancel) in [
            ([0.5, 0.55, 0.3], true),
            ([0.5, -0.3, 0.3], false),
            ([-0.35, -0.3, 0.3], true),
        ] {
            // Under SEARCH a common factor stops CALCFX.
            let result = objective.evaluate_searching(x, searched).unwrap();
            assert_eq!(result.is_none(), roots_cancel, "{x:?}");
            // With JFAC=1 CALCFX does not look for one and completes.
            objective.evaluate(x, searched).unwrap();
            assert_eq!(objective.coefficients().previous, Some(x));
        }
        // INIT=2 zeroes the root counts and takes coefficients without TRANSC.
        let result = objective
            .evaluate_searching(
                [-0.35, -0.3, 0.3],
                ArmaCalcControls {
                    init: Init::Imported2,
                    ifn: 11,
                },
            )
            .unwrap();
        assert!(result.is_some());
    }
}
