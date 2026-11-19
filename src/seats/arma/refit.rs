//! STAVAL's starting values and the SEARCH run of the ARMA family
//! (`ansub1.f`), and the import of a fitted MA(2).
//!
//! STAVAL starts a regular AR part from ratios of autocorrelations and
//! removes it from the autocorrelations the MA starts read
//! ([`starting_parameters`]). The other functions are its branches for a
//! model without that part: a regular MA(1) or MA(2), a seasonal MA, a
//! seasonal AR. [`search_arma`] runs SEARCH on the family's CALCFX from
//! those starts.

use super::model::ArmaLayout;
use super::residual::{
    ArmaCalcContinuation, ArmaCalcControls, ArmaObjective, ArmaPolynomials, coordinate,
};
use crate::Frequency;
use crate::seats::failure::{Error, Failure, Stage, usable};
use crate::seats::import::{MA_BOUND, trans0_clip};
use crate::seats::refit::search::{BoundRule, SearchObjective, search};
use crate::seats::refit::start::{contract_start, degree_one_start, seasonal_start};
use crate::seats::residual::ResidualResult;
use crate::seats::retry::Init;

/// SEARCH's covariance tail does not run for this family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArmaRefitCovariance {
    UnavailableAllFixed,
    NotEstimated,
}

/// SEARCH's result for an ARMA model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmaRefitResult<const N: usize = 3> {
    /// The coordinates of the last completed CALCFX call SEARCH accepted.
    pub parameters: [f64; N],
    pub scaled_residuals: Vec<f64>,
    pub determinant_scale: f64,
    pub objective: f64,
    /// The coordinates SEARCH fixed, or was given fixed.
    pub bound_mask: [bool; N],
    /// SEARCH's count of bound fixes, which CHMODEL tests.
    pub fixed: usize,
    /// IFN: the completed CALCFX calls.
    pub evaluations: usize,
    /// The coefficients CALCFX held when SEARCH returned. After a common
    /// factor stopped an evaluation they are that evaluation's, not those of
    /// `parameters`.
    pub coefficients: ArmaPolynomials<N>,
    pub covariance: ArmaRefitCovariance,
}

/// CALCFX as SEARCH calls it: INIT=0, with IFN counting the completed calls.
struct ArmaSearchObjective<'a, const N: usize> {
    objective: ArmaObjective<'a, N>,
    evaluations: usize,
}

impl<const N: usize> ArmaSearchObjective<'_, N> {
    const fn controls(&self) -> ArmaCalcControls {
        ArmaCalcControls {
            init: Init::Refit0,
            ifn: self.evaluations,
        }
    }
}

impl<const N: usize> SearchObjective<N> for ArmaSearchObjective<'_, N> {
    fn calcfx(&mut self, point: [f64; N]) -> Result<ResidualResult, Error> {
        let residuals = self
            .objective
            .evaluate(point, self.controls())
            .map_err(|error| error.restage(Stage::Refit))?;
        self.evaluations += 1;
        Ok(residuals)
    }

    fn interruptible(&mut self, point: [f64; N]) -> Result<Option<ResidualResult>, Error> {
        let completed = self
            .objective
            .evaluate_searching(point, self.controls())
            .map_err(|error| error.restage(Stage::Refit))?;
        if completed.is_some() {
            self.evaluations += 1;
        }
        Ok(completed)
    }

    fn evaluations(&self) -> usize {
        self.evaluations
    }
}

/// SEARCH on the differenced series `wd` from `start`, holding the
/// coordinates `fixed` marks. It also returns the CALCFX state SEARCH left,
/// which FCAST's backward CALCFX resumes.
pub(crate) fn search_arma<const N: usize>(
    frequency: Frequency,
    wd: &[f64],
    start: [f64; N],
    fixed: [bool; N],
    layout: ArmaLayout,
) -> Result<(ArmaRefitResult<N>, ArmaCalcContinuation<N>), Error> {
    validate_coordinates(start, fixed, layout)?;
    let calcfx = ArmaSearchObjective {
        objective: ArmaObjective::new(frequency, wd, layout),
        evaluations: 0,
    };
    let (result, calcfx) = search(
        calcfx,
        start,
        fixed,
        BoundRule::from_orders(layout.orders()),
    )?;
    let refit = ArmaRefitResult {
        parameters: result.parameters,
        scaled_residuals: result.scaled_residuals,
        determinant_scale: result.determinant_scale,
        objective: result.objective,
        bound_mask: result.bound_mask,
        fixed: result.fixed,
        evaluations: result.evaluations,
        coefficients: calcfx.objective.coefficients(),
        covariance: if result.all_fixed {
            ArmaRefitCovariance::UnavailableAllFixed
        } else {
            ArmaRefitCovariance::NotEstimated
        },
    };
    Ok((refit, calcfx.objective.into_continuation()))
}

/// SEARCH's start check: each coordinate finite and within the unit bound,
/// and a free one inside ±.99.
fn validate_coordinates<const N: usize>(
    x: [f64; N],
    fixed: [bool; N],
    layout: ArmaLayout,
) -> Result<(), Error> {
    for j in 0..N {
        if !x[j].is_finite() || x[j].abs() > 1.0 || (!fixed[j] && !(-0.99..=0.99).contains(&x[j])) {
            let (quantity, index) = coordinate(layout, j);
            return Err(Failure::Unusable {
                quantity,
                index,
                value: x[j],
            }
            .at(Stage::Refit));
        }
    }
    Ok(())
}

/// The import of a fitted MA(2) `[th1, th2]` into TRANS1's coordinates.
pub(crate) fn import_ma2(parameters: [f64; 2]) -> Result<[f64; 2], Error> {
    // NMLMDL sign reversal, TRANS0 with IPRS=0/UR=1, then TRANS2 and ANALTS.
    let first = -(0.0 - parameters[0]);
    let mut second = -(0.0 - parameters[1]);
    if (1.0 - second).abs() < 1e-9 {
        second = 1.0;
    }
    // TRANS0 quirk: with PHITH(2) reset to UR=1 the quotient divides by
    // exactly zero (and a finite one can overflow). Build 62 keeps the IEEE
    // ±infinity, which the clip below maps to ±XL=.99.
    let x = first / (1.0 - second);
    let clip = |x: f64| trans0_clip(x, MA_BOUND);
    // 0/0 leaves a NaN that both of TRANS0's comparisons pass over, and
    // Build 62's SEATS then hangs. Here it is an error.
    let first = finite(clip(x), "MA2 import coordinate", 1)?;
    let second = clip(second);
    Ok([first * (1.0 - second), second])
}

/// STAVAL's start for a regular AR part, with the autocorrelations it
/// leaves for the MA starts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmaStartValues {
    /// `[phi1, th1, bth]`, before TRANS1.
    pub(crate) parameters: [f64; 3],
    /// The autocorrelations with the AR(1) `phi1` removed from lags 1 and 2
    /// and, for a seasonal MA, from the seasonal lag.
    pub(crate) conditioned_acf: [f64; 36],
}

/// STAVAL's Q=2 start in TRANS1 coordinates `[th1, th2, bth]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ma2Start {
    pub(super) coordinates: [f64; 3],
    /// ANALTS 5005–5006: with Q>1, SEARCH fixes the lower-order regular MA
    /// coordinate (E=1) when STAVAL's raw Th(1), before TRANS1, is within
    /// CEPS=1e-13 of zero. It then keeps its start value.
    pub(super) fixes_first: bool,
}

/// STAVAL's Q=1 start in TRANS1 coordinates `[th1, bth]`; `bth` is 0 without
/// a seasonal MA.
pub(crate) fn ma1_start(
    acf: &[f64; 36],
    frequency: Frequency,
    seasonal: bool,
) -> Result<[f64; 2], Error> {
    let first = degree_one_start(finite(acf[0], "MA1 ACF", 1)?);
    let second = if seasonal {
        seasonal_ma_start(acf, frequency)?
    } else {
        0.0
    };
    Ok([contract_start(first), second])
}

/// STAVAL's seasonal MA start as a TRANS1 coordinate.
pub(crate) fn seasonal_ma_start(acf: &[f64; 36], frequency: Frequency) -> Result<f64, Error> {
    let s = usize::from(frequency.periods_per_year());
    let rs = finite(acf[s - 1], "seasonal MA1 ACF", s)?;
    Ok(contract_start(seasonal_start(rs)))
}

/// STAVAL's Q=2 start from `acf`, which has any regular AR part removed
/// already.
pub(crate) fn ma2_start(
    acf: &[f64; 36],
    frequency: Frequency,
    seasonal: bool,
) -> Result<Ma2Start, Error> {
    // ANSUB1 STAVAL:2537–2557. gfortran fuses only b*b with the rounded
    // 4*r1*r1 product here.
    let mut r1 = finite(acf[0], "MA2 ACF", 1)?;
    let mut r2 = finite(acf[1], "MA2 ACF", 2)?;
    if r2.abs() >= 0.5 {
        r2 = 0.45_f64.copysign(r2);
    }
    let doubled = r2 + r2;
    let b = 1.0 + doubled;
    if r2 > 0.16666 {
        let a = r2 * (1.0 - doubled);
        if r1 * r1 >= 4.0 * a {
            r1 = (1.8 * a.sqrt()).copysign(r1);
        }
    } else if r1.abs() >= 0.5 * b {
        r1 = (b * 0.45).copysign(r1);
    }
    let x = finite(
        ((doubled - 1.0) - b.mul_add(b, -(4.0 * (r1 * r1))).sqrt()) / dividecheck(doubled),
        "MA2 start value",
        2,
    )?;
    let mut second = finite(
        (0.5 * x) * (1.0 - (1.0 - 4.0 / dividecheck(x * x)).sqrt()),
        "MA2 start second",
        2,
    )?;
    let first = finite(
        (r1 * second) / dividecheck(r2 * (1.0 - second)),
        "MA2 start first",
        1,
    )?;
    let fixes_first = first.abs() <= 1e-13;
    // TRANS1:1688–1700 transforms the pair together; it does not clip each
    // coefficient as it does an MA(1).
    if (1.0 - second).abs() < 1e-9 {
        second = 0.9999999;
    }
    let first = finite(first / (1.0 - second), "MA2 start coordinate", 1)?;
    if !seasonal {
        return Ok(Ma2Start {
            coordinates: [first, second, 0.0].map(contract_start),
            fixes_first,
        });
    }
    let rs = finite(
        acf[usize::from(frequency.periods_per_year()) - 1],
        "MA2 seasonal ACF",
        0,
    )?;
    Ok(Ma2Start {
        coordinates: [first, second, seasonal_start(rs)].map(contract_start),
        fixes_first,
    })
}

/// `value` when it is finite, else [`Failure::Unusable`] naming it with
/// `index`, at [`Stage::Refit`].
fn finite(value: f64, quantity: &'static str, index: usize) -> Result<f64, Error> {
    usable(quantity, Some(index), value).map_err(|failure| failure.at(Stage::Refit))
}

/// DIVIDECHECK: a divisor below 1e-9 in magnitude, of either sign, becomes
/// +1e-9.
fn dividecheck(value: f64) -> f64 {
    if value.abs() < 1e-9 { 1e-9 } else { value }
}

/// STAVAL's start for (0,1,1)(1,1,0) in TRANS1 coordinates `[bphi, th1]`.
pub(crate) fn seasonal_ar_start(acf: &[f64; 36], frequency: Frequency) -> Result<[f64; 2], Error> {
    let r = finite(acf[0], "regular ACF", 1)?;
    let ma = finite(degree_one_start(r), "regular start", 1)?;
    Ok([seasonal_ar_coordinate(acf, frequency)?, contract_start(ma)])
}

/// STAVAL's seasonal AR start as a TRANS1 coordinate: the ratio of the
/// autocorrelations two seasons and one season apart, held to [0, .8].
pub(super) fn seasonal_ar_coordinate(acf: &[f64; 36], frequency: Frequency) -> Result<f64, Error> {
    let s = usize::from(frequency.periods_per_year());
    let seasonal = finite(
        finite(acf[2 * s - 1], "seasonal ACF", 2 * s)?
            / dividecheck(finite(acf[s - 1], "seasonal ACF", s)?),
        "seasonal AR ratio",
        2 * s,
    )?
    .clamp(0.0, 0.8);
    Ok(contract_start(seasonal))
}

/// STAVAL's regular AR start for regular MA order `q`, and the MA starts
/// from the autocorrelations it leaves. It removes the AR from the seasonal
/// lag only when `seasonal`.
pub(crate) fn starting_parameters(
    acf: &[f64; 36],
    frequency: Frequency,
    seasonal: bool,
    q: usize,
) -> Result<ArmaStartValues, Error> {
    let s = usize::from(frequency.periods_per_year());
    let (j2, j3) = if s == 12 {
        (q + 2, q + 3)
    } else {
        (s + q + 2, 2 * s + q + 2)
    };
    for lag in [1, 2, 3, j2 - 1, j2, j3 - 1, j3]
        .into_iter()
        .chain([s - 1, s, s + 1].into_iter().filter(|_| seasonal))
    {
        finite(acf[lag - 1], "supplied ACF", lag)?;
    }
    // STAVAL: Q=0 uses r(1) directly. Otherwise the first denominator is
    // raw, unlike the next two DIVIDECHECK calls. Its infinite result may
    // select the sign fallback or be clamped to 0.9, so only the start that
    // comes out of those branches is checked.
    let b1 = if q == 0 { acf[0] } else { acf[q] / acf[q - 1] };
    let b2 = finite(acf[j2 - 1] / dividecheck(acf[j2 - 2]), "AR ratio2", j2)?;
    let b3 = finite(acf[j3 - 1] / dividecheck(acf[j3 - 2]), "AR ratio3", j3)?;
    let mut phi = if b1 * b2 <= 0.0 || b2 * b3 <= 0.0 {
        f64::from(0.2_f32)
    } else {
        (b1 + b2 + b3) / 3.0
    };
    if phi >= 1.0 {
        phi = 0.9;
    }
    let phi = finite(phi, "AR start", 1)?;
    let c = finite(phi.mul_add(phi, 1.0), "AR conditioning scale", 1)?;
    let r0 = finite(c - (phi + phi) * acf[0], "AR-conditioned lag-zero term", 0)?;
    let rp1 = finite(
        c.mul_add(acf[0], -(phi * (1.0 + acf[1]))),
        "AR-conditioned lag-one term",
        1,
    )?;
    let rp2 = finite(
        c.mul_add(acf[1], -(phi * (acf[0] + acf[2]))),
        "AR-conditioned lag-two term",
        2,
    )?;
    let mut r = *acf;
    r[0] = finite(
        rp1 / dividecheck(r0),
        "conditioned lag-one autocorrelation",
        1,
    )?;
    r[1] = finite(
        rp2 / dividecheck(r0),
        "conditioned lag-two autocorrelation",
        2,
    )?;
    if seasonal {
        let rmq = finite(
            c.mul_add(r[s - 1], -(phi * (r[s - 2] + r[s]))),
            "AR-conditioned seasonal-lag term",
            s,
        )?;
        r[s - 1] = finite(
            rmq / dividecheck(r0),
            "conditioned seasonal autocorrelation",
            s,
        )?;
    }
    let th = if q == 0 { 0.0 } else { degree_one_start(r[0]) };
    let bth = if seasonal {
        seasonal_start(r[s - 1])
    } else {
        0.0
    };
    Ok(ArmaStartValues {
        parameters: [
            phi,
            finite(th, "regular start", 1)?,
            finite(bth, "seasonal start", s)?,
        ],
        conditioned_acf: r,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::arma::model::ArOrder;
    use crate::seats::refit::test_support::acf;

    /// SEARCH for the quarterly (1,1,1)(0,1,1) on a wave of `scale` radians
    /// a quarter plus a sawtooth.
    fn search_wave(scale: f64) -> ArmaRefitResult<3> {
        let wd = (0..64)
            .map(|i| (i as f64 * scale).sin() + 0.1 * ((i * 7 % 11) as f64))
            .collect::<Vec<_>>();
        search_arma(
            Frequency::Quarterly,
            &wd,
            [0.5, 0.55, 0.3],
            [false; 3],
            ArmaLayout::RegularSeasonal(ArOrder::One),
        )
        .unwrap()
        .0
    }

    #[test]
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        ignore = "pins exact results of the Apple silicon math library"
    )]
    fn three_coordinates_converge() {
        let refit = search_wave(0.31);
        assert_eq!(
            refit.parameters,
            [0.9197467636446237, 0.16255587326224566, 0.3725412628157334]
        );
        assert_eq!(refit.bound_mask, [false; 3]);
        assert_eq!(refit.fixed, 0);
        assert_eq!(refit.objective, 16.3671539472273);
        assert_eq!(refit.determinant_scale, 1.0049722714837954);
        assert_eq!(refit.evaluations, 37);
        assert_eq!(refit.scaled_residuals.len(), 68);
        assert_eq!(refit.scaled_residuals[0], 0.06211752075649984);
        assert_eq!(refit.scaled_residuals[1], -0.3545717911127099);
        assert_eq!(refit.scaled_residuals[67], -0.15201325256017223);
        assert_eq!(refit.coefficients.parameters, refit.parameters);
        assert_eq!(refit.coefficients.phist, [0.9197467636446237]);
        assert_eq!(
            refit.coefficients.thstar,
            [
                0.16255587326224566,
                0.0,
                0.0,
                0.3725412628157334,
                -0.06055877030323131
            ]
        );
        assert_eq!(refit.covariance, ArmaRefitCovariance::NotEstimated);
    }

    #[test]
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        ignore = "pins exact results of the Apple silicon math library"
    )]
    fn a_common_factor_ends_search_at_the_last_completed_evaluation() {
        // The AR coordinate is fixed at its bound, and the derivative call
        // that follows finds the AR and MA roots within 0.1 of each other.
        let refit = search_wave(1.3);
        assert_eq!(
            refit.parameters,
            [0.9703999999999999, 0.9089604644832863, -0.08795285939629922]
        );
        assert_eq!(refit.bound_mask, [true, false, false]);
        assert_eq!(refit.fixed, 1);
        assert_eq!(refit.objective, 38.37737241835522);
        assert_eq!(refit.determinant_scale, 1.0132967598197344);
        assert_eq!(refit.evaluations, 10);
        assert_eq!(refit.scaled_residuals.len(), 68);
        assert_eq!(refit.scaled_residuals[0], -0.0479387831680194);
        assert_eq!(refit.scaled_residuals[1], 0.09984024109639628);
        assert_eq!(refit.scaled_residuals[67], -0.18974731355184313);
        // CALCFX keeps the coefficients of the call the common factor
        // stopped, and the coordinates of the last one it completed.
        assert_eq!(
            refit.coefficients.parameters,
            [0.99, 0.9091584644832863, -0.08795285939629922]
        );
        assert_eq!(refit.coefficients.phist, [0.99]);
        assert_eq!(
            refit.coefficients.thstar,
            [
                0.9091584644832863,
                0.0,
                0.0,
                -0.08795285939629922,
                0.07996308659565378
            ]
        );
        assert_eq!(
            refit.coefficients.previous,
            Some([0.99, 0.9089604644832863, -0.08795285939629922])
        );
    }

    #[test]
    fn the_ar_start_leaves_its_autocorrelations_for_the_ma_starts() {
        let monthly = |seasonal, q| {
            let start = starting_parameters(&acf(), Frequency::Monthly, seasonal, q).unwrap();
            (
                start.parameters,
                [start.conditioned_acf[0], start.conditioned_acf[1]],
            )
        };
        assert_eq!(
            monthly(true, 1),
            (
                [0.6999999999999998, 0.08995278023848066, 0.49913151556871255],
                [-0.08923076923076909, -0.05353846153846142]
            )
        );
        assert_eq!(
            monthly(false, 2),
            (
                [0.6333333333333333, 0.032269269127518416, 0.0],
                [-0.03223570190641241, -0.01934142114384748]
            )
        );
        assert_eq!(
            monthly(true, 0),
            (
                [0.6, 0.0, 0.49907486255843553],
                [-8.18789480661053e-17, 2.0261570199409113e-17]
            )
        );
        let quarterly = |seasonal, q| {
            starting_parameters(&acf(), Frequency::Quarterly, seasonal, q)
                .unwrap()
                .parameters
        };
        assert_eq!(quarterly(true, 1), [0.6, 0.0, 0.4438340231115782]);
        assert_eq!(
            quarterly(false, 2),
            [0.7999999999999999, 0.15669646058689368, 0.0]
        );
    }

    #[test]
    fn ma_and_seasonal_ar_starts_are_trans1_coordinates() {
        let acf = acf();
        assert_eq!(
            ma1_start(&acf, Frequency::Monthly, true),
            Ok([-0.8909999874606729, 0.49869391667362745])
        );
        assert_eq!(
            ma1_start(&acf, Frequency::Quarterly, true),
            Ok([-0.8909999874606729, 0.41905574109809957])
        );
        assert_eq!(
            ma1_start(&acf, Frequency::Quarterly, false),
            Ok([-0.8909999874606729, 0.0])
        );
        assert_eq!(
            seasonal_ma_start(&acf, Frequency::Monthly),
            Ok(0.49869391667362745)
        );
        assert_eq!(
            seasonal_ar_start(&acf, Frequency::Monthly),
            Ok([0.0021767823359999986, -0.8909999874606729])
        );
        assert_eq!(
            seasonal_ar_start(&acf, Frequency::Quarterly),
            Ok([0.1296, -0.8909999874606729])
        );
    }

    /// Autocorrelations that are zero except at lags 1, 2, 4 and 12.
    fn sparse_acf(r1: f64, r2: f64) -> [f64; 36] {
        let mut acf = [0.0; 36];
        acf[0] = r1;
        acf[1] = r2;
        acf[3] = 0.2;
        acf[11] = -0.2;
        acf
    }

    #[test]
    fn the_ma2_start_follows_the_first_two_autocorrelations() {
        let start = |r1, r2, frequency, seasonal| {
            ma2_start(&sparse_acf(r1, r2), frequency, seasonal)
                .unwrap()
                .coordinates
        };
        assert_eq!(
            start(-0.45, 0.1, Frequency::Quarterly, true),
            [0.45141622964513617, -0.12753322857899985, 0.416515138991168]
        );
        assert_eq!(
            start(-0.45, 0.1, Frequency::Monthly, true),
            [0.45141622964513617, -0.12753322857899985, 0.583484861008832]
        );
        // Autocorrelations no MA(2) has are moved inside its region first.
        assert_eq!(
            start(0.6, 0.7, Frequency::Quarterly, false),
            [-0.20981416483776477, -0.8107247606588193, 0.0]
        );
        assert_eq!(
            ma2_start(&acf(), Frequency::Monthly, true)
                .unwrap()
                .coordinates,
            [
                -0.4064718799500425,
                -0.7294741511340003,
                0.49869391667362745
            ]
        );
    }

    #[test]
    fn a_zero_first_ma2_start_is_held_through_search() {
        let start = ma2_start(&sparse_acf(0.0, 0.3), Frequency::Quarterly, true).unwrap();
        assert_eq!(
            start.coordinates,
            [-0.0, -0.33333333333333326, 0.416515138991168]
        );
        assert!(start.fixes_first);
        let start = ma2_start(&sparse_acf(-0.45, 0.1), Frequency::Quarterly, true).unwrap();
        assert!(!start.fixes_first);
    }
}
