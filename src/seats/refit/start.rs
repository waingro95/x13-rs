//! STAVAL's starting values (`ansub1.f`) for SEARCH, and TRANS1's move of a
//! start away from SEARCH's bounds.
//!
//! [`starting_parameters`] is STAVAL for a model with a regular AR part and
//! no regular MA: it starts the AR coefficient from ratios of
//! autocorrelations, removes that AR from the seasonal-lag autocorrelation
//! and starts the seasonal MA from what is left. The MA starts,
//! [`degree_one_start`] and [`seasonal_start`], also serve the models
//! without an AR part.

use crate::Frequency;
use crate::seats::failure::usable;
use crate::seats::{Error, Stage};

/// STAVAL's starts for the first regular AR coefficient and the seasonal MA.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Q0StartValues {
    pub(crate) regular: f64,
    pub(crate) seasonal_ma: f64,
}

/// `value` when it is finite, else `Unusable` at the autocorrelation lag it
/// belongs to.
fn finite(value: f64, quantity: &'static str, lag: usize) -> Result<f64, Error> {
    usable(quantity, Some(lag), value).map_err(|failure| failure.at(Stage::Refit))
}

/// DIVIDECHECK: a divisor below 1e-9 in magnitude, of either sign, becomes
/// +1e-9.
fn dividecheck(value: f64) -> f64 {
    if value.abs() < 1e-9 { 1e-9 } else { value }
}

/// STAVAL with P>0 and Q=0, from the autocorrelations `acf` of the
/// differenced series (lag 1 first).
pub(crate) fn starting_parameters(
    acf: &[f64; 36],
    frequency: Frequency,
) -> Result<Q0StartValues, Error> {
    let s = usize::from(frequency.periods_per_year());
    let (j2, j3) = if s == 12 { (2, 3) } else { (6, 10) };
    for lag in [1, 2, 3, j2 - 1, j2, j3 - 1, j3, s - 1, s, s + 1] {
        finite(acf[lag - 1], "autocorrelation input", lag)?;
    }
    let phi = regular_start(acf, frequency)?;
    let c = finite(phi.mul_add(phi, 1.0), "AR conditioning scale", 1)?;
    // gfortran rounds STAVAL's doubled-Phi product before the subtraction in
    // r0. In rp1, rp2 and rmq it rounds Phi times the sum and fuses the
    // product with c.
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
    // STAVAL replaces r(1) and r(2) by these; with Q=0 it reads neither
    // again.
    finite(
        rp1 / dividecheck(r0),
        "conditioned lag-one autocorrelation",
        1,
    )?;
    finite(
        rp2 / dividecheck(r0),
        "conditioned lag-two autocorrelation",
        2,
    )?;
    let rmq = finite(
        c.mul_add(acf[s - 1], -(phi * (acf[s - 2] + acf[s]))),
        "AR-conditioned seasonal-lag term",
        s,
    )?;
    let seasonal_r = finite(
        rmq / dividecheck(r0),
        "conditioned seasonal autocorrelation",
        s,
    )?;
    let bth = if seasonal_r.abs() < 0.5 {
        let estimate = ((-4.0_f64).mul_add(seasonal_r * seasonal_r, 1.0).sqrt() - 1.0)
            / dividecheck(2.0 * seasonal_r);
        0.4_f64.mul_add(estimate, 0.5)
    } else {
        -0.9_f64.copysign(seasonal_r)
    };
    finite(bth, "seasonal start", s)?;
    Ok(Q0StartValues {
        regular: phi,
        seasonal_ma: bth,
    })
}

/// STAVAL's start for the first regular AR coefficient when Q=0: the mean of
/// three autocorrelation ratios, or single-precision .2 when they change
/// sign. The ratios' lags depend on the frequency.
pub(crate) fn regular_start(acf: &[f64; 36], frequency: Frequency) -> Result<f64, Error> {
    let (j2, j3) = if frequency == Frequency::Monthly {
        (2, 3)
    } else {
        (6, 10)
    };
    for lag in [1, j2 - 1, j2, j3 - 1, j3] {
        finite(acf[lag - 1], "autocorrelation input", lag)?;
    }
    let b1 = acf[0];
    let b2 = finite(acf[j2 - 1] / dividecheck(acf[j2 - 2]), "AR ratio2", j2)?;
    let b3 = finite(acf[j3 - 1] / dividecheck(acf[j3 - 2]), "AR ratio3", j3)?;
    let mut phi = if b1 * b2 <= 0.0 || b2 * b3 <= 0.0 {
        f64::from(0.2_f32)
    } else {
        finite((b1 + b2 + b3) / 3.0, "AR average", 1)?
    };
    if phi >= 1.0 {
        phi = 0.9;
    }
    Ok(phi)
}

/// STAVAL's degree-one MA start from the lag's autocorrelation `r`.
pub(crate) fn degree_one_start(r: f64) -> f64 {
    if r.abs() < 0.5 {
        let denominator = r + r;
        let denominator = if denominator.abs() < 1e-9 {
            1e-9
        } else {
            denominator
        };
        ((-4.0_f64).mul_add(r * r, 1.0).sqrt() - 1.0) / denominator
    } else {
        -0.9_f64.copysign(r)
    }
}

/// STAVAL's seasonal MA start from the seasonal-lag autocorrelation `r`.
pub(crate) fn seasonal_start(r: f64) -> f64 {
    if r.abs() < 0.5 {
        0.4_f64.mul_add(degree_one_start(r), 0.5)
    } else {
        -0.9_f64.copysign(r)
    }
}

/// TRANS1's last step: a start outside `.95*xmin + .05*xmax` and
/// `.05*xmin + .95*xmax` moves to the nearer of the two.
pub(crate) fn contract_start(parameter: f64) -> f64 {
    // TRANS1's .95 and .05 are single precision. gfortran rounds the xmax
    // product and fuses the xmin one.
    let a = f64::from(0.95_f32);
    let b = f64::from(0.05_f32);
    let lower = a.mul_add(-0.99, b * 0.99);
    let upper = b.mul_add(-0.99, a * 0.99);
    parameter.max(lower).min(upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::refit::test_support::acf;

    #[test]
    fn ma_starts_invert_a_small_autocorrelation() {
        assert_eq!(degree_one_start(0.2), -0.20871215252208003);
        assert_eq!(degree_one_start(-0.3), 0.33333333333333326);
        assert_eq!(degree_one_start(0.49), -0.8173495026313019);
        assert_eq!(degree_one_start(0.0), 0.0);
        assert_eq!(degree_one_start(1e-10), 0.0);
        assert_eq!(seasonal_start(0.2), 0.416515138991168);
        assert_eq!(seasonal_start(-0.3), 0.6333333333333333);
        assert_eq!(seasonal_start(0.49), 0.17306019894747923);
        assert_eq!(seasonal_start(0.0), 0.5);
    }

    #[test]
    fn ma_starts_are_point_nine_from_half_an_autocorrelation_up() {
        for start in [degree_one_start, seasonal_start] {
            assert_eq!(start(0.5), -0.9);
            assert_eq!(start(0.8), -0.9);
            assert_eq!(start(-0.5), 0.9);
        }
    }

    #[test]
    fn a_start_near_a_bound_moves_inside() {
        assert_eq!(contract_start(0.5), 0.5);
        assert_eq!(contract_start(0.95), 0.8909999874606728);
        assert_eq!(contract_start(1.0), 0.8909999874606728);
        assert_eq!(contract_start(-1.0), -0.8909999874606729);
        assert_eq!(contract_start(f64::NAN), -0.8909999874606729);
    }

    #[test]
    fn the_ar_start_averages_three_ratios_of_one_sign() {
        assert_eq!(regular_start(&acf(), Frequency::Monthly), Ok(0.6));
        assert_eq!(regular_start(&acf(), Frequency::Quarterly), Ok(0.6));
    }

    #[test]
    fn the_ar_start_is_single_precision_point_two_when_the_ratios_change_sign() {
        let mut acf = acf();
        acf[1] = -0.2;
        assert_eq!(
            regular_start(&acf, Frequency::Monthly),
            Ok(0.20000000298023224)
        );
        // The quarterly ratios use lags 5, 6, 9 and 10.
        assert_eq!(regular_start(&acf, Frequency::Quarterly), Ok(0.6));
    }

    #[test]
    fn the_seasonal_ma_start_follows_the_ar_conditioned_seasonal_lag() {
        assert_eq!(
            starting_parameters(&acf(), Frequency::Monthly),
            Ok(Q0StartValues {
                regular: 0.6,
                seasonal_ma: 0.49907486255843553,
            })
        );
        assert_eq!(
            starting_parameters(&acf(), Frequency::Quarterly),
            Ok(Q0StartValues {
                regular: 0.6,
                seasonal_ma: 0.4438340231115782,
            })
        );
    }
}
