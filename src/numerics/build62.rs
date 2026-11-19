//! Constants that Build 62 writes as decimal literals, one definition each.
//!
//! Each constant keeps the decimal literal exactly as the Fortran source
//! writes it, never the nearest `std::f64::consts` value: a one-ULP
//! difference in these literals changes results. A constant here is shared
//! only where the Fortran routines use the same literal for the same purpose;
//! equal values with different meanings (for example `enorm.f`'s `rdwarf`,
//! which also happens to be `3.834d-20`) stay in the module that uses them.

/// `dpmpar.f` `dmach(1)`: the machine epsilon `2.220446d-16` as written in the
/// source, not `f64::EPSILON` (`2.220446049250313e-16`).
pub(crate) const BUILD_62_MACHINE_EPSILON: f64 = 2.220_446e-16;

/// `dpmpar.f` `dmach(2)`: the smallest positive magnitude `2.225074d-308`,
/// used for `ddot.f`'s log guard, `lmpar.f`'s `dwarf` and `ttest.f`'s `xl`.
pub(crate) const BUILD_62_MIN_POSITIVE: f64 = 2.225_074e-308;

/// The shortened pi literal `3.14159265358979` that `rgarma.f`, `spgrh`,
/// `hndtrn.f`, `setcv.f` and `lkshnk` write instead of a full-precision pi.
///
/// Its binary64 encoding is `0x400921FB54442D11`, seven ULPs below
/// `std::f64::consts::PI`.
#[expect(
    clippy::approx_constant,
    reason = "The shortened source literal is the point of this constant."
)]
pub(crate) const BUILD_62_PI: f64 = 3.14159265358979;

/// `idotlr.f` (`rbmse`), `sdxtrm.f` and `histx.f` divide a median absolute
/// deviation by `0.6745D0`, the standard normal 0.75 quantile to four places,
/// to estimate a standard deviation.
pub(crate) const BUILD_62_MAD_SCALE: f64 = 0.6745;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_epsilon_is_the_source_literal_not_the_ieee_one() {
        assert_eq!(
            BUILD_62_MACHINE_EPSILON,
            "2.220446e-16".parse::<f64>().unwrap()
        );
        assert_ne!(BUILD_62_MACHINE_EPSILON, f64::EPSILON);
    }

    #[test]
    fn min_positive_is_the_source_literal_not_the_ieee_one() {
        assert_eq!(
            BUILD_62_MIN_POSITIVE,
            "2.225074e-308".parse::<f64>().unwrap()
        );
        assert_ne!(BUILD_62_MIN_POSITIVE, f64::MIN_POSITIVE);
    }

    #[test]
    fn pi_is_the_shortened_source_literal() {
        assert_eq!(BUILD_62_PI.to_bits(), 0x4009_21fb_5444_2d11);
        assert_ne!(BUILD_62_PI, std::f64::consts::PI);
    }
}
