use super::*;
use crate::seats::arma::refit::import_ma2;
use crate::seats::decomposition::root_allocation::{
    RegularArCoefficients, cubic_coefficients, import_regular_ar,
};
use crate::seats::failure::{Error, Failure, Stage};

/// Whether `result` is a NaN imported regular AR coefficient, at `index` when
/// one is given.
fn is_nan_import<T>(result: Result<T, Error>, index: Option<usize>) -> bool {
    matches!(
        result,
        Err(Error {
            stage: Stage::CanonicalDecomposition,
            failure: Failure::Unusable {
                quantity: "imported regular AR coefficient",
                index: Some(at),
                value,
            },
        }) if value.is_nan() && index.is_none_or(|index| index == at)
    )
}

#[test]
fn degree_one_groups_keep_signed_zeros_and_nans_and_clip_at_the_cuts() {
    let ma_group = |p| trans0_degree_one(p, MA_BOUND);
    let ar_group = |p| trans0_degree_one(p, AR_BOUND);
    let low = 0.9702_f64;
    let high = 0.9702_f64.next_up().next_up();
    // (p, MA group, AR group): the groups differ only in the clip value.
    for (p, ma, ar) in [
        (0.0, 0.0, 0.0),
        (-0.0, -0.0, -0.0),
        (0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (low, low, low),
        (low.next_up(), 0.99, 1.0),
        (-low.next_up(), -low.next_up(), -low.next_up()),
        (-high, -0.99, -1.0),
        (0.98, 0.99, 1.0),
        (-0.98, -0.99, -1.0),
        (f64::INFINITY, 0.99, 1.0),
        (f64::NEG_INFINITY, -0.99, -1.0),
        (f64::MAX, 0.99, 1.0),
        (-f64::MAX, -0.99, -1.0),
    ] {
        assert_eq!(ma_group(p).to_bits(), f64::to_bits(ma), "MA p={p:e}");
        assert_eq!(ar_group(p).to_bits(), f64::to_bits(ar), "AR p={p:e}");
    }
    for nan in [f64::NAN, -f64::NAN] {
        assert_eq!(ma_group(nan).to_bits(), nan.to_bits());
        assert_eq!(ar_group(nan).to_bits(), nan.to_bits());
    }
}

// TRANS0 (transc.f) sets PHITH(2)=UR=1 when |1-PHITH(2)|<1e-9, so the first
// coordinate divides by exactly zero. Build 62 keeps the IEEE ±infinity and
// its clip loop maps it to ±UR (AR, J<=IPRS) or ±XL=.99 (MA); TRANS2 then
// writes C(1)=X(1)*(1-X(2)). The 0/0 case leaves a NaN, on which Build 62's
// SEATS hangs, so it stays a typed error.
const UNIT_BAND: [f64; 3] = [1.0, 1.0 - 5e-10, 1.0 + 5e-10];

#[test]
fn an_ar2_import_clips_an_infinite_coordinate() {
    for second in UNIT_BAND {
        for (first, c1) in [(0.3, 0.0), (1e-10, 0.0), (-0.3, -0.0), (-1e-10, -0.0)] {
            let Ok(RegularArCoefficients::Two(imported)) =
                import_regular_ar(RegularArCoefficients::Two([first, second]))
            else {
                panic!("phi=({first:e}, {second:e}) was not imported");
            };
            assert_eq!(
                imported.map(f64::to_bits),
                [f64::to_bits(c1), 1.0_f64.to_bits()],
                "phi=({first:e}, {second:e})"
            );
        }
        for first in [0.0, -0.0] {
            assert!(is_nan_import(
                import_regular_ar(RegularArCoefficients::Two([first, second])),
                Some(0)
            ));
        }
    }
    // Outside the band an overflowing quotient follows the same IEEE clip:
    // X(1) = -UR, so C(1) = -1 * (1 - 0.5).
    assert_eq!(
        import_regular_ar(RegularArCoefficients::Two([-f64::MAX, 0.5])).unwrap(),
        RegularArCoefficients::Two([-0.5, 0.5])
    );
}

#[test]
fn an_ma2_import_clips_an_infinite_coordinate() {
    let c1 = 0.99_f64 * (1.0 - 0.99);
    assert_eq!(c1.to_bits(), 0.00990000000000001_f64.to_bits());
    for second in UNIT_BAND {
        for (first, expected) in [(0.3, c1), (1e-10, c1), (-0.3, -c1), (-1e-10, -c1)] {
            let imported = import_ma2([first, second])
                .unwrap_or_else(|error| panic!("theta=({first:e}, {second:e}): {error}"));
            assert_eq!(
                imported.map(f64::to_bits),
                [expected.to_bits(), 0.99_f64.to_bits()],
                "theta=({first:e}, {second:e})"
            );
        }
        for first in [0.0, -0.0] {
            assert!(matches!(
                import_ma2([first, second]),
                Err(Error {
                    stage: Stage::Refit,
                    failure: Failure::Unusable {
                        quantity: "MA2 import coordinate",
                        index: Some(1),
                        value,
                    },
                }) if value.is_nan()
            ));
        }
    }
    assert_eq!(
        import_ma2([f64::MAX, 0.5]).unwrap(),
        [0.99 * (1.0 - 0.5), 0.5]
    );
}

#[test]
fn an_ar3_import_clips_infinite_coordinates() {
    let import = |phi: [f64; 3]| import_regular_ar(RegularArCoefficients::Three(phi));
    // Each case names the clipped TRANS0 coordinates X(1..3) TRANS2 receives.
    let assert_clips = |phi: [f64; 3], x: [f64; 3]| {
        let Ok(RegularArCoefficients::Three(imported)) = import(phi) else {
            panic!("phi={phi:?} was not imported");
        };
        assert_eq!(
            imported.map(f64::to_bits),
            cubic_coefficients(x).map(f64::to_bits),
            "phi={phi:?}"
        );
        imported
    };
    let assert_nan = |phi: [f64; 3]| {
        assert!(is_nan_import(import(phi), None), "phi={phi:?}");
    };
    // PHITH(3) reset to 1: X(1) = ±Inf clips to ±UR and X(2) = X(2)/±Inf - 1
    // is exactly -1.
    for third in UNIT_BAND {
        for (first, second, x1) in [(0.3, 0.2, 1.0), (1e-10, 0.0, 1.0), (-0.3, -0.2, -1.0)] {
            let c = assert_clips([first, second, third], [x1, -1.0, 1.0]);
            assert_eq!(c.map(f64::to_bits), [-1.0_f64, 1.0, 1.0].map(f64::to_bits));
        }
        assert_nan([0.3, -0.3, third]);
    }
    // PHITH(3) = -1: X(2) = 1 + (PHITH(1) - PHITH(2))/0 = ±Inf, carried through
    // the second division by 1 + X(1) and clipped to ±UR.
    let x1 = 0.5 * ((0.2 + 0.1) / 2.0 + 1.0);
    assert_clips([0.2, 0.1, -1.0], [x1, 1.0, -1.0]);
    assert_clips([0.1, 0.2, -1.0], [x1, -1.0, -1.0]);
    assert_clips([-3.0, -4.0, -1.0], [-1.0, -1.0, -1.0]);
    assert_nan([0.2, 0.2, -1.0]);
    // X(1) within 1e-9 of -1 is reset to -UR, so X(2)/(1 + X(1)) divides by +0.
    for first in [-0.25, -0.25 + 5e-10, -0.25 - 5e-10] {
        let c = assert_clips([first, -1.25, 0.5], [-1.0, 1.0, 0.5]);
        assert_eq!(c.map(f64::to_bits), [-1.5_f64, 0.0, 0.5].map(f64::to_bits));
    }
    assert_clips([-1.75, 0.25, 0.5], [-1.0, -1.0, 0.5]);
    assert_nan([-1.5, 0.0, 0.5]);
    // Outside every band an overflowing sum follows the same IEEE clip, unless
    // both sums overflow and X(2)/(1 + X(1)) is inf/inf.
    assert_clips([f64::MAX, f64::MAX, 0.5], [1.0, -1.0, 0.5]);
    assert_nan([f64::MAX, -0.1 * f64::MAX, 0.5]);
}
