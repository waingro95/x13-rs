//! The import of the regARIMA coefficients into SEATS.
//!
//! SEATS does not take the AR and MA coefficients of the regARIMA fit as they
//! are. NMLMDL (`nmlmdl.f`) reverses their signs and sends each AR and MA
//! group through TRANS0 (`transc.f`), which maps the group to the coordinates
//! SEARCH works in and sets a coordinate near or beyond a bound to that
//! bound; TRANS2 maps the coordinates back. A model at the edge of
//! invertibility or stationarity so enters SEATS at the bound. The SEATS
//! modules call the result the imported coefficients.
//!
//! This module holds the bounds and the clip of a group of degree one. The
//! families with groups of degree two and three apply the clip in their own
//! imports.

/// TRANS0's MA limit XL: an MA group outside the cut is set to ±XL.
pub(super) const MA_BOUND: f64 = 0.99;
/// TRANS0's AR limit UR (J<=IPRS): an AR group outside the cut is set to ±UR.
pub(super) const AR_BOUND: f64 = 1.0;

/// Degree-one TRANS0 then TRANS2 (transc.f:195–202,221–235 and 58–62,127–129)
/// on one parameter group: TRANS0 sees `PHITH = −p` and TRANS2 returns `−C`.
/// `bound` is [`MA_BOUND`] for an MA group (IPRS=0) and [`AR_BOUND`] for an
/// AR group.
pub(super) fn trans0_degree_one(p: f64, bound: f64) -> f64 {
    -trans0_clip(-p, bound)
}

/// TRANS0's clip of one degree-one value. The cut is always taken on the
/// ±XL scale ([`ma_bound_ratio`]); only the value it snaps to is `±bound`.
/// A NaN fails both comparisons and passes through.
pub(super) fn trans0_clip(x: f64, bound: f64) -> f64 {
    let xtest = ma_bound_ratio(x);
    if xtest < 0.01 {
        -bound
    } else if xtest > 0.99 {
        bound
    } else {
        x
    }
}

/// TRANS0's normalized position of `x` on `[−XL, XL]`.
fn ma_bound_ratio(x: f64) -> f64 {
    let xmin = -MA_BOUND;
    let xmax = MA_BOUND;
    (x - xmin) / (xmax - xmin)
}

#[cfg(test)]
mod tests;
