//! Series and autocorrelations for the SEARCH and STAVAL tests. They use no
//! library function, so they are the same on every platform.

/// Autocorrelations that decay by 0.6 a lag, half as large again at every
/// fourth lag.
pub(in crate::seats) fn acf() -> [f64; 36] {
    let mut acf = [0.0; 36];
    let mut power = 1.0;
    for (lag, value) in acf.iter_mut().enumerate() {
        power *= 0.6;
        *value = if lag % 4 == 3 { 1.5 * power } else { power };
    }
    acf
}

/// Uniform noise on [-0.5, 0.5) from a fixed linear congruential generator.
pub(in crate::seats) fn noise(len: usize) -> Vec<f64> {
    let mut state = 1_u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// `noise` differenced at `lag`. Its MA root on the unit circle puts the
/// coordinate for that lag on SEARCH's upper bound.
pub(in crate::seats) fn differenced_noise(len: usize, lag: usize) -> Vec<f64> {
    let e = noise(len + lag);
    (lag..len + lag).map(|t| e[t] - e[t - lag]).collect()
}

/// The running sum of `noise`.
pub(in crate::seats) fn random_walk(len: usize) -> Vec<f64> {
    noise(len)
        .into_iter()
        .scan(0.0, |level, value| {
            *level += value;
            Some(*level)
        })
        .collect()
}

/// Values near 1 and -1 in turn: an AR(1) coefficient close to -1.
pub(in crate::seats) fn alternating(len: usize) -> Vec<f64> {
    noise(len)
        .into_iter()
        .enumerate()
        .map(|(t, e)| {
            if t % 2 == 0 {
                1.0 + 0.2 * e
            } else {
                -1.0 - 0.2 * e
            }
        })
        .collect()
}
