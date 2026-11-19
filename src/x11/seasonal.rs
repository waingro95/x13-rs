//! Seasonal subseries filters from Build 62 `vsfb.f` and `endsf.f`.
//!
//! - [`SeasonalFilterSet`] holds one filter per calendar month or quarter,
//!   January (or the first quarter) first, never rotated by the series start.
//!   It has exactly `period` entries.
//! - [`ResolvedSeasonalFilter`] is what `vsfb.f` runs: a declared
//!   [`x11::SeasonalFilter`] after the provisional MSR and X-11 default
//!   requests have been resolved. Declared sets keep the public type because
//!   MSR selection, the quality history and error reports read the request;
//!   smoothing, centering and shrinkage take only resolved sets, so
//!   [`smooth_subseries`] needs no arm for a provisional request.

use std::ops::Index;

use crate::{Frequency, numerics, x11};

/// A seasonal filter `vsfb.f` can run: a declared filter after MSR and the
/// X-11 default have been resolved for the current section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResolvedSeasonalFilter {
    S3x1,
    S3x3,
    S3x5,
    S3x9,
    S3x15,
    Stable,
}

impl ResolvedSeasonalFilter {
    /// `vsfb.f`: the declared filter, or `provisional` for an MSR or X-11
    /// default request.
    pub(super) fn resolve(declared: x11::SeasonalFilter, provisional: Self) -> Self {
        match declared {
            x11::SeasonalFilter::S3x1 => Self::S3x1,
            x11::SeasonalFilter::S3x3 => Self::S3x3,
            x11::SeasonalFilter::S3x5 => Self::S3x5,
            x11::SeasonalFilter::S3x9 => Self::S3x9,
            x11::SeasonalFilter::S3x15 => Self::S3x15,
            x11::SeasonalFilter::Stable => Self::Stable,
            x11::SeasonalFilter::Msr | x11::SeasonalFilter::X11Default => provisional,
        }
    }
}

impl From<ResolvedSeasonalFilter> for x11::SeasonalFilter {
    fn from(filter: ResolvedSeasonalFilter) -> Self {
        match filter {
            ResolvedSeasonalFilter::S3x1 => Self::S3x1,
            ResolvedSeasonalFilter::S3x3 => Self::S3x3,
            ResolvedSeasonalFilter::S3x5 => Self::S3x5,
            ResolvedSeasonalFilter::S3x9 => Self::S3x9,
            ResolvedSeasonalFilter::S3x15 => Self::S3x15,
            ResolvedSeasonalFilter::Stable => Self::Stable,
        }
    }
}

/// One filter per calendar period, in calendar order: exactly twelve monthly
/// or four quarterly entries. Index it by zero-based calendar period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SeasonalFilterSet<F> {
    Monthly([F; 12]),
    Quarterly([F; 4]),
}

impl<F: Copy> SeasonalFilterSet<F> {
    /// The same filter for every calendar period of `frequency`.
    pub(super) fn uniform(frequency: Frequency, filter: F) -> Self {
        match frequency {
            Frequency::Monthly => Self::Monthly([filter; 12]),
            Frequency::Quarterly => Self::Quarterly([filter; 4]),
        }
    }

    pub(super) fn frequency(&self) -> Frequency {
        match self {
            Self::Monthly(_) => Frequency::Monthly,
            Self::Quarterly(_) => Frequency::Quarterly,
        }
    }

    pub(super) fn as_slice(&self) -> &[F] {
        match self {
            Self::Monthly(filters) => filters,
            Self::Quarterly(filters) => filters,
        }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [F] {
        match self {
            Self::Monthly(filters) => filters,
            Self::Quarterly(filters) => filters,
        }
    }

    pub(super) fn map<G>(self, f: impl FnMut(F) -> G) -> SeasonalFilterSet<G> {
        match self {
            Self::Monthly(filters) => SeasonalFilterSet::Monthly(filters.map(f)),
            Self::Quarterly(filters) => SeasonalFilterSet::Quarterly(filters.map(f)),
        }
    }
}

impl SeasonalFilterSet<x11::SeasonalFilter> {
    /// `vsfb.f`: resolve the provisional requests without changing explicit
    /// filters. Declared Stable entries stay Stable for `vsfc.f` centering.
    pub(super) fn resolve(
        &self,
        provisional: ResolvedSeasonalFilter,
    ) -> SeasonalFilterSet<ResolvedSeasonalFilter> {
        self.map(|filter| ResolvedSeasonalFilter::resolve(filter, provisional))
    }
}

impl From<x11::SeasonalFilters> for SeasonalFilterSet<x11::SeasonalFilter> {
    fn from(filters: x11::SeasonalFilters) -> Self {
        match filters {
            x11::SeasonalFilters::Monthly(filters) => Self::Monthly(filters),
            x11::SeasonalFilters::Quarterly(filters) => Self::Quarterly(filters),
        }
    }
}

impl<F: Copy> Index<usize> for SeasonalFilterSet<F> {
    type Output = F;

    fn index(&self, calendar_index: usize) -> &F {
        &self.as_slice()[calendar_index]
    }
}

/// `vsfb.f`: smooths one calendar period's SI subseries with `filter`. A 3x15
/// filter takes the mean of a subseries shorter than 20, as the stable filter
/// always does.
pub(super) fn smooth_subseries(input: &[f64], filter: ResolvedSeasonalFilter) -> Vec<f64> {
    use ResolvedSeasonalFilter::*;
    match filter {
        Stable => vec![mean(input); input.len()],
        S3x15 if input.len() < 20 => vec![mean(input); input.len()],
        S3x5 => numerics::s3x5_subseries(input),
        S3x1 => s3x1(input),
        S3x3 => s3x3(input),
        S3x9 => end_weighted(input, 9, &END_WEIGHTS_3X9),
        S3x15 => end_weighted(input, 15, &END_WEIGHTS_3X15),
    }
}

fn s3x1(input: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; input.len()];
    let last = input.len() - 1;
    numerics::moving_average(input, &mut output, 0..input.len(), 1, 3);
    // vsfb.f: gfortran rounds the neighbour's product, then fuses the
    // endpoint's multiply-add.
    output[0] = input[0].mul_add(0.61, 0.39 * input[1]);
    output[last] = input[last].mul_add(0.61, 0.39 * input[last - 1]);
    output
}

fn s3x3(input: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; input.len()];
    let last = input.len() - 1;
    numerics::moving_average(input, &mut output, 0..input.len(), 3, 3);
    // vsfb.f: gfortran rounds `5 * term`, then fuses `11 * sum` into it.
    output[0] = (input[0] + input[1]).mul_add(11.0, 5.0 * input[2]) / 27.0;
    output[last] = (input[last] + input[last - 1]).mul_add(11.0, 5.0 * input[last - 2]) / 27.0;
    if input.len() == 3 {
        output[1] = (input[0] + input[1] + input[2]) / 3.0;
    } else {
        output[1] = input[3].mul_add(0.3, (input[0] + input[2]).mul_add(0.7, input[1])) / 2.7;
        output[last - 1] = input[last - 3].mul_add(
            0.3,
            (input[last] + input[last - 2]).mul_add(0.7, input[last - 1]),
        ) / 2.7;
    }
    output
}

/// S3x9 and S3x15: a 3x`terms` moving average with `endsf.f` endpoints.
fn end_weighted(input: &[f64], terms: usize, rows: &[&[f64]]) -> Vec<f64> {
    let mut output = vec![0.0; input.len()];
    numerics::moving_average(input, &mut output, 0..input.len(), 3, terms);
    apply_end_weights(input, &mut output, rows);
    output
}

/// Build 62 `totals.f` mean, including the sequential floating-point count.
fn mean(input: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for &value in input {
        total += value;
        count += 1.0;
    }
    total / count
}

/// `endsf.f`: endpoint `i` (0-based) from each end takes row `i` of the
/// table, whose `i + 1 + rows.len()` weights run from that end inward. A
/// subseries shorter than the row takes its mean instead.
fn apply_end_weights(input: &[f64], output: &mut [f64], rows: &[&[f64]]) {
    for (end_index, weights) in rows.iter().enumerate() {
        let back_index = input.len() - end_index - 1;
        if end_index > back_index {
            break;
        }
        if weights.len() > input.len() {
            output[end_index] = mean(input);
            if end_index != back_index {
                output[back_index] = output[end_index];
            }
        } else {
            // gfortran fuses each end's weighted sum in endsf.f, but not the
            // sum of the weights or the divisions.
            let (front, back, weight_sum) = weights.iter().zip(input).zip(input.iter().rev()).fold(
                (0.0, 0.0, 0.0),
                |(front, back, weight_sum), ((&weight, &first), &last)| {
                    (
                        weight.mul_add(first, front),
                        weight.mul_add(last, back),
                        weight_sum + weight,
                    )
                },
            );
            output[end_index] = front / weight_sum;
            if end_index == back_index {
                // endsf assigns both ends even when they coincide, dividing
                // the meeting value twice.
                output[end_index] /= weight_sum;
            } else {
                output[back_index] = back / weight_sum;
            }
        }
    }
}

/// `endsf.f`'s S3x9 endpoint weights, one row per endpoint.
const END_WEIGHTS_3X9: [&[f64]; 5] = [
    &[0.246, 0.221, 0.197, 0.173, 0.112, 0.051],
    &[0.208, 0.192, 0.176, 0.160, 0.144, 0.092, 0.028],
    &[0.173, 0.163, 0.154, 0.143, 0.133, 0.123, 0.079, 0.032],
    &[
        0.141, 0.137, 0.132, 0.128, 0.123, 0.117, 0.113, 0.075, 0.034,
    ],
    &[
        0.084, 0.120, 0.118, 0.117, 0.116, 0.114, 0.113, 0.111, 0.073, 0.034,
    ],
];

/// `endsf.f`'s S3x15 endpoint weights, one row per endpoint.
const END_WEIGHTS_3X15: [&[f64]; 8] = [
    &[
        0.16000, 0.16000, 0.16000, 0.16000, 0.16000, 0.06667, 0.06667, 0.04444, 0.02222,
    ],
    &[
        0.14667, 0.14667, 0.14667, 0.14667, 0.14667, 0.06667, 0.06667, 0.06667, 0.04444, 0.02220,
    ],
    &[
        0.13333, 0.13333, 0.13333, 0.13333, 0.13333, 0.06667, 0.06667, 0.06667, 0.06667, 0.04444,
        0.02223,
    ],
    &[
        0.12000, 0.12000, 0.12000, 0.12000, 0.12000, 0.06667, 0.06667, 0.06667, 0.06667, 0.06667,
        0.04444, 0.02221,
    ],
    &[
        0.10667, 0.10667, 0.10667, 0.10667, 0.10667, 0.06667, 0.06667, 0.06667, 0.06667, 0.06667,
        0.06667, 0.04444, 0.02219,
    ],
    &[
        0.09333, 0.09333, 0.09333, 0.09333, 0.09333, 0.06667, 0.06667, 0.06667, 0.06667, 0.06667,
        0.06667, 0.06667, 0.04444, 0.02222,
    ],
    &[
        0.08000, 0.08000, 0.08000, 0.08000, 0.08000, 0.06667, 0.06667, 0.06667, 0.06667, 0.06667,
        0.06667, 0.06667, 0.06667, 0.04444, 0.02220,
    ],
    &[
        0.04889, 0.07111, 0.07111, 0.07111, 0.07111, 0.06667, 0.06667, 0.06667, 0.06667, 0.06667,
        0.06667, 0.06667, 0.06667, 0.06667, 0.04444, 0.02220,
    ],
];

#[cfg(test)]
mod tests {
    use super::{
        END_WEIGHTS_3X9, END_WEIGHTS_3X15, ResolvedSeasonalFilter, apply_end_weights,
        smooth_subseries,
    };

    #[test]
    fn s3x1_endpoints_round_the_product_then_fuse() {
        // The bits are the endpoint expression with the 0.39 product rounded
        // and the sum then rounded once.
        let input = [
            0.8001176401388154,
            1.238076152304609,
            2.0,
            1.238076152304609,
            0.8001176401388154,
        ];
        let actual = smooth_subseries(&input, ResolvedSeasonalFilter::S3x1);
        assert_eq!(actual[0].to_bits(), 0x3fef11c9e1a5e427);
        assert_eq!(actual[4].to_bits(), 0x3fef11c9e1a5e427);
    }

    #[test]
    fn s3x3_endpoints_match_build_62() {
        // Build 62's vsfb.f gives these bits for this SI.
        let input = [
            2.2803468208092488,
            5.916184971098266,
            3.7890173410404624,
            1.6618497109826589,
            5.297687861271676,
            3.1705202312138727,
            1.0433526011560694,
            4.679190751445087,
        ];
        let expected = [
            0x401029fb4962b4c0,
            0x400f984e7b8885b9,
            0x4007592b761b8342,
            0x400814d38d586708,
        ];
        for (values, bits) in [
            (input, expected),
            (
                std::array::from_fn(|i| input[7 - i]),
                [expected[2], expected[3], expected[0], expected[1]],
            ),
        ] {
            let actual = smooth_subseries(&values, ResolvedSeasonalFilter::S3x3);
            assert_eq!(
                [
                    actual[0].to_bits(),
                    actual[1].to_bits(),
                    actual[7].to_bits(),
                    actual[6].to_bits()
                ],
                bits
            );
        }
    }

    #[test]
    fn s3x9_endpoints_match_build_62() {
        // An April SI subseries and the endpoint values Build 62's endsf.f
        // returns for it.
        let bits = [
            0x3ff0008d2cda21d9,
            0x3feffffffffffffe,
            0x3feffeb7d8a265ce,
            0x3feffd7d29758627,
            0x3ff001608695837b,
            0x3ff000d8d9b6e6e4,
            0x3ff0000000000001,
            0x3feffdd7f010428d,
            0x3feffb8131aecd92,
            0x3ff002ab594cdd9b,
            0x3ff001d3635ec4f3,
            0x3feffffffffffffe,
            0x3feff93629c40a8d,
            0x3fefea92dc10f667,
        ];
        let input = bits.map(f64::from_bits);
        let actual = smooth_subseries(&input, ResolvedSeasonalFilter::S3x9);
        assert_eq!(actual[0].to_bits(), 0x3feffffaaf78666c);
        assert_eq!(actual[10].to_bits(), 0x3feffc75fc38f84d);
        let reversed: Vec<_> = input.iter().rev().copied().collect();
        let actual = smooth_subseries(&reversed, ResolvedSeasonalFilter::S3x9);
        assert_eq!(actual[13].to_bits(), 0x3feffffaaf78666c);
        assert_eq!(actual[3].to_bits(), 0x3feffc75fc38f84d);
    }

    #[test]
    fn s3x15_endpoints_use_one_table_row_each() {
        // An impulse at one end picks out the first weight of each
        // endpoint's row.
        let first_weights = END_WEIGHTS_3X15.map(|row| row[0]);
        let mut input = [0.0; 24];
        input[0] = 1.0;
        let front = smooth_subseries(&input, ResolvedSeasonalFilter::S3x15);
        input.reverse();
        let back = smooth_subseries(&input, ResolvedSeasonalFilter::S3x15);
        for (i, weight) in first_weights.into_iter().enumerate() {
            assert!((front[i] - weight).abs() < 1e-15, "row {i}");
            assert!((back[23 - i] - weight).abs() < 1e-15, "row {i}");
        }
    }

    #[test]
    fn short_subseries_take_their_mean() {
        assert_eq!(
            smooth_subseries(&[2.0, 5.0, 8.0], ResolvedSeasonalFilter::S3x3)[1],
            5.0
        );
        // Every S3x9 endpoint row is longer than five values.
        let short = [2.0, 5.0, 9.0, 4.0, 7.0];
        let mut actual = vec![0.0; 5];
        apply_end_weights(&short, &mut actual, &END_WEIGHTS_3X9);
        assert_eq!(actual, vec![27.0 / 5.0; 5]);
        // The two ends meet at the middle of nine values, where the row has
        // ten weights.
        let meeting = [2.0, 5.0, 9.0, 4.0, 7.0, 11.0, 3.0, 8.0, 6.0];
        let mut actual = vec![0.0; 9];
        apply_end_weights(&meeting, &mut actual, &END_WEIGHTS_3X9);
        assert_eq!(actual[4], 55.0 / 9.0);
        // A 3x15 filter takes the mean of a subseries shorter than 20.
        assert_eq!(
            smooth_subseries(&short, ResolvedSeasonalFilter::S3x15),
            vec![27.0 / 5.0; 5]
        );
    }
}
