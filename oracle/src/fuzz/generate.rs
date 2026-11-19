//! Deterministic case generation from `(seed, index)`: a spec source (call-table
//! row, random builder spec or hand-named spec) plus a generated or
//! corpus-derived series with edge-case mutations.

use crate::parity::data::{self, SpecRow};
use crate::{Frequency, Series};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::OnceLock;

/// A gap, as Rust takes it. Build 62's side of a replay maps it back to the
/// `-99999` code (`crate::series::build_62_values`).
pub(crate) const MISSING: f64 = f64::NAN;

/// Whether a value is an observation rather than a gap. `MISSING` is NaN, so
/// comparing against it is always unequal.
fn observed(value: f64) -> bool {
    !value.is_nan()
}

// ---------------------------------------------------------------- PRNG
#[derive(Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
    pub(crate) fn usize(&mut self, n: usize) -> usize {
        self.below(n as u64) as usize
    }
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub(crate) fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }
    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub(crate) fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.usize(xs.len())]
    }
}

fn mix(a: u64, b: u64) -> u64 {
    let mut r = Rng::new(a ^ b.wrapping_mul(0x2545_F491_4F6C_DD1D));
    r.next_u64()
}

// ---------------------------------------------------------------- context
pub(crate) struct Context {
    pub(crate) rows: Vec<SpecRow>,
    pub(crate) corpus: Vec<Series>,
}

pub(crate) fn context() -> &'static Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity");
        let (rows, corpus) = data::load(&root).expect("load parity corpus");
        Context { rows, corpus }
    })
}

// ---------------------------------------------------------------- cases
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind")]
pub(crate) enum SpecSource {
    Row {
        id: String,
    },
    Random {
        seed: u64,
    },
    /// Hand-built repro: "x11reg|seatsreg|x11auto:p,d,q,P,D,Q:log|none".
    Named {
        name: String,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Case {
    pub(crate) index: u64,
    pub(crate) spec: SpecSource,
    pub(crate) class: String,
    pub(crate) frequency: Frequency,
    pub(crate) start_year: i32,
    pub(crate) start_subperiod: u8,
    /// Rust `{:?}` reprs so NaN/inf round-trip through JSON.
    pub(crate) values: Vec<String>,
}

impl Case {
    pub(crate) fn values(&self) -> Vec<f64> {
        self.values
            .iter()
            .map(|v| v.parse::<f64>().expect("value repr"))
            .collect()
    }
    pub(crate) fn set_values(&mut self, values: &[f64]) {
        self.values = values.iter().map(|v| format!("{v:?}")).collect();
    }
}

const MONTHLY_LENGTHS: &[usize] = &[
    0, 1, 2, 3, 5, 8, 11, 12, 13, 17, 23, 24, 25, 26, 29, 35, 36, 37, 38, 47, 48, 49, 50, 59, 60,
    61, 62, 71, 72, 73, 83, 84, 85, 95, 96, 97, 107, 108, 109, 119, 120, 121, 143, 144, 145, 180,
    240, 360, 500, 600, 779, 780, 781, 782, 800, 1000, 1200, 1500,
];
const QUARTERLY_LENGTHS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 27,
    28, 29, 31, 32, 33, 39, 40, 41, 48, 60, 80, 120, 200, 260, 320, 324, 325, 326, 327, 328, 329,
    330, 336, 340, 341, 342, 400, 500, 780,
];
const EDGE_YEARS: &[i32] = &[
    1,
    2,
    0,
    -1,
    -4713,
    1582,
    1583,
    1600,
    1752,
    1753,
    1800,
    1899,
    1900,
    1901,
    2099,
    2100,
    2101,
    2400,
    4099,
    9999,
    10000,
    99999,
    2_147_483_000,
    i32::MAX - 1,
    i32::MAX - 70,
    i32::MIN + 1,
    -2_147_483_000,
];

fn gen_length(rng: &mut Rng, freq: Frequency) -> usize {
    let s = if freq == Frequency::Monthly { 12 } else { 4 };
    match rng.below(10) {
        0..=5 => rng.pick(if freq == Frequency::Monthly {
            MONTHLY_LENGTHS
        } else {
            QUARTERLY_LENGTHS
        }),
        6..=8 => 1 + rng.usize(if freq == Frequency::Monthly { 800 } else { 345 }),
        _ => {
            let years = rng.range(1, 15) as usize;
            (years * s).saturating_add_signed(rng.range(-2, 2) as isize)
        }
    }
}

fn gen_start(rng: &mut Rng, freq: Frequency) -> (i32, u8) {
    let s = if freq == Frequency::Monthly { 12 } else { 4 };
    let sub = 1 + rng.below(s) as u8;
    if rng.chance(0.1) {
        (rng.pick(EDGE_YEARS), sub)
    } else {
        (rng.range(1900, 2025) as i32, sub)
    }
}

const LEVELS: &[f64] = &[
    100.0, 1.0, 1000.0, 0.001, 1e6, 1e-6, 1e9, 1e-12, 1e30, 1e-30, 1e100, 1e-100, 1e300, 1e-300,
    5e-324, 1.7e308, 42.0,
];

fn synthetic(rng: &mut Rng, freq: Frequency, n: usize) -> (Vec<f64>, String) {
    let s = if freq == Frequency::Monthly { 12 } else { 4 };
    let level = if rng.chance(0.7) {
        100.0
    } else {
        rng.pick(LEVELS)
    };
    let pattern: Vec<f64> = (0..s).map(|_| rng.normal()).collect();
    let amp = rng.pick(&[0.0, 0.01, 0.1, 0.3, 0.9]);
    let noise = rng.pick(&[0.0, 0.0, 1e-12, 0.001, 0.01, 0.1, 1.0]);
    let kind = rng.below(17);
    let name;
    let mut v = vec![0.0; n];
    match kind {
        0 => {
            name = "constant";
            v.iter_mut().for_each(|x| *x = level);
        }
        1 => {
            name = "linear_trend";
            let slope = rng.pick(&[0.01, 1.0, -1.0, 0.5, -0.1]);
            for (t, x) in v.iter_mut().enumerate() {
                *x = level * (1.0 + slope * t as f64 / n.max(1) as f64);
            }
        }
        2 => {
            name = "exp_growth";
            let g = rng.pick(&[0.001, 0.01, 0.05, 0.5, -0.05]);
            for (t, x) in v.iter_mut().enumerate() {
                *x = level * (g * t as f64).exp();
            }
        }
        3 => {
            name = "exact_seasonal_mult";
            for (t, x) in v.iter_mut().enumerate() {
                *x = level * (1.0 + amp * pattern[t % s].tanh());
            }
        }
        4 => {
            name = "exact_seasonal_add";
            for (t, x) in v.iter_mut().enumerate() {
                *x = level * amp * pattern[t % s];
            }
        }
        5 => {
            name = "trend_plus_seasonal";
            for (t, x) in v.iter_mut().enumerate() {
                *x = level * (1.0 + 0.002 * t as f64) * (1.0 + amp * pattern[t % s].tanh());
            }
        }
        6 => {
            name = "random_walk";
            let mut y = level;
            for x in v.iter_mut() {
                y += level * 0.01 * rng.normal();
                *x = y;
            }
        }
        7 => {
            name = "white_noise";
            for x in v.iter_mut() {
                *x = level * (1.0 + 0.1 * rng.normal());
            }
        }
        8 => {
            name = "single_spike";
            v.iter_mut().for_each(|x| *x = level);
            if n > 0 {
                let i = rng.usize(n);
                v[i] = level * rng.pick(&[0.0, 1e-6, 100.0, 1e6, -1.0]);
            }
        }
        9 => {
            name = "step";
            let at = rng.usize(n.max(1));
            let k = rng.pick(&[2.0, 10.0, 0.5, 1e-6, 1e6]);
            for (t, x) in v.iter_mut().enumerate() {
                *x = if t < at { level } else { level * k };
            }
        }
        10 => {
            name = "alternating_sign";
            for (t, x) in v.iter_mut().enumerate() {
                *x = if t % 2 == 0 { level } else { -level };
            }
        }
        11 => {
            name = "near_zero";
            for x in v.iter_mut() {
                *x = rng.pick(&[1e-10, 1e-300, 5e-324, 1e-15]) * (1.0 + 0.1 * rng.normal()).abs();
            }
        }
        12 => {
            name = "small_counts";
            for x in v.iter_mut() {
                *x = (rng.normal().abs() * rng.pick(&[1.0, 3.0])).floor();
            }
        }
        13 => {
            name = "all_missing";
            v.iter_mut().for_each(|x| *x = MISSING);
        }
        14 => {
            name = "mostly_missing";
            for x in v.iter_mut() {
                *x = if rng.chance(0.85) { MISSING } else { level };
            }
        }
        15 => {
            name = "piecewise_seasonal_break";
            let at = rng.usize(n.max(1));
            for (t, x) in v.iter_mut().enumerate() {
                let p = if t < at {
                    pattern[t % s]
                } else {
                    -pattern[t % s]
                };
                *x = level * (1.0 + 0.2 * p.tanh());
            }
        }
        _ => {
            name = "periodic_zero_season";
            for (t, x) in v.iter_mut().enumerate() {
                *x = if t % s == 0 {
                    0.0
                } else {
                    level * (1.0 + amp * pattern[t % s].tanh())
                };
            }
        }
    }
    if noise > 0.0 && !matches!(kind, 13 | 14) {
        for x in v.iter_mut() {
            *x *= 1.0 + noise * rng.normal();
        }
    }
    (v, format!("syn:{name}"))
}

fn corpus_window(
    rng: &mut Rng,
    freq: Frequency,
    n: usize,
) -> Option<(Vec<f64>, (i32, u8), String)> {
    let ctx = context();
    let pool: Vec<&Series> = ctx.corpus.iter().filter(|s| s.frequency == freq).collect();
    if pool.is_empty() {
        return None;
    }
    let series = pool[rng.usize(pool.len())];
    let len = series.values.len();
    let n = n.min(len);
    let offset = rng.usize(len - n + 1);
    let s = i64::from(freq.periods_per_year());
    let date =
        i64::from(series.start.year) * s + i64::from(series.start.subperiod) - 1 + offset as i64;
    let start = (date.div_euclid(s) as i32, (date.rem_euclid(s) + 1) as u8);
    Some((
        crate::series::rust_values(&series.values[offset..offset + n]),
        start,
        format!("corpus:{}", series.series_id),
    ))
}

fn mutate(rng: &mut Rng, freq: Frequency, v: &mut Vec<f64>, tags: &mut Vec<String>) {
    let s = if freq == Frequency::Monthly { 12 } else { 4 };
    let count = rng.pick(&[0, 0, 0, 0, 0, 1, 1, 1, 2, 3]);
    for _ in 0..count {
        let n = v.len();
        let m = rng.below(16);
        match m {
            0 => {
                let k = rng.pick(&[-300, -200, -100, -30, -12, -6, 6, 12, 30, 100, 200, 300]);
                let f = 10f64.powi(k);
                v.iter_mut().filter(|x| observed(**x)).for_each(|x| *x *= f);
                tags.push(format!("scale1e{k}"));
            }
            1 => {
                let mut sorted: Vec<f64> = v.iter().copied().filter(|x| observed(*x)).collect();
                sorted.sort_by(|a, b| a.total_cmp(b));
                if let Some(&med) = sorted.get(sorted.len() / 2) {
                    v.iter_mut()
                        .filter(|x| observed(**x))
                        .for_each(|x| *x -= med);
                }
                tags.push("demean".into());
            }
            2 => {
                v.iter_mut()
                    .filter(|x| observed(**x))
                    .for_each(|x| *x = -*x);
                tags.push("negate".into());
            }
            3 => {
                v.iter_mut()
                    .filter(|x| observed(**x))
                    .for_each(|x| *x = x.round());
                tags.push("round".into());
            }
            4 if n > 0 => {
                for _ in 0..rng.range(1, 3) {
                    let i = rng.usize(n);
                    if observed(v[i]) {
                        v[i] *= rng.pick(&[0.0, 100.0, 1e6, -1.0, 1e-6, 1e100]);
                    }
                }
                tags.push("spikes".into());
            }
            5 if n > 0 => {
                let pattern = rng.below(9);
                match pattern {
                    0 => {
                        let k = rng.usize(n.min(30)) + 1;
                        v[..k].iter_mut().for_each(|x| *x = MISSING);
                    }
                    1 => {
                        let k = rng.usize(n.min(30)) + 1;
                        let len = v.len();
                        v[len - k..].iter_mut().for_each(|x| *x = MISSING);
                    }
                    2 => {
                        let i = rng.usize(n);
                        v[i] = MISSING;
                    }
                    3 => {
                        for i in (0..n).step_by(2) {
                            v[i] = MISSING;
                        }
                    }
                    4 => {
                        for x in v.iter_mut() {
                            if rng.chance(0.3) {
                                *x = MISSING;
                            }
                        }
                    }
                    5 => {
                        let a = rng.usize(n);
                        let b = (a + s).min(n);
                        v[a..b].iter_mut().for_each(|x| *x = MISSING);
                    }
                    6 => {
                        let p = rng.usize(s);
                        for i in (p..n).step_by(s) {
                            v[i] = MISSING;
                        }
                    }
                    7 => {
                        v.iter_mut().for_each(|x| *x = MISSING);
                    }
                    _ => {
                        let a = rng.usize(n);
                        let b = (a + rng.usize(3 * s) + 1).min(n);
                        v[a..b].iter_mut().for_each(|x| *x = MISSING);
                    }
                }
                tags.push(format!("gaps{pattern}"));
            }
            6 if n > 0 => {
                for _ in 0..rng.range(1, 4) {
                    let i = rng.usize(n);
                    v[i] = 0.0;
                }
                tags.push("zeros".into());
            }
            7 if n > 0 && rng.chance(0.3) => {
                // NaN is a gap at the public API; the infinities are invalid.
                let i = rng.usize(n);
                v[i] = rng.pick(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
                tags.push("nonfinite".into());
            }
            8 if n > 2 => {
                let a = rng.usize(n);
                let b = (a + rng.usize(n - a) + 1).min(n);
                let c = v[a];
                v[a..b].iter_mut().for_each(|x| *x = c);
                tags.push("const_segment".into());
            }
            9 => {
                let k = rng.usize(n + 1);
                v.truncate(k);
                tags.push(format!("truncate{k}"));
            }
            10 if n > 0 => {
                let c = rng.pick(&[1e300, -1e300, 1e-300, f64::MAX, f64::MIN_POSITIVE]);
                let i = rng.usize(n);
                v[i] = c;
                tags.push("extreme_value".into());
            }
            11 if n >= 2 * s => {
                let a = rng.usize(n - s);
                let block: Vec<f64> = v[a..a + s].to_vec();
                for (k, x) in v.iter_mut().enumerate() {
                    if observed(*x) {
                        *x = block[k % s];
                    }
                }
                tags.push("repeat_year".into());
            }
            12 => {
                let c = rng.pick(&[1.0, 1e6, -1e6, 1e-9]);
                v.iter_mut().filter(|x| observed(**x)).for_each(|x| *x += c);
                tags.push("offset".into());
            }
            13 if n > 0 => {
                // Values equal or nearly equal to Build 62's missing code.
                // Build 62 reads the exact code as a gap and Rust as an
                // observation (a documented deviation); the neighbours are
                // observations to both.
                let i = rng.usize(n);
                v[i] = rng.pick(&[-99999.0, -99_999.000_000_1, -99_998.999_999_9, 99999.0]);
                tags.push("near_missing_code".into());
            }
            14 => {
                v.iter_mut()
                    .filter(|x| observed(**x))
                    .for_each(|x| *x = x.abs());
                tags.push("abs".into());
            }
            _ => {}
        }
    }
}

pub(crate) fn generate(seed: u64, index: u64) -> Case {
    let mut rng = Rng::new(mix(seed, index));
    let ctx = context();
    // Spec source: 65% table rows, 35% random builder specs.
    let (spec, row_freqs): (SpecSource, Option<Vec<Frequency>>) = if rng.chance(0.65) {
        let row = &ctx.rows[rng.usize(ctx.rows.len())];
        (
            SpecSource::Row { id: row.id.clone() },
            Some(row.frequencies.clone()),
        )
    } else {
        (
            SpecSource::Random {
                seed: rng.next_u64(),
            },
            None,
        )
    };
    let freq = match row_freqs {
        Some(f) if rng.chance(0.92) => f[rng.usize(f.len())],
        _ => {
            if rng.chance(0.6) {
                Frequency::Monthly
            } else {
                Frequency::Quarterly
            }
        }
    };
    let n = gen_length(&mut rng, freq);
    let mut start = gen_start(&mut rng, freq);
    let mut tags = Vec::new();
    let mut values = if rng.chance(0.5) {
        match corpus_window(&mut rng, freq, n) {
            Some((v, real_start, name)) => {
                if !rng.chance(0.1) {
                    start = real_start;
                }
                tags.push(name);
                v
            }
            None => vec![],
        }
    } else {
        let (v, name) = synthetic(&mut rng, freq, n);
        tags.push(name);
        v
    };
    mutate(&mut rng, freq, &mut values, &mut tags);
    let mut case = Case {
        index,
        spec,
        class: tags.join("+"),
        frequency: freq,
        start_year: start.0,
        start_subperiod: start.1,
        values: vec![],
    };
    case.set_values(&values);
    case
}
