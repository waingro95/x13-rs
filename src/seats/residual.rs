//! CALCFX's shared numeric kernels: the AR prefilter and the MA
//! initial-residual solve every family runs.
//!
//! Build 62's ANSUB1 and ANSUB11 hold near-identical CALCFX copies. They do
//! the same arithmetic (the K'K/K'a assembly, the Gauss-Jordan inversion with
//! its `1e-12` skip, `Detpri`, the solve and both recurrences) and differ in
//! two places only: how the K columns are built and the determinant floor.
//! [`calcfx`] is the one 0-based kernel for both. [`CalcfxMa`] selects the
//! column builder, and each caller passes its own floor, because the floors
//! come from differently written source expressions (a GCC `powi`, stored
//! bits or exact literals).
//!
//! Every operation keeps Build 62's order: each sum folds its terms in source
//! order onto the same starting value, and each `mul_add` keeps its operands.
//! Failures keep CALCFX's 1-based positions and names, because they reach
//! the SEATS failure text.
//!
//! The AR prefilter is shared the same way. Its callers check intermediate
//! values at different points and under different names, so [`ar_prefilter`]
//! hands each partial result to the caller's check.

use std::ops::{Index, IndexMut};

use super::airline::model::ResolvedAirline;
use super::failure::{Error, Failure, Stage, usable};
use crate::Frequency;

/// The airline's SEARCH objective: CALCFX at trial MA coefficients on one
/// differenced series.
pub(crate) struct AirlineObjective<'a> {
    frequency: Frequency,
    differenced: &'a [f64],
    /// The previous trial's coordinates.
    previous: Option<[f64; 2]>,
    /// The coefficients CALCFX holds.
    ma: [f64; 2],
    evaluations: usize,
}

impl<'a> AirlineObjective<'a> {
    pub(crate) fn new(frequency: Frequency, differenced: &'a [f64]) -> Self {
        Self {
            frequency,
            differenced,
            previous: None,
            ma: [0.0; 2],
            evaluations: 0,
        }
    }

    pub(crate) fn evaluate(&mut self, coordinates: [f64; 2]) -> Result<ResidualResult, Error> {
        for (index, value) in coordinates.iter().enumerate() {
            if !value.is_finite() || value.abs() > 1.0 {
                return Err(Failure::Unusable {
                    quantity: "restart coordinate",
                    index: Some(index),
                    value: *value,
                }
                .at(Stage::Residual));
            }
            // CALCFX converts a coordinate again only when `dpeq` finds it
            // changed since the previous trial.
            if self
                .previous
                .is_none_or(|previous| (previous[index] - value).abs() >= 3.834e-20)
            {
                self.ma[index] = *value;
            }
        }
        let result = ordered_residuals(
            usize::from(self.frequency.periods_per_year()),
            self.ma,
            self.differenced,
        )?;
        self.previous = Some(coordinates);
        self.evaluations += 1;
        Ok(result)
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }
}

/// analts.f `POBS`: SEATS holds at most 780 observations at any frequency.
/// The editor's tighter quarterly cap, 328 from a first-quarter start, is
/// enforced when the series is validated.
pub(super) const MAXIMUM_OBSERVATIONS: usize = 780;

/// One CALCFX solve.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResidualResult {
    /// The residuals times `Detpri`, initial residuals first.
    pub(crate) scaled_residuals: Vec<f64>,
    /// `Detpri`, the determinant of K'K to the power `1/(2·Nw)`.
    pub(crate) determinant_scale: f64,
    /// The sum of squares of the scaled residuals.
    pub(crate) objective: f64,
    /// The scaled residuals divided by `Detpri` again.
    pub(crate) normalized_residuals: Vec<f64>,
}

/// `value` when it is finite, else [`Failure::Unusable`] at
/// [`Stage::Residual`].
pub(super) fn finite(
    value: f64,
    quantity: &'static str,
    index: Option<usize>,
) -> Result<f64, Error> {
    usable(quantity, index, value).map_err(|failure| failure.at(Stage::Residual))
}

/// CALCFX's AR conditioning (ANSUB1:1116–1131). Output `i` is
/// `wd[i+p] − φ1·wd[i+p−1] − … − φp·wd[i]`, each subtraction fused and taken
/// in lag order. `checked(lag, i, partial)` receives every partial result,
/// so each family keeps its own finiteness checks and stage names.
pub(super) fn ar_prefilter<E>(
    phi: &[f64],
    wd: &[f64],
    checked: impl Fn(usize, usize, f64) -> Result<f64, E>,
) -> Result<Vec<f64>, E> {
    let p = phi.len();
    let mut filtered = Vec::with_capacity(wd.len().saturating_sub(p));
    for (index, window) in wd.windows(p + 1).enumerate() {
        let (lagged, current) = window.split_at(p);
        let value = phi.iter().zip(lagged.iter().rev()).enumerate().try_fold(
            current[0],
            |value, (lag, (&coefficient, &lagged))| {
                checked(lag, index, (-coefficient).mul_add(lagged, value))
            },
        )?;
        filtered.push(value);
    }
    Ok(filtered)
}

/// CALCFX for an airline model on its differenced series.
pub(super) fn resolved_residuals(
    model: &ResolvedAirline,
    differenced: &[f64],
) -> Result<ResidualResult, Error> {
    ordered_residuals(
        usize::from(model.frequency().periods_per_year()),
        model.ma_coefficients(),
        differenced,
    )
}

/// CALCFX for the airline's two MA coefficients. `differenced` comes from a
/// series of `differenced.len() + s + 1` observations in SEATS's range.
pub(super) fn ordered_residuals(
    s: usize,
    ma: [f64; 2],
    differenced: &[f64],
) -> Result<ResidualResult, Error> {
    debug_assert!((2 * s + 2..=MAXIMUM_OBSERVATIONS).contains(&(differenced.len() + s + 1)));
    ordered_ma_residuals(s, &ma[..1], Some(ma[1]), differenced)
}

/// ANSUB11's ordered CALCFX. Na=Nw-Pstar+Qstar includes the initial MA
/// residuals, so it can exceed the observed sample's 780 limit. Each family
/// validates that sample with its own differencing and AR degrees; only the
/// airline can recover it as `differenced.len()+s+1`. `differenced` is
/// finite.
pub(super) fn ordered_ma_residuals(
    s: usize,
    regular_ma: &[f64],
    seasonal_ma: Option<f64>,
    differenced: &[f64],
) -> Result<ResidualResult, Error> {
    debug_assert!(differenced.iter().all(|value| value.is_finite()));
    let ma = CalcfxMa::Ordered {
        period: s,
        regular: regular_ma,
        seasonal: seasonal_ma,
    };
    let threshold = ordered_threshold(regular_ma.len(), ma.initial_count(), s);
    calcfx(ma, differenced, threshold)
}

/// CALCFX's determinant floor `10**(−Qstar−3)` for the ordered layouts, as
/// Build 62 rounds it: GCC's `powi` for MA2, and the stored bits of that
/// reciprocal for the airline's Qstar of 5 and 13.
fn ordered_threshold(q: usize, qstar: usize, s: usize) -> f64 {
    if q == 2 {
        gcc_powi_reciprocal(10.0, qstar + 3)
    } else if s == 4 {
        f64::from_bits(0x3e45798ee2308c3a)
    } else {
        f64::from_bits(0x3c9cd2b297d889bc)
    }
}

/// GCC's integer-power helper for a negative exponent: repeated squaring,
/// then the reciprocal.
fn gcc_powi_reciprocal(mut base: f64, mut exponent: usize) -> f64 {
    let mut power = 1.0;
    while exponent != 0 {
        if exponent & 1 != 0 {
            power *= base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base *= base;
        }
    }
    1.0 / power
}

/// The MA side of one CALCFX solve: which lags `thstar` holds, and so how
/// the K columns are built.
#[derive(Clone, Copy)]
pub(super) enum CalcfxMa<'a> {
    /// ANSUB11's ordered layout: regular lags `1..=q`, then, with a seasonal
    /// MA, lag `s` and the cross lags `s+1..=s+q`. Qstar is `q`, or `s+q`.
    /// With `q = 0` this is ANSUB1's seasonal-only (Q=0, BQ=1) solve.
    Ordered {
        period: usize,
        regular: &'a [f64],
        seasonal: Option<f64>,
    },
    /// ANSUB1's regular-only solve (Q=1, BQ=0): one dense K column that the
    /// seasonal-column loop builds with s=1.
    SingleRegular { ma: f64 },
}

impl CalcfxMa<'_> {
    /// Qstar, the number of initial residuals the solve estimates.
    fn initial_count(self) -> usize {
        match self {
            Self::Ordered {
                period,
                regular,
                seasonal,
            } => seasonal.map_or(0, |_| period) + regular.len(),
            Self::SingleRegular { .. } => 1,
        }
    }

    /// The row step of a sparse K column.
    fn period(self) -> usize {
        match self {
            Self::Ordered { period, .. } => period,
            Self::SingleRegular { .. } => 1,
        }
    }

    /// `thstar`'s active lags in CALCFX's `ith` order, which is the order
    /// every recurrence sums them in.
    fn lags(self) -> Vec<MaLag> {
        match self {
            Self::Ordered {
                period,
                regular,
                seasonal,
            } => {
                let mut lags = regular
                    .iter()
                    .enumerate()
                    .map(|(k, &coefficient)| MaLag {
                        lag: k + 1,
                        coefficient,
                    })
                    .collect::<Vec<_>>();
                if let Some(seasonal) = seasonal {
                    lags.push(MaLag {
                        lag: period,
                        coefficient: seasonal,
                    });
                    lags.extend(regular.iter().enumerate().map(|(k, &coefficient)| MaLag {
                        lag: period + k + 1,
                        coefficient: -coefficient * seasonal,
                    }));
                }
                lags
            }
            Self::SingleRegular { ma } => vec![MaLag {
                lag: 1,
                coefficient: ma,
            }],
        }
    }

    /// K's columns, in CALCFX's build order (which decides the reported
    /// position when one overflows).
    fn build_columns(self, k: &mut KMatrix, lags: &[MaLag]) -> Result<(), Error> {
        match self {
            Self::Ordered {
                period,
                regular,
                seasonal,
            } => {
                let q = regular.len();
                // Q regular columns in each block, then shifted seasonal columns.
                for column in 0..q {
                    k.regular_column(column, lags)?;
                }
                if let Some(seasonal) = seasonal {
                    k.seasonal_column(q, seasonal, period)?;
                    for column in q + 1..period {
                        k.shifted_column(column, column - q, period);
                    }
                    for column in period..period + q {
                        k.regular_column(column, lags)?;
                    }
                }
            }
            Self::SingleRegular { ma } => {
                let (initial, n) = (k.initial, k.rows);
                let column = k.column_mut(0);
                for t in initial..n {
                    let value = finite(
                        ma.mul_add(column[t - 1], 0.0),
                        "seasonal K column",
                        Some(t + 1),
                    )?;
                    // CALCFX:1286 zeros tiny entries only in regular K columns.
                    column[t] = if value.abs() <= 1e-9 { 0.0 } else { value };
                }
            }
        }
        Ok(())
    }
}

/// One active lag of CALCFX's expanded MA polynomial `thstar`.
#[derive(Clone, Copy)]
struct MaLag {
    lag: usize,
    coefficient: f64,
}

/// Which rows of a K column can be nonzero.
#[derive(Clone, Copy, PartialEq, Eq)]
enum KColumn {
    /// Every row from Qstar on (CALCFX `jcol=0`).
    Dense,
    /// Rows `Qstar+offset`, then every `s` rows (CALCFX `jcol=offset+1`).
    Sparse { offset: usize },
}

/// A strided run of K rows: `start`, `start+step`, … below `end`.
#[derive(Clone, Copy)]
struct Rows {
    start: usize,
    end: usize,
    step: usize,
}

impl Rows {
    fn iter(self) -> impl Iterator<Item = usize> {
        (self.start..self.end).step_by(self.step)
    }
}

/// CALCFX's K (`am`): column `c` holds the derivative of each residual with
/// respect to initial residual `c`, stored by column. Rows `0..Qstar` start
/// as the identity.
struct KMatrix {
    rows: usize,
    initial: usize,
    values: Vec<f64>,
    kinds: Vec<KColumn>,
}

impl KMatrix {
    fn new(rows: usize, initial: usize) -> Self {
        let mut values = vec![0.0; rows * initial];
        for column in 0..initial {
            values[column * rows + column] = 1.0;
        }
        Self {
            rows,
            initial,
            values,
            kinds: vec![KColumn::Dense; initial],
        }
    }

    fn column(&self, column: usize) -> &[f64] {
        &self.values[column * self.rows..][..self.rows]
    }

    fn column_mut(&mut self, column: usize) -> &mut [f64] {
        &mut self.values[column * self.rows..][..self.rows]
    }

    /// The rows a column of this kind can be nonzero on.
    fn rows_of(&self, kind: KColumn, period: usize) -> Rows {
        match kind {
            KColumn::Dense => Rows {
                start: self.initial,
                end: self.rows,
                step: 1,
            },
            KColumn::Sparse { offset } => Rows {
                start: self.initial + offset,
                end: self.rows,
                step: period,
            },
        }
    }

    /// A regular MA column: the MA recursion of the column itself, with
    /// tiny entries zeroed.
    fn regular_column(&mut self, column: usize, lags: &[MaLag]) -> Result<(), Error> {
        let (initial, n) = (self.initial, self.rows);
        let values = self.column_mut(column);
        for t in initial..n {
            let mut sum = lags.iter().fold(0.0, |sum, term| {
                term.coefficient.mul_add(values[t - term.lag], sum)
            });
            if sum.abs() <= 1e-9 {
                sum = 0.0;
            }
            values[t] = finite(sum, "regular K column", Some(t + 1))?;
        }
        Ok(())
    }

    /// The seasonal MA column, nonzero every `period` rows from Qstar.
    fn seasonal_column(
        &mut self,
        column: usize,
        seasonal: f64,
        period: usize,
    ) -> Result<(), Error> {
        self.kinds[column] = KColumn::Sparse { offset: 0 };
        let (initial, n) = (self.initial, self.rows);
        let values = self.column_mut(column);
        for t in (initial..n).step_by(period) {
            values[t] = finite(
                seasonal.mul_add(values[t - period], 0.0),
                "seasonal K column",
                Some(t + 1),
            )?;
        }
        Ok(())
    }

    /// A shifted seasonal column: the previous column moved one row down,
    /// on its own sparse rows.
    fn shifted_column(&mut self, column: usize, offset: usize, period: usize) {
        let kind = KColumn::Sparse { offset };
        self.kinds[column] = kind;
        let rows = self.rows_of(kind, period);
        let (before, after) = self.values.split_at_mut(column * self.rows);
        let previous = &before[(column - 1) * self.rows..];
        for t in rows.iter() {
            after[t] = previous[t - 1];
        }
    }

    /// The rows two columns share, and CALCFX's stage name for their product;
    /// `None` when sparse columns on different rows cannot overlap.
    fn shared_rows(&self, i: usize, j: usize, period: usize) -> Option<(Rows, &'static str)> {
        match (self.kinds[i], self.kinds[j]) {
            (KColumn::Dense, KColumn::Dense) => {
                Some((self.rows_of(KColumn::Dense, period), "K'K regular"))
            }
            (KColumn::Sparse { offset: a }, KColumn::Sparse { offset: b }) if a != b => None,
            (sparse @ KColumn::Sparse { .. }, _) | (_, sparse @ KColumn::Sparse { .. }) => {
                Some((self.rows_of(sparse, period), "K'K sparse"))
            }
        }
    }
}

/// Row-major square scratch: CALCFX's `b`, holding K'K and then its inverse.
struct Square {
    size: usize,
    values: Vec<f64>,
}

impl Square {
    fn identity(size: usize) -> Self {
        let mut values = vec![0.0; size * size];
        for i in 0..size {
            values[i * size + i] = 1.0;
        }
        Self { size, values }
    }

    fn row(&self, row: usize) -> &[f64] {
        &self.values[row * self.size..][..self.size]
    }
}

impl Index<(usize, usize)> for Square {
    type Output = f64;

    fn index(&self, (row, column): (usize, usize)) -> &f64 {
        &self.values[row * self.size + column]
    }
}

impl IndexMut<(usize, usize)> for Square {
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut f64 {
        &mut self.values[row * self.size + column]
    }
}

/// One CALCFX TYPE=0 solve on AR-conditioned `u`. It initializes all of its
/// scratch on each call; the coefficients CALCFX keeps between the trials of
/// a search belong to the family objectives, `AirlineObjective` and its peers.
pub(super) fn calcfx(
    ma: CalcfxMa<'_>,
    conditioned: &[f64],
    threshold: f64,
) -> Result<ResidualResult, Error> {
    let qstar = ma.initial_count();
    let period = ma.period();
    let lags = ma.lags();
    let n = conditioned.len() + qstar;
    let mut residuals = vec![0.0; n];
    recur(&lags, conditioned, &mut residuals, "initial recurrence")?;

    let mut k = KMatrix::new(n, qstar);
    ma.build_columns(&mut k, &lags)?;

    // K'K's upper triangle and K'a, in CALCFX's (i,j,l) loop order.
    let mut kk = Square::identity(qstar);
    let mut ka = vec![0.0; qstar];
    for i in 0..qstar {
        let column_i = k.column(i);
        for j in 0..=i {
            let Some((rows, stage)) = k.shared_rows(i, j, period) else {
                continue;
            };
            let column_j = k.column(j);
            let sum = rows
                .iter()
                .fold(kk[(j, i)], |sum, t| column_i[t].mul_add(column_j[t], sum));
            kk[(j, i)] = finite(sum, stage, Some(i + 1))?;
        }
        ka[i] = k
            .rows_of(k.kinds[i], period)
            .iter()
            .fold(0.0, |sum, t| column_i[t].mul_add(residuals[t], sum));
        finite(ka[i], "K'a", Some(i + 1))?;
    }

    let det = invert(&mut kk, threshold)?;
    let detpri = finite(
        det.powf(0.5 / conditioned.len() as f64),
        "determinant",
        None,
    )?;
    if detpri <= 0.0 {
        return Err(Failure::Unusable {
            quantity: "determinant",
            index: None,
            value: detpri,
        }
        .at(Stage::Residual));
    }
    for (i, initial) in residuals[..qstar].iter_mut().enumerate() {
        let sum = kk
            .row(i)
            .iter()
            .zip(&ka)
            .fold(0.0, |sum, (&inverse, &ka)| (-inverse).mul_add(ka, sum));
        *initial = finite(sum, "initial residual solve", Some(i + 1))?;
    }
    recur(&lags, conditioned, &mut residuals, "final recurrence")?;

    let mut objective = 0.0;
    let mut scaled_residuals = Vec::with_capacity(n);
    let mut normalized_residuals = Vec::with_capacity(n);
    for (t, &residual) in residuals.iter().enumerate() {
        let scaled = finite(detpri * residual, "scaled residual", Some(t + 1))?;
        objective = finite(scaled.mul_add(scaled, objective), "objective", Some(t + 1))?;
        scaled_residuals.push(scaled);
        normalized_residuals.push(finite(scaled / detpri, "normalized residual", Some(t + 1))?);
    }
    Ok(ResidualResult {
        scaled_residuals,
        determinant_scale: detpri,
        objective,
        normalized_residuals,
    })
}

/// CALCFX's residual recursion `a(t) = u(t−Qstar) + Σ θ·a(t−lag)` for
/// `t ≥ Qstar`, each term fused onto the running sum in `ith` order.
fn recur(
    lags: &[MaLag],
    conditioned: &[f64],
    residuals: &mut [f64],
    quantity: &'static str,
) -> Result<(), Error> {
    let n = residuals.len();
    let qstar = n - conditioned.len();
    for (t, &u) in (qstar..n).zip(conditioned) {
        let value = lags.iter().fold(u, |sum, term| {
            term.coefficient.mul_add(residuals[t - term.lag], sum)
        });
        residuals[t] = finite(value, quantity, Some(t + 1))?;
    }
    Ok(())
}

/// CALCFX's in-place Gauss-Jordan inversion of K'K, which holds only its
/// upper triangle. It skips any row update whose multiplier is below
/// `1e-12`, refuses a pivot product below `threshold`, then mirrors the
/// inverse into the lower triangle. Returns the pivot product.
fn invert(b: &mut Square, threshold: f64) -> Result<f64, Error> {
    let size = b.size;
    let mut det = 1.0;
    for i in 0..size {
        let pivot = b[(i, i)];
        det *= pivot;
        if !det.is_finite() || det < threshold || pivot == 0.0 {
            return Err(Failure::Unusable {
                quantity: "determinant",
                index: Some(i + 1),
                value: det,
            }
            .at(Stage::Residual));
        }
        let g = finite(1.0 / pivot, "inverse pivot", Some(i + 1))?;
        b[(i, i)] = g;
        for j in 0..i {
            if b[(j, i)].abs() >= 1e-12 {
                let h = g * b[(j, i)];
                for k in j..i {
                    b[(j, k)] = h.mul_add(b[(k, i)], b[(j, k)]);
                }
                for k in i + 1..size {
                    b[(j, k)] = (-h).mul_add(b[(i, k)], b[(j, k)]);
                }
                b[(j, i)] = -h;
            }
        }
        for j in i + 1..size {
            if b[(i, j)].abs() >= 1e-12 {
                let h = g * b[(i, j)];
                for k in j..size {
                    b[(j, k)] = (-h).mul_add(b[(i, k)], b[(j, k)]);
                }
                b[(i, j)] = h;
            }
        }
    }
    for j in 0..size {
        for k in 0..j {
            b[(j, k)] = b[(k, j)];
        }
    }
    Ok(det)
}

#[cfg(test)]
mod tests;
