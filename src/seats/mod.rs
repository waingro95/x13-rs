//! SEATS: what a SEATS adjustment reports beyond its components.
//!
//! [`Spec::seats`](crate::Spec::seats) requests a SEATS adjustment, and the
//! [SEATS guide](crate::guide::seats) covers the models it runs and the
//! errors it returns. An adjustment's SEATS [`Diagnostics`] come from
//! [`Diagnostics::seats`](crate::Diagnostics::seats): the [`Model`] SEATS
//! decomposed, which can differ from the regARIMA fit, its mean, and whether
//! SEATS re-estimated it. The [`Transitory`] component (table S14), in the
//! units its [`TransitoryScale`] gives, comes from
//! [`Adjustment::seats_transitory`](crate::Adjustment::seats_transitory).

// The decomposition starts from the coefficients of a fitted regARIMA model
// and does not depend on the modules that estimate it. The modules follow
// Build 62's SEATS routines:
//
// - The run loop. `pipeline` is ANALTS: up to three passes over the
//   differenced series, with the `retry` decision between them, then the
//   decomposition of the completed pass. Every family runs it through the
//   hooks of `pipeline::Family`.
// - The families, one for each group of models ANALTS treats alike:
//   `airline` is `(0 1 1)(0 1 1)`; `second_difference` is `(0 2 1)(0 1 1)`;
//   `seasonal_ma` is `(0 1 0)(0 1 1)` or the nonseasonal `(0 1 1)`;
//   `ar1_seasonal_ma` is `(1 1 0)(0 1 1)` or the nonseasonal `(1 1 0)`;
//   `ar2` is `(2 1 0)(0 1 1)` or `(3 1 0)(0 1 1)`; `arma` is every other
//   supported model.
// - The steps of a pass: `import` (NMLMDL and TRANS0, which clip the
//   coefficients SEATS starts from), `mean` (the mean of the differenced
//   series and the residual mean test), `autocorrelation` (AUTO), `residual`
//   (CALCFX) and `refit` (SEARCH and STAVAL).
// - The completed pass: `tails` (FCAST), `decomposition` (SIGEX),
//   `extraction` (ESTBUR and DETCOMP), `log_transform` (SECOND), `effects`
//   (ADJREG and DETCOMP's restoration of the regression effects) and
//   `trading_day` (SIGEX's stochastic trading-day branch).
// - Shared numerics: `roots` (C02AEF) and `linear` (MLTSOL).
// - `failure`: what failed, and in which step.
//
// Build 62's SIGEX calls AUTOCOMP (ansub4) to build the Wiener-Kolmogorov
// filters and BFAC autocovariances behind its printed tables only. The saved
// components come from ESTBUR extraction, so none of that is computed here.

pub use crate::api::seats::{Diagnostics, Model, Transitory, TransitoryScale};

pub(crate) mod airline;
pub(crate) mod ar1_seasonal_ma;
pub(crate) mod ar2;
pub(crate) mod arma;
pub(crate) mod seasonal_ma;
pub(crate) mod second_difference;

mod autocorrelation;
pub(crate) mod decomposition;
pub(crate) mod effects;
mod extraction;
pub(crate) mod failure;
mod import;
mod linear;
mod log_transform;
pub(crate) mod mean;
pub(crate) mod pipeline;
pub(crate) mod refit;
mod residual;
mod retry;
mod roots;
mod tails;
pub(crate) mod trading_day;

pub(crate) use extraction::FinalComponents;
pub(crate) use failure::{Error, Failure, Stage};

/// An ARIMA model's orders, (p, d, q)(bp, bd, bq).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Orders {
    pub p: u8,
    pub d: u8,
    pub q: u8,
    pub bp: u8,
    pub bd: u8,
    pub bq: u8,
}

impl Orders {
    /// From the regular (p, d, q) and seasonal (bp, bd, bq) orders.
    pub(crate) const fn new([p, d, q]: [u8; 3], [bp, bd, bq]: [u8; 3]) -> Self {
        Self {
            p,
            d,
            q,
            bp,
            bd,
            bq,
        }
    }

    /// `[p, d, q, bp, bd, bq]` as the public model and error fields carry
    /// them.
    pub(crate) const fn to_array(self) -> [u8; 6] {
        [self.p, self.d, self.q, self.bp, self.bd, self.bq]
    }
}
