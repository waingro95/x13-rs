//! SIGEX's stochastic trading-day branch.
//!
//! The root allocation can find an AR root close to the trading-day
//! frequency and give it to the transitory component. SIGEX then treats that
//! component as a stochastic trading-day effect and keeps it out of the
//! seasonally adjusted series. Under the default `stochTD=-1` it does so only
//! for a model with a trading-day regression.

/// Whether the regARIMA model has a trading-day regression (`npatd > 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TradingDayPresence {
    Absent,
    Present,
}

/// How SIGEX treats the transitory component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectiveTradingDay {
    /// As a transitory component, which stays in the seasonally adjusted
    /// series.
    Ordinary,
    /// As a stochastic trading-day effect, which leaves it.
    Stochastic,
}

impl TradingDayPresence {
    /// `Present` when the model is configured with a trading-day regression.
    pub(crate) fn from_configured_basis(configured: bool) -> Self {
        if configured {
            Self::Present
        } else {
            Self::Absent
        }
    }

    /// The treatment of the transitory component, given whether the root
    /// allocation found a root close to the trading-day frequency.
    pub(super) fn resolve(self, close_to_trading_day: bool) -> EffectiveTradingDay {
        // SIGEX 579–581: the default stochTD=-1 clears IsCloseToTD when
        // npatd=0. The test is the regressor's presence, not the size of its
        // effect.
        if self == Self::Present && close_to_trading_day {
            EffectiveTradingDay::Stochastic
        } else {
            EffectiveTradingDay::Ordinary
        }
    }
}
