//! Outliers: the automatic search, outliers at known dates, and the outliers
//! the fitted model contains.

use crate::error::SpecProblem;
use crate::time::TimeAxis;
use crate::{Error, Period};
use std::num::NonZeroU8;

/// Kind of automatically detected regARIMA outlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutlierType {
    /// Additive outlier affecting one period.
    Additive,
    /// Permanent level shift.
    LevelShift,
    /// Temporary change that decays after its initial period.
    TemporaryChange,
}

display_phrases!(OutlierType {
    Additive => "additive outlier",
    LevelShift => "level shift",
    TemporaryChange => "temporary change",
});

impl OutlierType {
    /// Build 62's AO, LS, TC order: `idotlr.f` tests a date's candidates in
    /// this order, and automatic outliers sort by date, then by this rank.
    pub(crate) const BUILD_62_ORDER: [Self; 3] =
        [Self::Additive, Self::LevelShift, Self::TemporaryChange];

    /// Position of this type in [`Self::BUILD_62_ORDER`].
    pub(crate) const fn build_62_index(self) -> usize {
        match self {
            Self::Additive => 0,
            Self::LevelShift => 1,
            Self::TemporaryChange => 2,
        }
    }
}

/// A nonempty set of outlier classes for search or final-output selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutlierTypes {
    mask: NonZeroU8,
}

impl OutlierTypes {
    const ADDITIVE_MASK: u8 = 0b001;
    const LEVEL_SHIFT_MASK: u8 = 0b010;
    const TEMPORARY_CHANGE_MASK: u8 = 0b100;

    /// Additive outliers only.
    pub const ADDITIVE: Self = Self::from_mask(Self::ADDITIVE_MASK);
    /// Level shifts only.
    pub const LEVEL_SHIFT: Self = Self::from_mask(Self::LEVEL_SHIFT_MASK);
    /// Temporary changes only.
    pub const TEMPORARY_CHANGE: Self = Self::from_mask(Self::TEMPORARY_CHANGE_MASK);
    /// Additive outliers and level shifts, the X-13ARIMA-SEATS default.
    pub const ADDITIVE_AND_LEVEL_SHIFT: Self =
        Self::from_mask(Self::ADDITIVE_MASK | Self::LEVEL_SHIFT_MASK);
    /// Additive outliers and temporary changes.
    pub const ADDITIVE_AND_TEMPORARY_CHANGE: Self =
        Self::from_mask(Self::ADDITIVE_MASK | Self::TEMPORARY_CHANGE_MASK);
    /// Level shifts and temporary changes.
    pub const LEVEL_SHIFT_AND_TEMPORARY_CHANGE: Self =
        Self::from_mask(Self::LEVEL_SHIFT_MASK | Self::TEMPORARY_CHANGE_MASK);
    /// Additive outliers, level shifts, and temporary changes.
    pub const ALL: Self =
        Self::from_mask(Self::ADDITIVE_MASK | Self::LEVEL_SHIFT_MASK | Self::TEMPORARY_CHANGE_MASK);

    /// Returns the union of two nonempty outlier-type sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self::from_mask(self.mask.get() | other.mask.get())
    }

    /// Whether this set includes `kind`.
    #[must_use]
    pub const fn contains(self, kind: OutlierType) -> bool {
        let mask = match kind {
            OutlierType::Additive => Self::ADDITIVE_MASK,
            OutlierType::LevelShift => Self::LEVEL_SHIFT_MASK,
            OutlierType::TemporaryChange => Self::TEMPORARY_CHANGE_MASK,
        };
        self.mask.get() & mask != 0
    }

    #[expect(
        clippy::panic,
        reason = "every caller passes a nonzero *_MASK constant or an OR of them, or an OR of \
                  two NonZeroU8 masks, and an OR with a nonzero operand is nonzero"
    )]
    const fn from_mask(mask: u8) -> Self {
        let Some(mask) = NonZeroU8::new(mask) else {
            panic!("outlier type set must not be empty");
        };
        Self { mask }
    }
}

/// How the automatic outlier search adds candidates on each pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutlierMethod {
    /// Add the most significant candidate, then re-estimate before testing
    /// again (the default).
    AddOne,
    /// Add every candidate over its critical value, then re-estimate before
    /// testing again.
    AddAll,
}

/// A validated explicit outlier critical value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutlierCriticalValue {
    value: f64,
}

impl OutlierCriticalValue {
    /// `automd.f` `BIGCV`: the cutoff of the AO-only pass automdl runs when
    /// the spec requests no outlier search, so that no candidate qualifies.
    pub(crate) const UNREACHABLE: Self = Self { value: 1_000_001.0 };

    /// Constructs a finite, positive critical value.
    pub fn new(value: f64) -> Result<Self, Error> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::from(SpecProblem::OutlierCriticalValue));
        }
        Ok(Self { value })
    }

    /// Numeric critical value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

/// Automatic or explicit critical values for selected outlier types.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum OutlierCriticalValues {
    /// Derive each cutoff from the number of observations, as X-13ARIMA-SEATS
    /// does.
    Automatic,
    /// Use one explicit cutoff for every selected type.
    Shared(OutlierCriticalValue),
    /// Override selected types individually and use the automatic cutoff for
    /// each omitted type.
    PerType {
        /// Additive outlier cutoff, or the automatic one.
        additive: Option<OutlierCriticalValue>,
        /// Level shift cutoff, or the automatic one.
        level_shift: Option<OutlierCriticalValue>,
        /// Temporary change cutoff, or the automatic one.
        temporary_change: Option<OutlierCriticalValue>,
    },
}

impl OutlierCriticalValues {
    /// Uses one explicit cutoff for every selected outlier type.
    #[must_use]
    pub const fn shared(value: OutlierCriticalValue) -> Self {
        Self::Shared(value)
    }

    /// Supplies per-type overrides with automatic fallback for omitted types.
    #[must_use]
    pub const fn per_type(
        additive: Option<OutlierCriticalValue>,
        level_shift: Option<OutlierCriticalValue>,
        temporary_change: Option<OutlierCriticalValue>,
    ) -> Self {
        Self::PerType {
            additive,
            level_shift,
            temporary_change,
        }
    }
}

/// A validated temporary-change decay rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporaryChangeRate {
    value: f64,
}

impl TemporaryChangeRate {
    /// Constructs a finite rate strictly between zero and one.
    pub fn new(value: f64) -> Result<Self, Error> {
        if !value.is_finite() || value <= 0.0 || value >= 1.0 {
            return Err(Error::from(SpecProblem::TemporaryChangeRate));
        }
        Ok(Self { value })
    }

    /// Numeric decay rate.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

/// A user-specified outlier fitted without automatic detection or deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManualOutlier {
    kind: OutlierType,
    period: crate::Period,
}

impl ManualOutlier {
    /// Selects a dated AO, LS or TC. A TC decays at the request's rate
    /// ([`RegArimaSpec::with_temporary_change_rate`](crate::RegArimaSpec::with_temporary_change_rate)), or the
    /// frequency-dependent default. When adjusting, the date must have the
    /// input's frequency and lie in its span.
    #[must_use]
    pub const fn new(kind: OutlierType, period: crate::Period) -> Self {
        Self { kind, period }
    }

    /// Outlier type.
    #[must_use]
    pub const fn kind(self) -> OutlierType {
        self.kind
    }

    /// Period of the outlier.
    #[must_use]
    pub const fn period(self) -> crate::Period {
        self.period
    }

    pub(crate) fn validate_list(
        outliers: &[Self],
        axis: TimeAxis,
        count: usize,
    ) -> Result<(), Error> {
        for (position, &outlier) in outliers.iter().enumerate() {
            outlier.check_frequency(axis)?;
            let index = axis.periods(count)?.position(|date| date == outlier.period);
            if index.is_none()
                || (outlier.kind == OutlierType::LevelShift
                    && index.is_some_and(|index| index == 0 || index + 1 == count))
                || outliers[..position].contains(&outlier)
            {
                return Err(Error::InvalidManualOutlier {
                    kind: outlier.kind,
                    period: outlier.period,
                });
            }
        }
        Ok(())
    }

    /// Rejects a date of another frequency rather than letting it miss every
    /// axis date and read as an out-of-span outlier.
    pub(crate) fn check_frequency(self, axis: TimeAxis) -> Result<(), Error> {
        if self.period.frequency() == axis.frequency() {
            Ok(())
        } else {
            Err(Error::PeriodFrequencyMismatch {
                period: self.period,
                expected: axis.frequency(),
            })
        }
    }
}

/// Settings for the automatic outlier search, turned on with
/// [`RegArimaSpec::with_outliers`](crate::RegArimaSpec::with_outliers).
///
/// The search tests every date in its span for the selected outlier types, adds
/// those whose t-statistics exceed the critical value, re-estimates, and
/// repeats; outliers that are no longer significant are then removed.
/// [`OutlierSpec::new`], also the [`Default`], is the X-13ARIMA-SEATS default:
/// additive outliers and level shifts, one candidate added per pass
/// ([`OutlierMethod::AddOne`]), critical values from the number of
/// observations, over the whole model span.
///
/// ```
/// use x13::regarima::{
///     OutlierCriticalValue, OutlierCriticalValues, OutlierSpec, OutlierType, OutlierTypes,
/// };
///
/// let search = OutlierSpec::new()
///     .with_types(OutlierTypes::ALL)
///     .with_critical_values(OutlierCriticalValues::shared(OutlierCriticalValue::new(3.5)?));
/// assert!(search.types().contains(OutlierType::TemporaryChange));
/// # Ok::<(), x13::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct OutlierSpec {
    types: OutlierTypes,
    method: OutlierMethod,
    critical_values: OutlierCriticalValues,
    span: Option<crate::api::PeriodSpan>,
}

impl OutlierSpec {
    /// The X-13ARIMA-SEATS default search.
    pub const fn new() -> Self {
        Self {
            types: OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT,
            method: OutlierMethod::AddOne,
            critical_values: OutlierCriticalValues::Automatic,
            span: None,
        }
    }

    /// Restricts automatic detection to these inclusive dates within the model span.
    pub const fn with_span(mut self, span: crate::api::PeriodSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Requested search span, or the whole model span by default.
    #[must_use]
    pub const fn span(self) -> Option<crate::api::PeriodSpan> {
        self.span
    }

    pub(crate) fn resolved_span(
        self,
        axis: TimeAxis,
        count: usize,
    ) -> Result<std::ops::Range<usize>, Error> {
        self.span
            .map_or(Ok(0..count), |span| span.resolve_outlier_span(axis, count))
    }

    /// Selects the nonempty set of candidate types.
    pub const fn with_types(mut self, types: OutlierTypes) -> Self {
        self.types = types;
        self
    }

    /// Selects the forward-addition method.
    pub const fn with_method(mut self, method: OutlierMethod) -> Self {
        self.method = method;
        self
    }

    /// Selects automatic or explicit critical values.
    pub const fn with_critical_values(mut self, values: OutlierCriticalValues) -> Self {
        self.critical_values = values;
        self
    }

    /// Selected candidate types.
    #[must_use]
    pub const fn types(self) -> OutlierTypes {
        self.types
    }

    /// Forward-addition method.
    #[must_use]
    pub const fn method(self) -> OutlierMethod {
        self.method
    }

    /// Critical-value policy.
    #[must_use]
    pub const fn critical_values(self) -> OutlierCriticalValues {
        self.critical_values
    }
}

impl Default for OutlierSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// One specified or automatically detected outlier in the final regARIMA model.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedOutlier {
    pub(crate) kind: OutlierType,
    pub(crate) period: Period,
    pub(crate) estimate: f64,
    pub(crate) standard_error: Option<f64>,
    pub(crate) t_value: Option<f64>,
}

impl DetectedOutlier {
    /// Fitted outlier kind.
    #[must_use]
    pub const fn kind(&self) -> OutlierType {
        self.kind
    }

    /// Calendar period containing the outlier.
    #[must_use]
    pub const fn period(&self) -> Period {
        self.period
    }

    /// Final fitted outlier coefficient.
    #[must_use]
    pub const fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Final non-robust coefficient standard error, when available.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }

    /// Final non-robust coefficient t value, when uncertainty is available.
    #[must_use]
    pub const fn t_value(&self) -> Option<f64> {
        self.t_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArimaModel, ArimaOrder, Frequency, RegArimaSpec, Series, Spec, x11};

    #[test]
    fn manual_outliers_of_another_frequency_are_rejected() {
        // March 2001 must not match 2001Q3, nor 2001Q2 February 2001.
        let quarterly = TimeAxis::quarterly(2000, 1).unwrap();
        let march = crate::Period::monthly(2001, 3).unwrap();
        let monthly = TimeAxis::monthly(2000, 1).unwrap();
        let second_quarter = crate::Period::quarterly(2001, 2).unwrap();
        for (axis, period, expected) in [
            (quarterly, march, Frequency::Quarterly),
            (monthly, second_quarter, Frequency::Monthly),
        ] {
            for kind in [
                OutlierType::Additive,
                OutlierType::LevelShift,
                OutlierType::TemporaryChange,
            ] {
                assert_eq!(
                    ManualOutlier::validate_list(&[ManualOutlier::new(kind, period)], axis, 40),
                    Err(Error::PeriodFrequencyMismatch { period, expected })
                );
            }
        }
        // Through `adjust`, before the model is fitted.
        let order = ArimaOrder::new(0, 1, 1).unwrap();
        let request = RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
            .with_manual_outliers(&[ManualOutlier::new(OutlierType::Additive, march)]);
        assert_eq!(
            crate::adjust(
                Series::new(quarterly.start(), &[100.0; 40]),
                &Spec::x11_regarima(request, x11::Options::new().with_mode(x11::Mode::Additive))
            )
            .unwrap_err(),
            Error::PeriodFrequencyMismatch {
                period: march,
                expected: Frequency::Quarterly,
            }
        );
    }
}
