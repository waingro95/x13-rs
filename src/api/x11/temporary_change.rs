/// Placement of fitted temporary-change effects in final X-11 components.
///
/// This controls final restoration only. It does not change outlier search,
/// model estimation, or the fitted outliers exposed in the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TemporaryChangeAllocation {
    /// Restore fitted temporary changes to the final irregular (the default).
    #[default]
    Irregular,
    /// Restore fitted temporary changes to the final trend, unless final
    /// outlier removal selects temporary changes.
    Trend,
}
