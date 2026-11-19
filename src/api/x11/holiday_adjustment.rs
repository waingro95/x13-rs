/// Placement of fitted holiday effects in final X-11 components.
///
/// This controls final restoration only. It does not change regression
/// estimation, preadjustment, or the fitted calendar effects in the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HolidayAdjustment {
    /// Remove fitted holidays from the final adjusted series (the default).
    #[default]
    Remove,
    /// Retain fitted holidays in the final adjusted and irregular series.
    Retain,
}
