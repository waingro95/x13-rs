use crate::regarima::{OutlierType, OutlierTypes};

/// Whether fitted outliers stay in the final X-11 output, by outlier type.
///
/// This does not alter outlier search, model estimation, or the stored fitted
/// outliers. Removal affects the final adjusted series; level shifts also
/// affect the final trend. Direct X-11 and fits without selected classes are
/// unchanged.
///
/// Pseudo-additive regARIMA X-11 allows no outliers in the model. There, as in
/// X-13ARIMA-SEATS, a removal request on a series without gaps with an
/// automatic model or an excluded constant is refused before fitting
/// ([`Error::InvalidSpec`](crate::Error::InvalidSpec)), and with an included
/// constant it has no effect. On a series with gaps, removing additive
/// outliers is refused with the same error, and removing other classes has no
/// effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FinalOutliers {
    /// Retain fitted outliers in the final components (the default).
    #[default]
    Retain,
    /// Remove these fitted classes from the final output.
    Remove(OutlierTypes),
}

impl FinalOutliers {
    pub(crate) const fn removes(self, kind: OutlierType) -> bool {
        match self {
            Self::Retain => false,
            Self::Remove(types) => types.contains(kind),
        }
    }
}
