//! The [`Error`](enum@Error) that [`adjust`](crate::adjust) returns, its
//! coarse [`ErrorKind`], and the payloads of its four catch-all variants.
//!
//! [`Error`](enum@Error) and [`ErrorKind`] are also at the crate root. A
//! payload type describes its failure through `Display` and has no
//! accessors: route on [`Error::kind`] and report the message.

pub use crate::api::{Error, ErrorKind, InternalError, InvalidSpec, NumericalFailure, Unsupported};

pub(crate) use crate::api::error::{
    AdjustmentOutput, Component, ExactLikelihoodFailure, ModelSelectionConstantTest,
    NumericalDetail, NumericalOperation, SpecProblem, TransformSelectionUnavailableReason,
    UnsupportedConfiguration, UnsupportedDetail, X11ExtremeScaleUnavailableReason, X11ExtremeStage,
    X11ExtremeWeightingPass, X11LogAdditiveStage, X11LogAdditiveUnavailableReason,
    X11PseudoAdditiveStage, X11PseudoAdditiveUnavailableReason, X11ShrinkageStage,
    X11ShrinkageUnavailableReason, X11TrendUnavailableReason,
};
