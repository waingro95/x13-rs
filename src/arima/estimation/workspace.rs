use crate::regarima::ResolvedArimaModel;

/// The MINPACK scaling diagonal `rgarma.f` carries from one fit of a model
/// to the next, one entry per ARMA parameter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmaWorkspace {
    model: ResolvedArimaModel,
    diagonal: Vec<f64>,
}

impl ArmaWorkspace {
    pub(crate) fn reset(model: ResolvedArimaModel) -> Self {
        let regular = model.regular();
        let seasonal = model.seasonal();
        let parameter_count = usize::from(regular.p())
            + usize::from(seasonal.p())
            + usize::from(regular.q())
            + usize::from(seasonal.q());
        Self {
            model,
            diagonal: vec![0.0; parameter_count],
        }
    }

    pub(crate) fn model(&self) -> ResolvedArimaModel {
        self.model
    }

    pub(crate) fn diagonal(&self) -> &[f64] {
        &self.diagonal
    }

    pub(crate) fn replace_diagonal(&mut self, values: Vec<f64>) {
        debug_assert_eq!(values.len(), self.diagonal.len());
        self.diagonal = values;
    }
}

/// Whether the ARMA covariance step found a singular matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArmaCovarianceDiagnostic {
    None,
    Singular,
}

#[cfg(test)]
mod tests;
