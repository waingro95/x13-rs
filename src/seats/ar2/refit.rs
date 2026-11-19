//! The SEARCH refit of the AR(2)/AR(3) coordinates and the seasonal MA.

use super::residual::Ar2Objective;
use super::transform::{Ar2Coefficients, Ar2Coordinates};
use crate::Frequency;
use crate::seats::failure::{Error, Stage};
use crate::seats::refit::search::{BoundRule, SearchObjective};
use crate::seats::residual::ResidualResult;

/// What SEARCH left: the final coordinates, their coefficients, the residuals
/// with Detpri and the objective there, and SEARCH's bound mask and count of
/// fixed coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ar2RefitResult {
    pub(crate) coordinates: Ar2Coordinates,
    pub(crate) coefficients: Ar2Coefficients,
    pub(crate) scaled_residuals: Vec<f64>,
    pub(crate) determinant_scale: f64,
    pub(crate) objective: f64,
    pub(crate) bound_mask: Vec<bool>,
    pub(crate) fixed: usize,
}

/// SEARCH for `(2 1 0)(0 1 1)` (N=3) and `(3 1 0)(0 1 1)` (N=4) on the
/// differenced series `wd`, from `start` with the `fixed` coordinates held.
pub(crate) fn search<const N: usize>(
    frequency: Frequency,
    wd: &[f64],
    start: [f64; N],
    fixed: [bool; N],
) -> Result<Ar2RefitResult, Error>
where
    Ar2Coordinates: From<[f64; N]>,
{
    let regular = Ar2Coordinates::from(start).regular;
    let (result, objective) = crate::seats::refit::search::search(
        Ar2Objective::new(frequency, wd, regular),
        start,
        fixed,
        BoundRule::from_orders(crate::seats::Orders::new(
            [regular.degree(), 1, 0],
            [0, 1, 1],
        )),
    )?;
    Ok(Ar2RefitResult {
        coordinates: Ar2Coordinates::from(result.parameters),
        coefficients: objective.coefficients(),
        scaled_residuals: result.scaled_residuals,
        determinant_scale: result.determinant_scale,
        objective: result.objective,
        bound_mask: result.bound_mask.to_vec(),
        fixed: result.fixed,
    })
}

impl<const N: usize> SearchObjective<N> for Ar2Objective<'_>
where
    Ar2Coordinates: From<[f64; N]>,
{
    fn calcfx(&mut self, point: [f64; N]) -> Result<ResidualResult, Error> {
        self.evaluate(Ar2Coordinates::from(point))
            .map_err(|error| error.restage(Stage::Refit))
    }
    fn evaluations(&self) -> usize {
        self.evaluations()
    }
}
