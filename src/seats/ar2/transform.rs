//! STAVAL's starting values and the TRANS1 and TRANSC coordinate transforms
//! for a regular AR(2) or AR(3) with seasonal MA.

use crate::Frequency;
use crate::seats::decomposition::root_allocation::RegularArCoefficients;
use crate::seats::failure::{Error, Stage, usable};
use crate::seats::refit::start::contract_start;

/// The regular AR coefficients and the seasonal MA of `(2 1 0)(0 1 1)` or
/// `(3 1 0)(0 1 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ar2Coefficients {
    pub(crate) regular: RegularArCoefficients,
    pub(crate) seasonal_ma: f64,
}

/// The same model in SEARCH's coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ar2Coordinates {
    pub(crate) regular: RegularArCoefficients,
    pub(crate) seasonal_ma: f64,
}

impl Ar2Coefficients {
    /// Regular AR(2) or AR(3) coefficients, then the seasonal MA. Any other
    /// parameter count is not this family's layout.
    pub(crate) fn from_parameters(values: &[f64]) -> Option<Self> {
        let (regular, seasonal_ma) = match *values {
            [a, b, seasonal_ma] => (RegularArCoefficients::Two([a, b]), seasonal_ma),
            [a, b, c, seasonal_ma] => (RegularArCoefficients::Three([a, b, c]), seasonal_ma),
            _ => return None,
        };
        Some(Self {
            regular,
            seasonal_ma,
        })
    }
}

impl From<[f64; 3]> for Ar2Coordinates {
    fn from([a, b, seasonal_ma]: [f64; 3]) -> Self {
        Self {
            regular: RegularArCoefficients::Two([a, b]),
            seasonal_ma,
        }
    }
}

impl From<[f64; 4]> for Ar2Coordinates {
    fn from([a, b, c, seasonal_ma]: [f64; 4]) -> Self {
        Self {
            regular: RegularArCoefficients::Three([a, b, c]),
            seasonal_ma,
        }
    }
}

impl Ar2Coordinates {
    /// The regular AR coordinates, then the seasonal MA.
    pub(crate) fn values(self) -> Vec<f64> {
        self.regular
            .as_slice()
            .iter()
            .copied()
            .chain([self.seasonal_ma])
            .collect()
    }
}

/// TRANS1's result for the regular AR group: the coefficients after its snap
/// and their SEARCH coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ar2RegularStart {
    pub(crate) coefficients: RegularArCoefficients,
    pub(crate) coordinates: RegularArCoefficients,
}

/// TRANSC's result for the regular AR group: the coefficients and, for an
/// AR(2) with a nonnegative discriminant, its two real roots.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ar2RegularPolynomial {
    pub(crate) coefficients: RegularArCoefficients,
    pub(crate) real_roots: Option<[f64; 2]>,
}

/// `value` when it is finite, else [`Failure::Unusable`](crate::seats::Failure::Unusable)
/// naming it at [`Stage::Refit`].
fn finite(value: f64, quantity: &'static str, index: Option<usize>) -> Result<f64, Error> {
    usable(quantity, index, value).map_err(|failure| failure.at(Stage::Refit))
}

/// STAVAL's starting coefficients for an AR(`degree`) with seasonal MA, from
/// the autocorrelations `acf` of the differenced series.
pub(crate) fn starting_parameters(
    acf: &[f64; 36],
    frequency: Frequency,
    degree: usize,
) -> Result<Ar2Coefficients, Error> {
    let start = crate::seats::refit::start::starting_parameters(acf, frequency)?;
    // STAVAL:2507–2514: each further AR coefficient starts at half the one
    // before it.
    Ok(Ar2Coefficients {
        regular: if degree == 2 {
            RegularArCoefficients::Two([start.regular, 0.5 * start.regular])
        } else {
            RegularArCoefficients::Three([
                start.regular,
                0.5 * start.regular,
                0.5 * (0.5 * start.regular),
            ])
        },
        seasonal_ma: start.seasonal_ma,
    })
}

/// TRANS1's in-place PHI mutation, which precedes its divisions.
///
/// ANALTS calls TRANS1 on every pass, before it branches on INIT. With INIT=2
/// this is all of TRANS1 that reaches the residuals: CALCFX calls TRANSC only
/// when INIT<2, so the transformed x it also writes is never read. After the
/// import snap, a `(3 1 0)` PHI can make the Phi2 division 0/0; Build 62 stores
/// the NaN and never reads it.
pub(crate) fn trans1_coefficients(coefficients: RegularArCoefficients) -> RegularArCoefficients {
    match coefficients {
        RegularArCoefficients::Two(mut c) => {
            // TRANS1:1700–1703: strict threshold and input mutation precede division.
            if (1.0 - c[1]).abs() < 1e-9 {
                c[1] = 0.9999999;
            }
            RegularArCoefficients::Two(c)
        }
        RegularArCoefficients::Three(mut c) => {
            // TRANS1:1689–1691: mutate Phi3 before the divisions.
            if (c[2] - 1.0).abs() < 1e-9 {
                c[2] = 0.9999999_f64.copysign(c[2]);
            }
            RegularArCoefficients::Three(c)
        }
    }
}

/// TRANS1's snap, then its divisions with finiteness checks. Only the INIT=0
/// starts use it, because their transformed x feeds SEARCH.
pub(crate) fn transform_regular_start(
    coefficients: RegularArCoefficients,
) -> Result<Ar2RegularStart, Error> {
    for (index, &value) in coefficients.as_slice().iter().enumerate() {
        finite(value, "regular AR start value", Some(index))?;
    }
    let coefficients = trans1_coefficients(coefficients);
    let raw = match coefficients {
        RegularArCoefficients::Two(c) => {
            let first = finite(c[0] / (1.0 - c[1]), "regular AR start coordinate", Some(0))?;
            RegularArCoefficients::Two([first, c[1]])
        }
        RegularArCoefficients::Three(c) => {
            let mut x1 = finite(
                0.5 * ((c[0] + c[1]) / (1.0 - c[2]) + 1.0),
                "regular AR start coordinate",
                Some(0),
            )?;
            let x2 = finite(
                1.0 + (c[0] - c[1]) / (1.0 + c[2]),
                "regular AR start coordinate",
                Some(1),
            )?;
            // TRANS1:1694–1697: x1's snap precedes the Phi2 division.
            if (x1 + 1.0).abs() < 1e-9 {
                x1 = -0.9999999;
            }
            let x2 = finite(
                x2 / (1.0 + x1) - 1.0,
                "regular AR start coordinate",
                Some(1),
            )?;
            RegularArCoefficients::Three([x1, x2, c[2]])
        }
    };
    Ok(Ar2RegularStart {
        coefficients,
        coordinates: raw.map(contract_start),
    })
}

/// The seasonal MA's SEARCH coordinate: TRANS1 contracts the starting
/// `coefficient` into its bounds.
pub(crate) fn transform_seasonal_start(coefficient: f64) -> Result<f64, Error> {
    finite(coefficient, "seasonal MA start value", None)?;
    Ok(contract_start(coefficient))
}

/// TRANSC for the regular AR group: the coefficients at SEARCH's
/// `coordinates`.
pub(crate) fn regular_coefficients(
    coordinates: RegularArCoefficients,
) -> Result<Ar2RegularPolynomial, Error> {
    let [x1, x2] = match coordinates {
        RegularArCoefficients::Two(v) => v,
        RegularArCoefficients::Three(v) => {
            for (index, value) in v.into_iter().enumerate() {
                finite(value, "regular AR coordinate", Some(index))?;
            }
            let c = crate::seats::decomposition::root_allocation::cubic_coefficients(v);
            for (index, value) in c.into_iter().enumerate() {
                finite(value, "regular AR coefficient", Some(index))?;
            }
            return Ok(Ar2RegularPolynomial {
                coefficients: RegularArCoefficients::Three(c),
                real_roots: None,
            });
        }
    };
    finite(x1, "regular AR coordinate", Some(0))?;
    let second = finite(x2, "regular AR coordinate", Some(1))?;
    let first = finite(x1 * (1.0 - second), "regular AR coefficient", Some(0))?;
    // TRANSC rounds 4*c2, then fuses the c1*c1 addition.
    let discriminant = finite(
        first.mul_add(first, 4.0 * second),
        "regular AR discriminant",
        None,
    )?;
    let real_roots = if discriminant >= 0.0 {
        let root = discriminant.sqrt();
        Some([0.5 * (first + root), 0.5 * (first - root)])
    } else {
        None
    };
    Ok(Ar2RegularPolynomial {
        coefficients: RegularArCoefficients::Two([first, second]),
        real_roots,
    })
}

/// The real roots TRANSC (ANSUB1:2394–2446) computes for an AR(3). Only the
/// common-factor check of a model with a regular MA reads them; the
/// decomposition takes its roots from RPQ.
pub(crate) fn cubic_detector_roots(c: [f64; 3]) -> Result<Vec<f64>, Error> {
    let check = |value, quantity| finite(value, quantity, None);
    let square = c[0] * c[0];
    let d = check(square / 3.0 + c[1], "common-factor cubic d")?;
    let e = check(
        (square * c[0]).mul_add(2.0, (9.0 * c[0]) * c[1]) / 27.0 + c[2],
        "common-factor cubic e",
    )?;
    let discriminant = check(
        ((d * d) * d).mul_add(4.0, -(27.0 * (e * e))),
        "common-factor cubic discriminant",
    )?;
    let mut y = check(
        if discriminant > 0.0 {
            -e / d
        } else if e > 0.0 {
            1.0 - c[0] / 3.0
        } else {
            -1.0 - c[0] / 3.0
        },
        "common-factor cubic start",
    )?;
    // TRANSC takes at most eleven Newton steps and keeps the last y when it
    // prints its iteration warning.
    for _ in 0..11 {
        let numerator = y.mul_add(y * y, -(d * y)) - e;
        let denominator = (3.0 * y).mul_add(y, -d);
        let delta = check(numerator / denominator, "common-factor cubic Newton step")?;
        y = check(y - delta, "common-factor cubic root")?;
        if delta.abs() <= 0.00005 {
            break;
        }
    }
    let mut roots = vec![y];
    if discriminant >= 0.0 {
        // TRANSC carries a NaN here into comparisons that are then false;
        // this crate returns an error instead.
        let radical = check(
            y.mul_add(y, -(4.0 * e / y)).sqrt(),
            "common-factor cubic remaining roots",
        )?;
        roots.extend([0.5 * (-y + radical), 0.5 * (-y - radical)]);
    }
    for root in &mut roots {
        *root = check(*root + c[0] / 3.0, "common-factor cubic root shift")?;
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_split_into_regular_ar_and_seasonal_ma() {
        let split = |values: &[f64]| {
            Ar2Coefficients::from_parameters(values)
                .map(|parameters| (parameters.regular, parameters.seasonal_ma))
        };
        assert_eq!(
            split(&[0.1, 0.2, 0.3]),
            Some((RegularArCoefficients::Two([0.1, 0.2]), 0.3))
        );
        assert_eq!(
            split(&[0.1, 0.2, 0.3, 0.4]),
            Some((RegularArCoefficients::Three([0.1, 0.2, 0.3]), 0.4))
        );
        assert_eq!(split(&[0.1, 0.2]), None);
        assert_eq!(split(&[0.1, 0.2, 0.3, 0.4, 0.5]), None);
    }

    #[test]
    fn coordinates_read_back_in_order() {
        assert_eq!(
            Ar2Coordinates::from([0.1, 0.2, 0.3]).values(),
            [0.1, 0.2, 0.3]
        );
        assert_eq!(
            Ar2Coordinates::from([0.1, 0.2, 0.3, 0.4]).values(),
            [0.1, 0.2, 0.3, 0.4]
        );
    }
}
