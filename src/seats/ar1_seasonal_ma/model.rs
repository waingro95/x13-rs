//! The AR(1) models `(1 1 0)(0 1 1)` and `(1 1 0)(0 0 0)`: the import from
//! regARIMA, the sizes a pass reads and the component polynomials.
//!
//! [`import_ar1_seasonal_ma`] follows NMLMDL: the AR and the seasonal MA
//! coefficient each go through TRANS0's degree-one clip, and ANALTS then
//! reverses both signs. [`ar1_seasonal_ma_geometry`] takes RPQ's root of the
//! AR polynomial, which F1RST allocates to the trend, the seasonal or the
//! transitory component, and builds SIGEX's polynomials.

use crate::Frequency;
use crate::numerics::BUILD_62_PI;
use crate::seats::decomposition::polynomial::{DenominatorFactors, SpectralInputs, convolve};
use crate::seats::failure::{Error, Stage, usable};
use crate::seats::import::{AR_BOUND, MA_BOUND, trans0_degree_one};

/// The regARIMA estimate of an AR(1) model: the AR coefficient and, with a
/// seasonal MA, its coefficient; both finite.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ar1SeasonalMaInput {
    frequency: Frequency,
    parameters: [f64; 2],
    seasonal: bool,
}

impl Ar1SeasonalMaInput {
    /// `(1 1 0)(0 1 1)`, with the AR and seasonal MA coefficients.
    pub(crate) fn new(frequency: Frequency, parameters: [f64; 2]) -> Result<Self, Error> {
        validate_parameters("fitted model coefficient", parameters)?;
        Ok(Self {
            frequency,
            parameters,
            seasonal: true,
        })
    }

    /// `(1 1 0)(0 0 0)`, with the AR coefficient.
    pub(crate) fn nonseasonal(frequency: Frequency, phi: f64) -> Result<Self, Error> {
        let mut input = Self::new(frequency, [phi, 0.0])?;
        input.seasonal = false;
        Ok(input)
    }
}

/// An AR(1) model in the SEATS convention: the AR and seasonal MA
/// coefficients of the imported model, or of a later refit. Both are finite,
/// and the seasonal MA is zero when the model has none.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedAr1SeasonalMa {
    frequency: Frequency,
    parameters: [f64; 2],
    seasonal: bool,
}

impl ResolvedAr1SeasonalMa {
    /// The model of SEARCH's coefficients.
    pub(crate) fn from_refit(
        frequency: Frequency,
        parameters: [f64; 2],
        seasonal: bool,
    ) -> Result<Self, Error> {
        validate_parameters("refit coefficient", parameters)?;
        Ok(Self {
            frequency,
            parameters,
            seasonal,
        })
    }
    pub(crate) fn frequency(&self) -> Frequency {
        self.frequency
    }
    pub(crate) fn parameters(&self) -> [f64; 2] {
        self.parameters
    }
    pub(crate) fn has_seasonal(&self) -> bool {
        self.seasonal
    }
    pub(super) fn differences(&self) -> crate::seats::tails::DifferenceOrder {
        if self.seasonal {
            crate::seats::tails::DifferenceOrder::First
        } else {
            crate::seats::tails::DifferenceOrder::NonseasonalFirst
        }
    }
}

pub(crate) fn import_ar1_seasonal_ma(input: &Ar1SeasonalMaInput) -> ResolvedAr1SeasonalMa {
    // nmlmdl.f:80–139 subtracts from zero and sends the two groups through
    // TRANS0/TRANS2 separately.
    let phi = trans0_degree_one(0.0 - input.parameters[0], AR_BOUND);
    let bth = trans0_degree_one(0.0 - input.parameters[1], MA_BOUND);
    // analts.f:1125–1135 reverses both signs into the SEATS convention.
    ResolvedAr1SeasonalMa {
        frequency: input.frequency,
        parameters: [-phi, -bth],
        seasonal: input.seasonal,
    }
}

/// Refuses a nonfinite AR or seasonal MA coefficient, reported as
/// `quantity`.
fn validate_parameters(quantity: &'static str, parameters: [f64; 2]) -> Result<(), Error> {
    for (index, value) in parameters.into_iter().enumerate() {
        usable(quantity, Some(index), value).map_err(|failure| failure.at(Stage::Import))?;
    }
    Ok(())
}

/// RPQ's description of the one real root of the AR polynomial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Ar1RealRoot {
    pub(crate) real: f64,
    pub(crate) imaginary: f64,
    pub(crate) modulus: f64,
    pub(crate) argument_degrees: f64,
    pub(crate) period: f64,
}

/// The AR root and the polynomials SPECTRUM factors.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ar1Geometry {
    pub(crate) root: Ar1RealRoot,
    pub(crate) inputs: SpectralInputs,
}

/// RPQ's root of the model's AR polynomial, allocated by F1RST, and SIGEX's
/// polynomials for it.
pub(crate) fn ar1_seasonal_ma_geometry(
    model: &ResolvedAr1SeasonalMa,
) -> Result<Ar1Geometry, Error> {
    let ar_polynomial = [1.0, -model.parameters[0]];
    // RPQ (ansub2.f:60–65) takes the root of a degree-one polynomial
    // directly, without C02AEF or trimming the degree.
    let real = -ar_polynomial[1];
    let imaginary = 0.0;
    let modulus = real.mul_add(real, imaginary * imaginary).sqrt();
    usable("root modulus", None, modulus).map_err(|failure| failure.at(Stage::Import))?;
    // ansub2.f:131–175, with Build 62's pi and its single-precision period.
    let ratio = if modulus < 1e-8 { 0.0 } else { real / modulus };
    let angle = if ratio.abs() <= 1.0 {
        ratio.acos()
    } else if real < 0.0 {
        BUILD_62_PI
    } else {
        0.0
    };
    let root = Ar1RealRoot {
        real,
        imaginary,
        modulus,
        argument_degrees: 180.0 * angle / BUILD_62_PI,
        period: crate::seats::roots::REAL_ROOT_PERIOD,
    };

    degree_one_geometry_body(
        model.frequency,
        root,
        None,
        model.seasonal.then_some(model.parameters[1]),
        model.differences(),
    )
}

/// F1RST's allocation of one real AR `root` and SIGEX's polynomials, with
/// the given MA coefficients. The root and the MA coefficients need not come
/// from the same fit.
pub(crate) fn degree_one_geometry_body(
    frequency: Frequency,
    root: Ar1RealRoot,
    regular_ma: Option<f64>,
    seasonal_ma: Option<f64>,
    differences: crate::seats::tails::DifferenceOrder,
) -> Result<Ar1Geometry, Error> {
    let period = usize::from(frequency.periods_per_year());
    let real = root.real;
    let ar_polynomial = [1.0, -real];
    for value in [real].into_iter().chain(seasonal_ma).chain(regular_ma) {
        usable("degree-one geometry", None, value).map_err(|failure| failure.at(Stage::Import))?;
    }

    // sigex.f:477–523: each difference updates the coefficients from the
    // highest down; the seasonal sum follows.
    let mut trend_nonstationary = vec![1.0];
    for i in 1..=differences.ordinary() + usize::from(differences.seasonal()) {
        trend_nonstationary.push(0.0);
        for k in (1..=i).rev() {
            trend_nonstationary[k] -= trend_nonstationary[k - 1];
        }
    }
    let mut trend_stationary = vec![1.0];
    let mut seasonal_nonstationary = if differences.seasonal() {
        convolve(&vec![1.0; period], &[1.0])
    } else {
        vec![1.0]
    };
    let mut seasonal_stationary = vec![1.0];
    let mut cycle_stationary = vec![1.0];
    let cycle_nonstationary = vec![1.0];

    // F1RST (sigsub.f:66–103). A root within 1e-6 of a unit root joins the
    // nonstationary factor; otherwise a negative root is seasonal from
    // modulus .5 with a seasonal difference and from .9 without one, a
    // positive root is trend from .5, and the rest is transitory. A negative
    // zero counts as nonpositive.
    if real <= 0.0 {
        if (1.0 + real).abs() < 1e-6 {
            seasonal_nonstationary = convolve(&ar_polynomial, &seasonal_nonstationary);
        } else if real.abs() >= if differences.seasonal() { 0.5 } else { 0.9 } {
            seasonal_stationary = convolve(&ar_polynomial, &seasonal_stationary);
        } else {
            cycle_stationary = convolve(&ar_polynomial, &cycle_stationary);
        }
    } else if (1.0 - real).abs() < 1e-6 {
        trend_nonstationary = convolve(&ar_polynomial, &trend_nonstationary);
    } else if real.abs() >= 0.5 {
        trend_stationary = convolve(&ar_polynomial, &trend_stationary);
    } else {
        cycle_stationary = convolve(&ar_polynomial, &cycle_stationary);
    }

    // sigex.f:582–622: each CONV keeps its operand order.
    let trend_combined = convolve(&trend_stationary, &trend_nonstationary);
    let seasonal_combined = convolve(&seasonal_stationary, &seasonal_nonstationary);
    let cycle_combined = convolve(&cycle_stationary, &cycle_nonstationary);
    let trend_cycle = convolve(&trend_combined, &cycle_combined);
    let seasonal_cycle = convolve(&seasonal_combined, &cycle_combined);
    let trend_seasonal = convolve(&trend_combined, &seasonal_combined);
    let aggregate_denominator = convolve(&seasonal_combined, &trend_cycle);
    let adjustment_stationary = convolve(&cycle_stationary, &trend_stationary);
    let adjustment_nonstationary = convolve(&cycle_nonstationary, &trend_nonstationary);
    let theta = regular_ma.map_or_else(|| vec![1.0], |theta| vec![1.0, -theta]);
    let seasonal_theta = if let Some(ma) = seasonal_ma {
        let mut theta = vec![0.0; period + 1];
        theta[0] = 1.0;
        theta[period] = -ma;
        theta
    } else {
        vec![1.0]
    };
    let aggregate_ma = convolve(&theta, &seasonal_theta);
    Ok(Ar1Geometry {
        root,
        inputs: SpectralInputs {
            theta,
            seasonal_theta,
            aggregate_ma,
            trend: DenominatorFactors {
                stationary: trend_stationary,
                nonstationary: trend_nonstationary,
                combined: trend_combined,
            },
            seasonal: DenominatorFactors {
                stationary: seasonal_stationary,
                nonstationary: seasonal_nonstationary,
                combined: seasonal_combined,
            },
            cycle: DenominatorFactors {
                stationary: cycle_stationary,
                nonstationary: cycle_nonstationary,
                combined: cycle_combined,
            },
            trend_cycle,
            seasonal_cycle,
            trend_seasonal,
            adjustment_stationary,
            adjustment_nonstationary,
            aggregate_denominator,
        },
    })
}
