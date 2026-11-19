//! The models of the ARMA family: the orders of each layout and where its
//! coefficients sit, the import of a fitted model (NMLMDL's sign reversal
//! and TRANS0), and ANALTS's checks on the model before its loop.

use crate::Frequency;
use crate::seats::decomposition::root_allocation::RegularArCoefficients;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::import::{AR_BOUND, MA_BOUND, trans0_degree_one};

/// A regular AR order the family covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArOrder {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
}

impl ArOrder {
    /// `None` above AR(3).
    pub(crate) const fn new(order: u8) -> Option<Self> {
        match order {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            3 => Some(Self::Three),
            _ => None,
        }
    }

    pub(crate) const fn count(self) -> u8 {
        self as u8
    }
}

/// A regular MA or regular differencing order the family covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LowOrder {
    Zero = 0,
    One = 1,
    Two = 2,
}

impl LowOrder {
    /// `None` above order 2.
    pub(crate) const fn new(order: u8) -> Option<Self> {
        match order {
            0 => Some(Self::Zero),
            1 => Some(Self::One),
            2 => Some(Self::Two),
            _ => None,
        }
    }

    pub(crate) const fn count(self) -> u8 {
        self as u8
    }
}

/// The regular AR coordinates of a parameter vector, split the way the
/// transforms treat them: one lag alone, two or three as a polynomial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RegularArGroup {
    Empty,
    Single(f64),
    Polynomial(RegularArCoefficients),
}

/// The orders of a model. The coordinates are the regular AR coefficients,
/// the seasonal AR, the regular MA and the seasonal MA, in that order; their
/// count alone does not say which is which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArmaLayout {
    /// `(0 1 1)(1 1 0)`.
    SeasonalAr,
    /// `(p 1 1)(0 1 1)`, p from 1 to 3.
    RegularSeasonal(ArOrder),
    /// `(p 1 1)(0 0 1)`, p from 1 to 3.
    RegularSeasonalUndifferenced(ArOrder),
    /// `(0 1 2)(0 1 1)`.
    SeasonalMa2,
    /// `(p 1 2)(0 1 1)`, p 1 or 2.
    RegularSeasonalMa2(ArOrder),
    /// `(p 1 1)(0 0 0)`, p from 1 to 3.
    Nonseasonal(ArOrder),
    /// `(p 1 2)(0 0 0)`, p from 0 to 2.
    NonseasonalMa2(ArOrder),
    /// Orders given one by one, for the models no variant above names; at
    /// most six coefficients.
    Integrated {
        ar: ArOrder,
        ma: LowOrder,
        seasonal_ar: bool,
        ordinary_difference: LowOrder,
        seasonal_difference: bool,
        seasonal_ma: bool,
    },
}

impl ArmaLayout {
    /// Whether the family runs this layout: its AR order is one the variant
    /// has and, for `Integrated`, it has at most six coordinates.
    pub(crate) fn is_supported(self) -> bool {
        match self {
            Self::RegularSeasonal(ar)
            | Self::RegularSeasonalUndifferenced(ar)
            | Self::Nonseasonal(ar) => ar != ArOrder::Zero,
            Self::RegularSeasonalMa2(ar) => matches!(ar, ArOrder::One | ArOrder::Two),
            Self::NonseasonalMa2(ar) => ar != ArOrder::Three,
            Self::SeasonalAr | Self::SeasonalMa2 => true,
            Self::Integrated { .. } => self.coordinates() <= 6,
        }
    }
    /// The same layout with no seasonal MA, for ChangeModel's BQ removal.
    /// `None` for a layout that has no such form.
    pub(crate) fn without_seasonal_ma(mut self) -> Option<Self> {
        let Self::Integrated { seasonal_ma, .. } = &mut self else {
            return None;
        };
        *seasonal_ma = false;
        Some(self)
    }
    pub(crate) fn differences(self) -> crate::seats::tails::DifferenceOrder {
        if let Self::Integrated {
            ordinary_difference,
            seasonal_difference,
            ..
        } = self
        {
            return match (ordinary_difference, seasonal_difference) {
                (LowOrder::Zero, false) => crate::seats::tails::DifferenceOrder::Stationary,
                (LowOrder::Zero, true) => crate::seats::tails::DifferenceOrder::SeasonalOnly,
                (LowOrder::One, true) => crate::seats::tails::DifferenceOrder::First,
                (LowOrder::One, false) => crate::seats::tails::DifferenceOrder::NonseasonalFirst,
                (LowOrder::Two, true) => crate::seats::tails::DifferenceOrder::Second,
                (LowOrder::Two, false) => crate::seats::tails::DifferenceOrder::NonseasonalSecond,
            };
        }
        if matches!(
            self,
            Self::Nonseasonal(_) | Self::NonseasonalMa2(_) | Self::RegularSeasonalUndifferenced(_)
        ) {
            crate::seats::tails::DifferenceOrder::NonseasonalFirst
        } else {
            crate::seats::tails::DifferenceOrder::First
        }
    }
    pub(crate) fn orders(self) -> crate::seats::Orders {
        use crate::seats::Orders;
        match self {
            Self::SeasonalAr => Orders::new([0, 1, 1], [1, 1, 0]),
            Self::RegularSeasonal(ar) => Orders::new([ar.count(), 1, 1], [0, 1, 1]),
            Self::RegularSeasonalUndifferenced(ar) => Orders::new([ar.count(), 1, 1], [0, 0, 1]),
            Self::SeasonalMa2 => Orders::new([0, 1, 2], [0, 1, 1]),
            Self::RegularSeasonalMa2(ar) => Orders::new([ar.count(), 1, 2], [0, 1, 1]),
            Self::Nonseasonal(ar) => Orders::new([ar.count(), 1, 1], [0, 0, 0]),
            Self::NonseasonalMa2(ar) => Orders::new([ar.count(), 1, 2], [0, 0, 0]),
            Self::Integrated {
                ar,
                ma,
                seasonal_ar,
                ordinary_difference,
                seasonal_difference,
                seasonal_ma,
            } => Orders {
                p: ar.count(),
                d: ordinary_difference.count(),
                q: ma.count(),
                bp: u8::from(seasonal_ar),
                bd: u8::from(seasonal_difference),
                bq: u8::from(seasonal_ma),
            },
        }
    }
    /// The parameter count: regular AR, seasonal AR, regular MA, seasonal MA.
    pub(crate) fn coordinates(self) -> usize {
        self.ar_coordinates() + self.regular_ma_count() + usize::from(self.has_seasonal_ma())
    }
    pub(super) fn regular_ar(self) -> ArOrder {
        match self {
            Self::SeasonalAr | Self::SeasonalMa2 => ArOrder::Zero,
            Self::RegularSeasonal(ar)
            | Self::RegularSeasonalUndifferenced(ar)
            | Self::RegularSeasonalMa2(ar)
            | Self::Nonseasonal(ar)
            | Self::NonseasonalMa2(ar)
            | Self::Integrated { ar, .. } => ar,
        }
    }
    pub(super) fn ar_coordinates(self) -> usize {
        self.regular_ar_coordinates() + usize::from(self.has_seasonal_ar())
    }
    pub(crate) fn has_seasonal_ar(self) -> bool {
        matches!(
            self,
            Self::SeasonalAr
                | Self::Integrated {
                    seasonal_ar: true,
                    ..
                }
        )
    }
    pub(crate) fn regular_ar_coordinates(self) -> usize {
        usize::from(self.regular_ar().count())
    }
    /// The regular AR coordinates at the front of `parameters`.
    pub(super) fn regular_ar_group(self, parameters: &[f64]) -> RegularArGroup {
        match self.regular_ar() {
            ArOrder::Zero => RegularArGroup::Empty,
            ArOrder::One => RegularArGroup::Single(parameters[0]),
            ArOrder::Two => RegularArGroup::Polynomial(RegularArCoefficients::Two([
                parameters[0],
                parameters[1],
            ])),
            ArOrder::Three => RegularArGroup::Polynomial(RegularArCoefficients::Three([
                parameters[0],
                parameters[1],
                parameters[2],
            ])),
        }
    }
    pub(crate) fn regular_ma_count(self) -> usize {
        if let Self::Integrated { ma, .. } = self {
            return usize::from(ma.count());
        }
        if matches!(
            self,
            Self::SeasonalMa2 | Self::RegularSeasonalMa2(_) | Self::NonseasonalMa2(_)
        ) {
            2
        } else {
            1
        }
    }
    pub(crate) fn has_seasonal_ma(self) -> bool {
        if let Self::Integrated { seasonal_ma, .. } = self {
            return seasonal_ma;
        }
        matches!(
            self,
            Self::RegularSeasonal(_)
                | Self::RegularSeasonalUndifferenced(_)
                | Self::SeasonalMa2
                | Self::RegularSeasonalMa2(_)
        )
    }
    pub(crate) fn seasonal_difference(self) -> bool {
        if let Self::Integrated {
            seasonal_difference,
            ..
        } = self
        {
            return seasonal_difference;
        }
        !matches!(
            self,
            Self::Nonseasonal(_) | Self::NonseasonalMa2(_) | Self::RegularSeasonalUndifferenced(_)
        )
    }
}

/// A fitted model: its layout and its coefficients in coordinate order,
/// with X-13's signs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmaInput<const N: usize = 3> {
    pub(crate) frequency: Frequency,
    pub(crate) parameters: [f64; N],
    pub(crate) layout: ArmaLayout,
}
impl<const N: usize> ArmaInput<N> {
    pub(crate) fn with_layout(
        frequency: Frequency,
        parameters: [f64; N],
        layout: ArmaLayout,
    ) -> Result<Self, Error> {
        if !layout.is_supported() || N != layout.coordinates() {
            return Err(Failure::Unusable {
                quantity: "model coordinate count",
                index: None,
                value: N as f64,
            }
            .at(Stage::Import));
        }
        for (index, value) in parameters.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(Failure::Unusable {
                    quantity: "fitted model coefficient",
                    index: Some(index),
                    value,
                }
                .at(Stage::Import));
            }
        }
        Ok(Self {
            frequency,
            parameters,
            layout,
        })
    }
}

/// A fitted model and its import.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ImportedArma<const N: usize = 3> {
    pub(crate) original: ArmaInput<N>,
    /// The coefficients as ANALTS holds them after NMLMDL and TRANS0.
    pub(crate) parameters: [f64; N],
}

/// The import of a fitted model: NMLMDL reverses each sign, TRANS0 moves a
/// coefficient group inside its bounds, and ANALTS reverses the sign back.
pub(crate) fn import_arma<const N: usize>(input: &ArmaInput<N>) -> Result<ImportedArma<N>, Error> {
    let mut parameters = input.parameters;
    match input.layout.regular_ar_group(&parameters) {
        RegularArGroup::Empty => {}
        RegularArGroup::Single(phi) => {
            parameters[0] = -trans0_degree_one(0.0 - phi, AR_BOUND);
        }
        RegularArGroup::Polynomial(regular) => {
            let ar = crate::seats::decomposition::root_allocation::import_regular_ar(regular)?;
            parameters[..ar.as_slice().len()].copy_from_slice(ar.as_slice());
        }
    }
    let p = input.layout.regular_ar_coordinates();
    if input.layout.has_seasonal_ar() {
        parameters[p] = -trans0_degree_one(0.0 - parameters[p], AR_BOUND);
    }
    let p = input.layout.ar_coordinates();
    let q = input.layout.regular_ma_count();
    if q == 2 {
        let imported = super::refit::import_ma2([parameters[p], parameters[p + 1]])?;
        parameters[p..p + q].copy_from_slice(&imported);
    }
    for value in &mut parameters[p + if q == 2 { q } else { 0 }..] {
        *value = -trans0_degree_one(0.0 - *value, MA_BOUND);
    }
    Ok(ImportedArma {
        original: input.clone(),
        parameters,
    })
}

/// ANALTS 1764–1777, labels 5015/5016. With INIT≠0 (the imported model, after
/// the pre-loop ChangeModel left it unchanged), a regular AR and MA of equal
/// nonzero order whose coefficients all agree within CEPS=1e-13, or else the
/// BP=BQ=1 seasonal pair, is a degenerate model. ANALTS then stops before any
/// decomposition, so SEATS performs no signal extraction: Build 62 exits zero
/// with no SEATS tables, and this is the `DegenerateModel` error. The engine
/// calls it only when neither pre-loop ChangeModel changed the model: a
/// change resets INIT and so skips the test. ANALTS compares the
/// sign-reversed Th and Phi; negation is exact, so X-13 signs compare alike.
pub(crate) fn degenerate_model<const N: usize>(imported: &ImportedArma<N>) -> Result<(), Error> {
    const CEPS: f64 = 1e-13;
    let layout = imported.original.layout;
    let parameters = &imported.parameters;
    let p = layout.regular_ar_coordinates();
    let q = layout.regular_ma_count();
    let ma_start = layout.ar_coordinates();
    let degenerate = |seasonal, ar: &[f64], ma: &[f64]| {
        let equal = ar
            .iter()
            .zip(ma)
            .all(|(&phi, &th)| (th - phi).abs() <= CEPS);
        if equal {
            Err(Failure::DegenerateModel {
                seasonal,
                ar: ar.to_vec(),
                ma: ma.to_vec(),
            }
            .at(Stage::Import))
        } else {
            Ok(())
        }
    };
    if q != 0 && p == q {
        degenerate(false, &parameters[..p], &parameters[ma_start..ma_start + q])?;
    }
    if layout.has_seasonal_ar() && layout.has_seasonal_ma() {
        degenerate(true, &parameters[p..p + 1], &parameters[N - 1..])?;
    }
    Ok(())
}

/// The residual degrees of freedom left after differencing, AR conditioning,
/// the free coordinates and the mean, or [`Failure::NoDegreesOfFreedom`] when
/// none are left. The series span is already within CheckLen's range.
pub(crate) fn arma_degrees_of_freedom<const N: usize>(
    frequency: Frequency,
    observed: usize,
    mean: crate::seats::mean::RequestedMean,
    layout: ArmaLayout,
) -> Result<usize, Error> {
    let s = usize::from(frequency.periods_per_year());
    let p = layout.regular_ar_coordinates() + usize::from(layout.has_seasonal_ar()) * s;
    let seasonal_difference = usize::from(layout.seasonal_difference()) * s;
    let ordinary_difference = layout.differences().ordinary();
    let (minimum, maximum) = if s == 12 { (36, 780) } else { (16, 328) };
    let consumed = seasonal_difference
        + ordinary_difference
        + p
        + N
        + usize::from(mean == crate::seats::mean::RequestedMean::Yes);
    debug_assert!((minimum..=maximum).contains(&observed));
    if observed <= consumed {
        return Err(Failure::NoDegreesOfFreedom { observed, consumed }.at(Stage::Import));
    }
    Ok(observed - consumed)
}

/// The roots and polynomials of a model with one regular AR and one regular
/// MA coefficient, from RPQ's root of the forward fit and backward CALCFX's
/// MA coefficients.
pub(crate) fn arma_geometry<const N: usize>(
    frequency: Frequency,
    forward_root: &crate::seats::ar1_seasonal_ma::model::Ar1RealRoot,
    refreshed: &super::residual::ArmaPolynomials<N>,
) -> Result<crate::seats::ar1_seasonal_ma::model::Ar1Geometry, Error> {
    crate::seats::ar1_seasonal_ma::model::degree_one_geometry_body(
        frequency,
        *forward_root,
        Some(refreshed.parameters[1]),
        if refreshed.layout.has_seasonal_ma() {
            Some(refreshed.parameters[2])
        } else {
            None
        },
        refreshed.layout.differences(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seats::mean::RequestedMean;

    #[test]
    fn an_order_above_the_limit_is_none() {
        for order in 0..=3 {
            assert_eq!(ArOrder::new(order).map(ArOrder::count), Some(order));
        }
        assert_eq!(ArOrder::new(4), None);
        for order in 0..=2 {
            assert_eq!(LowOrder::new(order).map(LowOrder::count), Some(order));
        }
        assert_eq!(LowOrder::new(3), None);
    }

    fn rejected<const N: usize>(layout: ArmaLayout) -> bool {
        ArmaInput::with_layout(Frequency::Monthly, [0.0; N], layout)
            == Err(Error {
                stage: Stage::Import,
                failure: Failure::Unusable {
                    quantity: "model coordinate count",
                    index: None,
                    value: N as f64,
                },
            })
    }

    #[test]
    fn input_takes_only_a_supported_layout_of_its_coordinate_count() {
        use ArmaLayout::*;
        // A count that is not the layout's.
        assert!(rejected::<3>(SeasonalAr));
        assert!(rejected::<2>(RegularSeasonal(ArOrder::One)));
        assert!(rejected::<4>(Nonseasonal(ArOrder::One)));
        // The layout's count, with an order no family has.
        assert!(rejected::<2>(RegularSeasonal(ArOrder::Zero)));
        assert!(rejected::<2>(RegularSeasonalUndifferenced(ArOrder::Zero)));
        assert!(rejected::<1>(Nonseasonal(ArOrder::Zero)));
        assert!(rejected::<3>(RegularSeasonalMa2(ArOrder::Zero)));
        assert!(rejected::<6>(RegularSeasonalMa2(ArOrder::Three)));
        assert!(rejected::<5>(NonseasonalMa2(ArOrder::Three)));
        assert!(rejected::<7>(Integrated {
            ar: ArOrder::Three,
            ma: LowOrder::Two,
            seasonal_ar: true,
            ordinary_difference: LowOrder::One,
            seasonal_difference: true,
            seasonal_ma: true,
        }));
        assert!(
            ArmaInput::with_layout(
                Frequency::Monthly,
                [0.0; 5],
                RegularSeasonal(ArOrder::Three)
            )
            .is_ok()
        );
    }

    #[test]
    fn regular_ar_group_follows_the_ar_order() {
        let parameters = [0.1, 0.2, 0.3, 0.4];
        let group = |layout: ArmaLayout| layout.regular_ar_group(&parameters);
        assert_eq!(group(ArmaLayout::SeasonalAr), RegularArGroup::Empty);
        assert_eq!(
            group(ArmaLayout::NonseasonalMa2(ArOrder::Zero)),
            RegularArGroup::Empty
        );
        assert_eq!(
            group(ArmaLayout::Nonseasonal(ArOrder::One)),
            RegularArGroup::Single(0.1)
        );
        assert_eq!(
            group(ArmaLayout::RegularSeasonal(ArOrder::Two)),
            RegularArGroup::Polynomial(RegularArCoefficients::Two([0.1, 0.2]))
        );
        assert_eq!(
            group(ArmaLayout::RegularSeasonal(ArOrder::Three)),
            RegularArGroup::Polynomial(RegularArCoefficients::Three([0.1, 0.2, 0.3]))
        );
    }

    #[test]
    fn only_an_integrated_layout_drops_its_seasonal_ma() {
        let layout = ArmaLayout::Integrated {
            ar: ArOrder::One,
            ma: LowOrder::Zero,
            seasonal_ar: true,
            ordinary_difference: LowOrder::Zero,
            seasonal_difference: false,
            seasonal_ma: true,
        };
        let reduced = layout.without_seasonal_ma().expect("integrated layout");
        assert!(!reduced.has_seasonal_ma());
        assert_eq!(reduced.coordinates(), 2);
        assert_eq!(
            ArmaLayout::RegularSeasonal(ArOrder::One).without_seasonal_ma(),
            None
        );
    }

    #[test]
    fn a_series_must_leave_a_residual_degree_of_freedom() {
        let layout = ArmaLayout::Integrated {
            ar: ArOrder::Three,
            ma: LowOrder::Zero,
            seasonal_ar: true,
            ordinary_difference: LowOrder::Two,
            seasonal_difference: true,
            seasonal_ma: false,
        };
        // Six differences, seven conditioning lags, four fitted parameters:
        // 16 observations underflow; 17 leave zero degrees of freedom.
        for (mean, consumed, invalid, first_valid) in [
            (RequestedMean::No, 17, [16, 17], 18),
            (RequestedMean::Yes, 18, [17, 18], 19),
        ] {
            for observed in invalid {
                assert_eq!(
                    arma_degrees_of_freedom::<4>(Frequency::Quarterly, observed, mean, layout),
                    Err(Failure::NoDegreesOfFreedom { observed, consumed }.at(Stage::Import))
                );
            }
            let degrees_of_freedom =
                arma_degrees_of_freedom::<4>(Frequency::Quarterly, first_valid, mean, layout)
                    .expect("one residual degree of freedom");
            assert_eq!(degrees_of_freedom, 1);
        }
    }

    #[test]
    fn a_nonfinite_coefficient_is_unusable_at_import() {
        assert_eq!(
            ArmaInput::with_layout(
                Frequency::Monthly,
                [0.1, f64::INFINITY, 0.2],
                ArmaLayout::RegularSeasonal(ArOrder::One)
            ),
            Err(Failure::Unusable {
                quantity: "fitted model coefficient",
                index: Some(1),
                value: f64::INFINITY,
            }
            .at(Stage::Import))
        );
    }

    fn imported<const N: usize>(layout: ArmaLayout, parameters: [f64; N]) -> ImportedArma<N> {
        ImportedArma {
            original: ArmaInput {
                frequency: Frequency::Monthly,
                parameters,
                layout,
            },
            parameters,
        }
    }

    #[test]
    fn equal_ar_and_ma_coefficients_are_a_degenerate_model() {
        let regular = ArmaLayout::RegularSeasonal(ArOrder::One);
        assert_eq!(
            degenerate_model(&imported(regular, [0.4, 0.4, -0.3])),
            Err(Failure::DegenerateModel {
                seasonal: false,
                ar: vec![0.4],
                ma: vec![0.4],
            }
            .at(Stage::Import))
        );
        assert_eq!(
            degenerate_model(&imported(regular, [0.4, 0.3, -0.3])),
            Ok(())
        );
        let seasonal = ArmaLayout::Integrated {
            ar: ArOrder::Zero,
            ma: LowOrder::One,
            seasonal_ar: true,
            ordinary_difference: LowOrder::One,
            seasonal_difference: false,
            seasonal_ma: true,
        };
        assert_eq!(
            degenerate_model(&imported(seasonal, [0.5, -0.2, 0.5])),
            Err(Failure::DegenerateModel {
                seasonal: true,
                ar: vec![0.5],
                ma: vec![0.5],
            }
            .at(Stage::Import))
        );
    }
}
