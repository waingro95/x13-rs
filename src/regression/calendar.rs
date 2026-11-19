//! Trading-day and length-of-period regressors (`td6var.f`, `tdset.f`,
//! `td7var.f`).

use super::model::{ExtensionRule, RegressionColumn, RegressorKind};
use crate::error::SpecProblem;
use crate::regarima::{
    CalendarBasis, CalendarFeature, FlowLengthTreatment, LengthEffect, TradingDayContrast,
    TradingDayVariable, Weekday,
};
use crate::time::TimeAxis;
use crate::{Error, Frequency, Transform};

/// `td6var.f`/`tdset.f` `fouryr`: the weekday shift of a year by its
/// position in the four-year leap cycle.
const FOUR_YEAR: [i32; 4] = [0, 2, 3, 4];
/// `td6var.f`/`tdset.f` `year/4-441`: four-year cycles counted from 1764.
const FOUR_YEAR_CYCLE_EPOCH: i32 = 441;
/// `td6var.f`/`tdset.f` `(year-1601)/100`: centuries counted from 1601, for
/// the Gregorian century leap-year correction.
const GREGORIAN_CENTURY_EPOCH: i32 = 1601;
const MONTH_LENGTHS: [[usize; 12]; 2] = [
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
];
const MONTH_FIRST_DAY_OFFSETS: [[usize; 12]; 2] = [
    [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5],
    [0, 3, 4, 0, 2, 5, 0, 3, 6, 1, 4, 6],
];
const MONTH_WEEKDAY_COUNTS: [[i32; 7]; 4] = [
    [4, 4, 4, 4, 4, 4, 4],
    [5, 4, 4, 4, 4, 4, 4],
    [5, 5, 4, 4, 4, 4, 4],
    [5, 5, 5, 4, 4, 4, 4],
];
const QUARTER_LENGTHS: [[usize; 4]; 2] = [[90, 91, 92, 92], [91, 91, 92, 92]];
const QUARTER_FIRST_DAY_OFFSETS: [[usize; 4]; 2] = [[0, 6, 6, 0], [0, 0, 0, 1]];
const QUARTER_WEEKDAY_COUNTS: [[i32; 7]; 3] = [
    [13, 13, 13, 13, 13, 13, 12],
    [13, 13, 13, 13, 13, 13, 13],
    [14, 13, 13, 13, 13, 13, 13],
];
const STOCK_ONE_COEFFICIENT_WEIGHTS: [f64; 4] = [-0.6, -0.2, 0.2, 0.6];
const WEEKDAYS: [Weekday; 6] = [
    Weekday::Monday,
    Weekday::Tuesday,
    Weekday::Wednesday,
    Weekday::Thursday,
    Weekday::Friday,
    Weekday::Saturday,
];
const AVERAGE_MONTH_LENGTH: f64 = 30.4375;
const AVERAGE_QUARTER_LENGTH: f64 = 91.3125;

/// The columns a calendar basis estimates and the fixed length factor it
/// applies instead of estimating, if any.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GeneratedCalendar {
    pub(crate) columns: Vec<RegressionColumn>,
    pub(crate) fixed_length: Vec<f64>,
    pub(crate) fixed_length_effect: Option<LengthEffect>,
}

/// Builds the trading-day and length columns of `basis` over `count`
/// periods of `axis`.
pub(crate) fn generate_calendar_basis_with_context(
    basis: CalendarBasis,
    transform: Transform,
    context: crate::api::FlowLengthContext,
    axis: TimeAxis,
    count: usize,
) -> Result<GeneratedCalendar, Error> {
    let mut columns = Vec::new();
    let mut fixed_length = Vec::new();
    let automatic_length = match context {
        crate::api::FlowLengthContext::LeapYear => LengthEffect::LeapYear,
        crate::api::FlowLengthContext::LengthOfPeriod => LengthEffect::LengthOfPeriod,
    };

    match basis {
        CalendarBasis::FlowTradingDay(flow) => {
            append_trading_day_columns(flow.contrast(), None, axis, count, &mut columns)?;
            match flow.length_treatment() {
                FlowLengthTreatment::LeapYear => append_length_treatment(
                    automatic_length,
                    transform,
                    axis,
                    count,
                    &mut columns,
                    &mut fixed_length,
                )?,
                FlowLengthTreatment::NoLengthEffect => {}
                FlowLengthTreatment::LengthOfPeriod => append_estimated_length_column(
                    LengthEffect::LengthOfPeriod,
                    axis,
                    count,
                    &mut columns,
                )?,
            }
        }
        CalendarBasis::StockTradingDay(stock) => {
            if axis.frequency() != Frequency::Monthly {
                return Err(Error::from(SpecProblem::UnsupportedCalendarFrequency {
                    feature: CalendarFeature::StockTradingDay,
                    frequency: axis.frequency(),
                }));
            }
            append_trading_day_columns(
                stock.contrast(),
                Some(stock.sample_day()),
                axis,
                count,
                &mut columns,
            )?;
        }
        CalendarBasis::Length(effect) => {
            append_estimated_length_column(effect, axis, count, &mut columns)?
        }
    }

    let fixed_length_effect = if fixed_length.is_empty() {
        None
    } else {
        Some(automatic_length)
    };

    debug_assert_eq!(columns.len(), calendar_basis_column_count(basis, transform));
    Ok(GeneratedCalendar {
        columns,
        fixed_length,
        fixed_length_effect,
    })
}

/// The number of columns `generate_calendar_basis_with_context` estimates
/// for `basis`, without evaluating them: a log leap-year treatment is a fixed
/// prior, not a column.
pub(crate) fn calendar_basis_column_count(basis: CalendarBasis, transform: Transform) -> usize {
    let trading_days = |contrast| match contrast {
        TradingDayContrast::SixCoefficient => WEEKDAYS.len(),
        TradingDayContrast::WeekdayWeekend => 1,
    };
    match basis {
        CalendarBasis::FlowTradingDay(flow) => {
            let length = match (flow.length_treatment(), transform) {
                (FlowLengthTreatment::LeapYear, Transform::Log)
                | (FlowLengthTreatment::NoLengthEffect, _) => 0,
                (FlowLengthTreatment::LeapYear, Transform::None)
                | (FlowLengthTreatment::LengthOfPeriod, _) => 1,
            };
            trading_days(flow.contrast()) + length
        }
        CalendarBasis::StockTradingDay(stock) => trading_days(stock.contrast()),
        CalendarBasis::Length(_) => 1,
    }
}

/// The fixed length-of-period factor over `count` periods of `axis`.
pub(crate) fn evaluate_fixed_length(
    effect: LengthEffect,
    axis: TimeAxis,
    count: usize,
) -> Result<Vec<f64>, Error> {
    Ok(axis
        .periods(count)?
        .map(|period| {
            length_value(
                effect,
                true,
                period.year(),
                period.subperiod(),
                axis.frequency(),
            )
        })
        .collect())
}

/// One trading-day regressor over `count` periods of `axis`.
pub(crate) fn evaluate_trading_day(
    variable: TradingDayVariable,
    axis: TimeAxis,
    count: usize,
) -> Result<Vec<f64>, Error> {
    axis.periods(count)?
        .map(|period| {
            trading_day_value(
                variable,
                period.year(),
                period.subperiod(),
                axis.frequency(),
            )
        })
        .collect()
}

/// One estimated length regressor over `count` periods of `axis`.
pub(crate) fn evaluate_length(
    effect: LengthEffect,
    axis: TimeAxis,
    count: usize,
) -> Result<Vec<f64>, Error> {
    Ok(axis
        .periods(count)?
        .map(|period| {
            length_value(
                effect,
                false,
                period.year(),
                period.subperiod(),
                axis.frequency(),
            )
        })
        .collect())
}

fn append_trading_day_columns(
    contrast: TradingDayContrast,
    stock_sample_day: Option<u8>,
    axis: TimeAxis,
    count: usize,
    columns: &mut Vec<RegressionColumn>,
) -> Result<(), Error> {
    let variables = match (contrast, stock_sample_day) {
        (TradingDayContrast::SixCoefficient, None) => WEEKDAYS
            .into_iter()
            .map(|weekday| TradingDayVariable::FlowSixCoefficient { weekday })
            .collect::<Vec<_>>(),
        (TradingDayContrast::WeekdayWeekend, None) => {
            vec![TradingDayVariable::FlowWeekdayWeekend]
        }
        (TradingDayContrast::SixCoefficient, Some(sample_day)) => WEEKDAYS
            .into_iter()
            .map(|weekday| TradingDayVariable::StockSixCoefficient {
                weekday,
                sample_day,
            })
            .collect::<Vec<_>>(),
        (TradingDayContrast::WeekdayWeekend, Some(sample_day)) => {
            vec![TradingDayVariable::StockWeekdayWeekend { sample_day }]
        }
    };

    for variable in variables {
        let observed = evaluate_trading_day(variable, axis, count)?;
        columns.push(RegressionColumn::new(
            RegressorKind::TradingDay(variable),
            observed,
            ExtensionRule::TradingDay(variable),
        ));
    }
    Ok(())
}

fn append_length_treatment(
    effect: LengthEffect,
    transform: Transform,
    axis: TimeAxis,
    count: usize,
    columns: &mut Vec<RegressionColumn>,
    fixed_length: &mut Vec<f64>,
) -> Result<(), Error> {
    match transform {
        Transform::None => append_estimated_length_column(effect, axis, count, columns),
        Transform::Log => {
            *fixed_length = evaluate_fixed_length(effect, axis, count)?;
            Ok(())
        }
    }
}

fn append_estimated_length_column(
    effect: LengthEffect,
    axis: TimeAxis,
    count: usize,
    columns: &mut Vec<RegressionColumn>,
) -> Result<(), Error> {
    let observed = evaluate_length(effect, axis, count)?;
    columns.push(RegressionColumn::new(
        RegressorKind::Length {
            effect,
            frequency: axis.frequency(),
        },
        observed,
        ExtensionRule::Length(effect),
    ));
    Ok(())
}

fn trading_day_value(
    variable: TradingDayVariable,
    year: i32,
    subperiod: u8,
    frequency: Frequency,
) -> Result<f64, Error> {
    if year < crate::api::MINIMUM_TRADING_DAY_YEAR {
        return Err(Error::CalendarYearOutOfRange {
            feature: match variable {
                TradingDayVariable::FlowSixCoefficient { .. }
                | TradingDayVariable::FlowWeekdayWeekend => CalendarFeature::TradingDay,
                TradingDayVariable::StockSixCoefficient { .. }
                | TradingDayVariable::StockWeekdayWeekend { .. } => {
                    CalendarFeature::StockTradingDay
                }
            },
            year,
            minimum: crate::api::MINIMUM_TRADING_DAY_YEAR,
            // Floor guard only: `api::spec::validate_year_range` rejects years
            // above 9999 (and reports that maximum) before this can run.
            maximum: None,
        });
    }
    let sample_day = match variable {
        TradingDayVariable::StockSixCoefficient { sample_day, .. }
        | TradingDayVariable::StockWeekdayWeekend { sample_day } => Some(sample_day),
        TradingDayVariable::FlowSixCoefficient { .. } | TradingDayVariable::FlowWeekdayWeekend => {
            None
        }
    };
    if sample_day.is_some() && frequency != Frequency::Monthly {
        return Err(Error::from(SpecProblem::UnsupportedCalendarFrequency {
            feature: CalendarFeature::StockTradingDay,
            frequency,
        }));
    }
    let counts = weekday_counts(year, subperiod, frequency, sample_day);
    Ok(match variable {
        TradingDayVariable::FlowSixCoefficient { weekday }
        | TradingDayVariable::StockSixCoefficient { weekday, .. } => {
            counts[weekday_index(weekday)] as f64 - counts[0] as f64
        }
        TradingDayVariable::FlowWeekdayWeekend => {
            let mut value = -2.5 * f64::from(counts[0] + counts[6]);
            for count in &counts[1..=5] {
                value += f64::from(*count);
            }
            value
        }
        TradingDayVariable::StockWeekdayWeekend { .. } => {
            let mut value = f64::from(counts[5] - counts[0]);
            for (weight, count) in STOCK_ONE_COEFFICIENT_WEIGHTS.iter().zip(&counts[1..=4]) {
                value += *weight * f64::from(*count - counts[0]);
            }
            value
        }
    })
}

fn weekday_counts(
    year: i32,
    subperiod: u8,
    frequency: Frequency,
    stock_sample_day: Option<u8>,
) -> [i32; 7] {
    let leap = usize::from(is_leap_year(year));
    let index = usize::from(subperiod - 1);
    let first_weekday = first_weekday_of_year(year);
    let (length, first_offset, base_counts) = match frequency {
        Frequency::Monthly => {
            let length = MONTH_LENGTHS[leap][index];
            (
                length,
                MONTH_FIRST_DAY_OFFSETS[leap][index],
                MONTH_WEEKDAY_COUNTS[length - 28],
            )
        }
        Frequency::Quarterly => {
            let length = QUARTER_LENGTHS[leap][index];
            (
                length,
                QUARTER_FIRST_DAY_OFFSETS[leap][index],
                QUARTER_WEEKDAY_COUNTS[length - 90],
            )
        }
    };
    let period_first_weekday = (first_weekday + first_offset) % 7;

    if let Some(sample_day) = stock_sample_day {
        let selected_day = usize::from(sample_day).min(length);
        let selected_weekday = (period_first_weekday + selected_day - 1) % 7;
        let mut counts = [0; 7];
        counts[selected_weekday] = 1;
        return counts;
    }

    let mut counts = [0; 7];
    for (offset, count) in base_counts.into_iter().enumerate() {
        counts[(offset + period_first_weekday) % 7] = count;
    }
    counts
}

fn length_value(
    effect: LengthEffect,
    ratio: bool,
    year: i32,
    subperiod: u8,
    frequency: Frequency,
) -> f64 {
    let leap = is_leap_year(year);
    match (effect, ratio, frequency) {
        (LengthEffect::LeapYear, false, Frequency::Monthly) => {
            if subperiod != 2 {
                0.0
            } else if leap {
                0.75
            } else {
                -0.25
            }
        }
        (LengthEffect::LeapYear, true, Frequency::Monthly) => {
            if subperiod != 2 {
                1.0
            } else if leap {
                29.0 / 28.25
            } else {
                28.0 / 28.25
            }
        }
        (LengthEffect::LeapYear, false, Frequency::Quarterly) => {
            if subperiod != 1 {
                0.0
            } else if leap {
                0.75
            } else {
                -0.25
            }
        }
        (LengthEffect::LeapYear, true, Frequency::Quarterly) => {
            if subperiod != 1 {
                1.0
            } else if leap {
                91.0 / 90.25
            } else {
                90.0 / 90.25
            }
        }
        (LengthEffect::LengthOfPeriod, false, Frequency::Monthly) => {
            period_length(year, subperiod, frequency) as f64 - AVERAGE_MONTH_LENGTH
        }
        (LengthEffect::LengthOfPeriod, true, Frequency::Monthly) => {
            period_length(year, subperiod, frequency) as f64 / AVERAGE_MONTH_LENGTH
        }
        (LengthEffect::LengthOfPeriod, false, Frequency::Quarterly) => {
            period_length(year, subperiod, frequency) as f64 - AVERAGE_QUARTER_LENGTH
        }
        (LengthEffect::LengthOfPeriod, true, Frequency::Quarterly) => {
            period_length(year, subperiod, frequency) as f64 / AVERAGE_QUARTER_LENGTH
        }
    }
}

fn period_length(year: i32, subperiod: u8, frequency: Frequency) -> usize {
    let leap = usize::from(is_leap_year(year));
    let index = usize::from(subperiod - 1);
    match frequency {
        Frequency::Monthly => MONTH_LENGTHS[leap][index],
        Frequency::Quarterly => QUARTER_LENGTHS[leap][index],
    }
}

/// `td6var.f`/`tdset.f` `sdoyr`: the weekday of January 1.
fn first_weekday_of_year(year: i32) -> usize {
    let mut weekday =
        5 * (year / 4 - FOUR_YEAR_CYCLE_EPOCH) + FOUR_YEAR[year.rem_euclid(4) as usize];
    let mut century_adjustment = (year - GREGORIAN_CENTURY_EPOCH) / 100;
    century_adjustment = century_adjustment - century_adjustment / 4 - 1;
    weekday -= century_adjustment;
    weekday.rem_euclid(7) as usize
}

pub(super) const fn is_leap_year(year: i32) -> bool {
    (year % 100 != 0 && year % 4 == 0) || year % 400 == 0
}

const fn weekday_index(weekday: Weekday) -> usize {
    match weekday {
        Weekday::Monday => 1,
        Weekday::Tuesday => 2,
        Weekday::Wednesday => 3,
        Weekday::Thursday => 4,
        Weekday::Friday => 5,
        Weekday::Saturday => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::{GeneratedCalendar, generate_calendar_basis_with_context};
    use crate::api::FlowLengthContext;
    use crate::regarima::{
        CalendarBasis, FlowLengthTreatment, FlowTradingDay, LengthEffect, RegressionVariable,
        TradingDayContrast, TradingDayVariable, Weekday,
    };
    use crate::regression::model::RegressionModel;
    use crate::time::TimeAxis;
    use crate::{Error, Transform};

    /// The basis under the default leap-year context.
    fn generate_calendar_basis(
        basis: CalendarBasis,
        transform: Transform,
        axis: TimeAxis,
        count: usize,
    ) -> Result<GeneratedCalendar, Error> {
        generate_calendar_basis_with_context(
            basis,
            transform,
            FlowLengthContext::LeapYear,
            axis,
            count,
        )
    }

    #[test]
    fn trend_only_length_context_changes_only_automatic_flow_length() {
        let flow = FlowTradingDay::new(TradingDayContrast::WeekdayWeekend);
        for (axis, expected, fixed) in [
            (
                TimeAxis::monthly(2000, 1).unwrap(),
                [0.5625, -1.4375, 0.5625],
                [31.0 / 30.4375, 29.0 / 30.4375, 31.0 / 30.4375],
            ),
            (
                TimeAxis::quarterly(2000, 1).unwrap(),
                [-0.3125, -0.3125, 0.6875],
                [91.0 / 91.3125, 91.0 / 91.3125, 92.0 / 91.3125],
            ),
        ] {
            let identity = generate_calendar_basis_with_context(
                CalendarBasis::FlowTradingDay(flow),
                Transform::None,
                FlowLengthContext::LengthOfPeriod,
                axis,
                3,
            )
            .unwrap();
            let mut model = RegressionModel::new();
            for column in identity.columns {
                model.push(column).unwrap();
            }
            let (kind, values) = model.columns().last().unwrap();
            assert_eq!(
                kind,
                crate::regression::model::RegressorKind::Length {
                    effect: LengthEffect::LengthOfPeriod,
                    frequency: axis.frequency()
                }
            );
            assert_eq!(values, expected);
            let log = generate_calendar_basis_with_context(
                CalendarBasis::FlowTradingDay(flow),
                Transform::Log,
                FlowLengthContext::LengthOfPeriod,
                axis,
                3,
            )
            .unwrap();
            assert_eq!(log.fixed_length, fixed);
            assert_eq!(log.fixed_length_effect, Some(LengthEffect::LengthOfPeriod));
            for basis in [
                CalendarBasis::Length(LengthEffect::LeapYear),
                CalendarBasis::Length(LengthEffect::LengthOfPeriod),
                CalendarBasis::FlowTradingDay(
                    flow.with_length_treatment(FlowLengthTreatment::NoLengthEffect),
                ),
                CalendarBasis::FlowTradingDay(
                    flow.with_length_treatment(FlowLengthTreatment::LengthOfPeriod),
                ),
            ] {
                for transform in [Transform::None, Transform::Log] {
                    assert_eq!(
                        generate_calendar_basis_with_context(
                            basis,
                            transform,
                            FlowLengthContext::LengthOfPeriod,
                            axis,
                            3
                        )
                        .unwrap(),
                        generate_calendar_basis(basis, transform, axis, 3).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn leap_and_length_values_match_build_62_arithmetic() {
        let leap =
            CalendarBasis::FlowTradingDay(FlowTradingDay::new(TradingDayContrast::WeekdayWeekend));
        let identity = generate_calendar_basis(
            leap,
            Transform::None,
            TimeAxis::monthly(1900, 1).unwrap(),
            1_260,
        )
        .unwrap();
        let mut model = RegressionModel::new();
        for column in identity.columns {
            model.push(column).unwrap();
        }
        let matrix = model.matrix(&vec![0.0; 1_260]).unwrap();
        let leap_column = matrix
            .values()
            .chunks_exact(matrix.column_count())
            .map(|row| row[1])
            .collect::<Vec<_>>();
        assert_eq!(leap_column[1], -0.25); // February 1900
        assert_eq!(leap_column[49], 0.75); // February 1904
        assert_eq!(leap_column[1_201], 0.75); // February 2000
        assert_eq!(leap_column[0], 0.0);

        let fixed = generate_calendar_basis(
            leap,
            Transform::Log,
            TimeAxis::monthly(1999, 12).unwrap(),
            15,
        )
        .unwrap();
        assert_eq!(fixed.fixed_length[2], 29.0 / 28.25);
        assert_eq!(fixed.fixed_length[14], 28.0 / 28.25);
        assert!(
            fixed
                .fixed_length
                .iter()
                .enumerate()
                .all(|(index, value)| index == 2 || index == 14 || *value == 1.0)
        );

        let monthly_length = generate_calendar_basis(
            CalendarBasis::Length(LengthEffect::LengthOfPeriod),
            Transform::None,
            TimeAxis::monthly(2000, 1).unwrap(),
            3,
        )
        .unwrap();
        let mut model = RegressionModel::new();
        for column in monthly_length.columns {
            model.push(column).unwrap();
        }
        assert_eq!(
            model.matrix(&[0.0; 3]).unwrap().values(),
            [
                31.0 - 30.4375,
                0.0,
                29.0 - 30.4375,
                0.0,
                31.0 - 30.4375,
                0.0,
            ]
        );
    }

    #[test]
    fn log_length_columns_follow_build_62_picktd_scope() {
        // Treating every log length variable as a fixed prior would make every
        // estimated case below lose its last column. regvar.f calls td7var with
        // Mltadd=F, so the values stay its centered estimated-regressor
        // values under log transforms.
        let fixed_leap = generate_calendar_basis(
            CalendarBasis::FlowTradingDay(FlowTradingDay::new(TradingDayContrast::WeekdayWeekend)),
            Transform::Log,
            TimeAxis::monthly(2000, 1).unwrap(),
            3,
        )
        .unwrap();
        assert_eq!(fixed_leap.columns.len(), 1);
        assert_eq!(fixed_leap.fixed_length, [1.0, 29.0 / 28.25, 1.0]);

        type LogLengthCase = (
            CalendarBasis,
            Transform,
            TimeAxis,
            usize,
            usize,
            &'static [f64],
            &'static [f64],
        );
        let cases: &[LogLengthCase] = &[
            (
                CalendarBasis::FlowTradingDay(
                    FlowTradingDay::new(TradingDayContrast::WeekdayWeekend)
                        .with_length_treatment(FlowLengthTreatment::LengthOfPeriod),
                ),
                Transform::Log,
                TimeAxis::monthly(2000, 1).unwrap(),
                3,
                2,
                &[],
                &[31.0 - 30.4375, 29.0 - 30.4375, 31.0 - 30.4375],
            ),
            (
                CalendarBasis::Length(LengthEffect::LeapYear),
                Transform::Log,
                TimeAxis::quarterly(2000, 1).unwrap(),
                5,
                1,
                &[],
                &[0.75, 0.0, 0.0, 0.0, -0.25],
            ),
            (
                CalendarBasis::Length(LengthEffect::LengthOfPeriod),
                Transform::Log,
                TimeAxis::quarterly(2000, 1).unwrap(),
                4,
                1,
                &[],
                &[
                    91.0 - 91.3125,
                    91.0 - 91.3125,
                    92.0 - 91.3125,
                    92.0 - 91.3125,
                ],
            ),
        ];

        for (basis, transform, axis, count, column_count, expected_fixed, expected_length) in cases
        {
            let generated = generate_calendar_basis(*basis, *transform, *axis, *count).unwrap();
            assert_eq!(generated.fixed_length, *expected_fixed);
            assert_eq!(generated.columns.len(), *column_count);

            let mut model = RegressionModel::new();
            for column in generated.columns {
                model.push(column).unwrap();
            }
            let matrix = model.matrix(&vec![0.0; *count]).unwrap();
            let actual = matrix
                .values()
                .chunks_exact(matrix.column_count())
                .map(|row| row[row.len() - 2])
                .collect::<Vec<_>>();
            assert_eq!(actual, *expected_length);
            assert_eq!(
                model.evaluate(*axis, *count).unwrap().last().unwrap(),
                *expected_length
            );
        }
    }

    #[test]
    fn six_coefficient_metadata_is_monday_through_saturday() {
        let generated = generate_calendar_basis(
            CalendarBasis::FlowTradingDay(
                FlowTradingDay::new(TradingDayContrast::SixCoefficient)
                    .with_length_treatment(FlowLengthTreatment::NoLengthEffect),
            ),
            Transform::None,
            TimeAxis::monthly(2000, 1).unwrap(),
            1,
        )
        .unwrap();
        let mut model = RegressionModel::new();
        for column in generated.columns {
            model.push(column).unwrap();
        }
        assert_eq!(
            model
                .kinds()
                .map(|kind| kind.variable())
                .collect::<Vec<_>>(),
            [
                Weekday::Monday,
                Weekday::Tuesday,
                Weekday::Wednesday,
                Weekday::Thursday,
                Weekday::Friday,
                Weekday::Saturday,
            ]
            .map(|weekday| {
                RegressionVariable::TradingDay(TradingDayVariable::FlowSixCoefficient { weekday })
            })
        );
    }
}
