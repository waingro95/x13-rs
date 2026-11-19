//! Easter, Labor Day and Thanksgiving regressors (`adestr.f`, `estrmu.f`,
//! `adlabr.f`, `adthnk.f`).

use super::calendar::is_leap_year;
use super::model::{ExtensionRule, RegressionColumn, RegressorKind};
use crate::easter_dates::EASTER_DATE_OFFSETS;
use crate::error::SpecProblem;
use crate::regarima::{CalendarFeature, CalendarSpec, HolidayVariable};
use crate::time::TimeAxis;
use crate::{Error, Frequency};

const CUMULATIVE_MONTH_DAYS: [[i32; 13]; 2] = [
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
    [0, 31, 60, 91, 121, 152, 182, 214, 244, 274, 305, 335, 366],
];
const CUMULATIVE_QUARTER_DAYS: [[i32; 5]; 2] = [[0, 90, 181, 273, 365], [0, 91, 182, 274, 366]];

// `estrmu.f`: long-run means for February, March, and April.
const EASTER_MEANS: [[f64; 26]; 3] = [
    [
        0.00368,
        0.002083333,
        0.001130435,
        0.0002727273,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.6576, 0.6450833, 0.6311304, 0.6162727, 0.5999048, 0.583, 0.5661053, 0.549, 0.5318824,
        0.514625, 0.4973333, 0.4807143, 0.4643077, 0.4476667, 0.4305455, 0.4136, 0.3975556, 0.382,
        0.3654286, 0.3483333, 0.3304, 0.3125, 0.2966667, 0.281, 0.266, 0.232,
    ],
    [
        0.33872, 0.3528333, 0.3677391, 0.3834545, 0.4000952, 0.417, 0.4338947, 0.451, 0.4681176,
        0.485375, 0.5026667, 0.5192857, 0.5356923, 0.5523333, 0.5694545, 0.5864, 0.6024444, 0.618,
        0.6345714, 0.6516667, 0.6696, 0.6875, 0.7033333, 0.719, 0.734, 0.768,
    ],
];

// `adlabr.f`: offsets from August 31 for years 1901 through 2100.
const LABOR_DAY_DATE_OFFSETS: [i32; 200] = [
    2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5,
    4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7,
    6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2,
    1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4,
    3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6,
    5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1,
    7, 6, 5, 3, 2, 1, 7, 5,
];
const LABOR_DAY_MEANS: [[f64; 25]; 2] = [
    [
        0.8800, 0.8750, 0.8696, 0.8636, 0.8571, 0.8500, 0.8421, 0.8333, 0.8235, 0.8125, 0.8000,
        0.7857, 0.7692, 0.7500, 0.7273, 0.7000, 0.6667, 0.6250, 0.5714, 0.5000, 0.4286, 0.3571,
        0.2857, 0.2143, 0.1429,
    ],
    [
        0.1200, 0.1250, 0.1304, 0.1364, 0.1429, 0.1500, 0.1579, 0.1667, 0.1765, 0.1875, 0.2000,
        0.2143, 0.2308, 0.2500, 0.2727, 0.3000, 0.3333, 0.3750, 0.4286, 0.5000, 0.5714, 0.6429,
        0.7143, 0.7857, 0.8571,
    ],
];

// `adthnk.f`: offsets from November 21 for years 1901 through 2100.
const THANKSGIVING_DATE_OFFSETS: [i32; 200] = [
    7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3,
    2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5,
    4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7,
    6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2,
    1, 7, 6, 4, 3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4,
    3, 2, 1, 6, 5, 4, 3, 1, 7, 6, 5, 3, 2, 1, 7, 5, 4, 3, 2, 7, 6, 5, 4, 2, 1, 7, 6, 4, 3, 2, 1, 6,
    5, 4, 3, 1, 7, 6, 5, 3,
];
const THANKSGIVING_MEANS: [[f64; 25]; 2] = [
    [
        0.4884, 0.4773, 0.4656, 0.4534, 0.4406, 0.4273, 0.4132, 0.3985, 0.3830, 0.3667, 0.3494,
        0.3313, 0.3120, 0.2917, 0.2700, 0.2471, 0.2226, 0.1684, 0.1384, 0.1062, 0.0776, 0.0530,
        0.0326, 0.0167, 0.0057,
    ],
    [
        0.5116, 0.5227, 0.5344, 0.5466, 0.5594, 0.5727, 0.5868, 0.6015, 0.6170, 0.6333, 0.6506,
        0.6687, 0.6880, 0.7083, 0.7300, 0.7529, 0.7774, 0.8316, 0.8616, 0.8938, 0.9224, 0.9470,
        0.9674, 0.9833, 0.9943,
    ],
];

fn requested_holidays(calendar: CalendarSpec) -> impl Iterator<Item = HolidayVariable> {
    [
        calendar.easter().map(HolidayVariable::Easter),
        calendar.labor_day().map(HolidayVariable::LaborDay),
        calendar.thanksgiving().map(HolidayVariable::Thanksgiving),
    ]
    .into_iter()
    .flatten()
}

/// The number of columns `generate_holidays` adds, without evaluating them.
pub(crate) fn holiday_column_count(calendar: CalendarSpec) -> usize {
    requested_holidays(calendar).count()
}

/// A column for each holiday `calendar` requests, over `count` periods of
/// `axis`.
pub(crate) fn generate_holidays(
    calendar: CalendarSpec,
    axis: TimeAxis,
    count: usize,
) -> Result<Vec<RegressionColumn>, Error> {
    let mut columns = Vec::new();
    for holiday in requested_holidays(calendar) {
        let observed = evaluate_holiday(holiday, axis, count)?;
        columns.push(RegressionColumn::new(
            RegressorKind::Holiday(holiday),
            observed,
            ExtensionRule::Holiday(holiday),
        ));
    }
    Ok(columns)
}

/// One holiday regressor over `count` periods of `axis`. Labor Day and
/// Thanksgiving are monthly only.
pub(crate) fn evaluate_holiday(
    holiday: HolidayVariable,
    axis: TimeAxis,
    count: usize,
) -> Result<Vec<f64>, Error> {
    if axis.frequency() != Frequency::Monthly
        && matches!(
            holiday,
            HolidayVariable::LaborDay(_) | HolidayVariable::Thanksgiving(_)
        )
    {
        return Err(Error::from(SpecProblem::UnsupportedCalendarFrequency {
            feature: holiday_feature(holiday),
            frequency: axis.frequency(),
        }));
    }
    axis.periods(count)?
        .map(|period| holiday_value(holiday, period.year(), period.subperiod(), axis.frequency()))
        .collect()
}

fn holiday_value(
    holiday: HolidayVariable,
    year: i32,
    subperiod: u8,
    frequency: Frequency,
) -> Result<f64, Error> {
    let (minimum, maximum) = match holiday {
        HolidayVariable::Easter(_) | HolidayVariable::LaborDay(_) => (1901, 2100),
        HolidayVariable::Thanksgiving(_) => (1939, 2100),
    };
    if year < minimum || year > maximum {
        return Err(Error::CalendarYearOutOfRange {
            feature: holiday_feature(holiday),
            year,
            minimum,
            maximum: Some(maximum),
        });
    }
    Ok(match holiday {
        HolidayVariable::Easter(effect) => {
            easter_value(year, subperiod, frequency, effect.window())
        }
        HolidayVariable::LaborDay(effect) => labor_day_value(year, subperiod, effect.window()),
        HolidayVariable::Thanksgiving(effect) => {
            thanksgiving_value(year, subperiod, effect.offset())
        }
    })
}

/// `trnaic.f` forces `Elong=F` only for its log candidate. `estrmu.f`
/// pads the observed+forecast span to full years and averages at most 29 years.
pub(crate) fn candidate_easter_values(
    effect: crate::regarima::Easter,
    axis: TimeAxis,
    observed_count: usize,
    forecast_count: u16,
) -> Result<Vec<f64>, Error> {
    let full_count = observed_count
        .checked_add(usize::from(forecast_count))
        .ok_or(Error::internal("Easter regressor span length overflowed"))?;
    if full_count == 0 {
        return Ok(Vec::new());
    }
    let last = axis.period_at(full_count - 1)?;
    for year in [axis.start().year(), last.year()] {
        if !(1901..=2100).contains(&year) {
            return Err(Error::CalendarYearOutOfRange {
                feature: CalendarFeature::Easter,
                year,
                minimum: 1901,
                maximum: Some(2100),
            });
        }
    }
    let years = (last.year() - axis.start().year() + 1).min(29);
    let frequency = axis.frequency();
    let mut means = [0.0; 12];
    for subperiod in 1..=frequency.periods_per_year() {
        let relevant = match frequency {
            Frequency::Monthly => (2..=4).contains(&subperiod),
            Frequency::Quarterly => subperiod <= 2,
        };
        if relevant {
            let mut sum = 0.0;
            for year in axis.start().year()..axis.start().year() + years {
                sum += raw_easter_value(year, subperiod, frequency, effect.window());
            }
            means[usize::from(subperiod - 1)] = sum / f64::from(years);
        }
    }
    axis.periods(observed_count)?
        .map(|period| {
            Ok(raw_easter_value(
                period.year(),
                period.subperiod(),
                frequency,
                effect.window(),
            ) - means[usize::from(period.subperiod() - 1)])
        })
        .collect()
}

fn easter_value(year: i32, subperiod: u8, frequency: Frequency, window: u8) -> f64 {
    raw_easter_value(year, subperiod, frequency, window) - easter_mean(window, subperiod, frequency)
}

fn raw_easter_value(year: i32, subperiod: u8, frequency: Frequency, window: u8) -> f64 {
    let leap = usize::from(is_leap_year(year));
    let period = usize::from(subperiod - 1);
    let offset = i32::from(EASTER_DATE_OFFSETS[date_offset_index(year)]);
    let ((period_begin, period_end), easter) = match frequency {
        Frequency::Monthly => (
            day_span(&CUMULATIVE_MONTH_DAYS[leap], period),
            CUMULATIVE_MONTH_DAYS[leap][2] + 22 + offset,
        ),
        Frequency::Quarterly => (
            day_span(&CUMULATIVE_QUARTER_DAYS[leap], period),
            CUMULATIVE_QUARTER_DAYS[leap][1] - 9 + offset,
        ),
    };
    let window = i32::from(window);
    let (window_begin, window_end) = if window > 0 {
        (easter - window, easter - 1)
    } else {
        (easter, easter)
    };
    let overlap_begin = period_begin.max(window_begin);
    let overlap_end = period_end.min(window_end);
    let mut value = if overlap_begin <= overlap_end {
        f64::from(overlap_end - overlap_begin + 1)
    } else {
        0.0
    };
    if window > 0 {
        value /= f64::from(window);
    }
    value
}

fn easter_mean(window: u8, subperiod: u8, frequency: Frequency) -> f64 {
    let index = window_mean_index(window);
    match (frequency, subperiod) {
        (Frequency::Monthly, 2) => EASTER_MEANS[0][index],
        (Frequency::Monthly, 3) => EASTER_MEANS[1][index],
        (Frequency::Monthly, 4) => EASTER_MEANS[2][index],
        (Frequency::Quarterly, 1) => EASTER_MEANS[0][index] + EASTER_MEANS[1][index],
        (Frequency::Quarterly, 2) => EASTER_MEANS[2][index],
        _ => 0.0,
    }
}

fn labor_day_value(year: i32, month: u8, window: u8) -> f64 {
    let leap = usize::from(is_leap_year(year));
    let month = usize::from(month - 1);
    let (period_begin, period_end) = day_span(&CUMULATIVE_MONTH_DAYS[leap], month);
    let labor_day =
        CUMULATIVE_MONTH_DAYS[leap][8] + LABOR_DAY_DATE_OFFSETS[date_offset_index(year)];
    let overlap_begin = period_begin.max(labor_day - i32::from(window));
    let overlap_end = period_end.min(labor_day - 1);
    let mut value = if overlap_begin <= overlap_end {
        f64::from(overlap_end - overlap_begin + 1) / f64::from(window)
    } else {
        0.0
    };
    let mean_index = window_mean_index(window);
    if month == 7 {
        value -= LABOR_DAY_MEANS[0][mean_index];
    } else if month == 8 {
        value -= LABOR_DAY_MEANS[1][mean_index];
    }
    value
}

fn thanksgiving_value(year: i32, month: u8, offset: i8) -> f64 {
    let leap = usize::from(is_leap_year(year));
    let month = usize::from(month - 1);
    let (period_begin, period_end) = day_span(&CUMULATIVE_MONTH_DAYS[leap], month);
    let thanksgiving =
        CUMULATIVE_MONTH_DAYS[leap][10] + 21 + THANKSGIVING_DATE_OFFSETS[date_offset_index(year)];
    let christmas = CUMULATIVE_MONTH_DAYS[leap][11] + 25;
    let overlap_begin = period_begin.max(thanksgiving - i32::from(offset));
    let overlap_end = period_end.min(christmas - 1);
    let window = christmas - thanksgiving + i32::from(offset);
    let mut value = if overlap_begin <= overlap_end {
        f64::from(overlap_end - overlap_begin + 1) / f64::from(window)
    } else {
        0.0
    };
    let mean_index = thanksgiving_mean_index(offset);
    if month == 10 {
        value -= THANKSGIVING_MEANS[0][mean_index];
    } else if month == 11 {
        value -= THANKSGIVING_MEANS[1][mean_index];
    }
    value
}

/// The first and last day of year of the 0-based `period`, from a
/// cumulative day table.
fn day_span(cumulative_days: &[i32], period: usize) -> (i32, i32) {
    (cumulative_days[period] + 1, cumulative_days[period + 1])
}

/// The date-offset tables' index: they start at 1901.
fn date_offset_index(year: i32) -> usize {
    (year - 1901) as usize
}

/// The Easter and Labor Day mean tables' column: windows run from 25 days
/// down.
fn window_mean_index(window: u8) -> usize {
    25 - usize::from(window)
}

/// `THANKSGIVING_MEANS`' column: offsets run from 17 down to 1, then from -1
/// down to -8. There is no offset 0.
fn thanksgiving_mean_index(offset: i8) -> usize {
    let offset = i32::from(offset);
    (if offset < 0 { 16 - offset } else { 17 - offset }) as usize
}

const fn holiday_feature(holiday: HolidayVariable) -> CalendarFeature {
    match holiday {
        HolidayVariable::Easter(_) => CalendarFeature::Easter,
        HolidayVariable::LaborDay(_) => CalendarFeature::LaborDay,
        HolidayVariable::Thanksgiving(_) => CalendarFeature::Thanksgiving,
    }
}

#[cfg(test)]
mod tests {
    use super::generate_holidays;

    use crate::regarima::{
        CalendarBasis, CalendarSpec, Easter, FlowLengthTreatment, FlowTradingDay, LaborDay,
        Thanksgiving, TradingDayContrast,
    };
    use crate::regression::model::RegressionModel;
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, Error, RegArimaSpec};

    #[test]
    fn candidate_easter_means_pad_partial_years_and_include_forecasts() {
        let axis = TimeAxis::monthly(2020, 7).unwrap();
        let easter = Easter::new(8).unwrap();
        // Easter 2020: April 12 (all eight days in April).
        // Easter 2021: April 4 (five days in March, three in April).
        let observed = super::candidate_easter_values(easter, axis, 9, 0).unwrap();
        assert_eq!(&observed[..8], &[0.0; 8]);
        assert_eq!(observed[8], 5.0 / 16.0);
        // A year of forecasts includes 2022 (Easter April 17), reducing the
        // March mean from (0 + 5/8)/2 to (0 + 5/8 + 0)/3.
        let extended = super::candidate_easter_values(easter, axis, 9, 12).unwrap();
        assert_eq!(extended[8], 5.0 / 8.0 - (5.0 / 8.0) / 3.0);
    }

    #[test]
    fn extension_years_are_validated_before_returning_columns() {
        let calendar = CalendarSpec::new().with_easter(Easter::new(8).unwrap());
        assert_eq!(
            generate_holidays(calendar, TimeAxis::monthly(2100, 12).unwrap(), 2),
            Err(Error::CalendarYearOutOfRange {
                feature: crate::regarima::CalendarFeature::Easter,
                year: 2101,
                minimum: 1901,
                maximum: Some(2100),
            })
        );
    }

    #[test]
    fn easter_columns_subtract_the_long_run_means() {
        let calendar = CalendarSpec::new().with_easter(Easter::new(8).unwrap());
        let columns = generate_holidays(calendar, TimeAxis::monthly(1999, 3).unwrap(), 2).unwrap();
        let mut model = RegressionModel::new();
        model.push(columns.into_iter().next().unwrap()).unwrap();
        assert_eq!(
            model.matrix(&[0.0; 2]).unwrap().values(),
            [5.0 / 8.0 - 0.382, 0.0, 3.0 / 8.0 - 0.618, 0.0]
        );

        let easter_day = generate_holidays(
            CalendarSpec::new().with_easter(Easter::new(0).unwrap()),
            TimeAxis::monthly(2000, 3).unwrap(),
            2,
        )
        .unwrap();
        let mut model = RegressionModel::new();
        model.push(easter_day.into_iter().next().unwrap()).unwrap();
        assert_eq!(
            model.matrix(&[0.0; 2]).unwrap().values(),
            [-0.232, 0.0, 1.0 - 0.768, 0.0]
        );
    }

    #[test]
    fn regression_model_orders_constant_basis_then_holidays() {
        let calendar = CalendarSpec::new()
            .with_thanksgiving(Thanksgiving::new(1).unwrap())
            .with_basis(CalendarBasis::FlowTradingDay(
                FlowTradingDay::new(TradingDayContrast::WeekdayWeekend)
                    .with_length_treatment(FlowLengthTreatment::NoLengthEffect),
            ))
            .with_easter(Easter::new(8).unwrap())
            .with_labor_day(LaborDay::new(8).unwrap());
        let none = ArimaOrder::new(0, 0, 0).unwrap();
        let spec = RegArimaSpec::new(ArimaModel::new(none, none).unwrap())
            .with_constant(true)
            .with_calendar(calendar);
        let model = RegressionModel::from_spec(
            spec.resolve_explicit().unwrap(),
            TimeAxis::monthly(2000, 1).unwrap(),
            24,
        )
        .unwrap();

        assert_eq!(
            model.kinds().map(|kind| kind.name()).collect::<Vec<_>>(),
            [
                "const",
                "Weekday",
                "Easter[8]",
                "Labor[8]",
                "Thanksgiving[1]",
            ]
        );
    }
}
