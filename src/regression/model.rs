//! The regression columns of a regARIMA model, how each is rebuilt over a
//! longer span, and Build 62's rules for where a new column goes (`adrgef.f`,
//! `addotl.f`, `tdaic.f`, `easaic.f`).

use crate::api::ResolvedRegArimaSpec;
use crate::arima::model::ExpandedModel;
use crate::numerics::polynomial::rational_expand;
use crate::regarima::{
    HolidayVariable, LengthEffect, OutlierType, RegressionVariable, ResolvedArimaModel,
    TradingDayVariable,
};
use crate::regression::design::RegressionMatrix;
use crate::time::TimeAxis;
use crate::{Error, Frequency, Period};

/// Build 62's `PB`: the most regression variables a model may hold.
const MAX_REGRESSION_VARIABLES: usize = 80;

/// The regression columns of a model in Build 62's order, with the fixed
/// length-of-period factor and the outliers and regressors the request
/// supplied.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionModel {
    columns: Vec<RegressionColumn>,
    fixed_length: Vec<f64>,
    fixed_length_effect: Option<LengthEffect>,
    manual_outliers: Vec<crate::regarima::ManualOutlier>,
    user_regressors: Vec<crate::regarima::UserRegressor>,
}

impl RegressionModel {
    /// A model with no columns.
    pub(crate) const fn new() -> Self {
        Self {
            columns: Vec::new(),
            fixed_length: Vec::new(),
            fixed_length_effect: None,
            manual_outliers: Vec::new(),
            user_regressors: Vec::new(),
        }
    }

    /// The constant, calendar and holiday columns `spec` asks for, over
    /// `observed_count` periods of `axis`.
    pub(crate) fn from_spec(
        spec: ResolvedRegArimaSpec,
        axis: TimeAxis,
        observed_count: usize,
    ) -> Result<Self, Error> {
        let mut regression = if spec.includes_constant() {
            Self::with_constant(spec.model(), axis.frequency(), observed_count)?
        } else {
            Self::new()
        };
        if let Some(calendar) = spec.calendar() {
            if let Some(basis) = calendar.basis() {
                let generated = super::calendar::generate_calendar_basis_with_context(
                    basis,
                    spec.transform(),
                    spec.flow_length_context(),
                    axis,
                    observed_count,
                )?;
                regression.fixed_length = generated.fixed_length;
                regression.fixed_length_effect = generated.fixed_length_effect;
                for column in generated.columns {
                    regression.push(column)?;
                }
            }
            for column in super::holidays::generate_holidays(calendar, axis, observed_count)? {
                regression.push(column)?;
            }
        }
        debug_assert_eq!(regression.columns.len(), spec_column_count(spec));
        Ok(regression)
    }

    /// adrgef's PB capacity stop for the regressors Build 62 registers while
    /// it reads the input: gtinpt.f the spec's constant, calendar and holiday
    /// regressors, then each group in `groups` (manual outliers, user
    /// regressors, then editor.f's missing-value regressors). Registration
    /// only counts: regvar evaluates the columns later, inside arima, which
    /// x11ari.f reaches after its legacy-Easter prepass (xrgdrv) and its Same
    /// bypass. So this evaluates nothing, and reports the same `actual` as the
    /// group-wise checks `from_spec` and the `add_*` methods make later.
    pub(crate) fn ensure_registration_capacity(
        spec: ResolvedRegArimaSpec,
        groups: impl IntoIterator<Item = usize>,
    ) -> Result<(), Error> {
        let mut registered = spec_column_count(spec);
        for group in groups {
            registered = capacity_after(registered, group)?;
        }
        Ok(())
    }

    /// A model holding only the constant column of `model`.
    pub(crate) fn with_constant(
        model: impl Into<ResolvedArimaModel>,
        frequency: Frequency,
        observed_count: usize,
    ) -> Result<Self, Error> {
        let model = model.into();
        let extension = ExtensionRule::Constant { model, frequency };
        let observed = constant_values(model, frequency, observed_count);
        let mut regression = Self::new();
        regression.push(RegressionColumn::new(
            RegressorKind::Constant,
            observed,
            extension,
        ))?;
        Ok(regression)
    }

    /// tdaic.f deletes the old TD/length group and appends its replacement
    /// after the surviving columns, including missing-value regressors.
    pub(crate) fn replace_aic_trading_day(
        &mut self,
        basis: Option<crate::regarima::CalendarBasis>,
        spec: ResolvedRegArimaSpec,
        axis: TimeAxis,
        count: usize,
    ) -> Result<(), Error> {
        self.columns.retain(|column| {
            !matches!(
                column.kind,
                RegressorKind::TradingDay(_) | RegressorKind::Length { .. }
            )
        });
        self.fixed_length.clear();
        self.fixed_length_effect = None;
        if let Some(basis) = basis {
            let mut generated = super::calendar::generate_calendar_basis_with_context(
                basis,
                spec.transform(),
                spec.flow_length_context(),
                axis,
                count,
            )?;
            // tdaic adds the estimated leap-year column before addtd, unlike
            // the initial user-specified basis order in adpdrg.
            generated
                .columns
                .sort_by_key(|column| !matches!(column.kind, RegressorKind::Length { .. }));
            self.fixed_length = generated.fixed_length;
            self.fixed_length_effect = generated.fixed_length_effect;
            for column in generated.columns {
                self.push(column)?;
            }
        }
        Ok(())
    }

    /// easaic.f deletes and re-appends Easter without reordering other groups.
    pub(crate) fn replace_aic_easter(
        &mut self,
        easter: Option<crate::regarima::Easter>,
        axis: TimeAxis,
        count: usize,
    ) -> Result<(), Error> {
        self.columns.retain(|column| {
            !matches!(
                column.kind,
                RegressorKind::Holiday(HolidayVariable::Easter(_))
            )
        });
        if let Some(easter) = easter {
            for column in super::holidays::generate_holidays(
                crate::regarima::CalendarSpec::new().with_easter(easter),
                axis,
                count,
            )? {
                self.push(column)?;
            }
        }
        Ok(())
    }

    /// `getreg.f`/`adrgef.f` append supplied groups after built-in/specified
    /// groups. Repeated types join their first group, retaining supplied names.
    pub(crate) fn add_user_regressors(
        &mut self,
        regressors: &[crate::regarima::UserRegressor],
        axis: TimeAxis,
        count: usize,
    ) -> Result<(), Error> {
        self.ensure_additional_capacity(regressors.len())?;
        let mut columns = Vec::with_capacity(regressors.len());
        for (first, regressor) in regressors.iter().enumerate() {
            if regressors[..first]
                .iter()
                .any(|prior| prior.kind() == regressor.kind())
            {
                continue;
            }
            for (index, member) in regressors.iter().enumerate() {
                if member.kind() == regressor.kind() {
                    columns.push(RegressionColumn::new(
                        RegressorKind::User {
                            index: self.user_regressors.len() + index,
                            kind: member.kind(),
                        },
                        member.values_over(axis, count)?.to_vec(),
                        ExtensionRule::User(member.clone()),
                    ));
                }
            }
        }
        self.columns.extend(columns);
        self.user_regressors.extend_from_slice(regressors);
        Ok(())
    }

    /// The user regressors in the order they were supplied, whatever order
    /// their columns take.
    pub(crate) fn user_regressors(&self) -> &[crate::regarima::UserRegressor] {
        &self.user_regressors
    }

    /// automd.f's `rmfix(...,2)`/`addfix(...,2)` round trip: addfix re-adds the
    /// non-user groups through adrgef in their saved order, then addusr appends
    /// the supplied groups, so user columns move behind e.g. missing values.
    pub(crate) fn move_user_regressors_last(&mut self) {
        let (user, mut columns): (Vec<_>, Vec<_>) = std::mem::take(&mut self.columns)
            .into_iter()
            .partition(|column| matches!(column.kind, RegressorKind::User { .. }));
        columns.extend(user);
        self.columns = columns;
    }

    /// Adds an additive-outlier column for each missing observation index
    /// (`editor.f`).
    pub(crate) fn add_missing(
        &mut self,
        missing: &[usize],
        axis: TimeAxis,
        count: usize,
    ) -> Result<(), Error> {
        self.ensure_additional_capacity(missing.len())?;
        for &index in missing {
            let period = axis.period_at(index)?;
            let mut observed = vec![0.0; count];
            observed[index] = 1.0;
            // editor/adrgef place the first Missing Value group among supplied
            // outlier groups by date (MV sorts after AO/LS/TC on equal dates).
            // Further MV columns append to that group, not between outliers.
            let target = self
                .columns
                .iter()
                .rposition(|column| matches!(column.kind, RegressorKind::MissingValue { .. }))
                .map(|last| last + 1)
                .unwrap_or_else(|| {
                    self.columns
                        .iter()
                        .position(|column| match column.kind {
                            RegressorKind::Outlier {
                                period: existing, ..
                            } => period < existing,
                            _ => false,
                        })
                        .unwrap_or(self.columns.len())
                });
            self.columns.insert(
                target,
                RegressionColumn::new(
                    RegressorKind::MissingValue {
                        period,
                        frequency: axis.frequency(),
                    },
                    observed,
                    ExtensionRule::Outlier {
                        kind: OutlierType::Additive,
                        period,
                        temporary_change_rate: 0.0,
                    },
                ),
            );
        }
        Ok(())
    }

    /// The outliers the request supplied, in the order they were added.
    pub(crate) fn manual_outliers(&self) -> &[crate::regarima::ManualOutlier] {
        &self.manual_outliers
    }

    /// Validates `outliers` against the series and appends a column for each.
    pub(crate) fn add_manual_outliers(
        &mut self,
        outliers: &[crate::regarima::ManualOutlier],
        axis: TimeAxis,
        count: usize,
        temporary_change_rate: f64,
    ) -> Result<(), Error> {
        self.ensure_additional_capacity(outliers.len())?;
        crate::regarima::ManualOutlier::validate_list(outliers, axis, count)?;
        self.append_manual_outliers(outliers, axis, count, temporary_change_rate)
    }

    /// Appends outliers already validated against the original series and
    /// pruned to the estimation span by `editor.f`'s rules. An LS at the
    /// estimation span's final date is valid even though an LS at the series
    /// end is not. `temporary_change_rate` is the request's resolved rate:
    /// `addotl.f` builds every TC column, specified or automatic, from the one
    /// global `Tcalfa`.
    pub(crate) fn add_validated_manual_outliers(
        &mut self,
        outliers: &[crate::regarima::ManualOutlier],
        axis: TimeAxis,
        count: usize,
        temporary_change_rate: f64,
    ) -> Result<(), Error> {
        self.ensure_additional_capacity(outliers.len())?;
        self.append_manual_outliers(outliers, axis, count, temporary_change_rate)
    }

    /// Appends a column for each outlier. The caller has checked capacity for
    /// all of them.
    fn append_manual_outliers(
        &mut self,
        outliers: &[crate::regarima::ManualOutlier],
        axis: TimeAxis,
        count: usize,
        temporary_change_rate: f64,
    ) -> Result<(), Error> {
        for &outlier in outliers {
            outlier.check_frequency(axis)?;
            let kind = outlier.kind();
            let period = outlier.period();
            if self.manual_outliers.contains(&outlier) {
                return Err(Error::InvalidManualOutlier { kind, period });
            }
            let frequency = axis.frequency();
            // adpdrg passes Userin=T to adrgef: preserve the supplied order,
            // unlike automatic outlier insertion, which sorts by date/type.
            self.columns.push(RegressionColumn::new(
                RegressorKind::Outlier {
                    kind,
                    period,
                    frequency,
                },
                evaluate_outlier(kind, period, temporary_change_rate, axis, count)?,
                ExtensionRule::Outlier {
                    kind,
                    period,
                    temporary_change_rate,
                },
            ));
            self.manual_outliers.push(outlier);
        }
        Ok(())
    }

    /// Adds, replaces or removes the constant column, leaving every other
    /// column and the fixed length factor where they are.
    pub(crate) fn set_constant(
        &mut self,
        include: bool,
        model: impl Into<ResolvedArimaModel>,
        frequency: Frequency,
        observed_count: usize,
    ) -> Result<(), Error> {
        let model = model.into();
        let index = self
            .columns
            .iter()
            .position(|column| column.kind == RegressorKind::Constant);
        match (include, index) {
            (false, Some(index)) => {
                self.columns.remove(index);
            }
            (false, None) => {}
            (true, existing) => {
                let column = RegressionColumn::new(
                    RegressorKind::Constant,
                    constant_values(model, frequency, observed_count),
                    ExtensionRule::Constant { model, frequency },
                );
                if let Some(index) = existing {
                    self.columns[index] = column;
                } else {
                    self.push(column)?;
                }
            }
        }
        Ok(())
    }

    /// The model for one candidate of the automatic transformation test: the
    /// log candidate's Easter column uses `trnaic.f`'s means.
    pub(crate) fn for_transform_candidate(
        spec: ResolvedRegArimaSpec,
        axis: TimeAxis,
        observed_count: usize,
    ) -> Result<Self, Error> {
        let mut model = Self::from_spec(spec, axis, observed_count)?;
        if spec.transform() == crate::Transform::Log {
            for column in &mut model.columns {
                if let RegressorKind::Holiday(HolidayVariable::Easter(effect)) = column.kind {
                    column.observed = super::holidays::candidate_easter_values(
                        effect,
                        axis,
                        observed_count,
                        spec.extensions().forecast_periods(axis.frequency()),
                    )?;
                }
            }
        }
        // These columns are only for candidate fitting; final fits and
        // extensions rebuild the ordinary long-run means.
        Ok(model)
    }

    /// Appends `column`, or fails at the capacity limit.
    pub(crate) fn push(&mut self, column: RegressionColumn) -> Result<(), Error> {
        self.ensure_additional_capacity(1)?;
        self.columns.push(column);
        Ok(())
    }

    /// Fails if `additional` more columns would pass the capacity limit.
    pub(crate) fn ensure_additional_capacity(&self, additional: usize) -> Result<(), Error> {
        capacity_after(self.columns.len(), additional).map(drop)
    }

    /// Inserts an automatically identified outlier among the outlier columns,
    /// ordered by date and then by Build 62's type index.
    pub(crate) fn insert_outlier(
        &mut self,
        kind: OutlierType,
        period: Period,
        frequency: Frequency,
        values: Vec<f64>,
        temporary_change_rate: f64,
    ) -> Result<(), Error> {
        self.ensure_additional_capacity(1)?;
        if self.columns.iter().any(|column| {
            matches!(
                column.kind,
                RegressorKind::Outlier {
                    kind: existing_kind,
                    period: existing_period,
                    ..
                } if existing_kind == kind && existing_period == period
            )
        }) {
            return Err(Error::internal(
                "outlier is already in the regression model",
            ));
        }
        let target = self
            .columns
            .iter()
            .position(|column| match column.kind {
                RegressorKind::Outlier {
                    kind: existing_kind,
                    period: existing_period,
                    ..
                } => {
                    (period, kind.build_62_index())
                        < (existing_period, existing_kind.build_62_index())
                }
                _ => false,
            })
            .unwrap_or_else(|| {
                self.columns
                    .iter()
                    .rposition(|column| matches!(column.kind, RegressorKind::Outlier { .. }))
                    .map_or(self.columns.len(), |last_outlier| last_outlier + 1)
            });
        self.columns.insert(
            target,
            RegressionColumn::new(
                RegressorKind::Outlier {
                    kind,
                    period,
                    frequency,
                },
                values,
                ExtensionRule::Outlier {
                    kind,
                    period,
                    temporary_change_rate,
                },
            ),
        );
        Ok(())
    }

    /// Removes the outlier column of this type and date; `false` if there is
    /// none.
    pub(crate) fn remove_outlier(&mut self, kind: OutlierType, period: Period) -> bool {
        let Some(index) = self.columns.iter().position(|column| {
            matches!(
                column.kind,
                RegressorKind::Outlier {
                    kind: existing_kind,
                    period: existing_period,
                    ..
                } if existing_kind == kind && existing_period == period
            )
        }) else {
            return false;
        };
        self.columns.remove(index);
        true
    }

    /// The `[X:y]` matrix of the columns and `response`.
    pub(crate) fn matrix(&self, response: &[f64]) -> Result<RegressionMatrix, Error> {
        let columns = self
            .columns
            .iter()
            .map(|column| column.observed.as_slice())
            .collect::<Vec<_>>();
        RegressionMatrix::from_columns(&columns, response)
    }

    /// Each column's values over `count` periods of `axis`.
    pub(crate) fn evaluate(&self, axis: TimeAxis, count: usize) -> Result<Vec<Vec<f64>>, Error> {
        self.columns
            .iter()
            .map(|column| column.extension.evaluate(axis, count))
            .collect()
    }

    /// A copy of the model with every column and the fixed length factor
    /// rebuilt over `count` periods of `axis`.
    pub(crate) fn evaluated_over(&self, axis: TimeAxis, count: usize) -> Result<Self, Error> {
        let evaluated = self.evaluate(axis, count)?;
        let mut model = self.clone();
        for (column, values) in model.columns.iter_mut().zip(evaluated) {
            column.observed = values;
        }
        model.fixed_length = match model.fixed_length_effect {
            Some(effect) => super::calendar::evaluate_fixed_length(effect, axis, count)?,
            None => Vec::new(),
        };
        Ok(model)
    }

    /// The fixed length factor over `count` periods of `axis`: all ones when
    /// the model has none.
    pub(crate) fn evaluate_fixed_length(
        &self,
        axis: TimeAxis,
        count: usize,
    ) -> Result<Vec<f64>, Error> {
        match self.fixed_length_effect {
            Some(effect) => super::calendar::evaluate_fixed_length(effect, axis, count),
            None => Ok(vec![1.0; count]),
        }
    }

    /// The fixed length factor over the model's span; empty when the model
    /// has none.
    pub(crate) fn fixed_length(&self) -> &[f64] {
        &self.fixed_length
    }

    /// Each column's kind and values, in column order.
    pub(crate) fn columns(&self) -> impl ExactSizeIterator<Item = (RegressorKind, &[f64])> + '_ {
        self.columns
            .iter()
            .map(|column| (column.kind, column.observed.as_slice()))
    }

    /// The decay rate a temporary-change column was built with; `None` for
    /// any other column. A column whose event is the last observed date does
    /// not show its rate, so the model keeps it.
    pub(crate) fn temporary_change_rate(&self, column_index: usize) -> Option<f64> {
        match &self.columns.get(column_index)?.extension {
            ExtensionRule::Outlier {
                kind: OutlierType::TemporaryChange,
                temporary_change_rate,
                ..
            } => Some(*temporary_change_rate),
            _ => None,
        }
    }

    /// The number of periods the columns span.
    pub(crate) fn value_count(&self) -> usize {
        self.columns
            .first()
            .map_or(self.fixed_length.len(), |column| column.observed.len())
    }

    pub(crate) fn kinds(&self) -> impl ExactSizeIterator<Item = RegressorKind> + '_ {
        self.columns.iter().map(|column| column.kind)
    }

    pub(crate) const fn column_count(&self) -> usize {
        self.columns.len()
    }

    #[cfg(test)]
    fn column(&self, index: usize) -> &RegressionColumn {
        &self.columns[index]
    }
}

/// One regression column: what it is, its values, and how to rebuild them
/// over another span.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionColumn {
    kind: RegressorKind,
    observed: Vec<f64>,
    extension: ExtensionRule,
}

impl RegressionColumn {
    pub(crate) const fn new(
        kind: RegressorKind,
        observed: Vec<f64>,
        extension: ExtensionRule,
    ) -> Self {
        Self {
            kind,
            observed,
            extension,
        }
    }

    #[cfg(test)]
    fn observed(&self) -> &[f64] {
        &self.observed
    }
}

#[cfg(test)]
impl RegressionColumn {
    /// A column of `variable` holding values a test supplies directly, with
    /// the variable's own extension rule.
    pub(crate) fn with_values(variable: RegressionVariable, observed: Vec<f64>) -> Self {
        let (kind, extension) = match variable {
            RegressionVariable::Constant => {
                let order = crate::ArimaOrder::new(0, 0, 0).unwrap();
                let model = crate::ArimaModel::new(order, order).unwrap().into();
                let frequency = Frequency::Monthly;
                (
                    RegressorKind::Constant,
                    ExtensionRule::Constant { model, frequency },
                )
            }
            RegressionVariable::TradingDay(variable) => (
                RegressorKind::TradingDay(variable),
                ExtensionRule::TradingDay(variable),
            ),
            RegressionVariable::Length(effect) => (
                RegressorKind::Length {
                    effect,
                    frequency: Frequency::Monthly,
                },
                ExtensionRule::Length(effect),
            ),
            RegressionVariable::Holiday(holiday) => (
                RegressorKind::Holiday(holiday),
                ExtensionRule::Holiday(holiday),
            ),
            RegressionVariable::Outlier { kind, period } => (
                RegressorKind::Outlier {
                    kind,
                    period,
                    frequency: period.frequency(),
                },
                ExtensionRule::Outlier {
                    kind,
                    period,
                    temporary_change_rate: 0.7,
                },
            ),
            RegressionVariable::User { .. } | RegressionVariable::MissingValue { .. } => {
                panic!("add these columns with add_user_regressors or add_missing")
            }
        };
        Self::new(kind, observed, extension)
    }
}

/// How to rebuild a column over a span that includes forecasts or
/// backcasts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExtensionRule {
    User(crate::regarima::UserRegressor),
    Constant {
        model: ResolvedArimaModel,
        frequency: Frequency,
    },
    TradingDay(TradingDayVariable),
    Length(LengthEffect),
    Holiday(HolidayVariable),
    Outlier {
        kind: OutlierType,
        period: Period,
        temporary_change_rate: f64,
    },
}

impl ExtensionRule {
    fn evaluate(&self, axis: TimeAxis, count: usize) -> Result<Vec<f64>, Error> {
        match self {
            Self::User(regressor) => Ok(regressor.values_over(axis, count)?.to_vec()),
            Self::Constant { model, frequency } => Ok(constant_values(*model, *frequency, count)),
            Self::TradingDay(variable) => {
                super::calendar::evaluate_trading_day(*variable, axis, count)
            }
            Self::Length(effect) => super::calendar::evaluate_length(*effect, axis, count),
            Self::Holiday(holiday) => super::holidays::evaluate_holiday(*holiday, axis, count),
            Self::Outlier {
                kind,
                period,
                temporary_change_rate,
            } => evaluate_outlier(*kind, *period, *temporary_change_rate, axis, count),
        }
    }
}

/// The values of an outlier regressor over `count` periods of `axis`
/// (`addotl.f`). A temporary change decays by `temporary_change_rate` each
/// period.
pub(crate) fn evaluate_outlier(
    kind: OutlierType,
    event: Period,
    temporary_change_rate: f64,
    axis: TimeAxis,
    count: usize,
) -> Result<Vec<f64>, Error> {
    if axis.frequency() != event.frequency() {
        return Err(Error::internal(
            "outlier period frequency differs from the series",
        ));
    }
    let periods_per_year = i64::from(event.frequency().periods_per_year());
    let ordinal = |period: Period| {
        i64::from(period.year()) * periods_per_year + i64::from(period.subperiod() - 1)
    };
    let event_ordinal = ordinal(event);
    let start_distance = ordinal(axis.start()) - event_ordinal;
    let mut temporary_change = 1.0;
    let mut temporary_change_started = false;
    if kind == OutlierType::TemporaryChange && start_distance > 0 {
        for _ in 0..start_distance {
            temporary_change *= temporary_change_rate;
        }
        temporary_change_started = true;
    }

    Ok(axis
        .periods(count)?
        .map(|period| {
            let distance = ordinal(period) - event_ordinal;
            match kind {
                OutlierType::Additive => f64::from(distance == 0),
                OutlierType::LevelShift => {
                    if distance < 0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                OutlierType::TemporaryChange if distance < 0 => 0.0,
                OutlierType::TemporaryChange => {
                    if !temporary_change_started {
                        temporary_change = 1.0;
                        temporary_change_started = true;
                    }
                    let value = temporary_change;
                    temporary_change *= temporary_change_rate;
                    value
                }
            }
        })
        .collect())
}

/// What a column is, with the frequency its name needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegressorKind {
    Constant,
    User {
        index: usize,
        kind: crate::regarima::UserRegressorType,
    },
    TradingDay(TradingDayVariable),
    Length {
        effect: LengthEffect,
        frequency: Frequency,
    },
    Holiday(HolidayVariable),
    MissingValue {
        period: Period,
        frequency: Frequency,
    },
    Outlier {
        kind: OutlierType,
        period: Period,
        frequency: Frequency,
    },
}

impl RegressorKind {
    /// The public identity of the column.
    pub(crate) const fn variable(self) -> RegressionVariable {
        match self {
            Self::Constant => RegressionVariable::Constant,
            Self::User { index, kind } => RegressionVariable::User { index, kind },
            Self::TradingDay(variable) => RegressionVariable::TradingDay(variable),
            Self::Length { effect, .. } => RegressionVariable::Length(effect),
            Self::Holiday(variable) => RegressionVariable::Holiday(variable),
            Self::MissingValue { period, .. } => RegressionVariable::MissingValue { period },
            Self::Outlier { kind, period, .. } => RegressionVariable::Outlier { kind, period },
        }
    }

    /// The column's name in results.
    pub(crate) fn name(self) -> String {
        let frequency = match self {
            Self::Constant | Self::TradingDay(_) | Self::Holiday(_) | Self::User { .. } => {
                Frequency::Monthly
            }
            Self::Length { frequency, .. } => frequency,
            Self::Outlier { frequency, .. } => frequency,
            Self::MissingValue { frequency, .. } => frequency,
        };
        self.variable().stable_name(frequency)
    }
}

/// The columns `RegressionModel::from_spec` builds for `spec`, counted
/// without evaluating any of them.
fn spec_column_count(spec: ResolvedRegArimaSpec) -> usize {
    let calendar = spec.calendar().map_or(0, |calendar| {
        calendar.basis().map_or(0, |basis| {
            super::calendar::calendar_basis_column_count(basis, spec.transform())
        }) + super::holidays::holiday_column_count(calendar)
    });
    usize::from(spec.includes_constant()) + calendar
}

/// The column count after adding `additional` to `registered`, or
/// `TooManyRegressionVariables` past the PB limit.
fn capacity_after(registered: usize, additional: usize) -> Result<usize, Error> {
    let actual = registered + additional;
    if actual > MAX_REGRESSION_VARIABLES {
        return Err(Error::TooManyRegressionVariables {
            actual,
            maximum: MAX_REGRESSION_VARIABLES,
        });
    }
    Ok(actual)
}

/// The constant regressor: ones summed through the model's differencing
/// operators.
fn constant_values(model: ResolvedArimaModel, frequency: Frequency, count: usize) -> Vec<f64> {
    let model = ExpandedModel::from_spec(model, frequency);
    let operators = crate::arima::polynomials(model.differencing_operators(), &[]);
    rational_expand(&vec![1.0; count], count, &operators)
}

#[cfg(test)]
mod tests {
    use super::{RegressionColumn, RegressionModel, RegressorKind};
    use crate::regarima::{LengthEffect, OutlierType, RegressionVariable};
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, Error};

    #[test]
    fn validated_manual_outliers_still_reject_another_frequency() {
        // Callers run `validate_list` first; the insertion keeps its own
        // check so no internal list can slip in a date of another frequency.
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let march = crate::Period::monthly(2001, 3).unwrap();
        let mut model = RegressionModel::new();
        assert_eq!(
            model.add_validated_manual_outliers(
                &[crate::regarima::ManualOutlier::new(
                    OutlierType::Additive,
                    march
                )],
                axis,
                20,
                0.343,
            ),
            Err(Error::PeriodFrequencyMismatch {
                period: march,
                expected: crate::Frequency::Quarterly,
            })
        );
        assert_eq!(model.column_count(), 0);
    }

    #[test]
    fn user_groups_follow_first_kind_and_retain_supplied_indices() {
        use crate::Series;
        use crate::regarima::{UserRegressor, UserRegressorType};

        let start = TimeAxis::quarterly(2000, 4).unwrap();
        let types = [
            UserRegressorType::TradingDay,
            UserRegressorType::Holiday,
            UserRegressorType::TradingDay,
            UserRegressorType::User,
        ];
        let supplied = types
            .into_iter()
            .enumerate()
            .map(|(index, kind)| {
                let base = (index + 1) as f64;
                let column = [base, 2.0 * base, 3.0 * base, 4.0 * base];
                UserRegressor::new(kind, Series::new(start.start(), &column)).unwrap()
            })
            .collect::<Vec<_>>();
        let order = ArimaOrder::new(0, 0, 0).unwrap();
        let mut model = RegressionModel::with_constant(
            ArimaModel::new(order, order).unwrap(),
            start.frequency(),
            2,
        )
        .unwrap();
        model
            .add_user_regressors(&supplied, start.shift(1).unwrap(), 2)
            .unwrap();
        assert_eq!(model.user_regressors(), supplied);
        assert_eq!(
            model
                .kinds()
                .map(RegressorKind::variable)
                .collect::<Vec<_>>(),
            [
                RegressionVariable::Constant,
                RegressionVariable::User {
                    index: 0,
                    kind: types[0]
                },
                RegressionVariable::User {
                    index: 2,
                    kind: types[2]
                },
                RegressionVariable::User {
                    index: 1,
                    kind: types[1]
                },
                RegressionVariable::User {
                    index: 3,
                    kind: types[3]
                },
            ],
        );
        assert_eq!(
            model.kinds().map(RegressorKind::name).collect::<Vec<_>>(),
            ["const", "user1", "user3", "user2", "user4"],
        );
        assert_eq!(
            model.matrix(&[10.0, 20.0]).unwrap().values(),
            &[
                1.0, 2.0, 6.0, 4.0, 8.0, 10.0, 1.0, 3.0, 9.0, 6.0, 12.0, 20.0
            ],
        );
        let extended = model.evaluate(start, 4).unwrap();
        for (column, supplied_index) in [(1, 0), (2, 2), (3, 1), (4, 3)] {
            assert_eq!(extended[column], supplied[supplied_index].values());
        }
        assert!(matches!(
            model.evaluate(start, 5),
            Err(Error::UserRegressorCoverage { .. })
        ));
    }

    #[test]
    fn user_groups_move_behind_missing_values_and_the_constant() {
        use crate::Series;
        use crate::regarima::{UserRegressor, UserRegressorType};

        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let supplied = [UserRegressorType::TradingDay, UserRegressorType::Holiday]
            .map(|kind| UserRegressor::new(kind, Series::new(axis.start(), &[1.0; 4])).unwrap());
        let order = ArimaOrder::new(0, 1, 1).unwrap();
        let arima = ArimaModel::new(order, order).unwrap();
        let mut model = RegressionModel::new();
        model.add_user_regressors(&supplied, axis, 4).unwrap();
        model.add_missing(&[2], axis, 4).unwrap();
        model
            .set_constant(true, arima, axis.frequency(), 4)
            .unwrap();
        assert_eq!(
            model.kinds().map(RegressorKind::name).collect::<Vec<_>>(),
            ["user1", "user2", "MV2000.Mar", "const"],
        );
        model.move_user_regressors_last();
        assert_eq!(
            model.kinds().map(RegressorKind::name).collect::<Vec<_>>(),
            ["MV2000.Mar", "const", "user1", "user2"],
        );
        assert_eq!(model.user_regressors(), supplied);
    }

    #[test]
    fn moved_user_groups_stay_grouped_by_kind() {
        use crate::Series;
        use crate::regarima::{UserRegressor, UserRegressorType};

        // addusr's adrgef joins a repeated type to its existing group, as
        // add_user_regressors already did: TD, Holiday, TD -> [u1, u3, u2].
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let supplied = [
            UserRegressorType::TradingDay,
            UserRegressorType::Holiday,
            UserRegressorType::TradingDay,
        ]
        .map(|kind| UserRegressor::new(kind, Series::new(axis.start(), &[1.0; 4])).unwrap());
        let mut model = RegressionModel::new();
        model.add_user_regressors(&supplied, axis, 4).unwrap();
        model.add_missing(&[1], axis, 4).unwrap();
        model.move_user_regressors_last();
        assert_eq!(
            model.kinds().map(RegressorKind::name).collect::<Vec<_>>(),
            ["MV2000.Feb", "user1", "user3", "user2"],
        );
        assert_eq!(model.user_regressors(), supplied);
    }

    #[test]
    fn no_columns_and_ordered_columns_keep_the_response_last() {
        let response = [10.0, 20.0];
        assert_eq!(
            RegressionModel::new().matrix(&response).unwrap().values(),
            response
        );

        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Length(LengthEffect::LeapYear),
                vec![1.0, 2.0],
            ))
            .unwrap();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Length(LengthEffect::LengthOfPeriod),
                vec![3.0, 4.0],
            ))
            .unwrap();

        let matrix = model.matrix(&response).unwrap();
        assert_eq!(matrix.row_count(), 2);
        assert_eq!(matrix.column_count(), 3);
        assert_eq!(matrix.values(), [1.0, 3.0, 10.0, 2.0, 4.0, 20.0]);
        assert_eq!(
            model
                .kinds()
                .map(RegressorKind::variable)
                .collect::<Vec<_>>(),
            [
                RegressionVariable::Length(LengthEffect::LeapYear),
                RegressionVariable::Length(LengthEffect::LengthOfPeriod),
            ]
        );
    }

    #[test]
    fn constant_is_recomputed_over_an_extension_span() {
        let arima = ArimaModel::new(
            ArimaOrder::new(0, 1, 0).unwrap(),
            ArimaOrder::new(0, 1, 0).unwrap(),
        )
        .unwrap();
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let model = RegressionModel::with_constant(arima, axis.frequency(), 24).unwrap();

        assert_eq!(model.matrix(&[0.0; 24]).unwrap().column_count(), 2);
        let extended = model.evaluate(axis, 36).unwrap();
        assert_eq!(extended.len(), 1);
        assert_eq!(extended[0].len(), 36);
        assert_eq!(&extended[0][..24], model.column(0).observed());
    }

    #[test]
    fn set_constant_appends_replaces_and_removes_only_the_constant() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let airline = ArimaModel::new(
            ArimaOrder::new(0, 1, 1).unwrap(),
            ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        let stationary = ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 1).unwrap(),
        )
        .unwrap();
        let mut model = RegressionModel::new();
        model.fixed_length = vec![2.0, 3.0];
        model.fixed_length_effect = Some(LengthEffect::LeapYear);
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Length(LengthEffect::LeapYear),
                vec![4.0, 5.0],
            ))
            .unwrap();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Outlier {
                    kind: OutlierType::Additive,
                    period: axis.start(),
                },
                vec![1.0, 0.0],
            ))
            .unwrap();
        let original = model.clone();
        model
            .set_constant(true, airline, axis.frequency(), 2)
            .unwrap();
        assert_eq!(
            model.kinds().collect::<Vec<_>>(),
            [
                original.column(0).kind,
                original.column(1).kind,
                RegressorKind::Constant
            ]
        );
        let appended = model.clone();
        model
            .set_constant(true, airline, axis.frequency(), 2)
            .unwrap();
        assert_eq!(model, appended);
        model
            .set_constant(true, stationary, axis.frequency(), 3)
            .unwrap();
        assert_eq!(model.column(2).observed(), &[1.0, 1.0, 1.0]);
        assert_eq!(
            model.column(2).extension.evaluate(axis, 4).unwrap(),
            vec![1.0; 4]
        );
        model
            .set_constant(false, stationary, axis.frequency(), 3)
            .unwrap();
        assert_eq!(model, original);
        model
            .set_constant(false, stationary, axis.frequency(), 3)
            .unwrap();
        assert_eq!(model, original);
    }

    #[test]
    fn set_constant_capacity_applies_only_to_append() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let airline = ArimaModel::new(
            ArimaOrder::new(0, 1, 1).unwrap(),
            ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        let mut model = RegressionModel::new();
        for _ in 0..80 {
            model
                .push(RegressionColumn::with_values(
                    RegressionVariable::Length(LengthEffect::LeapYear),
                    vec![1.0],
                ))
                .unwrap();
        }
        assert_eq!(
            model.set_constant(true, airline, axis.frequency(), 1),
            Err(Error::TooManyRegressionVariables {
                actual: 81,
                maximum: 80
            })
        );
        assert_eq!(model.column_count(), 80);
        model.columns[79].kind = RegressorKind::Constant;
        model
            .set_constant(true, airline, axis.frequency(), 1)
            .unwrap();
        assert_eq!(model.column_count(), 80);
        model
            .set_constant(false, airline, axis.frequency(), 1)
            .unwrap();
        assert_eq!(model.column_count(), 79);
    }

    #[test]
    fn eighty_columns_are_allowed_and_the_eighty_first_is_rejected() {
        let mut model = RegressionModel::new();
        for _ in 0..80 {
            model
                .push(RegressionColumn::with_values(
                    RegressionVariable::Constant,
                    vec![1.0],
                ))
                .unwrap();
        }

        assert_eq!(
            model.push(RegressionColumn::with_values(
                RegressionVariable::Constant,
                vec![1.0]
            )),
            Err(Error::TooManyRegressionVariables {
                actual: 81,
                maximum: 80,
            })
        );
        assert_eq!(model.column_count(), 80);
    }

    #[test]
    fn capacity_check_counts_every_requested_column() {
        let mut model = RegressionModel::new();
        for _ in 0..79 {
            model
                .push(RegressionColumn::with_values(
                    RegressionVariable::Constant,
                    vec![1.0],
                ))
                .unwrap();
        }

        assert_eq!(model.ensure_additional_capacity(1), Ok(()));
        assert_eq!(
            model.ensure_additional_capacity(2),
            Err(Error::TooManyRegressionVariables {
                actual: 81,
                maximum: 80,
            })
        );
    }

    #[test]
    fn automatic_outliers_are_inserted_by_date_then_type() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let period_2 = axis.period_at(1).unwrap();
        let period_5 = axis.period_at(4).unwrap();
        let mut model = RegressionModel::new();
        for (kind, period) in [
            (OutlierType::TemporaryChange, period_5),
            (OutlierType::LevelShift, period_2),
            (OutlierType::Additive, period_5),
            (OutlierType::Additive, period_2),
            (OutlierType::LevelShift, period_5),
        ] {
            model
                .insert_outlier(kind, period, axis.frequency(), vec![0.0; 12], 0.7)
                .unwrap();
        }

        assert_eq!(
            model.kinds().collect::<Vec<_>>(),
            [
                RegressorKind::Outlier {
                    kind: OutlierType::Additive,
                    period: period_2,
                    frequency: axis.frequency(),
                },
                RegressorKind::Outlier {
                    kind: OutlierType::LevelShift,
                    period: period_2,
                    frequency: axis.frequency(),
                },
                RegressorKind::Outlier {
                    kind: OutlierType::Additive,
                    period: period_5,
                    frequency: axis.frequency(),
                },
                RegressorKind::Outlier {
                    kind: OutlierType::LevelShift,
                    period: period_5,
                    frequency: axis.frequency(),
                },
                RegressorKind::Outlier {
                    kind: OutlierType::TemporaryChange,
                    period: period_5,
                    frequency: axis.frequency(),
                },
            ]
        );
        assert!(model.remove_outlier(OutlierType::Additive, period_5));
        assert!(!model.remove_outlier(OutlierType::Additive, period_5));
    }

    #[test]
    fn later_outlier_stays_before_a_constant_after_the_existing_group() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let first = axis.period_at(2).unwrap();
        let later = axis.period_at(4).unwrap();
        let mut model = RegressionModel::new();
        let first_values = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        model
            .insert_outlier(
                OutlierType::Additive,
                first,
                axis.frequency(),
                first_values.clone(),
                0.7,
            )
            .unwrap();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Constant,
                vec![1.0; 6],
            ))
            .unwrap();
        model
            .insert_outlier(
                OutlierType::Additive,
                later,
                axis.frequency(),
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                0.7,
            )
            .unwrap();

        assert_eq!(
            model.kinds().collect::<Vec<_>>(),
            [
                RegressorKind::Outlier {
                    kind: OutlierType::Additive,
                    period: first,
                    frequency: axis.frequency(),
                },
                RegressorKind::Outlier {
                    kind: OutlierType::Additive,
                    period: later,
                    frequency: axis.frequency(),
                },
                RegressorKind::Constant,
            ]
        );
        assert_eq!(model.matrix(&[0.0; 6]).unwrap().column_count(), 4);
        assert_eq!(model.evaluate(axis, 6).unwrap()[0], first_values);
    }

    #[test]
    fn outlier_insertion_without_an_outlier_group_and_with_a_duplicate() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let first = axis.period_at(1).unwrap();
        let later = axis.period_at(3).unwrap();
        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Constant,
                vec![1.0; 6],
            ))
            .unwrap();
        model
            .insert_outlier(
                OutlierType::TemporaryChange,
                later,
                axis.frequency(),
                vec![0.0; 6],
                0.7,
            )
            .unwrap();
        model
            .insert_outlier(
                OutlierType::Additive,
                first,
                axis.frequency(),
                vec![0.0; 6],
                0.7,
            )
            .unwrap();
        model
            .insert_outlier(
                OutlierType::LevelShift,
                first,
                axis.frequency(),
                vec![0.0; 6],
                0.7,
            )
            .unwrap();
        model
            .insert_outlier(
                OutlierType::TemporaryChange,
                first,
                axis.frequency(),
                vec![0.0; 6],
                0.7,
            )
            .unwrap();
        assert_eq!(
            model.kinds().collect::<Vec<_>>(),
            [
                RegressorKind::Constant,
                RegressorKind::Outlier {
                    kind: OutlierType::Additive,
                    period: first,
                    frequency: axis.frequency()
                },
                RegressorKind::Outlier {
                    kind: OutlierType::LevelShift,
                    period: first,
                    frequency: axis.frequency()
                },
                RegressorKind::Outlier {
                    kind: OutlierType::TemporaryChange,
                    period: first,
                    frequency: axis.frequency()
                },
                RegressorKind::Outlier {
                    kind: OutlierType::TemporaryChange,
                    period: later,
                    frequency: axis.frequency()
                },
            ]
        );
        let unchanged = model.clone();
        assert!(matches!(
            model.insert_outlier(
                OutlierType::Additive,
                first,
                axis.frequency(),
                vec![0.0; 6],
                0.7
            ),
            Err(Error::Internal(_))
        ));
        assert_eq!(model, unchanged);
    }

    #[test]
    fn outlier_columns_are_rebuilt_over_a_longer_monthly_span() {
        let observed_axis = TimeAxis::monthly(2000, 1).unwrap();
        let event = observed_axis.period_at(1).unwrap();
        let mut model = RegressionModel::new();
        for (kind, values) in [
            (OutlierType::Additive, vec![0.0, 1.0, 0.0]),
            (OutlierType::LevelShift, vec![-1.0, 0.0, 0.0]),
            (OutlierType::TemporaryChange, vec![0.0, 1.0, 0.7]),
        ] {
            model
                .insert_outlier(kind, event, observed_axis.frequency(), values, 0.7)
                .unwrap();
        }

        let extended = model
            .evaluate(TimeAxis::monthly(1999, 12).unwrap(), 6)
            .unwrap();
        assert_eq!(extended[0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(extended[1], [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            extended[2],
            [0.0, 0.0, 1.0, 0.7, 0.7_f64 * 0.7, 0.7_f64 * 0.7 * 0.7]
        );

        let forecast_only = model
            .evaluate(TimeAxis::monthly(2000, 4).unwrap(), 3)
            .unwrap();
        assert_eq!(forecast_only[0], [0.0; 3]);
        assert_eq!(forecast_only[1], [0.0; 3]);
        assert_eq!(
            forecast_only[2],
            [
                0.7_f64 * 0.7,
                0.7_f64 * 0.7 * 0.7,
                0.7_f64 * 0.7 * 0.7 * 0.7,
            ]
        );
    }

    #[test]
    fn level_shift_definition_extends_across_quarterly_backcasts() {
        let observed_axis = TimeAxis::quarterly(2000, 2).unwrap();
        let event = observed_axis.start();
        let mut model = RegressionModel::new();
        model
            .insert_outlier(
                OutlierType::LevelShift,
                event,
                observed_axis.frequency(),
                vec![0.0, 0.0],
                0.7,
            )
            .unwrap();

        assert_eq!(
            model
                .evaluate(TimeAxis::quarterly(1999, 3).unwrap(), 6)
                .unwrap()[0],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]
        );
    }
}
