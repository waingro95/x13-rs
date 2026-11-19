//! The effect each group of fitted regressors has on the series, and the
//! steps that remove those effects from, or restore them to, the X-11
//! components (`adjreg`, `x11pt2.f`, `x11pt3.f`).

use std::ops::Range;

use super::model::RegressionModel;
use crate::numerics::blas1::axpy;
use crate::regarima::{CalendarEffectScale, CalendarEffects};
use crate::regarima::{OutlierType, OutlierTypes, RegressionVariable};
use crate::{Error, Period, Transform};

/// The fitted regression effects by group, over the model's full span.
///
/// Each `*_linear` effect is on the transformed scale. The `full_*` calendar
/// effects are on the scale the adjustment uses: factors under a log
/// transform, differences otherwise.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EvaluatedEffects {
    /// The observed span's index range within the `full_*` columns.
    observed: Range<usize>,
    scale: CalendarEffectScale,
    pub(crate) full_estimated: Vec<f64>,
    pub(crate) full_fixed_length: Vec<f64>,
    pub(crate) full_combined: Vec<f64>,
    /// Trading-day and standalone length columns (Build 62's Ftd group). Its
    /// presence also stands in for the calendar-linear flag, which Build 62
    /// always sets together with it.
    trading_day: EffectGroup,
    holiday: EffectGroup,
    additive_outlier: EffectGroup,
    level_shift: EffectGroup,
    temporary_change: EffectGroup,
    user: EffectGroup,
    /// Missing-value columns, which no adjustment branch tests for presence.
    full_missing_linear: Vec<f64>,
    /// The model carries a fixed length-of-period prior. It is not a fitted
    /// group, so it has its own flag.
    has_fixed_length: bool,
}

/// One regression effect group over the full span.
///
/// `linear` stays all zeros when the model has no column in the group, and
/// `remove_from_transformed_response` and the public accessors read it
/// unconditionally. `present` records whether the group has a column: Build 62's
/// adjustment routines (`x11pt2.f`, `x11pt3.f`, `adjreg`) branch on that, never
/// on the values, so an all-zero effect from a present column still takes the
/// present branch.
#[derive(Debug, Clone, PartialEq)]
struct EffectGroup {
    present: bool,
    linear: Vec<f64>,
}

impl EffectGroup {
    fn absent(count: usize) -> Self {
        Self {
            present: false,
            linear: vec![0.0; count],
        }
    }

    fn add(&mut self, coefficient: f64, values: &[f64]) {
        self.present = true;
        let count = self.linear.len();
        axpy(coefficient, &values[..count], &mut self.linear);
    }
}

/// Multiplies each column of `model` by its coefficient and sums the
/// products into Build 62's effect groups.
pub(crate) fn evaluate_regression_effects(
    model: &RegressionModel,
    coefficients: &[f64],
    transform: Transform,
    observed: Range<usize>,
) -> Result<EvaluatedEffects, Error> {
    let count = model.value_count();
    if coefficients.len() != model.columns().len()
        || observed.start > observed.end
        || observed.end > count
    {
        return Err(layout_error());
    }
    let has_fixed_length = !model.fixed_length().is_empty();

    let mut calendar_linear = vec![0.0; count];
    let mut trading_day = EffectGroup::absent(count);
    let mut holiday = EffectGroup::absent(count);
    let mut additive_outlier = EffectGroup::absent(count);
    let mut level_shift = EffectGroup::absent(count);
    let mut temporary_change = EffectGroup::absent(count);
    let mut missing_linear = vec![0.0; count];
    let mut user = EffectGroup::absent(count);
    let mut user_type_index = 0;
    for ((kind, values), coefficient) in model.columns().zip(coefficients) {
        let mut variable = kind.variable();
        if let RegressionVariable::User { index, kind } = variable {
            // chkadj/prtref advance iusr for holiday and generic user columns,
            // but not supplied TD. Preserve that remapping quirk for mixtures;
            // coefficient identities still describe the supplied column.
            match kind {
                crate::regarima::UserRegressorType::User => {
                    variable = RegressionVariable::User {
                        index,
                        kind: model.user_regressors()[user_type_index].kind(),
                    };
                    user_type_index += 1;
                }
                crate::regarima::UserRegressorType::Holiday => user_type_index += 1,
                crate::regarima::UserRegressorType::TradingDay => {}
            }
        }
        match variable {
            RegressionVariable::TradingDay(_)
            | RegressionVariable::User {
                kind: crate::regarima::UserRegressorType::TradingDay,
                ..
            } => {
                axpy(*coefficient, &values[..count], &mut calendar_linear);
                trading_day.add(*coefficient, values);
            }
            RegressionVariable::Length(_) => {
                // chkadj.f and prtref.f classify standalone LOM/LOQ/LPYEAR
                // with Ftd, so it participates in the final calendar factor.
                axpy(*coefficient, &values[..count], &mut calendar_linear);
                trading_day.add(*coefficient, values);
            }
            RegressionVariable::Holiday(_)
            | RegressionVariable::User {
                kind: crate::regarima::UserRegressorType::Holiday,
                ..
            } => {
                axpy(*coefficient, &values[..count], &mut calendar_linear);
                holiday.add(*coefficient, values);
            }
            RegressionVariable::Outlier { kind, .. } => {
                let group = match kind {
                    crate::regarima::OutlierType::Additive => &mut additive_outlier,
                    crate::regarima::OutlierType::LevelShift => &mut level_shift,
                    crate::regarima::OutlierType::TemporaryChange => &mut temporary_change,
                };
                group.add(*coefficient, values);
            }
            RegressionVariable::Constant => {}
            RegressionVariable::User {
                kind: crate::regarima::UserRegressorType::User,
                ..
            } => user.add(*coefficient, values),
            RegressionVariable::MissingValue { .. } => {
                axpy(*coefficient, &values[..count], &mut missing_linear);
            }
        }
    }

    let (scale, full_estimated, full_fixed_length, full_combined) = match transform {
        Transform::None => {
            let fixed = vec![0.0; count];
            (
                CalendarEffectScale::Difference,
                calendar_linear.clone(),
                fixed,
                calendar_linear,
            )
        }
        Transform::Log => {
            let estimated = exponentiate(calendar_linear);
            let fixed = if model.fixed_length().is_empty() {
                vec![1.0; count]
            } else if model.fixed_length().len() == count {
                model.fixed_length().to_vec()
            } else {
                return Err(layout_error());
            };
            let combined = estimated
                .iter()
                .zip(&fixed)
                .map(|(estimated, fixed)| estimated * fixed)
                .collect();
            (CalendarEffectScale::Factor, estimated, fixed, combined)
        }
    };

    Ok(EvaluatedEffects {
        observed,
        scale,
        full_estimated,
        full_fixed_length,
        full_combined,
        trading_day,
        holiday,
        additive_outlier,
        level_shift,
        temporary_change,
        user,
        full_missing_linear: missing_linear,
        has_fixed_length,
    })
}

const fn layout_error() -> Error {
    Error::internal("regression effect lengths differ from the model")
}

fn exponentiate(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(f64::exp).collect()
}

impl EvaluatedEffects {
    /// The calendar effects over the observed span, whose first period is
    /// `start`.
    pub(crate) fn calendar_effects(&self, start: Period) -> CalendarEffects {
        CalendarEffects {
            start,
            scale: self.scale,
            estimated: self.full_estimated[self.observed.clone()].to_vec(),
            fixed_length: self.full_fixed_length[self.observed.clone()].to_vec(),
            combined: self.full_combined[self.observed.clone()].to_vec(),
        }
    }

    /// The user-regressor effect.
    pub(crate) fn user_linear(&self) -> &[f64] {
        &self.user.linear
    }

    /// Whether the model has a user regressor.
    pub(crate) fn has_user(&self) -> bool {
        self.user.present
    }

    /// The missing-value regressors' effect.
    pub(crate) fn missing_linear(&self) -> &[f64] {
        &self.full_missing_linear
    }

    /// adjreg.f's Orixmv, before calendar and outlier removal. arima.f restores
    /// this entire observed series when missing values are present.
    pub(crate) fn missing_adjusted_original(
        &self,
        transformed: &[f64],
        transform: Transform,
        restore_prior: bool,
    ) -> Vec<f64> {
        transformed
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let value =
                    crate::transform::inverse(transform, value - self.full_missing_linear[i]);
                // adjreg restores Sprior into Orixmv only when Kfmt > 0.
                if transform == Transform::Log && restore_prior {
                    value * self.full_fixed_length[i]
                } else {
                    value
                }
            })
            .collect()
    }

    /// The additive-outlier effect.
    pub(crate) fn additive_outlier_linear(&self) -> &[f64] {
        &self.additive_outlier.linear
    }

    /// The level-shift effect.
    pub(crate) fn level_shift_linear(&self) -> &[f64] {
        &self.level_shift.linear
    }

    /// The temporary-change effect.
    pub(crate) fn temporary_change_linear(&self) -> &[f64] {
        &self.temporary_change.linear
    }

    /// The trading-day and estimated length effect.
    pub(crate) fn trading_day_linear(&self) -> &[f64] {
        &self.trading_day.linear
    }

    /// The holiday effect.
    pub(crate) fn holiday_linear(&self) -> &[f64] {
        &self.holiday.linear
    }

    /// x11pt1/2/3: H1 enters Faccal first, but enters Fachol after fitted holidays.
    pub(crate) fn legacy_calendar_factors(
        &self,
        observed: Range<usize>,
        h1: &[f64],
        holiday_adjustment: crate::x11::HolidayAdjustment,
    ) -> Result<(Vec<f64>, Vec<f64>), Error> {
        if observed.len() != h1.len() {
            return Err(layout_error());
        }
        let mut calendar = Vec::with_capacity(h1.len());
        let mut holiday = Vec::with_capacity(h1.len());
        for (i, &factor) in observed.zip(h1) {
            // Faccal starts at 1.0 and takes H1 first: 1.0 * H1 is H1.
            let mut combined = factor;
            if self.trading_day.present {
                combined *= self.trading_day.linear[i].exp() * self.full_fixed_length[i];
            }
            let fitted_holiday = if self.holiday.present {
                self.holiday.linear[i].exp()
            } else {
                1.0
            };
            if self.holiday.present {
                combined *= fitted_holiday;
            }
            let combined_holiday = fitted_holiday * factor;
            if holiday_adjustment == crate::x11::HolidayAdjustment::Retain {
                combined /= combined_holiday;
            }
            calendar.push(combined);
            holiday.push(combined_holiday);
        }
        Ok((calendar, holiday))
    }

    pub(crate) fn has_final_calendar(&self) -> bool {
        self.trading_day.present || self.holiday.present
    }
    pub(crate) fn remove_selected_outliers_from_final(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
        types: OutlierTypes,
    ) {
        debug_assert_eq!(values.len(), full_observed.len());
        if types.contains(OutlierType::Additive) && self.additive_outlier.present {
            remove_effect_group(
                values,
                &self.additive_outlier.linear[full_observed.clone()],
                transform,
            );
        }
        if types.contains(OutlierType::LevelShift) && self.level_shift.present {
            remove_effect_group(
                values,
                &self.level_shift.linear[full_observed.clone()],
                transform,
            );
        }
        if types.contains(OutlierType::TemporaryChange) && self.temporary_change.present {
            remove_effect_group(
                values,
                &self.temporary_change.linear[full_observed],
                transform,
            );
        }
    }

    /// Subtracts every group's effect from a transformed series spanning
    /// the whole model.
    pub(crate) fn remove_from_transformed_response(&self, values: &mut [f64]) -> Result<(), Error> {
        // Every group spans the full model, as the missing-value linear does.
        if values.len() != self.full_missing_linear.len() {
            return Err(layout_error());
        }
        for group in [
            &self.trading_day.linear,
            &self.level_shift.linear,
            &self.holiday.linear,
            &self.additive_outlier.linear,
            &self.temporary_change.linear,
            &self.user.linear,
            &self.full_missing_linear,
        ] {
            remove_linear_group(values, group);
        }
        Ok(())
    }

    pub(crate) fn remove_final_calendar_and_prior(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
        holiday_adjustment: crate::x11::HolidayAdjustment,
    ) {
        debug_assert_eq!(values.len(), full_observed.len());
        let trading_day = &self.trading_day.linear[full_observed.clone()];
        let holiday = &self.holiday.linear[full_observed.clone()];
        let fixed_length = &self.full_fixed_length[full_observed];
        match transform {
            Transform::None => {
                if self.trading_day.present || self.holiday.present {
                    for ((value, trading_day), holiday) in
                        values.iter_mut().zip(trading_day).zip(holiday)
                    {
                        // `x11pt2.f` builds Faccal from its additive identity,
                        // TD, then holiday; `x11pt3.f` subtracts it once.
                        let mut calendar = 0.0;
                        if self.trading_day.present {
                            calendar += trading_day;
                        }
                        if self.holiday.present {
                            calendar += holiday;
                        }
                        if self.holiday.present
                            && holiday_adjustment == crate::x11::HolidayAdjustment::Retain
                        {
                            calendar -= holiday;
                        }
                        *value -= calendar;
                    }
                }
            }
            Transform::Log => {
                if self.trading_day.present || self.holiday.present {
                    for (((value, trading_day), holiday), fixed_length) in values
                        .iter_mut()
                        .zip(trading_day)
                        .zip(holiday)
                        .zip(fixed_length)
                    {
                        let mut calendar = 1.0;
                        if self.trading_day.present {
                            // `tdlom.f` folds a flow-length prior into Factd
                            // before `x11pt2.f` multiplies it into Faccal.
                            calendar *= trading_day.exp() * fixed_length;
                        }
                        if self.holiday.present {
                            let holiday_factor = holiday.exp();
                            calendar *= holiday_factor;
                            if holiday_adjustment == crate::x11::HolidayAdjustment::Retain {
                                calendar /= holiday_factor;
                            }
                        }
                        *value /= calendar;
                    }
                }
                if !self.trading_day.present && self.has_fixed_length {
                    for (value, fixed_length) in values.iter_mut().zip(fixed_length) {
                        *value /= fixed_length;
                    }
                }
            }
        }
    }

    pub(crate) fn remove_outliers_from_irregular(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
    ) {
        if self.level_shift.present {
            remove_effect_group(
                values,
                &self.level_shift.linear[full_observed.clone()],
                transform,
            );
        }
        if self.additive_outlier.present {
            remove_effect_group(
                values,
                &self.additive_outlier.linear[full_observed.clone()],
                transform,
            );
        }
        if self.temporary_change.present {
            remove_effect_group(
                values,
                &self.temporary_change.linear[full_observed.clone()],
                transform,
            );
        }
        // x11pt3:587 removes Facusr after LS/AO/TC; unlike AO/TC it is
        // never restored into the final irregular. D11 retains it by default.
        if self.user.present {
            remove_effect_group(values, &self.user.linear[full_observed], transform);
        }
    }

    /// Part F consumes final Faccal and sp2, not public effect totals or E2.
    pub(crate) fn quality_adjustments(
        &self,
        transform: Transform,
        observed: Range<usize>,
        holiday_adjustment: crate::x11::HolidayAdjustment,
    ) -> crate::x11::quality::QualityAdjustments {
        let multiplicative = transform == Transform::Log;
        let effect = |group: &EffectGroup| {
            group.present.then(|| {
                group.linear[observed.clone()]
                    .iter()
                    .map(|&value| if multiplicative { value.exp() } else { value })
                    .collect::<Vec<_>>()
            })
        };
        let level_shift = effect(&self.level_shift);
        let additive_outlier = effect(&self.additive_outlier);
        let temporary_change = effect(&self.temporary_change);
        let user = effect(&self.user);
        let prior = self.has_fixed_length
            || self.level_shift.present
            || self.additive_outlier.present
            || self.temporary_change.present
            || self.user.present;
        let prior = prior.then(|| {
            // The supported fixed flow prior was folded into Factd by tdlom;
            // its initial Sprior is identity, but its Kfmt remains present.
            let mut prior = vec![if multiplicative { 1.0 } else { 0.0 }; observed.len()];
            for group in [&level_shift, &additive_outlier, &temporary_change, &user]
                .into_iter()
                .flatten()
            {
                for (value, &factor) in prior.iter_mut().zip(group) {
                    *value = if multiplicative {
                        factor * *value
                    } else {
                        factor + *value
                    };
                }
            }
            prior
        });
        let calendar = (self.trading_day.present || self.holiday.present).then(|| {
            observed
                .clone()
                .map(|i| {
                    if multiplicative {
                        let mut value = 1.0;
                        if self.trading_day.present {
                            value *= self.trading_day.linear[i].exp() * self.full_fixed_length[i];
                        }
                        if self.holiday.present {
                            let holiday = self.holiday.linear[i].exp();
                            value *= holiday;
                            if holiday_adjustment == crate::x11::HolidayAdjustment::Retain {
                                value /= holiday;
                            }
                        }
                        value
                    } else {
                        let mut value = 0.0;
                        if self.trading_day.present {
                            value += self.trading_day.linear[i];
                        }
                        if self.holiday.present {
                            value += self.holiday.linear[i];
                        }
                        if self.holiday.present
                            && holiday_adjustment == crate::x11::HolidayAdjustment::Retain
                        {
                            value -= self.holiday.linear[i];
                        }
                        value
                    }
                })
                .collect()
        });
        crate::x11::quality::QualityAdjustments {
            calendar,
            prior,
            level_shift,
            additive_outlier,
            temporary_change,
        }
    }

    pub(crate) fn restore_level_shift_to_trend(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
    ) {
        if self.level_shift.present {
            restore_effect_group(values, &self.level_shift.linear[full_observed], transform);
        }
    }

    /// x11pt3 Part E: the zero-C17 operand is Stc2 only for retained LS.
    /// This receives base Stc, before the ordinary final D12 restoration.
    pub(crate) fn restore_modified_replacement(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
        final_outliers: crate::x11::FinalOutliers,
        allocation: crate::x11::TemporaryChangeAllocation,
    ) {
        if self.level_shift.present && !final_outliers.removes(OutlierType::LevelShift) {
            self.restore_level_shift_to_trend(values, transform, full_observed.clone());
            if allocation == crate::x11::TemporaryChangeAllocation::Trend
                && !final_outliers.removes(OutlierType::TemporaryChange)
            {
                self.restore_temporary_changes(values, transform, full_observed.clone());
            }
        }
        // The only supported initial fixed prior is log flow-TD length.
        // tdlom folds it into Factd and resets Sprior to identity, so Nadj's
        // source multiplication here is by 1.0 and changes nothing. Estimated
        // length is not an initial prior and never reaches it.
        debug_assert!(
            !self.has_fixed_length || (self.trading_day.present && transform == Transform::Log)
        );
        if self.level_shift.present && final_outliers.removes(OutlierType::LevelShift) {
            remove_effect_group(values, &self.level_shift.linear[full_observed], transform);
        }
    }

    /// Retained AO then retained TC are removed from every E2 entry, including
    /// positive-C17 entries copied from final D11 and either TC allocation.
    pub(crate) fn remove_modified_outliers(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
        final_outliers: crate::x11::FinalOutliers,
    ) {
        if self.additive_outlier.present && !final_outliers.removes(OutlierType::Additive) {
            remove_effect_group(
                values,
                &self.additive_outlier.linear[full_observed.clone()],
                transform,
            );
        }
        if self.temporary_change.present && !final_outliers.removes(OutlierType::TemporaryChange) {
            remove_effect_group(
                values,
                &self.temporary_change.linear[full_observed],
                transform,
            );
        }
    }

    pub(crate) fn restore_additive_outliers_to_irregular(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
    ) {
        if self.additive_outlier.present {
            restore_effect_group(
                values,
                &self.additive_outlier.linear[full_observed],
                transform,
            );
        }
    }

    pub(crate) fn restore_temporary_changes(
        &self,
        values: &mut [f64],
        transform: Transform,
        full_observed: Range<usize>,
    ) {
        if self.temporary_change.present {
            restore_effect_group(
                values,
                &self.temporary_change.linear[full_observed],
                transform,
            );
        }
    }
}

fn remove_linear_group(values: &mut [f64], group: &[f64]) {
    debug_assert_eq!(values.len(), group.len());
    for (value, effect) in values.iter_mut().zip(group) {
        *value -= effect;
    }
}

fn remove_effect_group(values: &mut [f64], group: &[f64], transform: Transform) {
    debug_assert_eq!(values.len(), group.len());
    for (value, effect) in values.iter_mut().zip(group) {
        match transform {
            Transform::None => *value -= effect,
            Transform::Log => *value /= effect.exp(),
        }
    }
}

fn restore_effect_group(values: &mut [f64], group: &[f64], transform: Transform) {
    debug_assert_eq!(values.len(), group.len());
    // `addmul.f` puts the effect on the left; IEEE `+` and `*` commute.
    for (value, effect) in values.iter_mut().zip(group) {
        match transform {
            Transform::None => *value += effect,
            Transform::Log => *value *= effect.exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate_regression_effects, layout_error};
    use crate::regarima::{
        CalendarBasis, CalendarSpec, FlowTradingDay, OutlierType, RegressionVariable,
        TradingDayContrast,
    };
    use crate::regression::model::{RegressionColumn, RegressionModel};
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, RegArimaSpec, Transform, x11};

    #[test]
    fn mismatched_shapes_are_errors() {
        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Constant,
                vec![1.0],
            ))
            .unwrap();
        let failure = layout_error();
        for (coefficients, observed) in [(&[][..], 0..1), (&[0.0], 0..2)] {
            assert_eq!(
                evaluate_regression_effects(&model, coefficients, Transform::None, observed),
                Err(failure.clone())
            );
        }
        let effects = evaluate_regression_effects(&model, &[0.0], Transform::None, 0..1).unwrap();
        assert_eq!(
            effects.legacy_calendar_factors(0..1, &[], x11::HolidayAdjustment::Remove),
            Err(failure.clone())
        );
        assert_eq!(
            effects.remove_from_transformed_response(&mut [0.0, 0.0]),
            Err(failure)
        );
    }

    #[test]
    fn legacy_easter_factor_combines_with_trading_day_and_holiday_effects() {
        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Constant,
                vec![1.0],
            ))
            .unwrap();
        let mut effects =
            evaluate_regression_effects(&model, &[0.0], Transform::Log, 0..1).unwrap();
        let h1 = [0.3];
        assert!(!effects.has_final_calendar()); // Khol/H1 alone is not the D11 guard.
        assert_eq!(
            effects
                .legacy_calendar_factors(0..1, &h1, crate::x11::HolidayAdjustment::Remove)
                .unwrap(),
            (vec![0.3], vec![0.3])
        );
        assert_eq!(
            effects
                .legacy_calendar_factors(0..1, &h1, crate::x11::HolidayAdjustment::Retain)
                .unwrap(),
            (vec![1.0], vec![0.3])
        );
        effects.trading_day.present = true;
        effects.holiday.present = true;
        effects.trading_day.linear[0] = 0.7;
        effects.full_fixed_length[0] = 0.9;
        effects.holiday.linear[0] = -0.4;
        let td = 0.7_f64.exp() * 0.9;
        let holiday = (-0.4_f64).exp();
        let faccal = (1.0 * h1[0]) * td * holiday;
        let fachol = holiday * h1[0];
        assert!(effects.has_final_calendar());
        assert_eq!(
            effects
                .legacy_calendar_factors(0..1, &h1, crate::x11::HolidayAdjustment::Remove)
                .unwrap(),
            (vec![faccal], vec![fachol])
        );
        assert_eq!(
            effects
                .legacy_calendar_factors(0..1, &h1, crate::x11::HolidayAdjustment::Retain)
                .unwrap(),
            (vec![faccal / fachol], vec![fachol])
        );
    }

    #[test]
    fn quality_adjustments_keep_present_groups_and_sum_ls_ao_tc_in_order() {
        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Constant,
                vec![1.0],
            ))
            .unwrap();
        for transform in [Transform::None, Transform::Log] {
            let effects = evaluate_regression_effects(&model, &[0.0], transform, 0..1).unwrap();
            let q =
                effects.quality_adjustments(transform, 0..1, crate::x11::HolidayAdjustment::Remove);
            assert!(
                q.prior.is_none()
                    && q.additive_outlier.is_none()
                    && q.level_shift.is_none()
                    && q.temporary_change.is_none()
            );
        }
        push_outlier(&mut model, OutlierType::Additive, vec![1.0]);
        push_outlier(&mut model, OutlierType::LevelShift, vec![1.0]);
        push_outlier(&mut model, OutlierType::TemporaryChange, vec![1.0]);
        for transform in [Transform::None, Transform::Log] {
            let effects = evaluate_regression_effects(&model, &[0.0; 4], transform, 0..1).unwrap();
            let q =
                effects.quality_adjustments(transform, 0..1, crate::x11::HolidayAdjustment::Remove);
            let identity = if transform == Transform::Log {
                1.0
            } else {
                0.0
            };
            for group in [
                q.prior,
                q.additive_outlier,
                q.level_shift,
                q.temporary_change,
            ] {
                assert_eq!(group, Some(vec![identity]));
            }
        }
        // The columns are AO, LS, TC; Build 62 still sums sp2 as LS, AO, TC.
        let effects =
            evaluate_regression_effects(&model, &[0.0, -1e16, 1e16, 1.0], Transform::None, 0..1)
                .unwrap();
        assert_eq!(
            effects
                .quality_adjustments(Transform::None, 0..1, crate::x11::HolidayAdjustment::Remove)
                .prior,
            Some(vec![1.0])
        );
    }

    #[test]
    fn seasonality_modified_replacement_requires_retained_ls_not_tc_in_trend() {
        use crate::regarima::OutlierTypes;
        use crate::{x11::FinalOutliers as F, x11::TemporaryChangeAllocation as T};
        for (with_ls, final_policy, expected) in [
            (false, F::Retain, 70.0),
            (true, F::Retain, 77.0),
            (true, F::Remove(OutlierTypes::LEVEL_SHIFT), 67.0),
            (false, F::Remove(OutlierTypes::LEVEL_SHIFT), 70.0),
        ] {
            let mut model = RegressionModel::new();
            let mut coefficients = Vec::new();
            if with_ls {
                push_outlier(&mut model, OutlierType::LevelShift, vec![1.0]);
                coefficients.push(3.0);
            }
            push_outlier(&mut model, OutlierType::TemporaryChange, vec![1.0]);
            coefficients.push(4.0);
            let effects =
                evaluate_regression_effects(&model, &coefficients, Transform::None, 0..1).unwrap();
            let mut replacement = [70.0];
            effects.restore_modified_replacement(
                &mut replacement,
                Transform::None,
                0..1,
                final_policy,
                T::Trend,
            );
            assert_eq!(replacement, [expected]);
        }
    }

    #[test]
    fn seasonality_modified_common_removal_is_ao_then_tc_and_respects_final_class_flags() {
        use crate::regarima::OutlierTypes;
        use crate::x11::FinalOutliers as F;
        let mut model = RegressionModel::new();
        push_outlier(&mut model, OutlierType::Additive, vec![1.0]);
        push_outlier(&mut model, OutlierType::TemporaryChange, vec![1.0]);
        let effects =
            evaluate_regression_effects(&model, &[2.0, 4.0], Transform::None, 0..1).unwrap();
        for (policy, expected) in [
            (F::Retain, 84.0),
            (F::Remove(OutlierTypes::ADDITIVE), 86.0),
            (F::Remove(OutlierTypes::TEMPORARY_CHANGE), 88.0),
            (F::Remove(OutlierTypes::ALL), 90.0),
        ] {
            let mut values = [90.0];
            effects.remove_modified_outliers(&mut values, Transform::None, 0..1, policy);
            assert_eq!(values, [expected]);
        }
    }

    #[test]
    fn seasonality_modified_present_zero_ls_still_selects_tc_restored_trend() {
        let mut model = RegressionModel::new();
        push_outlier(&mut model, OutlierType::LevelShift, vec![1.0]);
        push_outlier(&mut model, OutlierType::TemporaryChange, vec![1.0]);
        let effects =
            evaluate_regression_effects(&model, &[0.0, 4.0], Transform::None, 0..1).unwrap();
        let mut values = [70.0];
        effects.restore_modified_replacement(
            &mut values,
            Transform::None,
            0..1,
            crate::x11::FinalOutliers::Retain,
            crate::x11::TemporaryChangeAllocation::Trend,
        );
        assert_eq!(values, [74.0]);
    }

    #[test]
    fn seasonality_modified_log_removal_keeps_separate_ao_then_tc_divisions() {
        let mut model = RegressionModel::new();
        push_outlier(&mut model, OutlierType::Additive, vec![1.0]);
        push_outlier(&mut model, OutlierType::TemporaryChange, vec![1.0]);
        let effects =
            evaluate_regression_effects(&model, &[0.1, 0.2], Transform::Log, 0..1).unwrap();
        let mut values = [1.0];
        effects.remove_modified_outliers(
            &mut values,
            Transform::Log,
            0..1,
            crate::x11::FinalOutliers::Retain,
        );
        let expected = (1.0_f64 / 0.1_f64.exp()) / 0.2_f64.exp();
        assert_eq!(values[0].to_bits(), expected.to_bits());
        assert_ne!(
            expected.to_bits(),
            (1.0_f64 / (0.1_f64.exp() * 0.2_f64.exp())).to_bits()
        );
    }

    #[test]
    fn seasonality_modified_flow_length_prior_is_identity_but_standalone_length_is_estimated() {
        for basis in [
            CalendarBasis::FlowTradingDay(FlowTradingDay::new(TradingDayContrast::WeekdayWeekend)),
            CalendarBasis::Length(crate::regarima::LengthEffect::LeapYear),
        ] {
            let model = RegressionModel::from_spec(
                RegArimaSpec::new(empty_model())
                    .with_transform(Transform::Log)
                    .with_calendar(CalendarSpec::new().with_basis(basis))
                    .resolve_explicit()
                    .unwrap(),
                TimeAxis::monthly(2000, 1).unwrap(),
                24,
            )
            .unwrap();
            let coefficients = vec![0.1; model.column_count()];
            let effects =
                evaluate_regression_effects(&model, &coefficients, Transform::Log, 0..24).unwrap();
            let fixed = matches!(basis, CalendarBasis::FlowTradingDay(_));
            assert_eq!(effects.has_fixed_length, fixed);
            let quality = effects.quality_adjustments(
                Transform::Log,
                0..24,
                crate::x11::HolidayAdjustment::Remove,
            );
            assert_eq!(quality.prior, fixed.then(|| vec![1.0; 24]));
            assert!(quality.calendar.is_some());
            if fixed {
                assert!(effects.full_fixed_length.iter().any(|v| *v != 1.0));
            }
            let mut values = [17.25; 24];
            effects.restore_modified_replacement(
                &mut values,
                Transform::Log,
                0..24,
                crate::x11::FinalOutliers::Retain,
                crate::x11::TemporaryChangeAllocation::Irregular,
            );
            assert_eq!(values, [17.25; 24]);
        }
    }

    #[test]
    fn noncalendar_columns_leave_exact_scale_identities() {
        let model =
            RegressionModel::with_constant(empty_model(), crate::Frequency::Monthly, 3).unwrap();

        let difference =
            evaluate_regression_effects(&model, &[2.0], Transform::None, 0..3).unwrap();
        assert_eq!(difference.full_estimated, [0.0; 3]);
        assert_eq!(difference.full_fixed_length, [0.0; 3]);
        assert_eq!(difference.full_combined, [0.0; 3]);

        let factor = evaluate_regression_effects(&model, &[2.0], Transform::Log, 0..3).unwrap();
        assert_eq!(factor.full_estimated, [1.0; 3]);
        assert_eq!(factor.full_fixed_length, [1.0; 3]);
        assert_eq!(factor.full_combined, [1.0; 3]);

        let mut final_calendar_adjusted = [10.0, 11.0, 12.0];
        factor.remove_final_calendar_and_prior(
            &mut final_calendar_adjusted,
            Transform::Log,
            0..3,
            crate::x11::HolidayAdjustment::Remove,
        );
        assert_eq!(final_calendar_adjusted, [10.0, 11.0, 12.0]);
    }

    #[test]
    fn log_standalone_length_is_an_estimated_calendar_factor() {
        let model = RegressionModel::from_spec(
            RegArimaSpec::new(empty_model())
                .with_transform(Transform::Log)
                .with_calendar(CalendarSpec::new().with_basis(CalendarBasis::Length(
                    crate::regarima::LengthEffect::LeapYear,
                )))
                .resolve_explicit()
                .unwrap(),
            TimeAxis::monthly(2000, 2).unwrap(),
            1,
        )
        .unwrap();
        let effects = evaluate_regression_effects(&model, &[0.5], Transform::Log, 0..1).unwrap();
        assert_eq!(effects.full_fixed_length, [1.0]);
        assert_eq!(effects.full_estimated, [0.375_f64.exp()]);
        let mut final_calendar_adjusted = [100.0];
        effects.remove_final_calendar_and_prior(
            &mut final_calendar_adjusted,
            Transform::Log,
            0..1,
            crate::x11::HolidayAdjustment::Remove,
        );

        assert_eq!(final_calendar_adjusted, [100.0 / 0.375_f64.exp()]);
    }

    #[test]
    fn final_identity_calendar_removal_uses_the_combined_faccal() {
        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::TradingDay(
                    crate::regarima::TradingDayVariable::FlowWeekdayWeekend,
                ),
                vec![1.0e16],
            ))
            .unwrap();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Holiday(crate::regarima::HolidayVariable::Easter(
                    crate::regarima::Easter::new(1).unwrap(),
                )),
                vec![-1.0e16],
            ))
            .unwrap();
        let effects =
            evaluate_regression_effects(&model, &[1.0, 1.0], Transform::None, 0..1).unwrap();
        let mut final_calendar_adjusted = [1.0];
        effects.remove_final_calendar_and_prior(
            &mut final_calendar_adjusted,
            Transform::None,
            0..1,
            crate::x11::HolidayAdjustment::Remove,
        );

        // `x11pt2.f` builds Faccal as ((0 + TD) + holiday), and `x11pt3.f`
        // subtracts that one combined value. Sequential removal instead gives
        // (1 - 1e16) - -1e16 == 0.
        assert_eq!(final_calendar_adjusted, [1.0]);
    }

    #[test]
    fn retained_holiday_cancels_from_combined_additive_calendar() {
        let mut model = RegressionModel::new();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::TradingDay(
                    crate::regarima::TradingDayVariable::FlowWeekdayWeekend,
                ),
                vec![1.0],
            ))
            .unwrap();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Holiday(crate::regarima::HolidayVariable::Easter(
                    crate::regarima::Easter::new(1).unwrap(),
                )),
                vec![1.0e16],
            ))
            .unwrap();
        let effects =
            evaluate_regression_effects(&model, &[1.0, 1.0], Transform::None, 0..1).unwrap();
        let mut values = [100.0];
        effects.remove_final_calendar_and_prior(
            &mut values,
            Transform::None,
            0..1,
            crate::x11::HolidayAdjustment::Retain,
        );
        assert_eq!(values, [100.0]);
        assert_ne!(values, [99.0]);
    }

    #[test]
    fn retained_holiday_divides_the_combined_log_factor_at_observed_offset() {
        let mut model = RegressionModel::new();
        for variable in [
            RegressionVariable::TradingDay(crate::regarima::TradingDayVariable::FlowWeekdayWeekend),
            RegressionVariable::Holiday(crate::regarima::HolidayVariable::Easter(
                crate::regarima::Easter::new(1).unwrap(),
            )),
        ] {
            model
                .push(RegressionColumn::with_values(variable, vec![0.0, 1.0, 0.0]))
                .unwrap();
        }
        let effects = evaluate_regression_effects(
            &model,
            &[1.4_f64.ln(), 1.5_f64.ln()],
            Transform::Log,
            1..2,
        )
        .unwrap();
        let trading_day = 1.4_f64.ln().exp();
        let holiday = 1.5_f64.ln().exp();
        let combined_then_cancelled = ((1.0 * trading_day) * holiday) / holiday;
        assert_ne!(combined_then_cancelled.to_bits(), trading_day.to_bits());
        let mut values = [100.0];
        effects.remove_final_calendar_and_prior(
            &mut values,
            Transform::Log,
            1..2,
            crate::x11::HolidayAdjustment::Retain,
        );
        assert_eq!(
            values[0].to_bits(),
            (100.0 / combined_then_cancelled).to_bits()
        );
    }

    #[test]
    fn absent_holiday_and_present_zero_holiday_keep_their_presence_paths() {
        for present in [false, true] {
            let mut model = RegressionModel::new();
            model
                .push(RegressionColumn::with_values(
                    RegressionVariable::TradingDay(
                        crate::regarima::TradingDayVariable::FlowWeekdayWeekend,
                    ),
                    vec![0.0],
                ))
                .unwrap();
            if present {
                model
                    .push(RegressionColumn::with_values(
                        RegressionVariable::Holiday(crate::regarima::HolidayVariable::Easter(
                            crate::regarima::Easter::new(1).unwrap(),
                        )),
                        vec![1.0],
                    ))
                    .unwrap();
            }
            let coefficients: &[f64] = if present { &[0.0, 0.0] } else { &[0.0] };
            for transform in [Transform::None, Transform::Log] {
                let effects =
                    evaluate_regression_effects(&model, coefficients, transform, 0..1).unwrap();
                let mut remove = [-0.0];
                let mut retain = [-0.0];
                effects.remove_final_calendar_and_prior(
                    &mut remove,
                    transform,
                    0..1,
                    crate::x11::HolidayAdjustment::Remove,
                );
                effects.remove_final_calendar_and_prior(
                    &mut retain,
                    transform,
                    0..1,
                    crate::x11::HolidayAdjustment::Retain,
                );
                assert_eq!(
                    remove[0].to_bits(),
                    retain[0].to_bits(),
                    "{present}/{transform:?}"
                );
            }
        }
    }

    #[test]
    fn outlier_effects_are_grouped_by_type_and_removed_from_the_response() {
        let mut model = RegressionModel::new();
        push_outlier(&mut model, OutlierType::Additive, vec![0.0, 1.0, 0.0, 0.0]);
        push_outlier(
            &mut model,
            OutlierType::LevelShift,
            vec![-1.0, -1.0, 0.0, 0.0],
        );
        push_outlier(
            &mut model,
            OutlierType::TemporaryChange,
            vec![0.0, 0.0, 1.0, 0.5],
        );

        let effects =
            evaluate_regression_effects(&model, &[2.0, 3.0, 4.0], Transform::None, 0..4).unwrap();
        assert_eq!(effects.additive_outlier.linear, [0.0, 2.0, 0.0, 0.0]);
        assert_eq!(effects.level_shift.linear, [-3.0, -3.0, 0.0, 0.0]);
        assert_eq!(effects.temporary_change.linear, [0.0, 0.0, 4.0, 2.0]);

        let mut adjusted = vec![100.0; 4];
        effects
            .remove_from_transformed_response(&mut adjusted)
            .unwrap();
        assert_eq!(adjusted, [103.0, 101.0, 96.0, 98.0]);
    }

    #[test]
    fn log_effects_remove_linear_groups_before_a_single_inverse() {
        let mut model = RegressionModel::new();
        push_outlier(&mut model, OutlierType::Additive, vec![0.0, 1.0]);
        push_outlier(&mut model, OutlierType::LevelShift, vec![-1.0, 0.0]);
        push_outlier(&mut model, OutlierType::TemporaryChange, vec![0.0, 1.0]);

        let effects =
            evaluate_regression_effects(&model, &[0.2, 0.3, 0.4], Transform::Log, 0..2).unwrap();
        assert_eq!(effects.additive_outlier.linear, [0.0, 0.2]);
        assert_eq!(effects.level_shift.linear, [-0.3, 0.0]);
        assert_eq!(effects.temporary_change.linear, [0.0, 0.4]);

        let mut transformed = vec![10.0, 10.0];
        effects
            .remove_from_transformed_response(&mut transformed)
            .unwrap();
        assert_eq!(transformed, [10.3, 9.4]);
        assert_eq!(
            transformed
                .into_iter()
                .map(|value| crate::transform::inverse(Transform::Log, value))
                .collect::<Vec<_>>(),
            [10.3_f64.exp(), 9.4_f64.exp()]
        );
    }

    fn push_outlier(model: &mut RegressionModel, kind: OutlierType, values: Vec<f64>) {
        let period = TimeAxis::monthly(2000, 1).unwrap().period_at(1).unwrap();
        model
            .push(RegressionColumn::with_values(
                RegressionVariable::Outlier { kind, period },
                values,
            ))
            .unwrap();
    }

    fn empty_model() -> ArimaModel {
        let none = ArimaOrder::new(0, 0, 0).unwrap();
        ArimaModel::new(none, none).unwrap()
    }
}
