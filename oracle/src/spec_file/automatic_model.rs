use super::{
    calendar_variables, render_outlier_settings, render_x11_options, render_x11_print, requests_msr,
};
use crate::{Frequency, OracleAutomaticModelTransform, OracleConfiguration, OracleError, X11Mode};
use std::fmt::Write as _;

pub(super) fn pseudo_additive_no_effects(configuration: &OracleConfiguration) -> bool {
    matches!(configuration, OracleConfiguration::AutomaticModelX11 {
        mode: Some(X11Mode::PseudoAdditive), calendar, outliers: None, ..
    } if calendar.is_none_or(|calendar| calendar.is_empty()))
}

pub(super) fn render_blocks(
    spec: &mut String,
    frequency: Frequency,
    configuration: &OracleConfiguration,
) -> Result<(), OracleError> {
    let OracleConfiguration::AutomaticModelX11 {
        transform,
        mode: requested_mode,
        calendar,
        outliers,
        forecast_horizon,
        backcast_horizon,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
        ..
    } = *configuration
    else {
        unreachable!("automatic-model renderer requires its configuration")
    };

    writeln!(spec, "transform {{").unwrap();
    let inferred_mode = match transform {
        OracleAutomaticModelTransform::Explicit { transform } => {
            writeln!(spec, "  function = {}", transform.spec_value()).unwrap();
            Some(match transform {
                crate::OracleTransform::None => X11Mode::Additive,
                crate::OracleTransform::Log => X11Mode::Multiplicative,
            })
        }
        OracleAutomaticModelTransform::Automatic { options } => {
            writeln!(spec, "  function = auto").unwrap();
            writeln!(spec, "  aicdiff = {}", options.aic_difference()).unwrap();
            writeln!(spec, "  print = aictransform").unwrap();
            None
        }
    };
    writeln!(spec, "  save = transformed").unwrap();
    writeln!(spec, "}}\n").unwrap();
    if pseudo_additive_no_effects(configuration) {
        // A pseudo-additive request with no calendar and no outliers:
        // `noapply` (getreg.f) keeps every regression class out of the X-11
        // prior adjustment, and the regression block stays so automdl can
        // still test the constant.
        writeln!(spec, "regression {{").unwrap();
        writeln!(
            spec,
            "  noapply = (td ao ls holiday userseasonal user tc so)"
        )
        .unwrap();
        super::render_temporary_change_allocation(
            spec,
            configuration.temporary_change_allocation(),
        );
        writeln!(spec, "  save = regressionmatrix").unwrap();
        writeln!(spec, "}}\n").unwrap();
    } else if let Some(calendar) = calendar {
        let variables = calendar_variables(calendar, frequency);
        writeln!(spec, "regression {{").unwrap();
        if variables.len() == 1 {
            writeln!(spec, "  variables = {}", variables[0]).unwrap();
        } else {
            writeln!(spec, "  variables = ({})", variables.join(" ")).unwrap();
        }
        super::render_temporary_change_allocation(
            spec,
            configuration.temporary_change_allocation(),
        );
        writeln!(spec, "  save = regressionmatrix").unwrap();
        writeln!(spec, "}}\n").unwrap();
    } else if configuration.temporary_change_allocation().is_some() {
        writeln!(spec, "regression {{").unwrap();
        super::render_temporary_change_allocation(
            spec,
            configuration.temporary_change_allocation(),
        );
        writeln!(spec, "  save = regressionmatrix").unwrap();
        writeln!(spec, "}}\n").unwrap();
    } else {
        writeln!(spec, "regression {{ save = regressionmatrix }}\n").unwrap();
    }
    writeln!(spec, "automdl {{ print = all savelog = all }}\n").unwrap();
    if let Some(outliers) = outliers {
        writeln!(spec, "outlier {{").unwrap();
        render_outlier_settings(spec, outliers);
        writeln!(spec, "}}\n").unwrap();
    }
    writeln!(spec, "estimate {{").unwrap();
    writeln!(spec, "  exact = arma").unwrap();
    writeln!(spec, "  save = (model estimates roots armacmatrix regcmatrix lkstats regressioneffects regressionresiduals residuals)").unwrap();
    writeln!(spec, "}}\n").unwrap();
    writeln!(spec, "check {{ qtype = ljungbox save = acf }}\n").unwrap();

    let forecast_horizon = forecast_horizon.resolve(frequency)?;
    writeln!(spec, "forecast {{").unwrap();
    writeln!(spec, "  maxlead = {forecast_horizon}").unwrap();
    writeln!(spec, "  maxback = {backcast_horizon}").unwrap();
    writeln!(spec, "  lognormal = no").unwrap();
    let mut saves = Vec::new();
    if forecast_horizon > 0 {
        saves.extend(["transformed", "variances", "forecasts"]);
    }
    if backcast_horizon > 0 {
        saves.extend(["transformedbcst", "backcasts"]);
    }
    if !saves.is_empty() {
        writeln!(spec, "  save = ({})", saves.join(" ")).unwrap();
    }
    writeln!(spec, "}}\n").unwrap();

    writeln!(spec, "x11 {{").unwrap();
    super::render_x11_decomposition(spec, configuration.decomposition());
    super::render_legacy_easter(spec, configuration.legacy_easter());
    if let Some(mode) = requested_mode.or(inferred_mode) {
        writeln!(spec, "  mode = {}", mode.spec_value()).unwrap();
    }
    render_x11_options(
        spec,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
    );
    super::render_x11_final_outliers(spec, configuration.final_outliers());
    super::render_x11_holiday_adjustment(spec, configuration.holiday_adjustment());
    super::render_x11_short_series_filters(spec, configuration.short_series_filters());
    super::render_x11_append_policy(spec);
    render_x11_print(
        spec,
        requests_msr(seasonal_filter, seasonal_filters),
        configuration.requests_automatic_calendar_sigma(),
    );
    super::render_x11_extreme_settings(
        spec,
        configuration.extreme_estimation_span(),
        configuration.extreme_scale(),
        configuration.extreme_estimator(),
        super::X11SavedTables {
            trend_only: configuration.is_trend_only(),
            extreme_weights: configuration.requests_x11_extreme_weights(),
            log_additive: requested_mode == Some(X11Mode::LogAdditive),
            shrinkage: configuration.seasonal_shrinkage(),
        },
    );
    writeln!(spec, "}}").unwrap();
    Ok(())
}
