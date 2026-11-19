//! Renders a [`RegArimaRequest`]: the spec of its X-11 equivalent, then the
//! blocks only a request can ask for.

use std::ops::Range;

use crate::{Method, OracleConfiguration, OracleError, RegArimaRequest, Series, user_regressors};

/// Whether the case runs Build 62 with `-s`.
pub(crate) fn uses_summary(configuration: &OracleConfiguration) -> bool {
    // Direct X-11 reads only saved components, and `-s` makes prtd8b fill a
    // fixed-size buffer of extreme dates that long series overflow.
    if matches!(configuration, OracleConfiguration::DirectX11 { .. }) {
        return false;
    }
    // arima.f's Lsumm changes the Q statistic passed to SEATS (ansub9.f), so
    // this has to agree with what the Rust SEATS pipeline assumes Build 62
    // was run with.
    !matches!(configuration, OracleConfiguration::RegArima { method: Method::Seats, request }
        if request.model.is_some() && request.outliers.is_none())
}

pub(super) fn render(
    series: &Series,
    method: Method,
    request: RegArimaRequest,
) -> Result<String, OracleError> {
    let mut spec = super::render(series, &request.x11_equivalent()?)?;
    if let Some(supplied) = request.user_regressors {
        let columns = supplied.columns(series)?;
        let fields = user_regressors::render(series, &columns)?;
        if spec.contains("regression {") {
            spec = spec.replacen("regression {", &format!("regression {{\n{fields}"), 1);
        } else {
            spec = spec.replacen(
                "arima {",
                &format!("regression {{\n{fields}  save = regressionmatrix\n}}\narima {{"),
                1,
            );
        }
    }
    if let Some(policy) = request.model_span {
        let dates = span_dates(series, policy.range(series)?);
        spec = spec.replacen(
            "series {\n",
            &format!("series {{\n  modelspan = ({dates})\n"),
            1,
        );
    }
    if let Some(policy) = request.outlier_span {
        if request.outliers.is_none() {
            return Err(invalid("outlier span requires automatic outlier detection"));
        }
        let range = policy.range(series)?;
        if let Some(model_policy) = request.model_span {
            let model = model_policy.range(series)?;
            if range.start < model.start || range.end > model.end {
                return Err(invalid("outlier span must be within the model span"));
            }
        }
        let dates = span_dates(series, range);
        spec = spec.replacen(
            "outlier {\n",
            &format!("outlier {{\n  span = ({dates})\n"),
            1,
        );
    }
    if request.aic_test {
        if !spec.contains("regression {\n") {
            return Err(invalid("aictest requires calendar regressors"));
        }
        spec = spec.replacen(
            "regression {\n",
            "regression {\n  aictest = (td easter)\n",
            1,
        );
    }
    if request.manual_outliers {
        let n = series.values.len();
        if n < 9 {
            return Err(invalid(
                "manual outlier positions require nine observations",
            ));
        }
        let s = usize::from(series.frequency.periods_per_year());
        let date = |index: usize| {
            let offset = usize::from(series.start.subperiod) - 1 + index;
            format!(
                "{}.{}",
                i64::from(series.start.year) + (offset / s) as i64,
                offset % s + 1
            )
        };
        // The call table gives these positions as one-based observation numbers.
        let variables = format!(
            "ao{} ls{} tc{}",
            date(n / 3 - 1),
            date(2 * n / 3 - 1),
            date(n - 7)
        );
        if spec.contains("regression {") {
            return Err(invalid(
                "manual outliers cannot overwrite configured regression",
            ));
        }
        let outlier_rate = request
            .outliers
            .is_some_and(|outliers| outliers.temporary_change_rate().is_some());
        if request.manual_temporary_change_rate.is_some() && outlier_rate {
            // getreg.f and gtotlr.f: "Cannot specify tcrate in both the
            // regression and outlier specs".
            return Err(invalid(
                "tcrate cannot be set in both the regression and outlier specs",
            ));
        }
        // getreg.f reads `tcrate` into the one global Tcalfa that every TC
        // column uses (addotl.f), so the rate belongs to the regression block.
        let rate = request
            .manual_temporary_change_rate
            .map_or_else(String::new, |rate| format!(" tcrate = {}", rate.value()));
        spec = spec.replacen(
            "arima {",
            &format!("regression {{ variables = ({variables}){rate} }}\narima {{"),
            1,
        );
    } else if request.manual_temporary_change_rate.is_some() {
        return Err(invalid("a manual TC rate requires manual outliers"));
    }
    if method == Method::Seats {
        let start = spec
            .find("\nx11 {\n")
            .ok_or_else(|| invalid("missing final X11 block"))?;
        spec.truncate(start);
        // `savelog = seatsmodel` logs the model SEATS decomposed with (seatdg.f
        // LSLSMD), which the harness compares. It writes only to the log.
        spec.push_str("\nseats { save = (s10 s11 s12 s13 s14) savelog = seatsmodel }\n");
        if !uses_summary(&OracleConfiguration::RegArima { method, request }) {
            // Without -s there is no UDG to read the model choices from: save
            // the original orders and print the resolved transformation
            // (prtnfn.f), which no saved table holds.
            spec = spec.replacen("estimate {\n", "estimate {\n  print = model\n", 1);
            spec = spec.replacen("save = (estimates", "save = (model estimates", 1);
        }
    }
    Ok(spec)
}

fn span_dates(series: &Series, range: Range<usize>) -> String {
    let frequency = usize::from(series.frequency.periods_per_year());
    let date = |index| {
        let offset = usize::from(series.start.subperiod) - 1 + index;
        format!(
            "{}.{}",
            i64::from(series.start.year) + (offset / frequency) as i64,
            offset % frequency + 1
        )
    };
    // getsrs.f and gtotlr.f count both endpoint dates (dfdate + 1).
    format!("{}, {}", date(range.start), date(range.end - 1))
}

fn invalid(message: &str) -> OracleError {
    OracleError::InvalidConfiguration(message.into())
}

#[cfg(test)]
mod tests {
    use crate::{
        Frequency, OracleConfiguration, Period, Series,
        configuration::regarima_request::{ModelSpanPolicy, OutlierSpanPolicy},
        spec_file,
    };

    fn request(method: &str, extra: serde_json::Value) -> OracleConfiguration {
        let mut request = serde_json::json!({
            "model":{"regular":{"p":0,"d":1,"q":1},"seasonal":{"p":0,"d":1,"q":1}},
            "transform":{"kind":"explicit","transform":"log"},
            "forecast_horizon":{"kind":"one_year"}, "backcast_horizon":0
        });
        for (key, value) in extra.as_object().unwrap() {
            request[key] = value.clone();
        }
        serde_json::from_value(serde_json::json!({
            "kind":"reg_arima", "method":method, "request":request
        }))
        .unwrap()
    }

    #[test]
    fn user_regressors_cover_their_future_years_not_the_forecast_horizon() {
        for frequency in [Frequency::Monthly, Frequency::Quarterly] {
            for (method, future_years) in [("x11", 1), ("seats", 3)] {
                let periods = usize::from(frequency.periods_per_year());
                let series = Series {
                    series_id: "supplied-coverage".into(),
                    frequency,
                    start: Period::new(2000, 2),
                    values: vec![100.0; periods * 10],
                };
                let config = request(
                    method,
                    serde_json::json!({
                        "user_regressors":{"kind":"working_days", "future_years":future_years}
                    }),
                );
                let rendered = spec_file::render(&series, &config).unwrap();
                assert!(rendered.contains("user = (user1)\n  usertype = (td)\n  start = 2000.2"));
                assert!(rendered.contains(&format!("maxlead = {periods}\n")));
                let matrix = rendered.split_once("  data = (\n").unwrap().1;
                let matrix = matrix.split_once("  )\n").unwrap().0;
                assert_eq!(
                    matrix.split_whitespace().count(),
                    periods * (10 + future_years)
                );
                assert_eq!(rendered.matches("regression {").count(), 1);
                assert!(rendered.contains(&format!("\n{method} {{")));
            }
        }
    }

    #[test]
    fn span_policies_render_inclusive_dates() {
        for (frequency, start, count, model_end, model_start, outlier_end, series_end) in [
            (
                Frequency::Monthly,
                Period::new(2000, 7),
                145,
                "2009.7",
                "2004.7",
                "2008.6",
                "2012.7",
            ),
            (
                Frequency::Quarterly,
                Period::new(2000, 2),
                41,
                "2007.2",
                "2003.3",
                "2006.4",
                "2010.2",
            ),
        ] {
            let series = Series {
                series_id: "span-test".into(),
                frequency,
                start,
                values: vec![100.; count],
            };
            let config = request(
                "x11",
                serde_json::json!({
                    "outliers":{"types":["additive","level_shift","temporary_change"],
                        "method":"add_one","critical_values":{"kind":"automatic"}}
                }),
            );
            let OracleConfiguration::RegArima {
                method,
                mut request,
            } = config
            else {
                unreachable!();
            };
            request.model_span = Some(ModelSpanPolicy::ExcludeLastThreeYears);
            let rendered = super::render(&series, method, request).unwrap();
            assert!(rendered.contains(&format!(
                "modelspan = ({}.{}, {model_end})",
                start.year, start.subperiod
            )));
            request.model_span = Some(ModelSpanPolicy::ExcludeFirstThird);
            let rendered = super::render(&series, method, request).unwrap();
            assert!(rendered.contains(&format!("modelspan = ({model_start}, {series_end})")));
            request.model_span = None;
            request.outlier_span = Some(OutlierSpanPolicy::MiddleThird);
            let rendered = super::render(&series, method, request).unwrap();
            assert!(rendered.contains(&format!(
                "outlier {{\n  span = ({model_start}, {outlier_end})"
            )));
            assert!(!rendered.contains("modelspan"));
        }
    }

    #[test]
    fn a_seats_request_renders_a_seats_block_instead_of_x11() {
        let series = Series {
            series_id: "test".into(),
            frequency: Frequency::Monthly,
            start: Period::new(2000, 1),
            values: vec![100.; 144],
        };
        let text = spec_file::render(&series, &request("seats", serde_json::json!({}))).unwrap();
        assert!(text.contains("seats {"));
        assert!(!text.contains("x11 {"));
        assert!(text.contains("model = (0 1 1)(0 1 1)"));
        assert!(text.contains("maxlead = 12"));
    }
}
