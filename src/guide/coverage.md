Which X-13ARIMA-SEATS capabilities this crate covers, and how far each is
verified.

Each table below covers one area of Census X-13ARIMA-SEATS Version 1.1 Build 62,
the reference program, in the order of its manual, with the spec arguments in
parentheses. **Verified** means the API has it and its results are compared
with the reference program's; the source repository lists the configurations
compared. **Supported** means the API has it but none of those configurations
uses it, so its results have not been compared. **Default only** means it
cannot be set and the X-13ARIMA-SEATS default applies; **Not supported** means
the API cannot express it or returns an
[`Unsupported`](crate::ErrorKind::Unsupported) error; **Not applicable** marks
file, report and program options that a library call replaces.

# Input series

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Monthly series (`period = 12`) | Verified | [`Period::monthly`] |
| Quarterly series (`period = 4`) | Verified | [`Period::quarterly`]; monthly data is not aggregated into quarters |
| Other periods (`period` up to 12) | Not supported | X-11 needs 4 or 12 in X-13ARIMA-SEATS too |
| Start date (`start`) | Verified | The [`Series`] passed to [`adjust`] dates the first value |
| Series length | Verified | X-11 needs three years and SEATS 36 months or 16 quarters; see [`Error::SeriesTooLong`] and [`Error::SeatsTooLong`] for the upper limits |
| Model span (`modelspan`) | Verified | [`RegArimaSpec::with_model_span`] with a [`PeriodSpan`]; a later start cancels backcasts |
| Analysis span (`span`) | Not applicable | Pass only the values to analyze, in a [`Series`] that starts at the first one |
| Missing observations | Verified | NaN; regARIMA requests estimate them, direct X-11 rejects them ([`Error::MissingValue`]) |
| Missing-value code (`missingcode`, `missingval`) | Not applicable | NaN is the only marker; `-99999.0` is an ordinary value, so convert such codes first |
| Data entry and formats (`data`, `file`, `format`, `name`, `title`, `precision`, `decimals`, `divpower`, `trimzero`, `yr2000`) | Not applicable | Values arrive as a slice of `f64` |
| Composite and indirect adjustment (`composite`, `comptype`, `compwt`) | Not supported | |

# Transformation

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| No transformation or logarithms (`function = none`, `function = log`) | Verified | [`Transform`] via [`RegArimaSpec::with_transform`]; logs need positive values |
| Automatic choice between none and logs (`function = auto`) | Verified | [`TransformSelection::automatic`] via [`RegArimaSpec::with_transform`]; result in [`Adjustment::transform_selection`] |
| AICC difference for the automatic choice (`aicdiff`) | Supported | [`AutoTransformOptions::with_aic_difference`]; only the default of -2 is verified |
| Other Box-Cox powers, square root, inverse and logistic (`power`, `function`) | Not supported | |
| Prior adjustment for length of period or leap year (`adjust`) | Not supported | Length and leap-year effects are available as regressors ([`CalendarBasis`]) |
| User prior-adjustment factors (`data`, `file`, `type`, `mode`) | Not supported | |

# Regression

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Constant (`const`) | Verified | [`RegArimaSpec::with_constant`]; fixed models only, as the automatic search decides its own |
| Fixed seasonal effects (`seasonal`, `sincos`) | Not supported | |
| Six-coefficient trading day (`td`, `tdnolpyear`) | Verified | [`FlowTradingDay`] with [`TradingDayContrast::SixCoefficient`]; [`FlowLengthTreatment`] sets the leap-year treatment |
| Six-coefficient trading day with length of month or quarter (`td` with `lom` or `loq`) | Verified | [`FlowLengthTreatment::LengthOfPeriod`] |
| One-coefficient trading day (`td1coef`) | Verified | [`TradingDayContrast::WeekdayWeekend`] |
| One-coefficient trading day without leap year (`td1nolpyear`), or with length of period | Supported | [`TradingDayContrast::WeekdayWeekend`] with [`FlowLengthTreatment::NoLengthEffect`] or [`FlowLengthTreatment::LengthOfPeriod`] |
| Leap year alone (`lpyear`) | Verified | [`CalendarBasis::Length`] with [`LengthEffect::LeapYear`] |
| Length of month or quarter alone (`lom`, `loq`) | Verified | [`CalendarBasis::Length`] with [`LengthEffect::LengthOfPeriod`] |
| Stock trading day (`tdstock`, `tdstock1coef`) | Verified | [`StockTradingDay`]; monthly only; sample day 1 to 31 |
| Stock length of month (`lomstock`) | Not supported | |
| Easter (`easter`) | Verified | [`Easter::new`]; window 0 to 25; years 1901 to 2100 |
| Statistics Canada and stock Easter (`sceaster`, `easterstock`) | Not supported | |
| Labor Day (`labor`) | Verified | [`LaborDay::new`]; monthly only; window 1 to 25 |
| Thanksgiving (`thank`) | Verified | [`Thanksgiving::new`]; monthly only; offset -8 to -1 or 1 to 17 |
| Outliers at known dates (`ao`, `ls`, `tc`) | Verified | [`ManualOutlier`] via [`RegArimaSpec::with_manual_outliers`]; needs a fixed model and transform and no outlier search |
| Other intervention regressors (`so`, `rp`, `tl`, `qi`, `qd`, `aos`, `lss`) | Not supported | A ramp or other fixed shape can be supplied as a user regressor |
| Change-of-regime regressors | Not supported | |
| User regressors (`user`, with `usertype` td, holiday or user) | Verified | [`UserRegressor`] with [`UserRegressorType`] via [`RegArimaSpec::with_user_regressors`]; values must cover forecasts and backcasts |
| Other user regressor types (`usertype` constant, seasonal, lom, loq, lpyear, holiday2 to holiday5, ao, ls, so, transitory) | Not supported | |
| Centering user regressors (`centeruser`) | Default only | Values are used as supplied |
| Initial or fixed coefficients (`b`) | Not supported | |
| AICC test of calendar effects (`aictest`) | Verified | [`CalendarSelection::AicTest`]; trading day and Easter only, with a fixed model and transform and no outlier search |
| AICC threshold for that test (`aicdiff`) | Default only | Zero |
| Other significance tests (`pvaictest`, `testalleaster`, `chi2test`, `chi2testcv`, `tlimit`) | Default only | |
| Easter means and unapplied effects (`eastermeans`, `noapply`) | Default only | |
| Temporary changes allocated to the trend (`trendtc`) | Verified | [`x11::Options::with_temporary_change_allocation`]; X-11 only |

# ARIMA model

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Seasonal ARIMA orders (`model`) | Verified | [`ArimaModel::new`] from [`ArimaOrder::new`]; regular orders up to (3 2 3), seasonal up to (2 1 2) |
| Larger orders and models with missing lags | Not supported | [`Error::InvalidSpec`] |
| Initial or fixed coefficients (`ar`, `ma`) | Not supported | |
| Model and forecasts without a seasonal adjustment | Not supported | Every request runs X-11 or SEATS |

# Automatic model selection

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Automatic model search (`automdl`) | Verified | [`RegArimaSpec::automatic_model`]; it also decides the constant; result in [`Adjustment::model_selection`] |
| Search controls (`maxorder`, `maxdiff`, `diff`, `ljungboxlimit`, `reducecv`, `acceptdefault`, `noautooutlier` and the other `automdl` arguments) | Default only | |

# Model picking

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Choice from a list of candidate models (`pickmdl`) | Not supported | Use [`RegArimaSpec::automatic_model`] or a fixed model |

# Estimation

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Maximum likelihood estimates of the coefficients | Verified | [`ModelResult::parameters`] and [`ModelResult::regression_coefficients`] |
| Likelihood form (`exact`) | Default only | Exact ARMA likelihood; conditional likelihood (`ma`, `none`) is not available |
| Convergence controls (`maxiter`, `maxnliter`, `tol`, `nltol`) | Default only | Iteration counts in [`ModelResult::convergence`] |
| Model read from a file (`file`, `fix`) | Not supported | |
| Other options (`outofsample`, `removeconstant`) | Default only | |
| Standard errors and covariances | Supported | [`ParameterEstimate::standard_error`], [`ModelResult::arma_covariance`], [`ModelResult::regression_covariance`] |
| Likelihood and information criteria (AIC, AICC, Hannan-Quinn, BIC) | Supported | [`ModelResult::log_likelihood`], [`ModelResult::information_criteria`] |
| Residuals and innovation variance | Supported | [`ModelResult::residuals`], [`ModelResult::innovation_variance`] |

# Outliers

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Automatic search for AO, LS and TC outliers (`types`) | Verified | [`RegArimaSpec::with_outliers`] with [`OutlierSpec::with_types`]; AO and LS by default; results in [`Adjustment::outliers`] |
| Search method (`method` addone or addall) | Verified | [`OutlierSpec::with_method`] |
| Critical values (`critical`, shared or per type) | Verified | [`OutlierSpec::with_critical_values`] with [`OutlierCriticalValues`]; the default depends on the series length |
| Search span (`span`) | Verified | [`OutlierSpec::with_span`] |
| Temporary-change decay rate (`tcrate`) | Verified | [`RegArimaSpec::with_temporary_change_rate`]; one rate for automatic and specified TCs |
| Critical-value controls (`criticalalpha`, `defaultcritical`, `almost`) | Default only | |
| Temporary level-shift tests (`lsrun`) | Not supported | |

# Forecasts

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Forecasts (`maxlead`) | Verified | [`ForecastSpec::new`] with [`ForecastHorizon`] via [`RegArimaSpec::with_forecast`]; one year by default, at most 120; [`Adjustment::forecasts`] |
| Backcasts (`maxback`) | Verified | [`ForecastSpec::new`]; none by default, at most 120; [`Adjustment::backcasts`]; SEATS ignores them |
| Prediction intervals and forecast standard errors | Supported | [`Forecasts::lower_95`], [`Forecasts::upper_95`], [`Forecasts::transformed_standard_errors`] |
| Interval coverage (`probability`) | Default only | 95 percent |
| Bias correction of log forecasts (`lognormal`) | Default only | No correction |
| Forecast origin before the end of the series (`exclude`) | Default only | Forecasts start after the last observation |

# Residual checks

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Residual ACF, PACF and Q statistics (`maxlag`, `qtype`, `qlimit`, `acflimit`) | Not supported | The residuals are in [`ModelResult::residuals`] |
| Normality tests and residual histogram | Not supported | |

# Model identification

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| ACF and PACF of the differenced series (`identify`) | Not supported | |

# X-11

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Decomposition mode (`mode` mult, add, pseudoadd or logadd) | Verified | [`x11::Options::with_mode`] with [`x11::Mode`]; with none set, [`Spec::x11_regarima`] follows the transform |
| One seasonal filter for every period (`seasonalma`) | Verified | [`x11::Options::with_seasonal_filter`] with [`x11::SeasonalFilter`]; MSR choice in [`x11::Diagnostics::msr_selection`] |
| A seasonal filter per period (`seasonalma` list) | Verified | [`x11::Options::with_seasonal_filters`] with [`x11::SeasonalFilters`] |
| Filters for short series (`sfshort`) | Verified | [`x11::Options::with_short_series_filters`] |
| Henderson trend filter (`trendma`) | Verified | [`x11::Options::with_trend_filter`] with [`x11::HendersonLength`]; odd, 3 to 101; length used in [`x11::Diagnostics::final_henderson_terms`] |
| I/C ratio for the trend filter (`trendic`) | Verified | [`x11::TrendFilter::HendersonWithIcRatio`] with [`x11::HendersonIcRatio`] |
| Seven-term trend endpoints (`true7term`) | Verified | [`x11::Options::with_seven_term_endpoints`] |
| Sigma limits for extreme values (`sigmalim`) | Verified | [`x11::Options::with_sigma_limits`] with [`x11::SigmaLimits`] |
| Extreme-value scale by calendar period (`calendarsigma`, `sigmavec`) | Verified | [`x11::Options::with_extreme_scale`] with [`x11::ExtremeScale`]; choice in [`x11::Diagnostics::calendar_sigma_selection`] |
| Extreme-value estimator (`extremeadj`) | Verified | [`x11::Options::with_extreme_estimator`] with [`x11::ExtremeEstimator`] |
| Extreme values from observed data only (`excludefcst`) | Verified | [`x11::Options::with_extreme_estimation_span`] |
| Trend-only decomposition (`type = trend`) | Verified | [`x11::Options::with_decomposition`] with [`x11::Decomposition::TrendOnly`] |
| Summary decomposition (`type = summary`) | Not supported | |
| Seasonal shrinkage (`shrink`) | Verified | [`x11::Options::with_seasonal_shrinkage`] |
| X-11 Easter estimate (`x11easter`) | Verified | [`x11::Options::with_easter_estimation`]; factors in [`Adjustment::x11_easter`]; not with a fitted Easter regressor or an automatic transform |
| Holiday effects kept in the adjusted series (`keepholiday`) | Verified | [`x11::Options::with_holiday_adjustment`] |
| Outliers removed from the final adjustment (`final` ao, ls or tc) | Verified | [`x11::Options::with_final_outliers`] with [`x11::FinalOutliers`] |
| User regressors removed from the final adjustment (`final = user`) | Not supported | |
| Centered seasonal factors (`centerseasonal`) | Default only | Not centered |
| Spectral taper (`taper`) | Default only | |
| Quality measures M1 to M11, Q and Q2 | Supported | [`x11::Diagnostics::quality`] |
| Seasonality tests (stable, moving, Kruskal-Wallis, combined, residual) | Supported | [`x11::Diagnostics::seasonality`] |

# X-11 regression

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Trading-day and holiday regression on the irregular, with its prior weights and outlier search (`x11regression`) | Not supported | Estimate calendar effects in the regARIMA model instead |

# SEATS

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| SEATS decomposition (`seats`) | Verified | [`Spec::seats`]; regular orders up to (3 2 2), seasonal up to (1 1 1) |
| Transitory component | Verified | [`Adjustment::seats_transitory`] |
| Model revised by SEATS | Verified | [`seats::Diagnostics::model`]; a revision this crate does not follow is an error of kind [`Unsupported`](crate::ErrorKind::Unsupported) |
| Mean, refit and parameter-bound flags | Supported | [`seats::Diagnostics`] |
| Tuning options (`noadmiss`, `qmax`, `rmod`, `xl`, `epsphi`, `epsiv`, `maxit`, `imean`, `bias`, `statseas`, `finite`) | Default only | |
| Hodrick-Prescott long-term trend and cycle (`hpcycle`, `hplan`, `hptarget`, `hprmls`) | Not supported | X-13ARIMA-SEATS computes them by default; this crate does not return them |
| Standard errors of components and revisions | Not supported | |
| Combined adjustment factors (table S16) | Not supported | |

# Benchmarking

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Forcing yearly totals of the adjusted series (`force`) | Not supported | |

# Sliding spans

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Stability analysis over overlapping spans (`slidingspans`) | Not supported | |

# Revision history

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Revisions of adjustments, trends, forecasts and model estimates (`history`) | Not supported | |

# Spectrum

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Spectral peaks of the X-11 series | Supported | [`x11::Diagnostics::spectral`]; monthly only, as in X-13ARIMA-SEATS |
| Spectral peaks of the regARIMA residuals | Supported | [`ModelResult::residual_spectral`]; monthly only, as in X-13ARIMA-SEATS |
| Spectra of a SEATS adjustment | Not supported | |
| Spectrum settings (`start`, `type`, `series`, `difference`, `siglevel`, `peakwidth`, `maxar`) | Default only | |
| QS seasonality statistics | Not supported | |
| Tukey spectrum peaks (`tukey120`) | Not supported | |

# Metadata

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Keys and values written to the output files (`metadata`) | Not applicable | |

# Output and saved tables

| X-13ARIMA-SEATS capability | Status | Notes |
| --- | --- | --- |
| Final components (X-11 tables D10 to D13, SEATS tables S10 to S13) | Verified | [`Adjustment::seasonally_adjusted`], [`Adjustment::seasonal`], [`Adjustment::trend_cycle`], [`Adjustment::irregular`] |
| Combined calendar effects | Supported | [`Adjustment::calendar_effects`] |
| Warnings | Supported | [`Adjustment::warnings`]; not compared with the reference program's messages |
| Printed tables and print levels (`print`, `savelog`, `title`, `print1stpass`) | Not applicable | Results are values in the [`Adjustment`] |
| Saved tables (`save`, `saveprecision`) | Not applicable | |
| Forecasts appended to tables (`appendfcst`, `appendbcst`) | Not applicable | [`Adjustment::forecasts`] and [`Adjustment::backcasts`] return them separately |
| Graphics files and batch runs over many series | Not applicable | Call [`adjust`] once per series |

# Coverage limits

- **Frequencies.** Only monthly and quarterly series are accepted.
  X-13ARIMA-SEATS accepts other periods up to 12.
- **Platform.** Results are checked against the reference program on
  aarch64-darwin (Apple silicon macOS) only. They use the platform libm, so on
  other platforms they can differ in the last bits, and a discrete choice that
  falls close to its threshold can go the other way.
- **Defaults.** Two defaults differ from those of X-13ARIMA-SEATS:
  - under SEATS, X-13ARIMA-SEATS forecasts three years, while [`RegArimaSpec`]
    defaults to one; request three years with [`ForecastSpec::new`] to match;
  - with no transform and no mode set, [`Spec::x11_regarima`] runs additive
    X-11, where the X-13ARIMA-SEATS default is multiplicative; set the mode
    with [`x11::Options::with_mode`] to match.

[`Adjustment`]: crate::Adjustment
[`Adjustment::backcasts`]: crate::Adjustment::backcasts
[`Adjustment::calendar_effects`]: crate::Adjustment::calendar_effects
[`Adjustment::forecasts`]: crate::Adjustment::forecasts
[`Adjustment::irregular`]: crate::Adjustment::irregular
[`Adjustment::model_selection`]: crate::Adjustment::model_selection
[`Adjustment::outliers`]: crate::Adjustment::outliers
[`Adjustment::seasonal`]: crate::Adjustment::seasonal
[`Adjustment::seasonally_adjusted`]: crate::Adjustment::seasonally_adjusted
[`Adjustment::seats_transitory`]: crate::Adjustment::seats_transitory
[`Adjustment::transform_selection`]: crate::Adjustment::transform_selection
[`Adjustment::trend_cycle`]: crate::Adjustment::trend_cycle
[`Adjustment::warnings`]: crate::Adjustment::warnings
[`Adjustment::x11_easter`]: crate::Adjustment::x11_easter
[`adjust`]: crate::adjust
[`ArimaModel::new`]: crate::ArimaModel::new
[`ArimaOrder::new`]: crate::ArimaOrder::new
[`AutoTransformOptions::with_aic_difference`]: crate::regarima::AutoTransformOptions::with_aic_difference
[`CalendarBasis`]: crate::regarima::CalendarBasis
[`CalendarBasis::Length`]: crate::regarima::CalendarBasis::Length
[`CalendarSelection::AicTest`]: crate::regarima::CalendarSelection::AicTest
[`Easter::new`]: crate::regarima::Easter::new
[`Error::InvalidSpec`]: crate::Error::InvalidSpec
[`Error::MissingValue`]: crate::Error::MissingValue
[`Error::SeatsTooLong`]: crate::Error::SeatsTooLong
[`Error::SeriesTooLong`]: crate::Error::SeriesTooLong
[`FlowLengthTreatment`]: crate::regarima::FlowLengthTreatment
[`FlowLengthTreatment::LengthOfPeriod`]: crate::regarima::FlowLengthTreatment::LengthOfPeriod
[`FlowLengthTreatment::NoLengthEffect`]: crate::regarima::FlowLengthTreatment::NoLengthEffect
[`FlowTradingDay`]: crate::regarima::FlowTradingDay
[`ForecastHorizon`]: crate::regarima::ForecastHorizon
[`ForecastSpec::new`]: crate::regarima::ForecastSpec::new
[`Forecasts::lower_95`]: crate::Forecasts::lower_95
[`Forecasts::transformed_standard_errors`]: crate::Forecasts::transformed_standard_errors
[`Forecasts::upper_95`]: crate::Forecasts::upper_95
[`x11::HendersonIcRatio`]: crate::x11::HendersonIcRatio
[`x11::HendersonLength`]: crate::x11::HendersonLength
[`LaborDay::new`]: crate::regarima::LaborDay::new
[`LengthEffect::LeapYear`]: crate::regarima::LengthEffect::LeapYear
[`LengthEffect::LengthOfPeriod`]: crate::regarima::LengthEffect::LengthOfPeriod
[`ManualOutlier`]: crate::regarima::ManualOutlier
[`ModelResult::arma_covariance`]: crate::ModelResult::arma_covariance
[`ModelResult::convergence`]: crate::ModelResult::convergence
[`ModelResult::information_criteria`]: crate::ModelResult::information_criteria
[`ModelResult::innovation_variance`]: crate::ModelResult::innovation_variance
[`ModelResult::log_likelihood`]: crate::ModelResult::log_likelihood
[`ModelResult::parameters`]: crate::ModelResult::parameters
[`ModelResult::regression_coefficients`]: crate::ModelResult::regression_coefficients
[`ModelResult::regression_covariance`]: crate::ModelResult::regression_covariance
[`ModelResult::residual_spectral`]: crate::ModelResult::residual_spectral
[`ModelResult::residuals`]: crate::ModelResult::residuals
[`OutlierCriticalValues`]: crate::regarima::OutlierCriticalValues
[`OutlierSpec::with_critical_values`]: crate::regarima::OutlierSpec::with_critical_values
[`OutlierSpec::with_method`]: crate::regarima::OutlierSpec::with_method
[`OutlierSpec::with_span`]: crate::regarima::OutlierSpec::with_span
[`OutlierSpec::with_types`]: crate::regarima::OutlierSpec::with_types
[`ParameterEstimate::standard_error`]: crate::regarima::ParameterEstimate::standard_error
[`Period::monthly`]: crate::Period::monthly
[`Period::quarterly`]: crate::Period::quarterly
[`PeriodSpan`]: crate::PeriodSpan
[`RegArimaSpec`]: crate::RegArimaSpec
[`RegArimaSpec::automatic_model`]: crate::RegArimaSpec::automatic_model
[`RegArimaSpec::with_constant`]: crate::RegArimaSpec::with_constant
[`RegArimaSpec::with_forecast`]: crate::RegArimaSpec::with_forecast
[`RegArimaSpec::with_manual_outliers`]: crate::RegArimaSpec::with_manual_outliers
[`RegArimaSpec::with_model_span`]: crate::RegArimaSpec::with_model_span
[`RegArimaSpec::with_outliers`]: crate::RegArimaSpec::with_outliers
[`RegArimaSpec::with_temporary_change_rate`]: crate::RegArimaSpec::with_temporary_change_rate
[`RegArimaSpec::with_transform`]: crate::RegArimaSpec::with_transform
[`RegArimaSpec::with_user_regressors`]: crate::RegArimaSpec::with_user_regressors
[`seats::Diagnostics`]: crate::seats::Diagnostics
[`seats::Diagnostics::model`]: crate::seats::Diagnostics::model
[`Series`]: crate::Series
[`x11::SigmaLimits`]: crate::x11::SigmaLimits
[`Spec::seats`]: crate::Spec::seats
[`Spec::x11_regarima`]: crate::Spec::x11_regarima
[`StockTradingDay`]: crate::regarima::StockTradingDay
[`Thanksgiving::new`]: crate::regarima::Thanksgiving::new
[`TradingDayContrast::SixCoefficient`]: crate::regarima::TradingDayContrast::SixCoefficient
[`TradingDayContrast::WeekdayWeekend`]: crate::regarima::TradingDayContrast::WeekdayWeekend
[`Transform`]: crate::Transform
[`TransformSelection::automatic`]: crate::regarima::TransformSelection::automatic
[`UserRegressor`]: crate::regarima::UserRegressor
[`UserRegressorType`]: crate::regarima::UserRegressorType
[`x11::Decomposition::TrendOnly`]: crate::x11::Decomposition::TrendOnly
[`x11::Diagnostics::calendar_sigma_selection`]: crate::x11::Diagnostics::calendar_sigma_selection
[`x11::Diagnostics::final_henderson_terms`]: crate::x11::Diagnostics::final_henderson_terms
[`x11::Diagnostics::msr_selection`]: crate::x11::Diagnostics::msr_selection
[`x11::Diagnostics::quality`]: crate::x11::Diagnostics::quality
[`x11::Diagnostics::seasonality`]: crate::x11::Diagnostics::seasonality
[`x11::Diagnostics::spectral`]: crate::x11::Diagnostics::spectral
[`x11::ExtremeEstimator`]: crate::x11::ExtremeEstimator
[`x11::ExtremeScale`]: crate::x11::ExtremeScale
[`x11::FinalOutliers`]: crate::x11::FinalOutliers
[`x11::Mode`]: crate::x11::Mode
[`x11::Options::with_decomposition`]: crate::x11::Options::with_decomposition
[`x11::Options::with_easter_estimation`]: crate::x11::Options::with_easter_estimation
[`x11::Options::with_extreme_estimation_span`]: crate::x11::Options::with_extreme_estimation_span
[`x11::Options::with_extreme_estimator`]: crate::x11::Options::with_extreme_estimator
[`x11::Options::with_extreme_scale`]: crate::x11::Options::with_extreme_scale
[`x11::Options::with_final_outliers`]: crate::x11::Options::with_final_outliers
[`x11::Options::with_holiday_adjustment`]: crate::x11::Options::with_holiday_adjustment
[`x11::Options::with_mode`]: crate::x11::Options::with_mode
[`x11::Options::with_seasonal_filter`]: crate::x11::Options::with_seasonal_filter
[`x11::Options::with_seasonal_filters`]: crate::x11::Options::with_seasonal_filters
[`x11::Options::with_seasonal_shrinkage`]: crate::x11::Options::with_seasonal_shrinkage
[`x11::Options::with_seven_term_endpoints`]: crate::x11::Options::with_seven_term_endpoints
[`x11::Options::with_short_series_filters`]: crate::x11::Options::with_short_series_filters
[`x11::Options::with_sigma_limits`]: crate::x11::Options::with_sigma_limits
[`x11::Options::with_temporary_change_allocation`]: crate::x11::Options::with_temporary_change_allocation
[`x11::Options::with_trend_filter`]: crate::x11::Options::with_trend_filter
[`x11::SeasonalFilter`]: crate::x11::SeasonalFilter
[`x11::SeasonalFilters`]: crate::x11::SeasonalFilters
[`x11::TrendFilter::HendersonWithIcRatio`]: crate::x11::TrendFilter::HendersonWithIcRatio
