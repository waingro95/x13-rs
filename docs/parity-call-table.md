# Parity call table

Each row is one call (a pipeline with a transform, model, regression effects or
option settings) that the parity harness runs through Build 62 and through this
crate for every series that fits it. `tests/parity/specs.toml` encodes every
row under the id shown here. Rows 1–94 cover X-11, X-11 with regARIMA and
SEATS. Rows 95–108 add model and outlier spans and supplied regressors. The
robustness rows, whose ids start with `U-`, cover options, model families and
inputs the numbered rows leave out.

A row passes when every series it applies to passes. The corpus is the FRED
series listed in `tests/parity/series.toml` (344 monthly and 161 quarterly),
plus the synthetic `SYN_` series that rows U-14 to U-21b name
(`tests/parity/synthetic.toml`).

## How to read the rows

**Baselines.** Each row varies one thing from its section's baseline unless it
says "combo":
- X-11 rows use multiplicative mode, an S3x5 seasonal filter, automatic
  Henderson and sigma limits 1.5/2.5.
- X-11 + regARIMA rows use log transform, the airline model
  (0,1,1)(0,1,1), multiplicative X-11 and one year of forecasts.
- SEATS rows use the airline model, log transform and one year of forecasts.

The harness gives both programs exactly the settings a row specifies. Where a
row leaves the X-11 seasonal filter unset, neither program receives one, and
both use their default, MSR selection.

**Data restrictions.** "pos" means only strictly positive series count. "≥0"
means nonnegative series. "—" means the call doesn't exist for that frequency.

### Outputs compared

A series passes a row only if **all** of these match:

| Row type | Outputs compared |
|---|---|
| X-11 | D10 seasonal factors, D11 seasonally adjusted, D12 trend, D13 irregular |
| X-11 + regARIMA | the X-11 outputs, plus chosen transform, model orders, ARMA and regression coefficients, outliers (type and date), point forecasts |
| SEATS | S10 seasonal, S11 seasonally adjusted, S12 trend, S13 irregular, S14 transitory (where present), plus the regARIMA answers above |

Two further comparisons use the same tolerances:
- point backcasts, on every row that requests them;
- on SEATS rows, the final SEATS model orders and ARMA coefficients that
  Build 62 logs with `savelog = seatsmodel`, including whether a SEATS model
  exists at all.

These are not compared:
- diagnostics (M1–M11/Q, seasonality tests, spectra);
- standard errors and forecast intervals;
- likelihood and AIC;
- warnings;
- the SEATS requested and final mean, refit flag, temporary mean removal and
  bound-fixed parameters, which Build 62 only prints.

## X-11, no regARIMA

| # | Call | Monthly | Quarterly | Data |
|---|---|---|---|---|
| 1 | x11: multiplicative | ✓ | ✓ | pos |
| 2 | x11: additive | ✓ | ✓ | any |
| 3 | x11: log-additive | ✓ | ✓ | pos |
| 4 | x11: pseudo-additive | ✓ | ✓ | ≥0 |
| 5 | x11: filter S3x1 | ✓ | ✓ | pos |
| 6 | x11: filter S3x3 | ✓ | ✓ | pos |
| 7 | x11: filter S3x9 | ✓ | ✓ | pos |
| 8 | x11: filter S3x15 | ✓ | ✓ | pos |
| 9 | x11: filter stable | ✓ | ✓ | pos |
| 10 | x11: filter MSR (Census default) | ✓ | ✓ | pos |
| 11 | x11: filter x11default | ✓ | ✓ | pos |
| 12 | x11: mixed per-period filters | ✓ | ✓ | pos |
| 13 | x11: short series (last 4 years) | ✓ | ✓ | pos |
| 14 | x11: short series, requested filter kept (sfshort=yes) | ✓ | ✓ | pos |
| 15 | x11: Henderson 9 | ✓ | ✓ | pos |
| 16 | x11: Henderson 13 | ✓ | ✓ | pos |
| 17 | x11: Henderson 23 | ✓ | ✓ | pos |
| 18 | x11: Henderson I/C ratio set | ✓ | ✓ | pos |
| 19 | x11: true 7-term endpoints | ✓ | ✓ | pos |
| 20 | x11: sigma limits 2.0/3.0 | ✓ | ✓ | pos |
| 21 | x11: calendar sigma, all | ✓ | ✓ | pos |
| 22 | x11: calendar sigma, significant | ✓ | ✓ | pos |
| 23 | x11: calendar sigma, grouped | ✓ | ✓ | pos |
| 24 | x11: extremes, weighted MAD | ✓ | ✓ | pos |
| 25 | x11: extremes, tau MAD | ✓ | ✓ | pos |
| 26 | x11: extremes, log weighted MAD | ✓ | ✓ | pos |
| 27 | x11: extremes, log tau MAD | ✓ | ✓ | pos |
| 28 | x11: trend only | ✓ | ✓ | pos |
| 29 | x11: global shrinkage | ✓ | ✓ | pos |
| 30 | x11: legacy Easter prepass | ✓ | — | pos |

## X-11 with regARIMA

| # | Call | Monthly | Quarterly | Data |
|---|---|---|---|---|
| 31 | x11+reg: airline, log | ✓ | ✓ | pos |
| 32 | x11+reg: airline, identity (additive) | ✓ | ✓ | any |
| 33 | x11+reg: airline + constant | ✓ | ✓ | pos |
| 34 | x11+reg: model (1,1,0)(0,1,1) | ✓ | ✓ | pos |
| 35 | x11+reg: model (2,1,1)(0,1,1) | ✓ | ✓ | pos |
| 36 | x11+reg: model (0,1,1)(1,1,1) | ✓ | ✓ | pos |
| 37 | x11+reg: automatic transform | ✓ | ✓ | any |
| 38 | x11+reg: automatic model | ✓ | ✓ | pos |
| 39 | x11+reg: no forecasts | ✓ | ✓ | pos |
| 40 | x11+reg: 3-year forecasts + 1-year backcasts | ✓ | ✓ | pos |
| 41 | x11+reg: trading day | ✓ | ✓ | pos |
| 42 | x11+reg: working day (1-coefficient TD) | ✓ | ✓ | pos |
| 43 | x11+reg: trading day, no leap-year effect | ✓ | ✓ | pos |
| 44 | x11+reg: trading day + length of period | ✓ | ✓ | pos |
| 45 | x11+reg: leap year only | ✓ | ✓ | pos |
| 46 | x11+reg: length of period only | ✓ | ✓ | pos |
| 47 | x11+reg: Easter[8] | ✓ | ✓ | pos |
| 48 | x11+reg: Labor Day[8] | ✓ | — | pos |
| 49 | x11+reg: Thanksgiving[1] | ✓ | — | pos |
| 50 | x11+reg: stock trading day (end of month) | ✓ | — | pos |
| 51 | x11+reg: AIC-tested trading day + Easter (aictest) | ✓ | ✓ | pos |
| 52 | x11+reg: outliers AO+LS | ✓ | ✓ | pos |
| 53 | x11+reg: outliers AO+LS+TC | ✓ | ✓ | pos |
| 54 | x11+reg: outliers add-all method | ✓ | ✓ | pos |
| 55 | x11+reg: outliers, critical value 3.5 | ✓ | ✓ | pos |
| 56 | x11+reg: outliers, TC rate 0.8 | ✓ | ✓ | pos |
| 57 | x11+reg: manual outliers (AO, LS, TC at fixed positions) | ✓ | ✓ | pos |
| 58 | x11+reg: outliers removed from final SA | ✓ | ✓ | pos |
| 59 | x11+reg: TC allocated to trend | ✓ | ✓ | pos |
| 60 | x11+reg: Easter kept in SA (keepholiday) | ✓ | ✓ | pos |
| 61 | x11+reg: local shrinkage | ✓ | ✓ | pos |
| 62 | x11+reg: pseudo-additive | ✓ | ✓ | ≥0 |
| 63 | x11+reg: combo — auto transform + auto model + TD + Easter + AO/LS/TC | ✓ | ✓ | any |

## SEATS (with regARIMA)

| # | Call | Monthly | Quarterly | Data |
|---|---|---|---|---|
| 64 | seats: airline (0,1,1)(0,1,1), log | ✓ | ✓ | pos |
| 65 | seats: seasonal MA (0,1,0)(0,1,1), log | ✓ | ✓ | pos |
| 66 | seats: AR1 (1,1,0)(0,1,1), log | ✓ | ✓ | pos |
| 67 | seats: ARMA11 (1,1,1)(0,1,1), log | ✓ | ✓ | pos |
| 68 | seats: second difference (0,2,1)(0,1,1), log | ✓ | ✓ | pos |
| 69 | seats: AR2 (2,1,0)(0,1,1), log | ✓ | ✓ | pos |
| 70 | seats: airline, identity | ✓ | ✓ | any |
| 71 | seats: seasonal MA, identity | ✓ | ✓ | any |
| 72 | seats: AR1, identity | ✓ | ✓ | any |
| 73 | seats: ARMA11, identity | ✓ | ✓ | any |
| 74 | seats: second difference, identity | ✓ | ✓ | any |
| 75 | seats: AR2, identity | ✓ | ✓ | any |
| 76 | seats: airline + constant | ✓ | ✓ | pos |
| 77 | seats: automatic transform | ✓ | ✓ | any |
| 78 | seats: trading day | ✓ | ✓ | pos |
| 79 | seats: leap year only | ✓ | ✓ | pos |
| 80 | seats: Easter[8] | ✓ | ✓ | pos |
| 81 | seats: trading day + Easter | ✓ | ✓ | pos |
| 82 | seats: AIC-tested trading day + Easter (aictest) | ✓ | ✓ | pos |
| 83 | seats: outliers AO+LS+TC | ✓ | ✓ | pos |
| 84 | seats: manual outliers (AO, LS, TC at fixed positions) | ✓ | ✓ | pos |
| 85 | seats: no forecasts | ✓ | ✓ | pos |
| 86 | seats: automatic model | ✓ | ✓ | pos |
| 87 | seats: combo — auto transform + auto model + TD + Easter + AO/LS/TC | ✓ | ✓ | any |
| 88 | seats: model (2,1,1)(0,1,1) | ✓ | ✓ | pos |
| 89 | seats: model (0,1,1)(1,1,0) | ✓ | ✓ | pos |
| 90 | seats: model (3,1,0)(0,1,1) | ✓ | ✓ | pos |
| 91 | seats: 3-year forecasts + 1-year backcasts | ✓ | ✓ | pos |
| 92 | seats: stock trading day | ✓ | — | pos |
| 93 | seats: Labor Day[8] | ✓ | — | pos |
| 94 | seats: Thanksgiving[1] | ✓ | — | pos |

## Spans

Rows 95–101 require positive data and at least ten years (120 monthly or
40 quarterly observations). They use log transformation, the airline model,
one year of forecasts and no backcasts, except rows 98/101 use row 38's
automatic model selection. Rows 97/98/100/101 search AO, LS and TC with
automatic critical values. There are no specified manual outliers.

All indices below are one-based and use the selected series length `n`.
One year is 12 months or 4 quarters; span endpoints are inclusive dates.

| # | Call | M | Q | Definition |
|---|---|---|---|---|
| 95 | x11+reg: model span ends 3 years early | ✓ | ✓ | Model span: series start through index `n − 3 years`. |
| 96 | x11+reg: model span drops the first third | ✓ | ✓ | Model span: index `⌊n/3⌋+1` through series end. |
| 97 | x11+reg: outlier span, middle third | ✓ | ✓ | Outlier span: indices `⌊n/3⌋+1` through `⌊2n/3⌋`. |
| 98 | x11+reg: automatic model, model span ends 3 years early | ✓ | ✓ | Row 95's model span with automatic model and outlier selection. |
| 99 | seats: model span ends 3 years early | ✓ | ✓ | Row 95 with SEATS. |
| 100 | seats: outlier span, middle third | ✓ | ✓ | Row 97 with SEATS. |
| 101 | seats: automatic model, model span ends 3 years early | ✓ | ✓ | Row 98 with SEATS. |

The model and regression effects are estimated on the model span and applied
to the whole series. Forecasts start at the series end. Outlier searches are
restricted to their span, within the model span.

## Supplied regressors

Rows 102–108 use positive data, log transformation, the airline model
(0,1,1)(0,1,1), no constant, one requested forecast year and no backcasts.
They retain row 57's input bounds (at most 780 months or 336 quarters),
without its manual outliers or the span rows' ten-year minimum.

Every supplied column starts at the series start. X11 rows 102–104 and 108
supply one future year (12 months or 4 quarters). SEATS rows 105–107 supply
three future years (36 months or 12 quarters), because Build 62 requires that
coverage for SEATS signal extraction.
The requested forecast setting remains one year. Centering uses the entire
supplied span, including those future values.

| # | Call | M | Q | Supplied columns and their `usertype` |
|---|---|---|---|---|
| 102 | x11+reg: supplied working days | ✓ | ✓ | Working-day column, `usertype=td`. |
| 103 | x11+reg: supplied Orthodox Easter holiday | ✓ | — | Fourteen-day holiday column, `usertype=holiday`. |
| 104 | x11+reg: supplied intervention ramp | ✓ | ✓ | Ramp column, `usertype=user`. |
| 105 | seats: supplied working days | ✓ | ✓ | Row 102 with SEATS and three supplied future years. |
| 106 | seats: supplied Orthodox Easter holiday | ✓ | — | Row 103 with SEATS and three supplied future years. |
| 107 | seats: supplied intervention ramp | ✓ | ✓ | Row 104 with SEATS and three supplied future years. |
| 108 | x11+reg: supplied working days + Orthodox Easter + ramp | ✓ | — | All three columns, in that order, with types `td holiday user`. |

Working days count Monday through Friday in each month, excluding January 1,
July 4, November 11 and December 25 only when the date is a weekday. Monthly
values subtract the mean for their calendar month over the supplied span.
Quarterly values first sum the raw monthly counts, then subtract the mean for
their quarter of the year over the supplied span.

The holiday column is the fraction of the fourteen days immediately before
Orthodox Easter that falls in each month, excluding Easter itself. For year
`y`, calculate `a=y%4`, `b=y%7`, `c=y%19`, `d=(19*c+15)%30`,
`e=(2*a+4*b-d+34)%7`; the Julian month is `floor((d+e+114)/31)` and day is
`(d+e+114)%31+1`. Add thirteen days to obtain the Gregorian date; this formula
applies to 1900–2099. Subtract each calendar month's mean over the full supplied
span from its monthly fraction.

For observed length `n`, the ramp is zero through one-based observation
`floor(n/2)`, then rises by `1/6` for each of the next six months or by `1/2`
for each of the next two quarters. It remains one thereafter, including all
supplied future periods. The ramp is not centered.

The harness generates these columns once per case and gives both programs the
same numbers. Build 62 reads them from its `regression` spec, named `user1`,
`user2`, `user3` in the order supplied, with an explicit `start` and inline
`data`. This crate receives dated values.

## Robustness rows (U-01a to U-22)

A letter suffix separates the variants of one option, and the numbering has
gaps: there is no row U-13. Each row varies its section's baseline. FRED rows
keep the usual input bounds: the last 780 months, and the last 336 quarters
(340 for direct X-11, 328 for SEATS, 324 for U-07a). They require positive
data unless the definition says otherwise. Rows U-14 to U-21b run only on the
synthetic series they name (`tests/parity/synthetic.toml`). A U-12 row's
family is the Rust `SeatsFamily` its orders resolve to (`selected_family` in
`src/api/seats.rs`).

| # | Call | M | Q | Definition |
|---|---|---|---|---|
| U-01a | x11: pseudo-additive + legacy Easter, retained | ✓ | — | Data ≥0. `x11easter = yes` with `keepholiday = yes`: the Easter effect stays in the final adjusted and irregular series. |
| U-01b | x11+reg: legacy Easter | ✓ | — | `x11easter = yes` on the X-11 + regARIMA baseline. |
| U-02a | x11: extremes on observed span, mult | ✓ | ✓ | `excludefcst = yes` on direct X-11, which has no forecast extension: checks that the option is inert there. |
| U-02b | x11+reg: extremes on observed span, add | ✓ | ✓ | Data any. Row 32 (identity, additive) with `excludefcst = yes`, so extreme values ignore the forecast extension. |
| U-02c | x11+reg: extremes on observed span, mult | ✓ | ✓ | The X-11 + regARIMA baseline with `excludefcst = yes`. |
| U-03 | x11+reg: log-additive | ✓ | ✓ | Log-additive X-11 mode. |
| U-04a | x11: mixed S3x1/MSR filters | ✓ | ✓ | Per-period filters. Monthly: S3x1, MSR, MSR, S3x5, MSR, MSR, S3x9, MSR, MSR, S3x3, MSR, MSR. Quarterly: S3x1, MSR, S3x5, MSR. |
| U-04b | x11: mixed MSR/stable/S3x15 filters | ✓ | ✓ | Monthly: MSR, S3x1, MSR, S3x3, stable, MSR, S3x15, MSR, x11default, S3x5, MSR, S3x9. Quarterly: MSR, S3x1, stable, MSR. |
| U-05a | x11+reg: Henderson 9 | ✓ | ✓ | Fixed 9-term Henderson trend. |
| U-05b | x11+reg: Henderson 13 | ✓ | ✓ | Fixed 13-term Henderson trend. |
| U-05c | x11+reg: Henderson 23 | ✓ | ✓ | Fixed 23-term Henderson trend. |
| U-05d | x11: automatic Henderson, true 7-term | ✓ | ✓ | Automatic Henderson selection with a true 7-term filter. |
| U-06a | x11+reg: (0,1,2)(0,1,1) | ✓ | ✓ | Explicit model. |
| U-06b | x11+reg: (0,1,3)(0,1,1) | ✓ | ✓ | Explicit model. |
| U-06c | x11+reg: (0,1,1)(0,1,2) | ✓ | ✓ | Explicit model. |
| U-06d | x11+reg: (0,1,1)(2,1,0) | ✓ | ✓ | Explicit model. |
| U-06e | x11+reg: (1,0,1)(0,1,1) + constant | ✓ | ✓ | Explicit model with a constant. |
| U-06f | x11+reg: (0,1,1)(0,0,1) | ✓ | ✓ | Explicit model. |
| U-06g | x11+reg: (1,1,1)(1,1,1), minimum span | ✓ | ✓ | The last 39 months or 15 quarters, at the minimum effective-observation boundary. |
| U-07a | x11+reg: row 63 + 12 backcasts (rmfix) | ✓ | ✓ | Data any. Row 63 with 12 backcasts (the quarterly run asks for more backcasts than forecasts). |
| U-07b | x11+reg: row 63, 6 forecasts + 12 backcasts (rmfix b > f) | ✓ | — | Data any. Monthly b > f: U-07a's monthly half has b = f, so only this row reads monthly zero padding with monthly calendar effects. |
| U-07c | x11+reg: automatic + supplied working days | ✓ | ✓ | Row 38's automatic model with row 102's working-day column. |
| U-07d | x11+reg: automatic + LS outliers | ✓ | ✓ | Automatic model; LS-only outlier search, automatic critical value. |
| U-07e | x11+reg: automatic + add-all, per-type critical | ✓ | ✓ | Automatic model; AO/LS/TC `addall` search with critical values 4.0/3.5/3.8. |
| U-07f | x11+reg: automatic, 24 forecasts | ✓ | — | Automatic model with 24 forecasts. |
| U-07g | x11+reg: automatic + AO/LS outliers + working days and ramp | ✓ | ✓ | Automatic model; AO/LS outlier search, automatic critical value; row 102's working-day column then row 104's ramp, types `td user`. |
| U-08a | x11+reg: outliers LS only | ✓ | ✓ | LS-only outlier search, automatic critical value. |
| U-08b | x11+reg: outliers TC only | ✓ | ✓ | TC-only outlier search, automatic critical value. |
| U-08c | x11+reg: outliers LS+TC, per-type critical | ✓ | ✓ | LS/TC search (no AO) with critical values 3.5/3.8. Rows 52 and 55 cover AO+LS; U-07e covers per-type with AO. |
| U-09a | x11+reg: stock TD, day 15 | ✓ | — | `tdstock[15]`. |
| U-09b | x11+reg: stock weekday/weekend TD, day 15 | ✓ | — | `tdstock1coef[15]`. |
| U-09c | x11+reg: Easter[1] | ✓ | ✓ | `easter[1]`. |
| U-09d | x11+reg: Easter[15] | ✓ | ✓ | `easter[15]`. |
| U-09e | x11+reg: Labor Day[15] | ✓ | — | `labor[15]`. |
| U-09f | x11+reg: Thanksgiving[-1] | ✓ | — | `thank[-1]`. |
| U-10 | x11+reg: TD + holidays + ramp, model span | ✓ | — | At least 120 months. Flow trading day, Easter[8], Labor Day[8], Thanksgiving[1], row 104's ramp, row 95's model span. |
| U-11 | x11+reg: manual outliers, TC rate 0.8 | ✓ | ✓ | Row 57's manual AO/LS/TC with `tcrate = 0.8`, so the manual TC regressor decays at the configured rate rather than the default (row 56 sets the rate for an outlier search instead). |
| U-12a | seats: (0,1,2)(0,1,1) | ✓ | ✓ | Explicit model; family Ma2Seasonal. |
| U-12b | seats: (1,1,2)(0,1,1) | ✓ | ✓ | Explicit model; family Arma12. |
| U-12c | seats: (2,1,2)(0,1,1) | ✓ | ✓ | Explicit model; family Arma22. |
| U-12d | seats: (3,1,1)(0,1,1) | ✓ | ✓ | Explicit model; family Arma31 with a seasonal difference. |
| U-12e | seats: (0,1,1)(0,0,0) | ✓ | ✓ | Explicit model; family NonseasonalMa. |
| U-12f | seats: (1,1,0)(0,0,0) | ✓ | ✓ | Explicit model; family NonseasonalAr. |
| U-12g | seats: (1,1,1)(0,0,0) | ✓ | ✓ | Explicit model; family NonseasonalArma11. |
| U-12h | seats: (2,1,1)(0,0,0) | ✓ | ✓ | Explicit model; family NonseasonalArma(3). |
| U-12i | seats: (1,0,0)(1,1,0) | ✓ | ✓ | Explicit model; family IntegratedArma(2), seasonal-AR layout. |
| U-12j | seats: (3,0,1)(0,1,1) | ✓ | ✓ | Explicit model; family IntegratedArma(5), d=0/D=1 layout. |
| U-12k | seats: (1,1,1)(0,0,1) | ✓ | ✓ | Explicit model; family Arma11 with seasonal (0,0,1). |
| U-12l | seats: (0,1,2)(0,0,0) | ✓ | ✓ | Explicit model; family NonseasonalArma(2). |
| U-12m | seats: (2,1,2)(0,0,0) | ✓ | ✓ | Explicit model; family NonseasonalArma(4). |
| U-12n | seats: (2,1,1)(0,0,1) | ✓ | ✓ | Explicit model; family Arma21 with seasonal (0,0,1). |
| U-12o | seats: (3,1,1)(0,0,1) | ✓ | ✓ | Explicit model; family Arma31 with seasonal (0,0,1). |
| U-12p | seats: (0,1,0)(0,1,0) | ✓ | ✓ | Explicit model; family IntegratedArma(0). |
| U-12q | seats: (3,1,1)(1,1,1) | ✓ | ✓ | Explicit model; family IntegratedArma(6), seasonal-AR layout. |
| U-12r | seats: (0,1,1)(0,0,1) | ✓ | ✓ | Explicit model; family IntegratedArma(2), d≥1 layout. |
| U-14 | seats: (3,1,0)(1,1,0), no transform | ✓ | — | SYN_U14A/B: well-separated AR roots (26-coefficient numerator). |
| U-15a | seats: (1,0,0), no transform | ✓ | — | SYN_U15A: zero-mean AR with φ̂ ≤ −0.9, no trend or cycle. |
| U-15b | seats: (2,0,0), no transform | ✓ | — | SYN_U15B: AR roots at ±60°, no trend or cycle. |
| U-16 | seats: (0,1,0)(1,1,0), no transform | ✓ | — | SYN_U16A/B: Φ near −1, forced refit. |
| U-17a | seats: (0,1,1)(1,0,1), no transform | ✓ | — | SYN_U17A: misfit reaching MODEL CHANGED TO (0,1,1)(1,0,0). |
| U-17b | seats: (1,1,0)(1,0,1), no transform | ✓ | — | SYN_U17B: a misfit that both programs reject. |
| U-18 | seats: (3,1,0)(0,1,1), no transform | ✓ | — | SYN_U18: φ3 ≈ −0.99 (unit-modulus snap). |
| U-19a | x11+reg: constant series, airline, additive | ✓ | — | SYN_U19, no transform. |
| U-19b | x11+reg: constant series, (1,0,0), additive | ✓ | — | SYN_U19, no transform. |
| U-19c | seats: constant series, airline | ✓ | — | SYN_U19, no transform; both programs reject it. |
| U-20a | x11+reg: model span after the start, backcasts, Easter[8] | ✓ | — | SYN_U20: 12 backcasts, row 96's model span, Easter near the calendar-year floor. |
| U-20b | x11+reg: U-20a without the model span | ✓ | — | SYN_U20; a listed deviation. |
| U-21 | seats: (0,0,0)(1,0,1) + constant, no transform | — | ✓ | SYN_U21: white noise. ChangeModel drops BQ and the refit leaves Φ = 0, so no component keeps an AR root. |
| U-21b | seats: (0,0,1)(1,0,1) + constant, no transform | — | ✓ | SYN_U21B: white noise. As U-21, with the regular MA kept: no transitory component may be invented, and the irregular must match. |
| U-22 | x11+reg: pseudo-additive, log | ✓ | ✓ | Row 62 with `function = log`. The only pseudo-additive form that estimates gaps. |

## Not in the table

Left out:
- `pickmdl`;
- other Box-Cox powers;
- `centerseasonal` and `x11regression`;
- SEATS tuning options (`noadmiss`, `qmax`, `rmod`, `xl`, …).

Series with gaps (those with a `missing` list in `tests/parity/series.toml`)
go through every row like any other series. Where Build 62 estimates the gaps,
Rust must match its numbers; where Build 62 refuses, Rust must return a
sensible error.

Sliding spans, revision history and frequencies other than monthly and
quarterly are out of scope.

Combo rows 63 and 87 always include trading day and Easter rather than
AIC-testing them; rows 51 and 82 cover the AIC test.

Manual-outlier rows place an AO at observation ⌊n/3⌋, an LS at ⌊2n/3⌋ and a TC
at n−6, with dates computed per series.
