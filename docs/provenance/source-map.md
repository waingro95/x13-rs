# Build 62 source map

Fortran routine names refer to the Census X-13ARIMA-SEATS Version 1.1 Build 62
ASCII source distribution. Rust paths are relative to the repository root.
Routines listed together are implemented together, and a Rust module may
implement only the branches the supported configurations need. This is a map
for finding code, not a claim that the whole program is translated: the
[call table](../parity-call-table.md) defines what is supported. The last
column notes the mapping and any Build 62 quirk the Rust reproduces, such as a
particular order of arithmetic.

## Shared numerics and regARIMA

| Fortran routine(s) | Rust module(s) | Mapping / reproduced quirk |
| --- | --- | --- |
| `dpeq` | `src/numerics/mod.rs` | Absolute `3.834e-20` branch comparison, not exact equality. |
| `dpmpar` | `src/numerics/build62.rs` | `BUILD_62_MACHINE_EPSILON` (`2.220446d-16`, not `f64::EPSILON`) and `BUILD_62_MIN_POSITIVE` (`2.225074d-308`); shared by `ddot`, `dppfa`, `lmpar`, `ttest` and the covariance and variance gates. The same module holds the shortened `BUILD_62_PI` and `BUILD_62_MAD_SCALE` (`.6745D0`, the MAD divisor in `idotlr`, `sdxtrm` and `histx`). |
| `daxpy`, `ddot` | `src/numerics/blas1.rs` | Slice and positive-stride (`step_by`) forms keeping the reference operand order and fused accumulation; DDOT skips logarithms only where finite magnitudes guarantee its guard is true. |
| `dppfa`, `dppsl`, `dppdi`, `dscal` | `src/numerics/packed.rs` | Packed upper-triangle Cholesky, solve (forward and backward halves, any number of right-hand-side columns) and inverse loop 0-based through the `packed_column`/`packed_diagonal` helpers, keeping the source operation order. A failed `dppfa` reports its `Info` column, so `olsreg` callers can accept `Info.eq.Ncxy`. |
| `lmdif`, `fdjac2`, `qrfac`, `qrsolv`, `lmpar`, `enorm` | `src/numerics/minpack.rs` | Preserve finite-difference steps, pivot ties, exit codes and separately rounded squares. |
| `shlsrt` | `src/numerics/order.rs` | Source Shell-sort order for median calculations. |
| `chisq`, `gauss` | `src/numerics/probability.rs` | Build 62's distribution approximations, saturation and even/odd recurrences. |
| `polyml`, `ratpos`, `ratneg` | `src/numerics/polynomial.rs`, `src/arima/operators.rs`, `src/arima/exact.rs` | Preserve sparse lag traversal and convolution/division order. |
| `acf`, `acfar` | `src/numerics/autocorrelation.rs` | ACF uses two normalizations; ACFAR uses one division by `N*C0`. |
| `getmdl`, `gtarma`, `mdlinp`, `mdlset`, `setopr`, `maxlag` | `src/arima/model.rs`, `src/arima/operators.rs` | Typed orders expand to seasonal lags before filtering. |
| `setmdl`, `upespm`, `strtvl` | `src/arima/model.rs`, `src/arima/estimation.rs` | Unspecified starts are 0.1; warm refits retain estimates and source near-unit MA setup. |
| `roots` | `src/arima/roots/polynomial.rs` | Wrapper translated: coefficient reversal, the constant −1 and leading-zero stripping. RPOLY (`rpoly`, `fxshfr`, `quadit`, `realit`, `quadsd`, `calcsc`, `nextk`, `newest`, `quad`; ACM TOMS 493) is not translated: `src/arima/roots/solver.rs` is an independent closed-form solver, described in `clean-room-roots.md`. |
| `chkrts` | `src/arima/model.rs`, `src/arima/exact.rs` | Model admissibility precedes exact filtering. |
| `arflt`, `armafl`, `mltpos`, `xpand` | `src/arima/exact.rs`, `src/arima/operators.rs` | Preserve conditional AR lag loss and exact MA startup rows. |
| `intgpg`, `uconv`, `euclid`, `exctma`, `bldcov` | `src/arima/exact.rs` | Initial-condition covariance uses the source operator and solve order. |
| `regvar` | `src/regression/design.rs`, `src/regression/model.rs` | Response is last in packed `[X:y]`; preserve regressor order and the 80-column limit. |
| `olsreg`, `xprmx`, `resid`, `resid2` | `src/regression/gls.rs`, `src/numerics/packed.rs` | Filtered GLS, covariance and residuals share the source packed matrix. |
| `rgarma`, `fcnar` | `src/arima/estimation.rs`, `src/arima/estimation/identification.rs` | Keep callback, retained fit and covariance states distinct; convergence depends on regression presence. |
| `rgarma` (`Nefobs.lt.Nextvl`) | `src/arima/estimation.rs`, `src/arima/estimation/attempt.rs` | Checked before the exact filter; explicit fits, `trnaic` candidates and calendar AIC tests reject (`argok=F`); automatic searches keep their own handling. |
| `stpitr` | `src/arima/estimation.rs`, `src/arima/estimation/identification.rs` | Build 62 checks the iteration cap before convergence; internal callers can keep a fit that stopped at the limit. |
| `yprmy`, `logdet`, `prlkhd` | `src/arima/likelihood.rs`, `src/arima/estimation.rs`, `src/arima/exact.rs` | Likelihood uses the retained response diagonal and determinant before covariance work. |
| `covar`, `svamcm`, `svrgcm` | `src/arima/covariance.rs`, `src/regression/gls.rs` | Parameter and regression covariance are computed separately. |
| `trnfcn`, `invfcn` | `src/transform.rs` | Explicit identity/log and original-scale interval restoration. |
| `trnaic`, `varlog` | `src/engine/automatic_transform.rs` | Log wins only on the strict AICC comparison; preserve candidate starts and internal forecast horizon. |
| `fcstxy`, `prtfct`, `scrmlt` | `src/arima/forecast.rs` | Original-fit recursion, uncertainty and regression variance; default log points have no mean bias correction. |
| `getsrs`, `gtotlr` (spans) | `src/api/span.rs`, `oracle/src/spec_file/regarima_request.rs` | Inclusive dated model bounds contain the outlier-search bounds. |
| `arima`, `setspn`, `regvar`, `x11pt1` (modelspan) | `src/engine/model_span.rs` | Restore the end before forecasts and the beginning afterward; keep fitted quantities and remove length before Easter priors. |
| `editor` (modelspan) | `src/engine/model_span.rs`, `src/api/spec.rs` | Reject outside-span gaps, prune supplied outliers and cancel backcasts when the model begins late; calendar years and supplied-column coverage use the cancelled horizon (`ForecastSpec::for_model_start`). |
| `editor`, `tfmts`, `setxpt` (input limits) | `src/api/mod.rs` | Reject a magnitude needing more than 21 print columns (`int(log10)+3`), fewer than three years, and `Posffc` over `PLEN` positions or `PYRS` years counted from the backcast year's first period. |
| `armafl`, `fcstxy` (restored span) | `src/arima/exact.rs`, `src/arima/forecast.rs` | Forecasts consume only the MA residual tail; retain its fitted initial factor without rebuilding unused AR-prefix rows. `fcstxy` filters with `Lckrts` false, so a noninvertible estimate still forecasts. |
| `mkback`, `revrse`, `putbak` | `src/arima/forecast.rs` | Backcasts reverse data and regression rows, then restore chronological order. |
| `x12run`, `x11ari`, `arima` | `src/engine/regarima_x11.rs`, `src/engine/automatic_model_x11.rs`, `src/engine/x11_configuration.rs`, `src/engine/direct_x11.rs`, `src/engine/seats/pipeline.rs` | Preprocessing, fit and decomposition sequencing lives outside standalone X-11. |
| `extend`, `setxpt` | `src/engine/post_fit.rs`, `src/engine/x11_configuration.rs` | Cancellation prevents X-11 insertion but retains forecasts already saved by `prtfct`/`mkback`. |
| `editor`, `gtinpt` (missing values) | `src/api/mod.rs`, `src/engine/missing.rs`, `src/regression/model.rs` | NaN marks a gap; a finite `-99999`, Build 62's missing-value code (`Mvcode`), is an ordinary observation. Gaps take Build 62's `1e9` filler (`Mvval`) before transforms, or the constant in a constant series, and ordered missing-value columns estimate them. |
| `x11pt1` (missing values) | `src/engine/missing.rs` | Reset the gap filler after prior-factor removal rather than transforming it. |

## Calendar regression and outliers

| Fortran routine(s) | Rust module(s) | Mapping / reproduced quirk |
| --- | --- | --- |
| `td6var` | `src/regression/calendar.rs` | Sunday-baseline contrasts, source weekday tables and stock-day clipping. |
| `td7var`, `adpdrg`, `rmlnvr` | `src/regression/calendar.rs` | Only default flow `td`/`td1coef` gets a fixed leap prior under log; other length columns remain estimated. |
| `rmlpyr`, `replyf`, `tdlom` | `src/regression/calendar.rs`, `src/arima/forecast.rs`, `src/regression/effects.rs` | Permanent log leap prior covers extensions and original-scale forecasts but stays out of adjusted values. |
| `estrmu`, `adestr` | `src/regression/holidays.rs` | Fitted Easter uses inline source date tables and window-dependent mean subtraction. |
| `adlabr`, `adthnk` | `src/regression/holidays.rs` | Labor Day/Thanksgiving use inline DATA, not `kdate.prm` (notably Thanksgiving 1940/1941). |
| `tdaic`, `easaic` | `src/engine/calendar_selection.rs`, `src/regression/model.rs` | TD then Easter; ties omit, each trial cold-starts at 0.1, and surviving columns keep order. |
| `addotl`, `adrgef` | `src/regression/model.rs`, `src/outliers/search.rs` | Specified outliers retain supplied order; the first missing value positions the contiguous MV group. The PB capacity stop counts at input, ahead of `x11ari`'s legacy-Easter prepass and Same bypass; `regvar` evaluates the columns after both. |
| `getreg`, `adrgef`, `regvar` (supplied columns) | `src/api/regarima/user_regressor.rs`, `src/regression/model.rs` | Dated values are sliced without extrapolation; columns group by first encountered type while retaining supplied identities. |
| `ctod` (inline supplied values) | `oracle/src/user_regressors.rs` | Choose decimal spellings whose digit-by-digit source parse preserves the generated binary64 input exactly. |
| `editor`, `regvar` (SEATS supplied coverage) | `src/api/seats.rs` | Require `max(12,3*Sp)` future values even when only one forecast year is requested. |
| `makotl`, `isfixd` | `src/outliers/candidates.rs`, `src/regression/model.rs` | AO/LS/TC columns and endpoint exclusions use model-span positions. |
| `gtotlr`, `setcv`, `setcvl`, `lassol` | `src/outliers/critical.rs`, `src/api/regarima/outliers.rs` | Default AO+LS/add-one, count-based cutoff and frequency-scaled TC decay. |
| `ppnd` (AS 111) | `src/outliers/critical.rs` | Not translated: Build 62's outputs at the two probabilities `setcv` uses are recorded as constants. |
| `medabs` | `src/outliers/critical.rs` | Median of absolute effective residuals without centering, divided by 0.6745 in search. |
| `ttest` | `src/outliers/search.rs` | Packed solves, date-major scan and strict comparisons retain source type order on ties. |
| `idotlr` | `src/outliers/search.rs`, `src/outliers/search/fit.rs`, `src/outliers/search/automatic.rs` | Start is `sign(propt**2,propt)`; search-span cutoffs/eligibility retain model-relative endpoints and unrestricted backward deletion. |
| `deltst`, `dlrgef` | `src/outliers/search.rs`, `src/regression/model.rs` | Backward deletion uses non-robust RMSE and removes at most one outlier per refit. |
| `prtref` | `src/regression/effects.rs` | Accumulate TD, holiday, AO, LS, TC and generic user separately in fitted column order. |
| `chkadj`, `prtref` (supplied mixtures) | `src/regression/effects.rs` | User-type cursor advances for holiday/generic columns but not supplied TD, including the resulting generic-column remapping. |
| `x11pt3`, `USRENTRY` (`ansub9`) (generic user) | `src/regression/effects.rs`, `src/seats/effects.rs` | Generic user effects remain in final SA, but not trend or irregular; TD/holiday effects are removed from SA. |
| `chkadj`, `USRENTRY` (`ansub9`), `SIGEX` (supplied TD) | `src/engine/seats/ar2.rs`, `src/engine/seats/arma.rs` | Supplied TD presence enables stochastic-TD root allocation even with no built-in calendar specification. |
| `adjreg` | `src/regression/effects.rs`, `src/engine/regarima_x11.rs`, `src/seats/effects.rs` | Remove groups sequentially on transformed data, then inverse-transform once. |
| `chkadj`, `rmpadj` | `src/engine/post_fit.rs`, `src/regression/effects.rs` | Final removal, holiday retention and TC-to-trend controls change restoration, not the fit. |

## Automatic model selection

| Fortran routine(s) | Rust module(s) | Mapping / reproduced quirk |
| --- | --- | --- |
| `automd` | `src/automdl/selector.rs`, `src/automdl/corrections.rs`, `src/automdl/terminal.rs` | Baseline, identification, fallback, correction and final steps follow Build 62's order. |
| `rmfix` | `src/automdl/preparation.rs`, `src/regression/model.rs` | Remove regression columns but subtract only nonconstant fitted effects before prepared-series identification. |
| `pass0` | `src/automdl/baseline.rs`, `src/automdl/constant.rs`, `src/automdl/preparation.rs` | Constant testing uses `sqrt(diag)*sqrt(variance)`, not a square root of the product. |
| `iddiff` | `src/automdl/differencing.rs` | Exact 200-step trial precedes conditional 1500 retry with retained starts; preserve seven-round limit. |
| `amdid`, `amdid2`, `hrest` | `src/automdl/hannan_rissanen.rs`, `src/automdl/search.rs` | Hannan–Rissanen lag/order grid and candidate tie order. |
| `amdest`, `amdfct`, `bestmd`, `mdlmch` | `src/automdl/search.rs`, `src/automdl/hannan_rissanen.rs`, `src/automdl/first_pass.rs` | Rank five slots by unrounded BIC2; ties retain encounter order and evicted candidates may be refit. |
| `rgarma`, `stpitr` (search status) | `src/arima/estimation/identification.rs`, `src/automdl/search.rs` | Zero-parameter/early-failed candidates inherit Convrg in invocation-local state despite fresh starts. |
| `tstmd2`, `chkurt`, `armats` | `src/automdl/coefficients.rs`, `src/automdl/roots.rs` | Unit-root tests operate on every factor; coefficient tests follow covariance estimation. |
| `mdlchk`, `acfdgn`, `pracf2` | `src/automdl/model_check.rs`, `src/automdl/diagnostics.rs` | Preserve lag truncation, fitted parameter count and residual-Q thresholds. |
| `bkdfmd`, `chksmd` | `src/automdl/fallback.rs`, `src/automdl/restoration.rs` | Fallback fits and restored diagnostics/procedural mean are distinct state. |
| `pass2` | `src/automdl/fallback.rs`, `src/automdl/fallback/second.rs` | Updating comparison values does not replace the saved reference fit. |
| `amidot` | `src/automdl/outlier_policy.rs`, `src/automdl/reentry.rs`, `src/outliers/search/automatic.rs` | Cutoff update `max(2.8,c-c*.14286)` can increase an already low cutoff. |
| `chkmu`, `genrtt` | `src/automdl/constant.rs`, `src/automdl/final_constant.rs` | Strict source t cutoffs; nonpositive standard error has a defined zero t statistic. |
| `chkrt1`, `chkrt2` | `src/automdl/roots.rs`, `src/automdl/final_roots.rs` | Root repairs run regular before seasonal and preserve source signed-imaginary tests. |
| `testodf` | `src/automdl/final_adjustments.rs` | Sum regular MA coefficients left-to-right; strict `abs(sum-1)<.001` triggers reduction/refit. |
| `automd` (coefficient re-entry) | `src/automdl/final_coefficients.rs`, `src/automdl/reentry.rs`, `src/automdl/corrections.rs` | Trailing-coefficient tie order and at most two multi-insignificant restarts are preserved. |
| `editor`, `x11ari` (Same) | `src/automdl/selector.rs`, `src/engine/automatic_model_x11.rs`, `src/engine/missing.rs`, `src/engine/regarima_x11.rs`, `src/engine/seats/pipeline.rs`, `src/engine/x11_configuration.rs`, `src/engine/post_fit.rs`, `src/numerics/mod.rs`, `src/api/mod.rs` | Constant-series bypass follows transformation, rejects all-zero input and cancels extensions; it skips estimation for explicit models too. With gaps Rust refuses instead, unless the observed values equal the 1e9 filler, which leaves the series constant, or the first value is a gap, which leaves `Same` false and runs the model. |

## X-11

| Fortran routine(s) | Rust module(s) | Mapping / reproduced quirk |
| --- | --- | --- |
| `x11int`, `x11pt1`, `x11pt2`, `x11pt3` | `src/x11/direct.rs`, `src/x11/direct/passes.rs`, `src/x11/direct/part_d.rs`, `src/engine/post_fit.rs` | B/C/D spans and state resets remain separate; final components use original observations. |
| `divsub`, `addmul` | `src/x11/direct/mode.rs`, `src/engine/post_fit.rs` | Observation-ordered subtraction/addition or division/multiplication. |
| `averag`, `hender` | `src/numerics/filters.rs` | Nested M-then-N sums and source Henderson constants/order. |
| `hndtrn`, `ends`, `hndend`, `apply` | `src/numerics/filters.rs` | Preserve central/endpoint accumulation; legacy H7 overwrites I/C with .001 and uses five-term endpoints. |
| `vtc` | `src/x11/direct/trend.rs`, `src/numerics/filters.rs` | Measured I/C and endpoint I/C are separate; source squares the ratio before multiplying the constant. |
| `editor` (X-11 options) | `src/x11/mod.rs`, `src/x11/direct/seasonal_factors.rs` | Short S3x15 requests resolve to Stable before later extension cancellation. |
| `vsfb`, `endsf`, `totals` | `src/x11/seasonal.rs`, `src/numerics/filters.rs` | Build 62's endpoint decimals and sequential floating-point count; short spans can use a temporary mean filter. |
| `vsfc`, `forcst` | `src/x11/direct/seasonal_factors.rs` | Centering and seasonal endpoint copying keep calendar alignment; Stable and ordinary endpoints differ. |
| `sfmsr` | `src/x11/msr.rs`, `src/x11/direct/part_d.rs` | Exclude forecasts, retain backcasts, truncate only the end to complete years; first-period S3x1 suppresses selection. |
| `vsfa`, `fis` | `src/x11/msr.rs` | Sum corrected changes before division; preserve the correction constants and the 999.99 sentinel. |
| `chktrn` | `src/x11/direct/trend.rs` | Replace nonpositive trend left-to-right in place, with observed directional bounds. |
| `xtrm`, `wtxtrm` | `src/x11/direct/extreme_weights.rs`, `src/x11/calendar_sigma.rs` | Two passes and calendar windows; temporary negative-weight sentinel is clamped only over estimation span. |
| `sdxtrm`, `rho2` | `src/x11/robust.rs`, `src/x11/direct/extreme_scale.rs`, `src/x11/direct/extreme_weights.rs` | Center remains 0/1; preserve second-pass exclusions, rho cutoffs and tau's post-loop index divisor. |
| `entsch`, `fvalue` (calendar sigma) | `src/x11/calendar_sigma.rs` | Significance decisions precede grouped/per-period replacement and retain source zero guards. |
| `si`, `replac` | `src/x11/direct/seasonal_factors.rs`, `src/x11/direct/extreme_weights.rs` | Start with `weight*value`, then add four neighbors sequentially; retain fewer-than-four fallback. |
| `logar`, `antilg`, `trbias` | `src/x11/log_additive.rs` | The full C13 includes the forecast and backcast extensions that the saved C13 omits; the bias computation uses the full span. |
| `x11pt1`, `x11pt2`, `x11pt3` (pseudo-additive) | `src/x11/pseudo_additive.rs`, `src/x11/direct/mode.rs`, `src/x11/direct/passes.rs` | Keep source B/C/D formulas and eligibility masks; D11 is original minus trend times seasonal-minus-one. |
| `shrink`, `glbshk`, `locshk`, `lkshnk` | `src/x11/shrinkage.rs` | Shrinkage uses the appended forecasts and Build 62's local/global scale order. |
| `holday`, `holidy`, `easter`, `chkeas` | `src/x11/easter.rs` | Preserve January-padded indicator writes, integer Easter counts, shifted March factors and two-sigma trimming. |
| `xrgdrv`, `ssprep`, `restor` (legacy Easter) | `src/engine/x11_configuration.rs`, `src/x11/direct.rs` | Two prepasses retain requested filters and carry resolved MSR/sigma/pseudo eligibility. |
| `ftest`, `mstest`, `kwtest`, `combft`, `fvalue` | `src/x11/seasonality.rs` | Build 62's probability constants and strict conclusion thresholds; fvalue's fallback also changes F. |
| `x11pt3` (restoration) | `src/engine/post_fit.rs` | D13 is independent of final D11 removal; restore LS to trend and AO/TC to irregular per controls. |
| `x11pt4`, `f3cal`, `f3gen` | `src/x11/quality.rs` | Keep intermediate eligibility, source roundtrips and masked percentage-change/variance ordering. |
| `spcdrv`, `gendff` | `src/x11/spectral.rs`, `src/engine/regarima_spectral.rs` | At most 96 observations; log before backward differences, but E3 stays unlogged and undifferenced. |
| `sumf`, `smeadl`, `crosco`, `sautco` | `src/numerics/spectral.rs` | Forward mean deletion and covariance normalized by N, not N-lag; unused cornom correlations are not reproduced. |
| `sicp2` | `src/numerics/spectral.rs` | Final label10 overwrites the AIC optimum with the requested maximum-order fit. |
| `spgrh`, `decibl`, `mkpeak`, `mkfreq` | `src/numerics/spectral.rs` | Separate cosine/sine loops, power/decibel conversion and Build 62's default frequency grids. |
| `idpeak`, `ispeak`, `mkmdsx` | `src/numerics/spectral.rs`, `src/x11/spectral.rs` | Median/range threshold and strict seasonal/TD peak comparisons; warning attribution stays separate. |

## SEATS

The ANSUB files hold several routines each; the names below say which. The
model families share the numerical modules rather than keeping their own
copies.

| Fortran routine(s) | Rust module(s) | Mapping / reproduced quirk |
| --- | --- | --- |
| `seats`, `USRENTRY` (`ansub9`) | `src/engine/seats/pipeline.rs`, `src/engine/seats/strategy.rs`, one module per family in `src/engine/seats/` | Import the fitted orders, coefficients, transform and mean; the regARIMA results stay separate. |
| `nmlmdl`, `TRANS0`, `TRANS2` (`transc`) | `src/seats/import.rs`, `src/seats/decomposition/root_allocation.rs`, `src/seats/arma/mod.rs`, `src/seats/arma/refit.rs` | Import snaps transformed tails even for fixed coefficients; each AR/MA group is transformed separately. An infinite TRANS0 coordinate in a degree-two or degree-three group (a reset to ±UR that zeroes a divisor, PHITH(3) = -1, or overflow) is clipped like Build 62's IEEE path; only 0/0 is a typed error. |
| `TRANS1`, `TRANSC` | `src/seats/ar2/transform.rs`, `src/seats/decomposition/root_allocation.rs`, `src/seats/arma/mod.rs` | Preserve grouped regular/seasonal coordinates, signs and degree-two/three root geometry. |
| `ANALTS` (family dispatch) | `src/seats/pipeline.rs`, `src/seats/airline/mod.rs`, `src/seats/seasonal_ma/mod.rs`, `src/seats/ar1_seasonal_ma/mod.rs`, `src/seats/arma/mod.rs`, `src/seats/ar2/mod.rs`, `src/seats/second_difference/mod.rs` | The model's own orders determine lag loss and working dimensions; no lower-order model stands in. |
| `ANALTS` (retry) | `src/seats/retry.rs`, `src/seats/pipeline.rs` | The imported residual Q, whether the run prints a summary, and state kept from an earlier retry can change refit decisions. |
| `ANALTS` (mean), `AUTO` | `src/seats/mean.rs`, `src/seats/autocorrelation.rs` | Mean removal changes the working residuals but not CALCFX's raw residuals. |
| `ANALTS` (5018) | `src/seats/arma/mod.rs` | Pure MA/noise returns after forward diagnostics; retain source log component scaling through `seatad`, without canonical decomposition. |
| `CALCFX` (`ansub1`) | `src/seats/residual.rs`, `src/seats/seasonal_ma/residual.rs`, `src/seats/ar1_seasonal_ma/residual.rs`, `src/seats/arma/residual.rs`, `src/seats/ar2/residual.rs` | INIT2/INIT0, true expanded lags and determinant-scaled residuals remain distinct. |
| `CALCFX` (root reuse) | `src/seats/arma/residual.rs` | Cache roots of current parameters when `x==y`; implements intended persistence missing from Fortran's local ap/aq SAVE. |
| `STAVAL` | `src/seats/refit/start.rs`, `src/seats/ar2/transform.rs`, `src/seats/arma/refit.rs`, `src/seats/seasonal_ma/refit.rs` | Q=0, MA and grouped AR starts retain their own ACF ratios and transforms. |
| `SEARCH` (`ansub1`) | `src/seats/refit/search.rs`, `src/seats/refit/one_coordinate.rs`, `src/seats/ar2/refit.rs`, `src/seats/ar1_seasonal_ma/refit.rs` | Ordered direction updates, finite differences and separately rounded products preserve refit decisions. |
| `ChangeModel`, `CHMODEL` | `src/seats/arma/mod.rs`, `src/seats/arma/model.rs`, `src/seats/ar2/mod.rs`, `src/engine/seats/arma.rs` | Initial BQ removal and pure nonseasonal MA(1)→noise refit with INIT=0. The post-SEARCH ChangeModel BQ drop restarts loop 10 with INIT=0 and the persistent M/IQM and IMEAN (`ChangeModelRestart`); other post-SEARCH rewrites, including CHMODEL's unit-root rewrites (P−1/D+1, BP−1/BD+1), remain errors of kind `Unsupported`. |
| `RPQ`, `F1RST` (`ansub2`, `sigsub`) | `src/seats/decomposition/root_allocation.rs`, `src/seats/roots.rs`, `src/seats/ar2/canonical.rs` | Allocate seasonal roots before regular roots; retain existing-seasonal-factor cutoff and promoted REAL constants. |
| `SIGEX` (TD allocation) | `src/seats/trading_day.rs`, `src/seats/decomposition/root_allocation.rs` | The configured TD basis and the polynomial's degree control stochastic-TD allocation. |
| `SIGEX`, `ANALTS` (polynomials) | `src/seats/decomposition/polynomial.rs`, `src/seats/ar2/canonical.rs`, `src/seats/second_difference/model.rs` | Expand at the model's lags, regular factor first, seasonal factor second. |
| `CONV` (`ansub2`) | `src/seats/decomposition/polynomial.rs` | First operand outer, copied second operand inner; gfortran's fused accumulation, starting from +0. |
| `CONJ`, `MULTFN` (`ansub2`) | `src/seats/decomposition/harmonic.rs` | Fold into absolute lag; CONV precedes CONJ and `(conv+conj)/2`; preserve zero tails. |
| `DIVFCN`, `PARFRA`, `CONVM`, `CONJM` (`ansub2`) | `src/seats/decomposition/rational.rs` | Descending harmonic quotient and systems of the polynomials' own degree; no dummy seasonal system when the seasonal factor is absent. |
| `MLTSOL` (`ansub2`) | `src/seats/linear.rs` | Reverse rows, strict first-largest unused pivot, in-loop tiny-cell clamp and source fused elimination. |
| `SPECTRU`, `DecompSpectrum` (`spectrum`) | `src/seats/decomposition/canonical.rs`, `src/seats/decomposition/adjusted.rs`, `src/seats/ar2/canonical.rs`, `src/seats/decomposition/minima.rs`, `src/seats/decomposition/rational.rs` | Preserve white-noise transfer and canonical subtraction order; negative irregular is not an admissible decomposition. With no trend, seasonal or cycle factor (zero seasonal AR), SIGEX's pstar still counts BP, so Qt=0 and no component takes RT. |
| `FUNC0`, `FBIS`, `MINIM`, `MINIMbis`, `GlobalMINIM` | `src/seats/decomposition/spectrum.rs` | FBIS compares raw angle and clamps low trend frequencies only with differencing; losing trials still overwrite convergence status. |
| `MAK1` (`spectrum`) | `src/seats/decomposition/factorization.rs` | Source nounit=0 factorization and ordered root reconstruction. |
| `SYMPOLY`, `C02AEF`, `C02AEZ`, `ROOTC` | `src/seats/roots.rs`, `src/seats/decomposition/factorization.rs` | Reciprocal/root grouping, source precision constants and explicit contraction sites. The Rust keeps the arithmetic of C02AEF and C02AEZ but not their structure. |
| `AUTOCOMP` (`ansub4`) | not translated (see `src/seats/mod.rs`) | Only printed tables read the Wiener-Kolmogorov filters. The component geometry and the convolution operand order determine them. |
| `BFAC`, `MPB`, `MPBF`, `getPSIE` (`ansub3`) | not translated | Used only by `AUTOCOMP`. Box-Jenkins tails regain their true signs, MPBF reverses its second operand before MPB, and the cutoffs differ from CONV's. |
| `FCAST` (`ansub1`) | `src/seats/tails.rs` | SEATS's internal extensions differ from the regARIMA forecasts the API returns; residual cells start at +0 and horizons have distinct floors. |
| `RATF`, `SESFCAST` (`ansub1`), ANALTS's residual variance | not translated | No returned output reads the psi weights, SEATS's forecast standard errors or the variance that scales them. The API's forecast standard errors come from the regARIMA forecast. |
| `ESTBUR` (`ansub3`) | `src/seats/extraction.rs` | Partition/endpoint systems, reversal and mean restoration use the model's AR/MA dimensions. |
| `DETCOMP`, `TDLOM`, `ADJREG` | `src/seats/effects.rs`, `src/engine/seats/effects.rs` | Calendar and outlier factors are applied separately, in order; LS returns to trend, AO/TC to irregular. |
| `SECOND`, `log1313` (`sigsub`, `ansub5`) | `src/seats/log_transform.rs` | Bias1 uses one increasing-index loop; working cycle may affect results even when S14 is absent. |
| `USRENTRY`, `seatad` | `src/seats/extraction.rs`, `src/seats/log_transform.rs`, `src/engine/seats/strategy.rs` | Component availability and percentage-to-factor conversion are distinct from mathematical zero. |

### Plain words in public messages

Public docs and error messages use plain words; these are the Build 62 names
behind them.

| In messages and docs | Build 62 |
| --- | --- |
| SEATS optimizer, normalized optimizer residual | `SEARCH` (`ansub1`) |
| autocorrelation …, Ljung-Box Q, seasonal Ljung-Box Q, lag-zero autocovariance | `AUTO` (`ansub1`): `C0`, `Qstat`, `QS` |
| starting-value autocorrelation lags | `STAVAL`'s read of `AUTO` lags |
| original mean, stationary mean, residual mean | `TMU`, `DMEAN` |
| variance mean, variance square sum, variance second moment, variance | `DVAR` |
| common-factor cubic … | `TRANSC`'s cubic detector |
| determinant | `Detpri` |
| differenced series (regularly, seasonally, centered); corrected mean | `WD`; `WM` |
| residual standard deviation; unwritten free-coefficient count | `rstd`; `SEARCH`'s `nv` |
| root modulus | `RPQ` |
| zero-MA objective; grouped regular AR coefficient transform | the `Qstar=0` objective; grouped `TRANSC` |
| AR conditioning scale, AR-conditioned lag-zero/lag-one/lag-two/seasonal-lag term, conditioned autocorrelations, MA2 start value | `STAVAL` locals `c`, `r0`, `rp1`, `rp2`, `rmq`, `r`, `x` |
| restart coordinate; from fresh starting values / from the imported model | `INIT=0` / `INIT=2` |
| SEATS input series; linearized series; SEATS trading-day, Easter, trend outlier and irregular outlier effects | `Oz`; `TRAM`; `PATD`, `PAEAST`, `PAOUTR`, `PAOUIR` |
| SEATS input exp/log | `INVFCN` into `OZ`/`Tram`; the `ANALTS` log |
| trading-day SA divisor, seasonally adjusted, bias-corrected trend | `SECOND` |
| Box-Jenkins MA; AR2 differenced AR polynomial | BJ-convention MA; `BPHIST` |
| root solver code | `IFAIL` |
| constant-term test (initial / after AR-root or overdifferencing repair; retests) | `chkmu`; `pass0` |
| MA / AR initial-state covariance not positive definite; ARMA autocovariances | `armafl` codes `PGPGER`, `PVWPER`, `PACFER` |
| Build 62's capacity in positions or calendar years | `srslen.prm` `PLEN`, `PYRS` |
| every automatic ARMA order candidate failed (`Error::Numerical`) | `amdid.f`: "Cannot make a choice of … ARMA order" |
| differencing reduces the series to zeros (`DegenerateSeries`) | `rgarma.f` `POBFN0`; `editor.f` `Same` with gaps |
| `Spec::x11`'s mode when none is set | `gtinpt.f` `Muladd=0` |

## Oracle-only interfaces

These Build 62 routines define the input and output formats that `x13-oracle`
writes and reads. Nothing here is part of the `x13` crate.

| Fortran routine(s) | Rust module(s) | Mapping / reproduced quirk |
| --- | --- | --- |
| `gtfree`, `gtfldt` | `oracle/src/spec_file/mod.rs` | Render roundtrip decimal inputs through file/free format, avoiding inline ctod accumulation differences. |
| `getx11`, `getreg`, `gtseat`, `gtauto`, `gtfcst`, `gtinpt` | `oracle/src/configuration/`, `oracle/src/spec_file/` | Render explicit supported specs; summary flag is conditional because it affects SEATS decisions. |
| `dtoc`, `savtbl`, `savmtx` | `oracle/src/saved_tables/series.rs`, `oracle/src/saved_tables/mod.rs` | Read saved 15-significant-digit cells, including three-digit exponents without an E marker. |
| `savitr`, `prtitr`, `prlkhd` | `oracle/src/parity/output.rs` | Final parameters and variance come from different routines at different precision. |
| `svolit`, `savotl`, `idotlr` (output) | `oracle/src/saved_tables/outlier.rs` | OIT records added/deleted events, FTS final candidates; printed intermediate candidates are not full precision. |
| `amdprt`, `prarma`, `savmdl` | `oracle/src/saved_tables/selected_model.rs`, `oracle/src/spec_file/automatic_model.rs` | Distinguish selected and final fitted orders; preserve exact discrete choices. |
| `seatpr`, `seatdg` | `oracle/src/parity/output.rs` | Parse S10–S14, each of which may be absent, and the effective model; a Build 62 rejection is not numerical output. |

Build 62's formatting and batch helpers, and the reports this crate does not
produce, have no Rust counterpart. Deviations and comparison tolerances are
listed in `tests/parity/deviations.toml` and `tests/parity/tolerances.toml`.
