//! Runs one case: Build 62 (cached), then Rust, then classifies the pair.
//!
//! # Both-reject classification
//!
//! A case where both implementations reject passes, but as its own
//! [`Status::BothRejected`], so the report can show how much of a row's pass
//! rate is agreement on refusal rather than on numbers. The pair of messages
//! is kept (and written to `target/parity-both-rejected.json`) so the Rust
//! error can be checked by eye.
//!
//! The harness judges "sensible" mechanically only in one direction: a
//! failed internal check (`ErrorKind::Internal`) and the catch-all numerical
//! failure say nothing about the input, so agreeing with a Build 62 refusal
//! through them is coincidence and the case fails as "insensible
//! rejection". Unsupported and panicking Rust
//! outcomes never count as rejections. Build 62's messages are free text, so
//! the typed error's variant is not matched to its reason; the rejection map
//! below gives the coarser judgment.
//!
//! # Rejection map
//!
//! `tests/parity/rejection-map.toml` maps each Build 62 rejection, by a
//! literal prefix of the first line of its message, to the `ErrorKind`s that
//! are a sensible Rust rejection of it.
//! - **A prefix**, because Build 62's messages are fixed format strings with
//!   series-specific insertions (`Adding MV1977.Jan exceeds ...`): the text
//!   before the first insertion names the cause, so one line covers a cause
//!   across series. Prefixes may not start one another, so each message has at
//!   most one entry.
//! - **The first line**, because the classifier joins Build 62's `ERROR:`
//!   lines in file order and the first line is what Build 62 reports first,
//!   the one line every case has. It is usually the cause, but not always:
//!   idotlr's format 1200 ("regARIMA model estimation in outlier
//!   identification procedure did not converge") is a wrapper whose second
//!   line ("Regression matrix singular because of ...") is the cause, so
//!   that entry's kinds follow the causes seen after it.
//! - **Kinds, not variants**: `Error::kind()` is the public classification by
//!   cause and does not change when variants are added or renamed. A kind is
//!   coarse, so a same-kind error with another cause also passes. The kind
//!   comes from the live typed error (`TypedRejection`), never from the
//!   record's Debug text, so the bit records stay as they are. `Unsupported`
//!   is never allowed: such an outcome is already a failure.
//! - **Enforced**: `rejection_map::ENFORCED` is on. A miss (unmapped, or a kind
//!   outside the allowed set) fails the case with a `rejection map` failure
//!   before deviations apply, so a deviation can still cover it. The report's
//!   "Both-reject off map" column counts those failures, so it reads the same
//!   whether or not the switch is on: with it off, a miss keeps its
//!   both-reject status, is counted there and is listed in
//!   `target/parity-both-rejected.json`. Turning the switch off lets a map
//!   edit be previewed before it fails cases.
//!
//! # Explicit SEATS refusals
//!
//! Build 62 exits zero on some SEATS refusals: ShowComp's innovation-variance
//! test (Lfatal, `seatsadj: no`) and ANALTS's degenerate-model stop
//! (5015/5016), which skips the decomposition. Both leave no SEATS tables, so
//! the output parse would fail. `build62_rejection` reports them as
//! rejections instead, letting the both-reject rules judge Rust's error.
//! ShowComp has its own error-file text. The degenerate stop has none that
//! holds at every frequency: SPCDRV's note runs only for monthly series
//! (x11ari.f calls it at Ny=12) and also fires on an all-zero stochastic SA
//! (Hvstsa), and seatdg.f's log line needs a SEATS `savelog`. Its signal is
//! therefore [`SeatsTables::NoneSaved`]: an exit-zero run that requested SEATS
//! and saved none of its component tables. Requiring that no table was saved
//! keeps any decomposition a success, including one whose SA is stored all
//! zero (Havesa false, so no S11, while S12 or S13 is still saved); it then
//! fails visibly in the parser instead of passing as a refusal.
//!
//! # Deadline
//!
//! A Build 62 run that outlives [`DEADLINE`] is treated as a hang and
//! fails the case (the known hangs are listed deviations). The slowest
//! legitimate runs (PI with automatic modelling) take 120–169 s alone, and
//! the harness runs one case per hardware thread, some on efficiency cores,
//! often beside other worktrees' tables, so the limit leaves about 3.5×
//! headroom. A real hang still ends as a timeout. Timeout markers are cached
//! per limit, so raising it reruns each cached hang once at the new limit,
//! and the hang deviations' `fortran` text names the limit, so it changes with
//! them.
//!
//! # Summary buffer overflow
//!
//! Build 62's `-s` summary makes x11pt3 call prtd8b, which lists each calendar
//! month's flagged years in a 400-character buffer (`outstr*(400)`) and
//! flushes it only past 512. A month with many regARIMA outliers and X-11
//! extremes overruns it: itoc.f refuses a year that no longer fits
//! (`Can't write <year> in 400 spaces`, then errorstop), and the unchecked
//! marker writes smash the stack (SIGABRT after the loop). Backcasts add
//! enough outliers to reach it on long series (REALLN and CONSUMER). The
//! overrun touches only a print buffer, and without `-s` Build 62 completes
//! with identical saved tables.
//!
//! [`run_with_summary_retry`] therefore reruns a `-s` X-11-method run without
//! `-s` when it dies with that signature, and uses the rerun. SEATS rows that
//! use `-s` are never rerun (prtd8b is X-11's, and their reference needs the
//! summary: [`SummaryMode`]). The `-s` UDG written before the death still
//! supplies the resolved transformation and model (`parse_output` reads them
//! there, and only a `-s` run writes them), minus its `errorstop` line. Every
//! other death or rejection of a `-s` run stays as it was.
//!
//! Two limits. The SIGABRT arm cannot tell prtd8b's abort from another abort
//! of a `-s` X-11 run; the rerun then reproduces the abort and the case fails
//! as before. And after an abort gfortran's unflushed UDG buffer is lost, so
//! the retained prefix can miss the model lines: the case then fails in the
//! parse, with the rerun named in the message ([`for_rerun`]), never with a
//! wrong value. The cache key is unchanged, so entries of runs that never
//! overflowed stay valid; the rerun's tables and the retained UDG prefix make
//! up the cached record. A cold cache pays for the dying `-s` run first (70 to
//! 110 s on CONSUMER), and for both runs twice, once per stability pass.
//!
//! # Rust drift on rejections
//!
//! Recorded Rust rejections are compared by `Error` variant (the leading
//! identifiers of the recorded Debug text, through the payload wrappers of
//! `Error::Numerical` and its like), not by payload, so a change of
//! discriminant is drift while a reworded or re-numbered payload is not.

use std::{
    fs,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use super::{
    cache::{Pair, cache_key, cached_pair},
    compare::{Failure, Headroom, Tolerances},
    data::SpecRow,
    output::{Output, bit_record, parse_output, rust_output},
};
use crate::{Frequency, Series};

/// Wall-clock limit for one Build 62 run; see "Deadline" above.
const DEADLINE: Duration = Duration::from_secs(600);

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Build62 {
    Success(Box<Output>),
    // The serde name versions cached rejections: change it when what counts as
    // a rejection changes, so stale cache entries are recomputed.
    #[serde(rename = "RejectedV3")]
    Rejected(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Rust {
    Success(Box<Output>),
    Rejected(String),
    Unsupported(String),
    Failed(String),
}

#[derive(Debug)]
pub(crate) enum Status {
    Pass,
    /// Build 62 and Rust both reject, and Rust's error is sensible. Counted as
    /// a pass, but reported separately: no output was compared.
    BothRejected {
        fortran: String,
        rust: String,
        /// `kind()` of the live typed error, for the rejection map.
        kind: x13::ErrorKind,
    },
    Deviation(Vec<usize>),
    Nondeterministic,
    Fail(Vec<Failure>),
}

impl Status {
    pub(crate) fn passes(&self) -> bool {
        matches!(
            self,
            Self::Pass | Self::BothRejected { .. } | Self::Deviation(_)
        )
    }
}

pub(crate) struct CaseResult {
    pub(crate) id: String,
    pub(crate) spec_id: String,
    pub(crate) frequency: Frequency,
    pub(crate) status: Status,
    pub(crate) headroom: Option<super::compare::Headroom>,
    pub(crate) drift: Drift,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct Drift {
    pub(crate) incremental: Option<Vec<Headroom>>,
    pub(crate) cumulative: Option<Vec<Headroom>>,
}

impl CaseResult {
    #[cfg(test)]
    pub(crate) fn test(id: &str, frequency: Frequency, status: Status) -> Self {
        Self {
            id: id.into(),
            spec_id: id.split_once('/').unwrap().1.into(),
            frequency,
            status,
            headroom: None,
            drift: Drift::default(),
        }
    }
}

pub(crate) struct ContextPaths<'a> {
    pub(crate) executable: &'a Path,
    pub(crate) cache: &'a Path,
    pub(crate) record: Option<&'a Path>,
    pub(crate) compare: Option<&'a Path>,
    pub(crate) tolerant: Option<&'a Path>,
    pub(crate) cumulative: Option<&'a Path>,
}

pub(crate) fn run_case(
    series: &Series,
    row: &SpecRow,
    paths: &ContextPaths<'_>,
    tolerances: &Tolerances,
) -> CaseResult {
    let id = format!("{}/{}", series.series_id, row.id);
    let mut headroom = None;
    let mut drift = Drift::default();
    let status = execute(series, row, paths, tolerances, &mut headroom, &mut drift).unwrap_or_else(
        |error| {
            Status::Fail(vec![Failure::text(
                "harness execution/parse",
                "successful execution and parse",
                format!("{error:#}"),
            )])
        },
    );
    CaseResult {
        id,
        spec_id: row.id.clone(),
        frequency: series.frequency,
        status,
        headroom,
        drift,
    }
}

fn execute(
    series: &Series,
    row: &SpecRow,
    paths: &ContextPaths<'_>,
    tolerances: &Tolerances,
    headroom: &mut Option<super::compare::Headroom>,
    drift: &mut Drift,
) -> Result<Status> {
    let config = if series.frequency == Frequency::Quarterly {
        row.quarterly_configuration
            .as_ref()
            .unwrap_or(&row.configuration)
    } else {
        &row.configuration
    };
    let spec = crate::spec_file::render(series, config)?;
    let input = crate::spec_file::render_input(series)?;
    let summary = crate::spec_file::uses_summary(config);
    let columns: &[&str] = if super::output::requests_backcasts(config)? {
        &["backcasts"]
    } else {
        &[]
    };
    let key = cache_key(
        &spec,
        &input,
        &paths.executable.to_string_lossy(),
        summary,
        columns,
    );
    let build62_started = Instant::now();
    let timeout = DEADLINE;
    let seats = config.method() == crate::Method::Seats;
    let mode = SummaryMode::of(summary, seats);
    let build62 = cached_pair(paths.cache, &key, timeout, || {
        let temp = tempfile::tempdir()?;
        let run = run_with_summary_retry(temp.path(), mode, |directory, summary| {
            Ok(crate::build62::run(crate::build62::Run {
                executable: paths.executable,
                directory,
                spec: &spec,
                input: &input,
                summary,
                timeout,
            })?)
        })?;
        let Build62Run {
            process,
            directory: raw,
            error_text,
            summary: udg,
            retried,
        } = run;
        ensure!(
            process.status.code().is_some(),
            "Build 62 was terminated by a signal"
        );
        if let Some(message) = build62_rejection(
            &String::from_utf8_lossy(&process.stdout),
            process.status.success(),
            &error_text,
            &udg,
            SeatsTables::of_run(seats, &raw)?,
        )? {
            return Ok(Build62::Rejected(message));
        }
        for_rerun(parse_output(&raw, series, config), retried)
            .map(Box::new)
            .map(Build62::Success)
    });
    if build62_started.elapsed().as_secs() >= 30 {
        eprintln!(
            "Slow Fortran/cache {}/{}: {:.1}s",
            series.series_id,
            row.id,
            build62_started.elapsed().as_secs_f64()
        );
    }
    // Rust still runs and is recorded when the Build 62 run or its parse fails.
    let rust_started = Instant::now();
    let adjusted = std::panic::catch_unwind(|| crate::rust_spec::adjust(series, config));
    let typed = match &adjusted {
        Ok(Err(crate::RustError::Adjustment(error))) => Some(TypedRejection::of(error)),
        _ => None,
    };
    if rust_started.elapsed().as_secs() >= 5 {
        eprintln!(
            "Slow Rust {}/{}: {:.1}s",
            series.series_id,
            row.id,
            rust_started.elapsed().as_secs_f64()
        );
    }
    let (rust, adjustment) = match adjusted {
        Ok(Ok(adjustment)) => match rust_output(series, &adjustment) {
            Ok(output) => (Rust::Success(Box::new(output)), Some(adjustment)),
            Err(error) => (Rust::Failed(error.to_string()), None),
        },
        Ok(Err(crate::RustError::Adjustment(error))) => (
            if error.kind() == x13::ErrorKind::Unsupported {
                Rust::Unsupported(format!("{error:?}"))
            } else {
                Rust::Rejected(format!("{error:?}"))
            },
            None,
        ),
        Ok(Err(crate::RustError::UnsupportedConfiguration)) => {
            (Rust::Unsupported("configuration".into()), None)
        }
        Ok(Err(error)) => (Rust::Failed(error.to_string()), None),
        Err(_) => (Rust::Failed("panic".into()), None),
    };
    let bits = bit_record(&rust)?;
    let relative = Path::new(&series.series_id).join(format!("{}.json", row.id));
    let record_mismatch = paths
        .compare
        .map(|root| {
            fs::read(root.join(&relative))
                .map(|expected| expected != bits)
                .unwrap_or(true)
        })
        .unwrap_or(false);
    if record_mismatch {
        return Ok(Status::Fail(vec![Failure::text(
            "Rust bit record",
            "identical recorded output",
            "missing or different output",
        )]));
    }
    let strict_arma_gate = match &build62 {
        Ok(Pair::Stable(Build62::Success(output))) => {
            output.model.as_ref().map(|m| m.strict_arma_gate)
        }
        _ => None,
    };
    let mut drift_failures = vec![];
    for (scope, root, metrics) in [
        ("incremental", paths.tolerant, &mut drift.incremental),
        ("cumulative", paths.cumulative, &mut drift.cumulative),
    ] {
        if let Some(root) = root {
            let compared = (|| {
                let recorded =
                    super::output::decode_bit_record::<Rust>(&fs::read(root.join(&relative))?)?;
                compare_recorded(
                    recorded,
                    &rust,
                    adjustment.as_ref(),
                    series,
                    tolerances,
                    strict_arma_gate,
                )
            })();
            match compared {
                Ok((mut failures, values)) => {
                    for failure in &mut failures {
                        failure.output = format!("Rust {scope} drift: {}", failure.output);
                    }
                    failures.extend(super::compare::drift_failures(scope, &values));
                    drift_failures.extend(failures);
                    *metrics = Some(values);
                }
                Err(error) => drift_failures.push(Failure::text(
                    &format!("Rust {scope} baseline"),
                    "valid recorded output",
                    format!("{error:#}"),
                )),
            }
        }
    }
    if !drift_failures.is_empty() {
        // As in the comparison with Build 62, keep one failure per output and scope.
        // The drift metrics hold the full maxima. One failure per output is
        // enough here: one per cell would run to gigabytes.
        let mut outputs = std::collections::BTreeSet::new();
        drift_failures.retain(|failure| outputs.insert(failure.output.clone()));
        return Ok(Status::Fail(drift_failures));
    }
    // Recording comes after every baseline is read, so a missing baseline
    // record fails the case before anything is written.
    if let Some(root) = paths.record {
        fs::create_dir_all(root.join(&series.series_id))?;
        write_new_record(&root.join(&relative), &bits)?;
    }
    let build62 = match build62 {
        Ok(build62) => build62,
        Err(error) => return Ok(build62_failure(error, &rust)),
    };
    let Pair::Stable(build62) = build62 else {
        return Ok(Status::Nondeterministic);
    };
    if let Some(status) = rejection_status(&build62, &rust, typed) {
        return Ok(status);
    }
    match (build62, rust, adjustment) {
        (Build62::Success(expected), Rust::Success(actual), Some(adjustment)) => {
            let (mut failures, mut metrics) =
                super::compare::compare(&expected, &actual, &adjustment, series, tolerances);
            let (unrecorded_failures, unrecorded_metrics) =
                super::compare::compare_unrecorded(&expected, &adjustment, series, tolerances)?;
            failures.extend(unrecorded_failures);
            metrics.extend(unrecorded_metrics);
            *headroom = super::compare::worst(&metrics);
            // All cells were compared. Retain the first differing index per
            // output, rather than thousands of redundant cells per failed case.
            let mut outputs = std::collections::BTreeSet::new();
            failures.retain(|failure| outputs.insert(failure.output.clone()));
            Ok(if failures.is_empty() {
                Status::Pass
            } else {
                Status::Fail(failures)
            })
        }
        (build62, rust, _) => Ok(Status::Fail(vec![Failure::text(
            "outcome",
            outcome_build62(&build62),
            outcome_rust(&rust),
        )])),
    }
}

fn write_new_record(path: &Path, bits: &[u8]) -> Result<()> {
    // `create_new` fails on an existing file, so a record is never overwritten.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bits)?;
    Ok(())
}

fn compare_recorded(
    recorded: Rust,
    actual: &Rust,
    adjustment: Option<&x13::Adjustment>,
    series: &Series,
    tolerances: &Tolerances,
    strict_arma_gate: Option<bool>,
) -> Result<(Vec<Failure>, Vec<Headroom>)> {
    match (recorded, actual) {
        (Rust::Success(mut expected), Rust::Success(actual)) => {
            if let Some(model) = &mut expected.model {
                // A Rust record holds the model, not the precision Build 62
                // printed it with, so take the gate from this run's Build 62 output.
                model.strict_arma_gate = strict_arma_gate.ok_or_else(|| {
                    anyhow::anyhow!("no Build 62 model precision for the drift comparison")
                })?;
            }
            Ok(super::compare::compare(
                &expected,
                actual,
                adjustment.ok_or_else(|| anyhow::anyhow!("missing successful Rust adjustment"))?,
                series,
                tolerances,
            ))
        }
        // The record keeps the error's Debug text, whose leading identifiers
        // name the `x13::Error` variant. Payloads may carry drifting bit
        // patterns.
        (Rust::Rejected(expected), Rust::Rejected(actual))
            if error_variant(&expected) == error_variant(actual) =>
        {
            Ok((vec![], vec![]))
        }
        (expected, actual) => Ok((
            vec![Failure::text(
                "outcome",
                outcome_rust(&expected),
                outcome_rust(actual),
            )],
            vec![],
        )),
    }
}

fn outcome_build62(value: &Build62) -> String {
    match value {
        Build62::Success(_) => "success".into(),
        Build62::Rejected(s) => format!("rejected: {s}"),
    }
}

fn outcome_rust(value: &Rust) -> String {
    match value {
        Rust::Success(_) => "success".into(),
        Rust::Rejected(s) => format!("rejected: {s}"),
        Rust::Unsupported(s) => format!("unsupported: {s}"),
        Rust::Failed(s) => format!("failed: {s}"),
    }
}

/// One of Build 62's human-readable reports, decoded lossily as `parse_output` does.
/// An absent report reads as empty; any other read error surfaces.
fn read_report(path: &Path) -> Result<String> {
    match fs::read(path) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(anyhow::Error::new(error).context(format!("reading {}", path.display()))),
    }
}

/// The variant an error's Debug text names: its leading identifiers, read
/// through tuple wrappers, so `Numerical(NumericalFailure(X11PseudoAdditive`
/// for a payload variant and `EmptyInput` for a plain one.
fn error_variant(debug: &str) -> &str {
    let end = debug
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '('))
        .unwrap_or(debug.len());
    &debug[..end]
}

/// Rust errors that are never a sensible rejection: a failed internal check
/// and the catch-all numerical failure. Neither says what is wrong with the
/// input, so agreeing with a Build 62 rejection would be coincidence.
fn insensible_rejection(error: &x13::Error) -> bool {
    error.kind() == x13::ErrorKind::Internal || unspecified_numerical_failure(&format!("{error:?}"))
}

/// Whether Debug text is that of the catch-all numerical failure. Its detail
/// is private to `x13`, so the text is the only way to tell it from a
/// numerical failure that names its cause.
fn unspecified_numerical_failure(debug: &str) -> bool {
    debug.starts_with("Numerical(NumericalFailure(Unspecified")
}

/// What the harness reads from a live typed Rust error before it becomes
/// record text: its kind for the rejection map, and whether it can be a
/// sensible rejection at all.
#[derive(Debug, Clone, Copy)]
struct TypedRejection {
    kind: x13::ErrorKind,
    sensible: bool,
}

impl TypedRejection {
    fn of(error: &x13::Error) -> Self {
        Self {
            kind: error.kind(),
            sensible: !insensible_rejection(error),
        }
    }
}

/// Classifies a case both implementations reject. `None` when either side
/// did anything else; unsupported or failed Rust outcomes never match.
/// `typed` is present whenever Rust's outcome is a rejection.
fn rejection_status(
    build62: &Build62,
    rust: &Rust,
    typed: Option<TypedRejection>,
) -> Option<Status> {
    let (Build62::Rejected(fortran), Rust::Rejected(error), Some(typed)) = (build62, rust, typed)
    else {
        return None;
    };
    Some(if typed.sensible {
        Status::BothRejected {
            fortran: fortran.clone(),
            rust: error.clone(),
            kind: typed.kind,
        }
    } else {
        Status::Fail(vec![Failure::text(
            "insensible rejection",
            outcome_build62(build62),
            outcome_rust(rust),
        )])
    })
}

fn build62_failure(error: anyhow::Error, rust: &Rust) -> Status {
    Status::Fail(vec![Failure::text(
        "Fortran execution/parser",
        format!("failed: {error:#}"),
        outcome_rust(rust),
    )])
}

/// Which SEATS component tables a Build 62 run saved, for
/// [`build62_rejection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeatsTables {
    /// The spec requests no SEATS decomposition.
    NotRequested,
    /// SEATS was requested and at least one component table was saved.
    Saved,
    /// SEATS was requested and no component table was saved.
    NoneSaved,
}

impl SeatsTables {
    /// Reads the run directory of a `-o output` run. The tables are the
    /// SEATS components the harness and the fuzz specs save (S10–S14).
    pub(crate) fn of_run(seats: bool, directory: &Path) -> Result<Self> {
        if !seats {
            return Ok(Self::NotRequested);
        }
        for table in ["s10", "s11", "s12", "s13", "s14"] {
            if directory.join(format!("output.{table}")).try_exists()? {
                return Ok(Self::Saved);
            }
        }
        Ok(Self::NoneSaved)
    }
}

/// How one case invokes Build 62: whether the first run gets `-s`, and
/// whether a `-s` run that dies in prtd8b may be rerun without it. They differ
/// for SEATS: its `-s` rows need the summary (the UDG, and arima.f's Lsumm gate
/// on the Q statistic), but prtd8b belongs to X-11, so they are never rerun.
#[derive(Debug, Clone, Copy)]
struct SummaryMode {
    with_summary: bool,
    retry_on_overflow: bool,
}

impl SummaryMode {
    /// `uses_summary` is [`super::configuration::uses_summary`] for the case.
    fn of(uses_summary: bool, seats: bool) -> Self {
        Self {
            with_summary: uses_summary,
            retry_on_overflow: uses_summary && !seats,
        }
    }
}

/// A finished Build 62 run: its process and run directory, plus the `.err`
/// and `.udg` text [`build62_rejection`] reads. `retried` marks a rerun
/// without `-s`.
struct Build62Run {
    process: std::process::Output,
    directory: std::path::PathBuf,
    error_text: String,
    summary: String,
    retried: bool,
}

/// Runs Build 62 through `run(directory, with_summary)` in a fresh directory
/// under `temp`. When a `-s` run of the X-11 method dies in prtd8b's summary
/// block (`mode.retry_on_overflow`), reruns it without `-s` and returns the
/// rerun; see "Summary buffer overflow" in the module docs.
fn run_with_summary_retry(
    temp: &Path,
    mode: SummaryMode,
    mut run: impl FnMut(&Path, bool) -> Result<std::process::Output>,
) -> Result<Build62Run> {
    let directory = temp.join("run");
    let process = run(&directory, mode.with_summary)?;
    let error_text = read_report(&directory.join("output.err"))?;
    let udg = read_report(&directory.join("output.udg"))?;
    if !(mode.retry_on_overflow
        && overflowed_summary_buffer(terminating_signal(&process.status), &error_text))
    {
        return Ok(Build62Run {
            process,
            directory,
            error_text,
            summary: udg,
            retried: false,
        });
    }
    let retry = temp.join("retry");
    let process = run(&retry, false)?;
    // The rerun writes no UDG, and `parse_output` takes the resolved
    // transformation and model from it. The abend's errorstop line is the
    // overflow itself, not a property of the adjustment.
    let udg = udg
        .lines()
        .filter(|line| line.trim() != "errorstop: yes")
        .flat_map(|line| [line, "\n"])
        .collect::<String>();
    fs::write(retry.join("output.udg"), &udg)?;
    Ok(Build62Run {
        error_text: read_report(&retry.join("output.err"))?,
        process,
        directory: retry,
        summary: udg,
        retried: true,
    })
}

/// Names the rerun when its parse fails. The retained UDG is what the abort
/// left on disk: after a SIGABRT, gfortran's unflushed buffer is lost, so the
/// `transform`, `aictrans` or `arimamdl` line `parse_output` needs can be
/// missing. That fails the case, never yields a wrong value.
fn for_rerun<T>(parsed: Result<T>, retried: bool) -> Result<T> {
    parsed.map_err(|error| {
        if retried {
            error.context(
                "parsing the rerun without -s, whose model choices come from the -s UDG \
                 written before prtd8b died (an abort can lose its buffered tail)",
            )
        } else {
            error
        }
    })
}

/// Whether a `-s` run died in prtd8b's summary block. Matches only itoc.f's
/// refusal to write a year into `outstr*(400)` (an errorstop, exit code zero),
/// or SIGABRT, which is the stack protector catching the unchecked marker
/// writes past column 400 when prtd8b returns. Any other signal, message or
/// errorstop is a different failure and is not retried.
fn overflowed_summary_buffer(signal: Option<i32>, error_file: &str) -> bool {
    const SIGABRT: i32 = 6;
    let itoc_refusal = error_file.lines().any(|line| {
        matches!(
            line.split_whitespace().collect::<Vec<_>>()[..],
            ["Error:", "Can't", "write", year, "in", "400", "spaces"] if year.parse::<i64>().is_ok()
        )
    });
    itoc_refusal || signal == Some(SIGABRT)
}

#[cfg(unix)]
fn terminating_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

#[cfg(not(unix))]
fn terminating_signal(_: &std::process::ExitStatus) -> Option<i32> {
    None
}

pub(crate) fn build62_rejection(
    stdout: &str,
    success: bool,
    error_file: &str,
    summary: &str,
    seats_tables: SeatsTables,
) -> Result<Option<String>> {
    let version = stdout
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("Version Number"));
    ensure!(
        version == Some(crate::build62::VERSION_BANNER),
        "missing or wrong Build 62 executable banner"
    );
    let stopped = summary.lines().any(|line| line.trim() == "errorstop: yes");
    // automd calls prterr(..., T) during search; itrerr still writes ERROR to
    // the .err file. Only a successful final automatic fit can supersede those
    // iteration-limit messages. Saved outputs must still parse and compare.
    let recovered_search = success
        && !stopped
        && summary
            .lines()
            .any(|line| line.trim() == "automodeltype: automdl")
        && summary.lines().any(|line| line.trim() == "converged: yes");
    let errors = error_file
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("ERROR:"))
        .filter(|line| {
            // SPCDRV writes these messages without setting Lfatal: only the
            // logged plot is unavailable. Still require successful execution
            // and parsing/comparison of every requested saved component.
            !(success
                && !stopped
                && matches!(
                    *line,
                    "ERROR: Spectral plot for the logged original series cannot be done for"
                        | "ERROR: Spectral plot for the logged seasonally adjusted series cannot"
                ))
        })
        .filter(|line| {
            !(recovered_search
                && matches!(
                    *line,
                    "ERROR: Estimation failed to converge -- maximum function evaluations reached."
                        | "ERROR: Estimation failed to converge -- maximum iterations reached."
                ))
        })
        .collect::<Vec<_>>();
    if !errors.is_empty() {
        return Ok(Some(errors.join("\n")));
    }
    // SEATS x11ari.f writes an unprefixed errext before abend.f records this
    // explicit rejection. A signal/timeout was already rejected by the caller.
    if stopped {
        return Ok(Some("Build 62 errorstop: yes".into()));
    }
    // ansub1.f sets Ierr for these failures; x11ari.f writes Errext and
    // calls abend even without -s (when no errorstop UDG marker is written).
    for line in error_file.lines().map(str::trim) {
        if matches!(
            line,
            "SEARCH : Model degenerate DET(A) <=0"
                | "SEARCH : Model degenerate Diagonal Element too small"
                | "CALCFX : Error DET(B) Zero or Negative"
                | "AUTO: Cannot generate autocorrelations from a series of zeros."
        ) {
            return Ok(Some(line.into()));
        }
    }
    // spectrum.f format 7052 precedes an alternate return: no decomposition
    // is produced, despite the console calling this a WARNING and exit 0.
    const INVALID_DECOMPOSITION: &str = "DECOMPOSITION INVALID,IRREGULAR SPECTRUM NEGATIVE";
    if error_file
        .lines()
        .any(|line| line.trim() == INVALID_DECOMPOSITION)
    {
        return Ok(Some(INVALID_DECOMPOSITION.into()));
    }
    // spectrum.f ShowComp format 9000 sets Lfatal (and `seatsadj: no`) when a
    // component innovation variance is above one or below zero; exit is still
    // zero. The transitory call shifts its arguments ("transitoryThe
    // innovation variance of the greater than one is"), hence `contains`.
    const UNSUITABLE_VARIANCE: &str = "The innovation variance of the";
    if let Some(line) = error_file
        .lines()
        .find(|line| line.contains(UNSUITABLE_VARIANCE))
    {
        return Ok(Some(line.trim().into()));
    }
    ensure!(
        success,
        "nonzero Build 62 exit without an explicit rejection"
    );
    // ANALTS can stop before any decomposition without Lfatal (5015/5016:
    // equal imported AR and MA, a degenerate model). SEATS then stores no
    // component, so seatpr saves no SEATS table. The message is Build 62's
    // own wording (SPCDRV's note, seatdg.f format 1110), which only some runs
    // print; the missing tables are the signal at every frequency.
    if seats_tables == SeatsTables::NoneSaved {
        return Ok(Some("SEATS cannot perform a signal extraction.".into()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_existing_record_is_never_overwritten() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("record.json");
        write_new_record(&path, b"first").unwrap();
        assert!(write_new_record(&path, b"second").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"first");
    }

    #[test]
    fn a_case_fails_on_cumulative_drift_and_on_a_missing_or_corrupt_record() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity");
        let tolerances: Tolerances =
            super::super::data::read_toml(&root.join("tolerances.toml")).unwrap();
        let series = Series {
            series_id: "test".into(),
            frequency: Frequency::Monthly,
            start: crate::Period::new(2000, 1),
            values: (0..144)
                .map(|i| 100. + (i % 12) as f64 + i as f64 * 0.2)
                .collect(),
        };
        let row: SpecRow = serde_json::from_value(serde_json::json!({
            "id":"test", "label":"drift test", "frequencies":["monthly"],
            "length":{"monthly":{"min":1,"max":780}},
            "configuration":{"kind":"direct_x11","mode":"additive","seasonal_filter":"s3x5"}
        }))
        .unwrap();
        let adjustment = crate::rust_spec::adjust(&series, &row.configuration).unwrap();
        let output = rust_output(&series, &adjustment).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let incremental = temp.path().join("incremental");
        let cumulative = temp.path().join("cumulative");
        let executable = Path::new("not-invoked-by-cache-test");
        let key = cache_key(
            &crate::spec_file::render(&series, &row.configuration).unwrap(),
            &crate::spec_file::render_input(&series).unwrap(),
            &executable.to_string_lossy(),
            false,
            &[],
        );
        cached_pair(&cache, &key, Duration::from_secs(180), || {
            Ok(Build62::Success(Box::new(output.clone())))
        })
        .unwrap();
        let record = bit_record(&Rust::Success(Box::new(output.clone()))).unwrap();
        for directory in [&incremental, &cumulative] {
            fs::create_dir_all(directory.join("test")).unwrap();
            write_new_record(&directory.join("test/test.json"), &record).unwrap();
        }
        let paths = ContextPaths {
            executable,
            cache: &cache,
            record: None,
            compare: None,
            tolerant: Some(&incremental),
            cumulative: Some(&cumulative),
        };
        assert!(matches!(
            run_case(&series, &row, &paths, &tolerances).status,
            Status::Pass
        ));
        let mut changed = output;
        changed.components.trend_cycle[0].value *= 1. + 5e-14;
        fs::write(
            cumulative.join("test/test.json"),
            bit_record(&Rust::Success(Box::new(changed))).unwrap(),
        )
        .unwrap();
        let result = run_case(&series, &row, &paths, &tolerances);
        assert!(
            result
                .drift
                .incremental
                .unwrap()
                .iter()
                .all(|m| m.ratio == 0.)
        );
        assert!(
            result
                .drift
                .cumulative
                .unwrap()
                .iter()
                .any(|m| m.ratio > 0.1)
        );
        let Status::Fail(failures) = result.status else {
            panic!("cumulative drift passed")
        };
        assert_eq!(failures.len(), 1);
        assert!(
            failures
                .iter()
                .all(|f| f.output.starts_with("Rust cumulative drift:"))
        );
        fs::write(cumulative.join("test/test.json"), b"corrupt record").unwrap();
        let Status::Fail(failures) = run_case(&series, &row, &paths, &tolerances).status else {
            panic!("corrupt record passed")
        };
        assert!(
            failures
                .iter()
                .any(|f| f.output == "Rust cumulative baseline")
        );
        let missing = temp.path().join("missing-case");
        fs::create_dir(&missing).unwrap();
        let paths = ContextPaths {
            cumulative: Some(&missing),
            ..paths
        };
        let Status::Fail(failures) = run_case(&series, &row, &paths, &tolerances).status else {
            panic!("missing record passed")
        };
        assert!(
            failures
                .iter()
                .any(|f| f.output == "Rust cumulative baseline")
        );
    }

    #[test]
    fn drift_uses_the_parity_tolerances_and_needs_equal_choices_and_shapes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity");
        let tolerances: Tolerances =
            super::super::data::read_toml(&root.join("tolerances.toml")).unwrap();
        let input = Series {
            series_id: "drift-test".into(),
            frequency: Frequency::Monthly,
            start: crate::Period::new(2000, 1),
            values: (0..144)
                .map(|i| {
                    100. + i as f64 * 0.07 + 12. * (i as f64 * std::f64::consts::TAU / 12.).sin()
                })
                .collect(),
        };
        let config = serde_json::from_value(
            serde_json::json!({"kind":"direct_x11", "mode":"additive", "seasonal_filter":"s3x5"}),
        )
        .unwrap();
        let adjustment = crate::rust_spec::adjust(&input, &config).unwrap();
        let mut expected = rust_output(&input, &adjustment).unwrap();
        // The comparator is independent of how these saved fields were fitted.
        // Supply a model and forecast to exercise every output gate in one test.
        expected.forecasts = vec![crate::SeriesPoint {
            period: crate::Period::new(2012, 1),
            value: 2.0,
        }];
        expected.transitory = Some(expected.forecasts.clone());
        expected.model = Some(super::super::output::Model {
            transform: "none".into(),
            orders: [[0, 1, 1], [0, 1, 1]],
            arma: vec![super::super::output::Coefficient {
                name: "ma".into(),
                value: 2.0,
            }],
            regression: vec![],
            outliers: vec![],
            strict_arma_gate: false,
        });
        let compare = |actual: Output, strict| {
            compare_recorded(
                Rust::Success(Box::new(expected.clone())),
                &Rust::Success(Box::new(actual)),
                Some(&adjustment),
                &input,
                &tolerances,
                strict,
            )
            .unwrap()
        };
        let (failures, metrics) = compare(expected.clone(), Some(true));
        assert!(failures.is_empty());
        assert_eq!(metrics.len(), 8);
        assert!(metrics.iter().all(|metric| metric.ratio == 0.0));
        let mut changed = expected.clone();
        changed.forecasts[0].value = f64::from_bits(2_f64.to_bits() + 2);
        changed.model.as_mut().unwrap().arma[0].value = changed.forecasts[0].value;
        let (failures, metrics) = compare(changed.clone(), Some(true));
        assert!(failures.is_empty());
        let forecast_ratio = metrics
            .iter()
            .find(|m| m.output == "forecasts")
            .unwrap()
            .ratio;
        assert!((forecast_ratio - 0.07401486830834377).abs() < 1e-16);
        assert_eq!(
            metrics.iter().find(|m| m.output == "arma").unwrap().ratio,
            forecast_ratio
        );
        assert!(super::super::compare::drift_failures("incremental", &metrics).is_empty());
        changed.forecasts[0].value = f64::from_bits(2_f64.to_bits() + 3);
        let (_, cumulative) = compare(changed, Some(true));
        assert_eq!(
            super::super::compare::drift_failures("cumulative", &cumulative).len(),
            1
        );
        for edit in [
            |o: &mut Output| o.model.as_mut().unwrap().transform = "log".into(),
            |o: &mut Output| o.model.as_mut().unwrap().orders[0][0] = 1,
            |o: &mut Output| {
                o.model
                    .as_mut()
                    .unwrap()
                    .outliers
                    .push(crate::OracleOutlierVariable {
                        outlier_type: crate::OracleOutlierType::Additive,
                        period: crate::Period::new(2001, 3),
                    })
            },
            |o: &mut Output| o.model = None,
            |o: &mut Output| o.transitory = None,
            |o: &mut Output| {
                o.forecasts.clear();
            },
            |o: &mut Output| {
                o.components.trend_cycle.pop();
            },
            |o: &mut Output| o.forecasts[0].period = crate::Period::new(2012, 2),
        ] {
            let mut changed = expected.clone();
            edit(&mut changed);
            assert!(!compare(changed, Some(true)).0.is_empty());
        }
        let rejected = |text: &str| Rust::Rejected(text.into());
        let typed = || rejected("Numerical(NumericalFailure(X11Trend { period: 2001-03 }))");
        assert!(
            compare_recorded(
                typed(),
                &rejected("Numerical(NumericalFailure(X11Trend { period: 2001-04 }))"),
                None,
                &input,
                &tolerances,
                None
            )
            .unwrap()
            .0
            .is_empty()
        );
        for other in [
            "EmptyInput",
            "Numerical(NumericalFailure(X11Easter { period: 2001-03 }))",
        ] {
            assert!(
                !compare_recorded(typed(), &rejected(other), None, &input, &tolerances, None)
                    .unwrap()
                    .0
                    .is_empty()
            );
        }
        let rejected = typed;
        assert!(
            !compare_recorded(
                rejected(),
                &Rust::Success(Box::new(expected.clone())),
                Some(&adjustment),
                &input,
                &tolerances,
                Some(true)
            )
            .unwrap()
            .0
            .is_empty()
        );
        assert!(
            compare_recorded(
                Rust::Success(Box::new(expected.clone())),
                &Rust::Success(Box::new(expected)),
                Some(&adjustment),
                &input,
                &tolerances,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn unsupported_is_a_failure_even_when_fortran_rejects() {
        let build62 = || Build62::Rejected("Build 62 error".into());
        let typed = |sensible| {
            Some(TypedRejection {
                kind: x13::ErrorKind::UnsuitableData,
                sensible,
            })
        };
        assert!(
            rejection_status(
                &build62(),
                &Rust::Unsupported("unimplemented".into()),
                typed(true)
            )
            .is_none()
        );
        assert!(matches!(
            rejection_status(
                &build62(),
                &Rust::Rejected("TypedRejection".into()),
                typed(true)
            ),
            Some(Status::BothRejected {
                kind: x13::ErrorKind::UnsuitableData,
                ..
            })
        ));
        assert!(matches!(
            rejection_status(
                &build62(),
                &Rust::Rejected("Numerical(NumericalFailure(Unspecified))".into()),
                typed(false)
            ),
            Some(Status::Fail(_))
        ));
        assert!(rejection_status(&build62(), &Rust::Failed("panic".into()), typed(true)).is_none());
    }

    #[test]
    fn a_rejection_carries_the_kind_of_the_rust_error() {
        let typed = TypedRejection::of(&x13::Error::EmptyInput);
        assert_eq!(typed.kind, x13::ErrorKind::InvalidInput);
        assert!(typed.sensible);
    }

    #[test]
    fn the_catch_all_numerical_failure_is_not_a_sensible_rejection() {
        // This crate cannot construct `Error::Internal` or `Error::Numerical`,
        // and no public input is known to reach either, so the catch-all's
        // text is checked directly.
        assert!(unspecified_numerical_failure(
            "Numerical(NumericalFailure(Unspecified { operation: Estimation, source: None }))"
        ));
        assert!(!unspecified_numerical_failure(
            "Numerical(NumericalFailure(X11Trend { period: 2001-03 }))"
        ));
        assert!(!insensible_rejection(&x13::Error::InvalidPeriod(
            x13::PeriodError::PeriodOverflow { index: 3 }
        )));
        assert!(!insensible_rejection(&x13::Error::EmptyInput));
    }

    #[test]
    fn process_and_parse_failures_are_not_rejections() {
        assert!(
            build62_rejection("wrong executable", true, "", "", SeatsTables::NotRequested).is_err()
        );
        assert!(
            build62_rejection(
                "Version Number 1.1 Build 62\n",
                false,
                "",
                "",
                SeatsTables::NotRequested
            )
            .is_err()
        );
        assert!(
            build62_rejection(
                "Version Number 1.1 Build 62\n",
                true,
                "WARNING: no seasonality",
                "",
                SeatsTables::NotRequested
            )
            .unwrap()
            .is_none()
        );
        assert!(
            build62_rejection(
                "Version Number 1.1 Build 62\n",
                true,
                "ERROR: model failed",
                "",
                SeatsTables::NotRequested
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn a_direct_x11_run_asks_for_no_summary() {
        assert!(!crate::spec_file::uses_summary(
            &crate::OracleConfiguration::direct_x11()
        ));
    }

    #[test]
    fn completed_automatic_search_does_not_reject_recovered_iteration_errors() {
        let banner = "Version Number 1.1 Build 62\n";
        let limit = "ERROR: Estimation failed to converge -- maximum function evaluations reached.";
        let completed = "automodeltype: automdl\nconverged: yes\n";
        assert!(
            build62_rejection(banner, true, limit, completed, SeatsTables::NotRequested)
                .unwrap()
                .is_none()
        );
        for summary in [
            "",
            "automodeltype: automdl\nconverged: no\n",
            "converged: yes\n",
            "automodeltype: automdl\nconverged: yes\nerrorstop: yes\n",
        ] {
            assert!(
                build62_rejection(banner, true, limit, summary, SeatsTables::NotRequested)
                    .unwrap()
                    .is_some()
            );
        }
        assert!(
            build62_rejection(
                banner,
                true,
                "ERROR: invalid input",
                completed,
                SeatsTables::NotRequested
            )
            .unwrap()
            .is_some()
        );
        assert!(
            build62_rejection(banner, false, limit, completed, SeatsTables::NotRequested)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn summary_buffer_overflow_is_recognised_narrowly() {
        // REALLN/U-07a: itoc.f refuses the year, abend, exit code zero.
        let itoc = "     Error messages generated from processing the X-13ARIMA-SEATS spec file\n     case.spc:\n\n\n  Error:  Can't write         2023  in          400  spaces\n";
        assert!(overflowed_summary_buffer(None, itoc));
        // CONSUMER/U-07a: the stack protector aborts the process.
        assert!(overflowed_summary_buffer(Some(6), ""));
        for (signal, error_file) in [
            (None, ""),
            (None, "SEARCH : Model degenerate DET(A) <=0"),
            (
                None,
                "ERROR: Estimation failed to converge -- maximum iterations reached.",
            ),
            (Some(11), ""),
            (Some(9), ""),
            // itoc with another buffer length, or another message.
            (
                None,
                "  Error:  Can't write         2023  in           80  spaces",
            ),
            (
                None,
                "  Error:  Can't write         2023  in         4000  spaces",
            ),
            (
                None,
                "  Error:  Can't write         abcd  in          400  spaces",
            ),
            (
                None,
                "  Error:  Can't read          2023  in          400  spaces",
            ),
        ] {
            assert!(
                !overflowed_summary_buffer(signal, error_file),
                "{signal:?} {error_file:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn summary_run_that_overflows_prtd8b_is_rerun_without_summary() {
        use std::os::unix::process::ExitStatusExt;
        use std::process::{ExitStatus, Output};

        let udg = "transform: Automatic selection\naictrans: Log(y)\narimamdl: (3 1 1)(1 0 1)\nd8b.01: 1962z\nerrorstop: yes\n";
        let itoc = "  Error:  Can't write         2023  in          400  spaces\n";
        // (error file, wait status) of the `-s` run.
        let overflows = [(itoc, 0), ("", 6)];
        for (error_file, wait_status) in overflows {
            let temp = tempfile::tempdir().unwrap();
            let mut calls = vec![];
            let run = run_with_summary_retry(
                temp.path(),
                SummaryMode::of(true, false),
                |raw, summary| {
                    calls.push(summary);
                    fs::create_dir(raw)?;
                    if summary {
                        fs::write(raw.join("output.err"), error_file)?;
                        fs::write(raw.join("output.udg"), udg)?;
                    } else {
                        fs::write(raw.join("output.d11"), "tables")?;
                    }
                    Ok(Output {
                        status: ExitStatus::from_raw(if summary { wait_status } else { 0 }),
                        stdout: vec![],
                        stderr: vec![],
                    })
                },
            )
            .unwrap();
            assert_eq!(calls, [true, false], "{error_file:?}");
            assert!(run.retried);
            assert!(run.process.status.success());
            assert_eq!(run.directory, temp.path().join("retry"));
            assert!(run.directory.join("output.d11").exists());
            // The rerun's own error file is read; the UDG keeps the model
            // lines and loses only the errorstop.
            assert_eq!(run.error_text, "");
            assert!(run.summary.contains("arimamdl: (3 1 1)(1 0 1)"));
            assert!(run.summary.contains("d8b.01: 1962z"));
            assert!(!run.summary.contains("errorstop"));
            assert_eq!(
                fs::read_to_string(run.directory.join("output.udg")).unwrap(),
                run.summary
            );
        }

        // An ordinary rejection, another signal, or a run without `-s` is
        // never rerun.
        let others = [
            (true, "SEARCH : Model degenerate DET(A) <=0\n", 1 << 8),
            (true, "", 11),
            (false, itoc, 0),
            (false, "", 6),
        ];
        for (summary, error_file, wait_status) in others {
            let temp = tempfile::tempdir().unwrap();
            let mut calls = 0;
            let run =
                run_with_summary_retry(temp.path(), SummaryMode::of(summary, false), |raw, _| {
                    calls += 1;
                    fs::create_dir(raw)?;
                    fs::write(raw.join("output.err"), error_file)?;
                    fs::write(raw.join("output.udg"), udg)?;
                    Ok(Output {
                        status: ExitStatus::from_raw(wait_status),
                        stdout: vec![],
                        stderr: vec![],
                    })
                })
                .unwrap();
            assert_eq!(calls, 1, "{summary} {error_file:?} {wait_status}");
            assert!(!run.retried);
            assert_eq!(run.directory, temp.path().join("run"));
            assert_eq!(run.error_text, error_file);
            assert_eq!(run.summary, udg);
        }
    }

    #[test]
    fn summary_mode_passes_summary_to_every_run_that_asks_for_it_but_retries_only_x11() {
        // A SEATS row that uses `-s` (arima.f's Lsumm gate feeds SEATS its Q
        // statistic) still gets `-s` on its first run; only the retry is X-11's.
        let seats = SummaryMode::of(true, true);
        assert!(seats.with_summary && !seats.retry_on_overflow);
        let x11 = SummaryMode::of(true, false);
        assert!(x11.with_summary && x11.retry_on_overflow);
        for seats in [false, true] {
            let none = SummaryMode::of(false, seats);
            assert!(!none.with_summary && !none.retry_on_overflow);
        }
    }

    #[cfg(unix)]
    #[test]
    fn seats_summary_run_gets_summary_and_is_never_rerun() {
        use std::os::unix::process::ExitStatusExt;
        use std::process::{ExitStatus, Output};

        let udg = "seatsadj: yes\nerrorstop: yes\n";
        let itoc = "  Error:  Can't write         2023  in          400  spaces\n";
        // Success, SIGABRT and an itoc refusal all end after the one run.
        for (error_file, wait_status) in [("", 0), ("", 6), (itoc, 0)] {
            let temp = tempfile::tempdir().unwrap();
            let mut calls = vec![];
            let run =
                run_with_summary_retry(temp.path(), SummaryMode::of(true, true), |raw, summary| {
                    calls.push(summary);
                    fs::create_dir(raw)?;
                    fs::write(raw.join("output.err"), error_file)?;
                    fs::write(raw.join("output.udg"), udg)?;
                    Ok(Output {
                        status: ExitStatus::from_raw(wait_status),
                        stdout: vec![],
                        stderr: vec![],
                    })
                })
                .unwrap();
            assert_eq!(calls, [true], "{error_file:?} {wait_status}");
            assert!(!run.retried);
            assert_eq!(run.directory, temp.path().join("run"));
            assert_eq!(run.summary, udg);
        }
    }

    #[test]
    fn a_rerun_that_cannot_be_parsed_says_it_was_a_rerun() {
        let parse: Result<()> = Err(anyhow::anyhow!("missing or duplicate arimamdl in UDG"));
        let message = format!("{:#}", for_rerun(parse, true).unwrap_err());
        assert!(message.contains("rerun without -s"), "{message}");
        assert!(message.contains("arimamdl"), "{message}");
        let message = format!(
            "{:#}",
            for_rerun::<()>(Err(anyhow::anyhow!("parse")), false).unwrap_err()
        );
        assert_eq!(message, "parse");
        assert!(for_rerun(Ok(1), true).is_ok());
    }

    #[test]
    fn an_errorstop_is_a_rejection_without_an_error_prefix() {
        let banner = "Version Number 1.1 Build 62\n";
        assert!(
            build62_rejection(
                banner,
                true,
                "SEARCH : Model degenerate DET(A) <=0",
                "arimamdl: (0 1 1)(0 1 1)\nerrorstop: yes\n",
                SeatsTables::NotRequested
            )
            .unwrap()
            .is_some()
        );
        assert!(
            build62_rejection(
                banner,
                true,
                "",
                "errorstop: no\n",
                SeatsTables::NotRequested
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn invalid_seats_decomposition_is_a_rejection_despite_exit_zero() {
        assert!(build62_rejection(
            "Version Number 1.1 Build 62\n", true,
            "\n DECOMPOSITION INVALID,IRREGULAR SPECTRUM NEGATIVE\n TRY ANOTHER MODEL OR, FOR AN APPROXIMATION, SET NOADMISS=YES.\n",
            "seatsadj: no\n", SeatsTables::NoneSaved).unwrap().is_some());
    }

    #[test]
    fn explicit_seats_refusals_are_rejections_despite_exit_zero() {
        let banner = "Version Number 1.1 Build 62\n";
        let classify = |success, error_file, tables| {
            build62_rejection(banner, success, error_file, "", tables)
        };
        for (error_file, expected) in [
            (
                "  The innovation variance of the trend is greater than one,\n  an indication",
                "The innovation variance of the trend is greater than one,",
            ),
            (
                "transitoryThe innovation variance of the greater than one is \n",
                "transitoryThe innovation variance of the greater than one is",
            ),
        ] {
            assert_eq!(
                classify(true, error_file, SeatsTables::NoneSaved)
                    .unwrap()
                    .as_deref(),
                Some(expected)
            );
        }
        // The degenerate stop is judged by the saved tables, not SPCDRV's
        // monthly-only note: a quarterly stop prints nothing, and a saved
        // table (an all-zero SA among them) keeps the run a success.
        let note = " NOTE: Spectral plot for the seasonally adjusted series cannot be done\n       when SEATS cannot perform a signal extraction.\n";
        for error_file in ["", note] {
            assert_eq!(
                classify(true, error_file, SeatsTables::NoneSaved)
                    .unwrap()
                    .as_deref(),
                Some("SEATS cannot perform a signal extraction.")
            );
            assert!(
                classify(true, error_file, SeatsTables::Saved)
                    .unwrap()
                    .is_none()
            );
            assert!(
                classify(true, error_file, SeatsTables::NotRequested)
                    .unwrap()
                    .is_none()
            );
            assert!(classify(false, error_file, SeatsTables::NoneSaved).is_err());
        }
    }

    #[test]
    fn a_build62_failure_keeps_the_rust_outcome() {
        let Status::Fail(failures) = build62_failure(
            anyhow::anyhow!("Build 62 crash"),
            &Rust::Rejected("ExactTypedError".into()),
        ) else {
            panic!("a Build 62 execution failure must fail");
        };
        assert!(failures[0].expected.contains("Build 62 crash"));
        assert_eq!(failures[0].actual, "rejected: ExactTypedError");
    }

    #[test]
    fn cached_timeout_remains_a_failure_even_when_rust_rejects() {
        let directory = tempfile::tempdir().unwrap();
        let timeout = Duration::from_secs(180);
        let cold = cached_pair::<Build62>(directory.path(), "case", timeout, || {
            Err(crate::OracleError::Timeout { timeout }.into())
        })
        .unwrap_err();
        let warm = cached_pair::<Build62>(directory.path(), "case", timeout, || {
            panic!("cached timeout must not rerun Fortran")
        })
        .unwrap_err();
        for error in [cold, warm] {
            let Status::Fail(failures) =
                build62_failure(error, &Rust::Rejected("ExactTypedError".into()))
            else {
                panic!("a timeout is neither a rejection nor a pass");
            };
            assert_eq!(failures[0].output, "Fortran execution/parser");
            assert_eq!(
                failures[0].expected,
                "failed: X-13 exceeded its 180s deadline"
            );
            assert_eq!(failures[0].actual, "rejected: ExactTypedError");
        }
    }
}
