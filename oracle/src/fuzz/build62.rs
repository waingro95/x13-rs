//! The Build 62 side of a replay: render the case as a Build 62 spec, run the
//! executable and classify the result as the parity harness does.
//!
//! Call-table rows render through the parity harness's own renderer. Named
//! specs render through the small mapping below. Random builder specs have no
//! Build 62 rendering: there is no general `Spec` to Build 62 spec writer, so
//! those cases report `skipped`.
//!
//! # Notes
//!
//! - **Classification is the harness's.** `build62_rejection` treats an
//!   exit-0 run whose error file says `DECOMPOSITION INVALID` as a rejection,
//!   because Build 62 writes no decomposition: Rust should return a typed
//!   error for such a case, not panic.
//! - **`-s` follows the harness.** Rows use `uses_summary`; named specs use the
//!   same rule by kind. The `errorstop` marker the classifier reads is written
//!   only with `-s`, and the flag adds diagnostics a run without it never
//!   computes, so a replay uses it exactly where the table does.
//! - **Outputs are not compared.** A replay compares outcome classes only. The
//!   parity table compares numbers; a case that needs that belongs there.
//! - **Named specs are a subset.** They support only part of the Build 62
//!   grammar. Extend `named.rs` and `render_named` together.

use super::generate::Case;
use super::generate::SpecSource;
use super::named::{self, Kind, Mode, Model, Named, Outliers};
use super::spec::row_configuration;
use crate::parity::case::{SeatsTables, build62_rejection};
use crate::{Frequency, Period, Series};
use serde::Serialize;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// How Build 62 handled a case.
#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "class", content = "detail", rename_all = "lowercase")]
pub(crate) enum Build62 {
    Ok,
    /// Build 62 stopped with an error message.
    Rejected(String),
    /// No result within the time limit.
    Hang,
    /// The case cannot be rendered for Build 62.
    Skipped(String),
    /// The run itself failed (signal, missing banner, unparsable output).
    Failed(String),
}

impl Build62 {
    /// The class name `repro` prints, the same as its JSON tag.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Rejected(_) => "rejected",
            Self::Hang => "hang",
            Self::Skipped(_) => "skipped",
            Self::Failed(_) => "failed",
        }
    }
}

struct Rendered {
    spec: String,
    input: String,
    summary: bool,
    seats: bool,
}

/// `X13AS_ASCII` or `x13as_ascii` on `PATH`, resolved to an absolute path.
pub(crate) fn executable() -> Result<PathBuf, crate::OracleError> {
    let requested = std::env::var_os("X13AS_ASCII")
        .map(PathBuf::from)
        .unwrap_or_else(|| "x13as_ascii".into());
    crate::build62::executable(&requested)
}

fn render(case: &Case) -> Result<Rendered, String> {
    let skip = |e: crate::OracleError| e.to_string();
    match &case.spec {
        SpecSource::Random { .. } => Err("random builder spec has no Build 62 rendering".into()),
        SpecSource::Row { id } => {
            let (series, config) = row_configuration(case, id)?;
            Ok(Rendered {
                spec: crate::spec_file::render(&series, config).map_err(skip)?,
                input: crate::spec_file::render_input(&series).map_err(skip)?,
                summary: crate::spec_file::uses_summary(config),
                seats: config.method() == crate::Method::Seats,
            })
        }
        SpecSource::Named { name } => {
            let named = named::parse(name)?;
            let series = Series {
                series_id: "fuzz".into(),
                frequency: case.frequency,
                start: Period::new(case.start_year, case.start_subperiod),
                values: crate::series::build_62_values(&case.values()),
            };
            if series.frequency == Frequency::Semiannual {
                return Err("semiannual".into());
            }
            Ok(Rendered {
                spec: render_named(&named, &series),
                input: crate::spec_file::render_input(&series).map_err(skip)?,
                // As the harness: direct X-11 never passes -s, and explicit-model
                // SEATS without an outlier search does not either.
                summary: match named.kind {
                    Kind::Direct(_) => false,
                    Kind::SeatsReg => named.model == Model::Auto || named.outliers.is_some(),
                    Kind::X11Reg { .. } => true,
                },
                seats: named.kind == Kind::SeatsReg,
            })
        }
    }
}

/// `Y.P` for the observation `offset` periods after the series start.
fn date_at(series: &Series, offset: usize) -> String {
    let per = i64::from(series.frequency.periods_per_year());
    let index =
        i64::from(series.start.year) * per + i64::from(series.start.subperiod) - 1 + offset as i64;
    format!("{}.{}", index.div_euclid(per), index.rem_euclid(per) + 1)
}

fn span_text(a: named::Date, b: named::Date) -> String {
    format!("({}.{}, {}.{})", a.year, a.period, b.year, b.period)
}

/// Build 62 spec text equivalent to a named Rust spec.
fn render_named(named: &Named, series: &Series) -> String {
    let mut spec = String::new();
    let per = series.frequency.periods_per_year();
    write!(
        spec,
        "series {{ title = \"fuzz\" start = {}.{} period = {per} file = \"input.dat\" format = free",
        series.start.year, series.start.subperiod
    )
    .unwrap();
    if let Some((a, b)) = named.span_offsets {
        write!(
            spec,
            " modelspan = ({}, {})",
            date_at(series, a),
            date_at(series, b)
        )
        .unwrap();
    } else if let Some((a, b)) = named.model_span {
        write!(spec, " modelspan = {}", span_text(a, b)).unwrap();
    }
    spec.push_str(" }\n");
    let x11_tail = |spec: &mut String| spec.push_str(" save = (d10 d11) }\n");
    if let Kind::Direct(mode) = named.kind {
        let mode = match mode {
            Mode::Add => "add",
            Mode::Mult => "mult",
            Mode::Pseudo => "pseudoadd",
            Mode::LogAdd => "logadd",
        };
        write!(spec, "x11 {{ mode = {mode}").unwrap();
        if let Some((lower, upper)) = named.sigma {
            write!(spec, " sigmalim = ({lower:e}, {upper:e})").unwrap();
        }
        if let Some((length, ratio)) = named.henderson {
            write!(spec, " trendma = {length} trendic = {ratio:e}").unwrap();
        }
        if named.seven_term {
            spec.push_str(" true7term = yes");
        }
        x11_tail(&mut spec);
        return spec;
    }
    writeln!(
        spec,
        "transform {{ function = {} }}",
        if named.log { "log" } else { "none" }
    )
    .unwrap();
    let columns = named.regressor_columns(series.values.len());
    if !named.manual_ao.is_empty() || !columns.is_empty() {
        spec.push_str("regression {\n");
        if !named.manual_ao.is_empty() {
            let variables: Vec<String> = named
                .manual_ao
                .iter()
                .map(|d| format!("ao{}.{}", d.year, d.period))
                .collect();
            writeln!(spec, "  variables = ({})", variables.join(" ")).unwrap();
        }
        if !columns.is_empty() {
            let names: Vec<String> = (1..=columns.len()).map(|i| format!("u{i}")).collect();
            spec.push_str("  user = (\n");
            for line in names.chunks(10) {
                writeln!(spec, "{}", line.join(" ")).unwrap();
            }
            writeln!(
                spec,
                ")\n  usertype = user start = {}.{}\n  data = (",
                series.start.year, series.start.subperiod
            )
            .unwrap();
            // getreg.f reads a flat period-major vector, one value per line.
            for row in 0..columns[0].len() {
                for column in &columns {
                    writeln!(spec, "{:.17e}", column[row]).unwrap();
                }
            }
            spec.push_str(")\n");
        }
        spec.push_str("}\n");
    }
    match named.model {
        Model::Auto => spec.push_str("automdl { }\n"),
        Model::Explicit([p, d, q, sp, sd, sq]) => {
            writeln!(spec, "arima {{ model = ({p} {d} {q})({sp} {sd} {sq}) }}").unwrap();
        }
    }
    if let Some(search) = named.outliers {
        spec.push_str("outlier { types = ");
        match search {
            Outliers::Auto => spec.push_str("(ao ls tc)"),
            Outliers::Shared(value) => write!(spec, "(ao ls tc) critical = {value:e}").unwrap(),
            Outliers::PerType { ao, ls, tc } => write!(
                spec,
                "({}) critical = ({ao:e}, {ls:e}, {:e})",
                if tc.is_some() { "ao ls tc" } else { "ao ls" },
                tc.unwrap_or(ls)
            )
            .unwrap(),
        }
        if let Some((a, b)) = named.outlier_span {
            write!(spec, " span = {}", span_text(a, b)).unwrap();
        }
        spec.push_str(" }\n");
    }
    spec.push_str("estimate { exact = arma }\n");
    writeln!(
        spec,
        "forecast {{ maxlead = {} maxback = 0 }}",
        named.forecast.unwrap_or(u16::from(per))
    )
    .unwrap();
    match named.kind {
        Kind::SeatsReg => spec.push_str("seats { save = (s10 s11 s12 s13) }\n"),
        _ => {
            write!(
                spec,
                "x11 {{ mode = {}",
                if named.log { "mult" } else { "add" }
            )
            .unwrap();
            if matches!(named.kind, Kind::X11Reg { easter: true }) {
                spec.push_str(" x11easter = yes");
            }
            spec.push_str(" seasonalma = s3x5");
            x11_tail(&mut spec);
        }
    }
    spec
}

/// Runs Build 62 on the case. The spec, input and error file of the run are
/// kept under `keep` for inspection.
pub(crate) fn run(case: &Case, executable: &Path, timeout: Duration, keep: &Path) -> Build62 {
    let rendered = match render(case) {
        Ok(rendered) => rendered,
        Err(reason) => return Build62::Skipped(reason),
    };
    let result = (|| -> Result<Build62, String> {
        let temp = tempfile::tempdir().map_err(|e| e.to_string())?;
        let raw = temp.path().join("run");
        let outcome = crate::build62::run(crate::build62::Run {
            executable,
            directory: &raw,
            spec: &rendered.spec,
            input: &rendered.input,
            summary: rendered.summary,
            timeout,
        });
        let _ = fs::create_dir_all(keep);
        let _ = fs::write(keep.join("case.spc"), &rendered.spec);
        let _ = fs::write(keep.join("input.dat"), &rendered.input);
        let process = match outcome {
            Ok(process) => process,
            Err(crate::OracleError::Timeout { .. }) => return Ok(Build62::Hang),
            Err(error) => return Err(error.to_string()),
        };
        let error_text = fs::read_to_string(raw.join("output.err")).unwrap_or_default();
        let _ = fs::write(keep.join("output.err"), &error_text);
        if process.status.code().is_none() {
            return Ok(Build62::Failed("terminated by a signal".into()));
        }
        let summary = fs::read_to_string(raw.join("output.udg")).unwrap_or_default();
        let rejection = build62_rejection(
            &String::from_utf8_lossy(&process.stdout),
            process.status.success(),
            &error_text,
            &summary,
            SeatsTables::of_run(rendered.seats, &raw).map_err(|e| format!("{e:#}"))?,
        )
        .map_err(|e| format!("{e:#}"))?;
        Ok(rejection.map_or(Build62::Ok, Build62::Rejected))
    })();
    result.unwrap_or_else(Build62::Failed)
}
