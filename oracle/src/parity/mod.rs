//! The parity harness: runs every row of the call table over the corpus,
//! compares Rust with Build 62, and writes the parity table.
//!
//! A case is one series under one row. For each case the harness renders the
//! row's Build 62 spec file, takes Build 62's output from the cache (running
//! the executable on a miss), runs the Rust library on the same input and
//! compares the two. A case passes when every compared output is within
//! tolerance, when both implementations reject the input and the rejection
//! map accepts Rust's error for Build 62's message
//! (`tests/parity/rejection-map.toml`), or when it matches a deviation listed
//! in `tests/parity/deviations.toml`.
//!
//! # Environment
//!
//! A relative path resolves against the workspace root.
//!
//! - `PARITY_TIER`: `full` (default) or `quick`, which keeps the first four
//!   applicable series of each frequency for each row.
//! - `PARITY_FILTER`: glob over `<series>/<row>` case ids.
//! - `PARITY_JOBS`: how many cases run at once, a positive integer (default:
//!   one per available core).
//! - `X13AS_ASCII`: the Build 62 executable, as a path or as a name to find
//!   on `PATH` (default `x13as_ascii`).
//! - `PARITY_CACHE_DIR`: the cache of Build 62 output (default
//!   `.cache/parity`). A fresh directory runs every selected case cold.
//! - `PARITY_RECORD_RUST`: a directory that receives the Rust bit records, one
//!   file per case holding every Rust output exactly. Existing records are
//!   never overwritten, so record into a fresh directory.
//! - `PARITY_COMPARE_RUST`: a recorded directory the Rust bit records must
//!   equal byte for byte. This is how a refactor shows it moved no number.
//! - `PARITY_COMPARE_RUST_TOLERANT` and `PARITY_COMPARE_RUST_CUMULATIVE`:
//!   recorded directories to measure drift against, always given together:
//!   the records of the previous change and those of the starting point. A
//!   case fails when an output moved from either by more than a tenth of its
//!   tolerance.
//!
//! # Output
//!
//! Written to the Cargo target directory:
//!
//! - `parity-report.md`: the parity table, which is also printed.
//! - `parity-failures.json`: the failures of each failing case, one per
//!   output.
//! - `parity-headroom.json`: for each row, the compared value that came
//!   closest to its tolerance.
//! - `parity-both-rejected.json`: the cases both implementations rejected,
//!   with both messages and the rejection map's verdict.
//! - `parity-drift.json`: the drift of every case, when drift baselines are
//!   given.

mod cache;
pub(crate) mod case;
mod compare;
pub(crate) mod data;
mod output;
mod rejection_map;
mod report;
mod synthetic;

use anyhow::{Context, Result, ensure};
use case::{CaseResult, ContextPaths, Status};
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

/// Runs the cases the environment selects (see the module docs) and writes
/// the parity table. Fails when a case fails, when no case is selected, or,
/// on a full unfiltered run, when a listed deviation no longer matches the
/// cases it names.
pub fn run_from_environment() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("missing repository root")?;
    let parity = root.join("tests/parity");
    let (rows, mut series) = data::load(&parity)?;
    let tolerances: compare::Tolerances = data::read_toml(&parity.join("tolerances.toml"))?;
    let deviations = load_deviations(&parity.join("deviations.toml"))?;
    let rejections = rejection_map::RejectionMap::load(&parity.join("rejection-map.toml"))?;
    let tier = env::var("PARITY_TIER").unwrap_or_else(|_| "full".into());
    ensure!(
        matches!(tier.as_str(), "full" | "quick"),
        "PARITY_TIER must be full or quick"
    );
    series.sort_by(|a, b| a.series_id.cmp(&b.series_id));
    let filter = env::var("PARITY_FILTER")
        .ok()
        .map(|pattern| globset::Glob::new(&pattern).map(|g| g.compile_matcher()))
        .transpose()?;
    // `cargo test` runs the harness in `oracle/`, so a relative path is taken
    // from the workspace root instead of the working directory.
    let path = |variable: &str| env::var_os(variable).map(|path| root.join(path));
    let requested_executable = match env::var_os("X13AS_ASCII").map(PathBuf::from) {
        None => "x13as_ascii".into(),
        Some(name) if name.components().count() == 1 => name,
        Some(path) => root.join(path),
    };
    let executable = crate::build62::executable(&requested_executable)?;
    let cache = path("PARITY_CACHE_DIR").unwrap_or_else(|| root.join(".cache/parity"));
    cache::check_platform(&cache, &cache::platform())?;
    let record = path("PARITY_RECORD_RUST");
    let compare = path("PARITY_COMPARE_RUST");
    let tolerant = path("PARITY_COMPARE_RUST_TOLERANT");
    let cumulative = path("PARITY_COMPARE_RUST_CUMULATIVE");
    validate_record_paths(
        record.as_deref(),
        compare.as_deref(),
        tolerant.as_deref(),
        cumulative.as_deref(),
    )?;
    let paths = ContextPaths {
        executable: &executable,
        cache: &cache,
        record: record.as_deref(),
        compare: compare.as_deref(),
        tolerant: tolerant.as_deref(),
        cumulative: cumulative.as_deref(),
    };
    let cases = select_cases(&rows, &series, tier == "quick", filter.as_ref())?;
    ensure!(!cases.is_empty(), "no applicable cases selected");
    let jobs = jobs(env::var_os("PARITY_JOBS").map(|value| value.to_string_lossy().into_owned()))?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?;
    let started = Instant::now();
    let completed = AtomicUsize::new(0);
    eprintln!("Parity {tier}: {} cases, {jobs} workers", cases.len());
    let mut results = pool.install(|| {
        cases
            .par_iter()
            .map(|(series, row)| {
                let result = case::run_case(series, row, &paths, &tolerances);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(50) || done == cases.len() {
                    eprintln!(
                        "Parity: {done}/{} completed in {:.1}s",
                        cases.len(),
                        started.elapsed().as_secs_f64()
                    );
                }
                result
            })
            .collect::<Vec<_>>()
    });
    for result in &mut results {
        // Before deviations, so one can cover an enforced rejection-map miss.
        rejection_map::apply(result, &rejections, rejection_map::ENFORCED);
        apply_deviations(result, &deviations);
    }
    let selected_rows = rows
        .iter()
        .filter(|r| filter.is_none() || cases.iter().any(|(_, row)| row.id == r.id))
        .map(|r| (r.id.as_str(), r.label.as_str(), r.frequencies.clone()))
        .collect::<Vec<_>>();
    let mut report = report::render(&selected_rows, &results, &rejections);
    let (stale, miscounted) = if tier == "full" && filter.is_none() {
        audit_deviations(&deviations, &results)
    } else {
        (vec![], vec![])
    };
    if !stale.is_empty() {
        report.push_str(&format!("\nStale deviation entries: {stale:?}\n"));
    }
    if !miscounted.is_empty() {
        report.push_str(&format!(
            "\nDeviation entries off their expected_matches (index, expected, actual): \
             {miscounted:?}\n"
        ));
    }
    let target = path("CARGO_TARGET_DIR").unwrap_or_else(|| root.join("target"));
    fs::create_dir_all(&target)?;
    fs::write(target.join("parity-report.md"), &report)?;
    let headroom = selected_rows
        .iter()
        .map(|(id, _, _)| {
            let worst = results
                .iter()
                .filter(|case| case.spec_id == *id)
                .filter_map(|case| case.headroom.as_ref().map(|h| (case, h)))
                .max_by(|(_, a), (_, b)| a.ratio.total_cmp(&b.ratio));
            serde_json::json!({"row": id, "worst": worst.map(|(case, h)| metric_record(&case.id, h))})
        })
        .collect::<Vec<_>>();
    fs::write(
        target.join("parity-headroom.json"),
        serde_json::to_vec_pretty(&headroom)?,
    )?;
    if tolerant.is_some() {
        let rows = selected_rows.iter().map(|(id, _, _)| {
            let maximum = |cumulative: bool| results.iter().filter(|case| case.spec_id == *id)
                .flat_map(|case| {
                    let values = if cumulative { &case.drift.cumulative } else { &case.drift.incremental };
                    values.iter().flatten().map(move |metric| (case, metric))
                })
                .max_by(|(_, a), (_, b)| a.ratio.total_cmp(&b.ratio))
                .map(|(case, metric)| metric_record(&case.id, metric));
            serde_json::json!({"row": id, "incremental": maximum(false), "cumulative": maximum(true)})
        }).collect::<Vec<_>>();
        let cases = results
            .iter()
            .map(|case| serde_json::json!({"case": case.id, "drift": case.drift}))
            .collect::<Vec<_>>();
        fs::write(
            target.join("parity-drift.json"),
            serde_json::to_vec(&serde_json::json!({
                "limit": compare::DRIFT_LIMIT, "incremental_baseline": tolerant,
                "cumulative_baseline": cumulative, "rows": rows, "cases": cases,
            }))?,
        )?;
    }
    // For grouping failures by hand. Every run rewrites it.
    let failures = results
        .iter()
        .filter_map(|r| match &r.status {
            Status::Fail(failures) => Some(serde_json::json!({"case":r.id,"failures":failures})),
            _ => None,
        })
        .collect::<Vec<_>>();
    fs::write(
        target.join("parity-failures.json"),
        serde_json::to_vec(&failures)?,
    )?;
    // For reviewing which Rust errors are sensible for which Build 62
    // rejections, with the rejection map's verdict on each.
    let both_rejected = results
        .iter()
        .filter_map(|r| match &r.status {
            Status::BothRejected {
                fortran,
                rust,
                kind,
            } => {
                let verdict = rejections.check(fortran, *kind);
                Some(serde_json::json!({
                    "case": r.id, "fortran": fortran, "rust": rust,
                    "kind": format!("{kind:?}"), "prefix": verdict.prefix(),
                    "map": verdict.label(),
                }))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    fs::write(
        target.join("parity-both-rejected.json"),
        serde_json::to_vec(&both_rejected)?,
    )?;
    eprintln!("Parity elapsed: {:.3}s", started.elapsed().as_secs_f64());
    println!("{report}");
    ensure!(stale.is_empty(), "stale deviation entries");
    ensure!(
        miscounted.is_empty(),
        "deviation entries matched an unexpected number of cases"
    );
    ensure!(
        !results.iter().any(|r| matches!(r.status, Status::Fail(_))),
        "parity matrix has failures"
    );
    ensure!(
        results.iter().any(|r| r.status.passes()),
        "no deterministic case passed"
    );
    Ok(())
}

/// The number of worker threads: `PARITY_JOBS` when set, otherwise one per
/// available core.
fn jobs(requested: Option<String>) -> Result<usize> {
    let Some(requested) = requested else {
        return Ok(std::thread::available_parallelism().map_or(1, usize::from));
    };
    requested
        .parse()
        .ok()
        .filter(|&jobs| jobs > 0)
        .with_context(|| format!("PARITY_JOBS must be a positive integer, not {requested:?}"))
}

fn metric_record(case: &str, metric: &compare::Headroom) -> serde_json::Value {
    let mut value = serde_json::to_value(metric).expect("finite or explicitly unbounded metric");
    value["case"] = case.into();
    value
}

fn validate_record_paths(
    record: Option<&Path>,
    exact: Option<&Path>,
    tolerant: Option<&Path>,
    cumulative: Option<&Path>,
) -> Result<()> {
    ensure!(
        tolerant.is_some() == cumulative.is_some(),
        "incremental and cumulative drift baselines are both required"
    );
    let baselines = [exact, tolerant, cumulative]
        .into_iter()
        .flatten()
        .map(|path| {
            ensure!(
                path.is_dir(),
                "missing baseline directory: {}",
                path.display()
            );
            Ok(path.canonicalize()?)
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(record) = record {
        fs::create_dir_all(record)?;
        let record = record.canonicalize()?;
        ensure!(
            !baselines
                .iter()
                .any(|baseline| record.starts_with(baseline) || baseline.starts_with(&record)),
            "record directory must not overlap a comparison baseline"
        );
    }
    Ok(())
}

fn select_cases<'a>(
    rows: &'a [data::SpecRow],
    sorted_series: &[crate::Series],
    quick: bool,
    filter: Option<&globset::GlobMatcher>,
) -> Result<Vec<(crate::Series, &'a data::SpecRow)>> {
    let mut cases = Vec::new();
    for row in rows {
        let mut monthly = 0;
        let mut quarterly = 0;
        for input in sorted_series {
            let Some(input) = row.select(input)? else {
                continue;
            };
            let count = if input.frequency == crate::Frequency::Monthly {
                &mut monthly
            } else {
                &mut quarterly
            };
            *count += 1;
            // Stable per-row IDs, selected after applicability and before a
            // developer's filter. Ineligible early IDs cannot starve a row.
            if quick && *count > 4 {
                continue;
            }
            let id = format!("{}/{}", input.series_id, row.id);
            if filter.is_none_or(|f| f.is_match(&id)) {
                cases.push((input, row));
            }
        }
    }
    Ok(cases)
}

/// Loads the matched deviations after checking every `[[outside_corpus]]` entry.
fn load_deviations(path: &Path) -> Result<Vec<CompiledDeviation>> {
    let deviations: Deviations = data::read_toml(path)?;
    deviations
        .outside_corpus
        .iter()
        .try_for_each(OutsideCorpusDeviation::validate)?;
    deviations
        .deviations
        .into_iter()
        .map(CompiledDeviation::new)
        .collect()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Deviations {
    deviations: Vec<Deviation>,
    #[serde(default)]
    outside_corpus: Vec<OutsideCorpusDeviation>,
}

/// A deviation that no corpus case reaches (for example one only the fuzzer
/// finds). It is listed for the record but never matched,
/// so the full table's stale-entry check does not apply to it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutsideCorpusDeviation {
    input: String,
    rust: String,
    fortran: String,
    reason: String,
}

impl OutsideCorpusDeviation {
    fn validate(&self) -> Result<()> {
        ensure!(
            [&self.input, &self.rust, &self.fortran, &self.reason]
                .iter()
                .all(|field| !field.trim().is_empty()),
            "outside-corpus deviation needs every field"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Deviation {
    pattern: String,
    output: String,
    rust: String,
    fortran: String,
    reason: String,
    /// The number of cases a wildcard pattern matched when it was listed.
    #[serde(default)]
    expected_matches: Option<usize>,
}

struct CompiledDeviation {
    case: globset::GlobMatcher,
    output: String,
    rust: globset::GlobMatcher,
    fortran: globset::GlobMatcher,
    expected_matches: Option<usize>,
}

impl CompiledDeviation {
    fn new(value: Deviation) -> Result<Self> {
        ensure!(!value.reason.trim().is_empty(), "deviation needs a reason");
        Ok(Self {
            case: globset::Glob::new(&value.pattern)?.compile_matcher(),
            output: value.output,
            rust: globset::Glob::new(&value.rust)?.compile_matcher(),
            fortran: globset::Glob::new(&value.fortran)?.compile_matcher(),
            expected_matches: value.expected_matches,
        })
    }
}

/// Checks the deviation entries against a full untiered run. An entry is stale
/// when no case matched it, and miscounted when it states `expected_matches`
/// and a different number of cases matched.
///
/// A wildcard pattern over series (`*/<row>`) stays short, but on its own it
/// would silently absorb new cases if a translation change widened the Rust
/// failure it names. The count turns that widening, or a fix that narrows
/// it, into a failure, as the stale check does for zero matches.
/// Filtered and quick runs see only part of the corpus, so they skip both.
fn audit_deviations(
    deviations: &[CompiledDeviation],
    results: &[CaseResult],
) -> (Vec<usize>, Vec<(usize, usize, usize)>) {
    let matches = |i: usize| {
        results
            .iter()
            .filter(|r| matches!(&r.status, Status::Deviation(ids) if ids.contains(&i)))
            .count()
    };
    let stale = (0..deviations.len()).filter(|&i| matches(i) == 0).collect();
    let miscounted = deviations
        .iter()
        .enumerate()
        .filter_map(|(i, deviation)| {
            let expected = deviation.expected_matches?;
            let actual = matches(i);
            (actual != expected).then_some((i, expected, actual))
        })
        .collect();
    (stale, miscounted)
}

fn apply_deviations(case: &mut CaseResult, deviations: &[CompiledDeviation]) {
    let Status::Fail(failures) = &case.status else {
        return;
    };
    if failures.is_empty() {
        return;
    }
    let matched = failures
        .iter()
        .map(|failure| {
            deviations.iter().position(|d| {
                d.case.is_match(&case.id)
                    && d.output == failure.output
                    && d.rust.is_match(&failure.actual)
                    && d.fortran.is_match(&failure.expected)
            })
        })
        .collect::<Option<Vec<_>>>();
    if let Some(indices) = matched {
        case.status = Status::Deviation(indices);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_jobs_must_be_a_positive_integer() {
        assert_eq!(jobs(Some("4".into())).unwrap(), 4);
        assert!(jobs(None).unwrap() >= 1);
        for invalid in ["0", "-2", "1.5", "four", ""] {
            let error = jobs(Some(invalid.into())).unwrap_err().to_string();
            assert!(error.contains("PARITY_JOBS"), "{invalid:?}: {error}");
        }
    }

    #[test]
    fn drift_baselines_come_in_pairs_and_never_hold_the_record_directory() {
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        fs::create_dir(&baseline).unwrap();
        assert!(validate_record_paths(None, None, Some(&baseline), None).is_err());
        assert!(validate_record_paths(None, None, None, Some(&baseline)).is_err());
        assert!(
            validate_record_paths(
                None,
                None,
                Some(&baseline),
                Some(&temp.path().join("missing"))
            )
            .is_err()
        );
        assert!(
            validate_record_paths(Some(&baseline), None, Some(&baseline), Some(&baseline)).is_err()
        );
        assert!(
            validate_record_paths(
                Some(&baseline.join("nested")),
                None,
                Some(&baseline),
                Some(&baseline)
            )
            .is_err()
        );
        assert!(
            validate_record_paths(
                Some(&temp.path().join("new")),
                None,
                Some(&baseline),
                Some(&baseline)
            )
            .is_ok()
        );
        let value = metric_record(
            "series/01",
            &compare::Headroom {
                output: "forecasts".into(),
                ratio: f64::INFINITY,
            },
        );
        assert_eq!(value["ratio"], "unbounded");
    }

    #[test]
    fn the_quick_tier_keeps_the_first_four_applicable_series() {
        let mut row = data::SpecRow::test_row("positive");
        row.requires_positive = true;
        let rows = [row];
        let series = (0..10)
            .map(|i| crate::Series {
                series_id: format!("S{i}"),
                frequency: crate::Frequency::Monthly,
                start: crate::Period::new(2000, 1),
                values: vec![if i < 4 { -1. } else { 1. }],
            })
            .collect::<Vec<_>>();
        let quick = select_cases(&rows, &series, true, None).unwrap();
        assert_eq!(
            quick
                .iter()
                .map(|(s, _)| s.series_id.as_str())
                .collect::<Vec<_>>(),
            ["S4", "S5", "S6", "S7"]
        );
        assert_eq!(select_cases(&rows, &series, false, None).unwrap().len(), 6);
        let late = globset::Glob::new("S9/*").unwrap().compile_matcher();
        assert!(
            select_cases(&rows, &series, true, Some(&late))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            select_cases(&rows, &series, false, Some(&late))
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn a_deviation_matches_the_case_the_output_and_both_results() {
        let rule = || {
            CompiledDeviation::new(Deviation {
                pattern: "SERIES/row".into(),
                output: "outcome".into(),
                rust: "rejected: ExactTypedError".into(),
                fortran: "rejected: ERROR: example".into(),
                reason: "test-only example, not a corpus deviation".into(),
                expected_matches: None,
            })
            .unwrap()
        };
        let failure = compare::Failure::text(
            "outcome",
            "rejected: ERROR: example",
            "rejected: ExactTypedError",
        );
        let mut case = CaseResult::test(
            "SERIES/row",
            crate::Frequency::Monthly,
            Status::Fail(vec![failure.clone()]),
        );
        apply_deviations(&mut case, &[rule()]);
        assert!(matches!(case.status, Status::Deviation(_)));
        for changed in [
            compare::Failure::text("other output", &failure.expected, &failure.actual),
            compare::Failure::text("outcome", "other Fortran error", &failure.actual),
            compare::Failure::text("outcome", &failure.expected, "other Rust error"),
        ] {
            case.status = Status::Fail(vec![changed]);
            apply_deviations(&mut case, &[rule()]);
            assert!(matches!(case.status, Status::Fail(_)));
        }
        case.id = "OTHER/row".into();
        case.status = Status::Fail(vec![failure]);
        apply_deviations(&mut case, &[rule()]);
        assert!(matches!(case.status, Status::Fail(_)));
    }

    #[test]
    fn full_run_audit_flags_stale_and_miscounted_deviations() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("deviations.toml");
        let entry = |pattern: &str, count: &str| {
            format!(
                "[[deviations]]\npattern = \"{pattern}\"\noutput = \"outcome\"\n\
                 rust = \"r\"\nfortran = \"f\"\nreason = \"why\"\n{count}"
            )
        };
        fs::write(
            &path,
            [
                entry("*/wild", "expected_matches = 2\n"),
                entry("*/low", "expected_matches = 1\n"),
                entry("A/plain", ""),
                entry("B/never", ""),
            ]
            .concat(),
        )
        .unwrap();
        let deviations = load_deviations(&path).unwrap();
        assert_eq!(
            deviations
                .iter()
                .map(|d| d.expected_matches)
                .collect::<Vec<_>>(),
            [Some(2), Some(1), None, None]
        );
        let case = |id: &str, ids: Vec<usize>| {
            CaseResult::test(id, crate::Frequency::Monthly, Status::Deviation(ids))
        };
        let mut results = vec![
            case("A/wild", vec![0]),
            case("B/wild", vec![0, 0]),
            case("A/low", vec![1]),
            case("B/low", vec![1]),
            case("A/plain", vec![2]),
        ];
        // A case counts once however many of its failures the entry matched.
        assert_eq!(
            audit_deviations(&deviations, &results),
            (vec![3], vec![(1, 1, 2)])
        );
        results.push(case("C/wild", vec![0]));
        results.push(case("B/never", vec![3]));
        results.remove(3);
        assert_eq!(
            audit_deviations(&deviations, &results),
            (vec![], vec![(0, 2, 3)])
        );
        fs::write(&path, entry("*/wild", "expected_matches = -1\n")).unwrap();
        assert!(load_deviations(&path).is_err());
    }

    #[test]
    fn malformed_outside_corpus_deviations_fail_to_load() {
        let temp = tempfile::tempdir().unwrap();
        let load = |body: &str| {
            let path = temp.path().join("deviations.toml");
            fs::write(&path, format!("deviations = []\n{body}")).unwrap();
            load_deviations(&path)
        };
        let entry = |reason: &str| {
            format!(
                "[[outside_corpus]]\ninput = \"i\"\nrust = \"r\"\nfortran = \"f\"\n\
                 reason = \"{reason}\"\n"
            )
        };
        assert!(load(&entry("why")).is_ok());
        assert!(load(&entry(" ")).is_err());
        assert!(load(&entry("why").replace("fortran = \"f\"\n", "")).is_err());
        assert!(load(&format!("{}case = \"c\"\n", entry("why"))).is_err());
        assert!(load(&entry("why").replace("outside_corpus", "outside_corpra")).is_err());
    }
}
