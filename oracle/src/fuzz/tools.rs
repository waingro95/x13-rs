//! Single-case tools: `gen`, `replay`, `minimize` and `conc`.

use super::generate::{Case, Rng, generate};
use super::outcome::{Outcome, evaluate, run_case, variant};
use super::repro::load_case;
use super::spec::build;
use crate::Frequency;
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Args)]
pub(crate) struct GenArgs {
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    index: u64,
    /// Write the case here instead of printing it.
    #[arg(long)]
    out: Option<PathBuf>,
}

pub(crate) fn gen_case(args: GenArgs) -> Result<()> {
    let text = serde_json::to_string(&generate(args.seed, args.index))?;
    match args.out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| parent.display().to_string())?;
            }
            fs::write(&path, text).with_context(|| path.display().to_string())
        }
        None => {
            println!("{text}");
            Ok(())
        }
    }
}

#[derive(Args)]
pub(crate) struct ReplayArgs {
    case: PathBuf,
    /// Print the first period and `Spec` built.
    #[arg(long)]
    spec: bool,
    /// Print the result components (and, with `--full`, the whole result).
    #[arg(long)]
    dump: bool,
    #[arg(long)]
    full: bool,
}

/// Runs one case in this process, with no time or memory limit. A hanging case
/// hangs this command; use `repro` for a supervised run.
pub(crate) fn replay(args: ReplayArgs) -> Result<()> {
    let case = load_case(&args.case)?;
    if args.spec {
        match build(&case) {
            Ok((spec, start)) => println!("start={start:?}\nspec={spec:#?}"),
            Err(error) => println!("build error {error}"),
        }
    }
    let mut sink = std::io::sink();
    let record = evaluate(0, &case, &mut sink);
    println!("{}", serde_json::to_string_pretty(&record.first)?);
    println!(
        "deterministic={} second={}",
        record.deterministic, record.second_kind
    );
    if args.dump
        && let Ok((spec, start)) = build(&case)
    {
        match x13::adjust(x13::Series::new(start, &case.values()), &spec) {
            Ok(adj) => {
                println!("seasonal={:?}", adj.seasonal());
                println!("sa={:?}", adj.seasonally_adjusted());
                println!("trend={:?}", adj.trend_cycle());
                println!("irregular={:?}", adj.irregular());
                println!("warnings={:?}", adj.warnings());
                if args.full {
                    println!("{adj:#?}");
                }
            }
            Err(error) => println!("error={error:?}"),
        }
    }
    Ok(())
}

#[derive(Args)]
pub(crate) struct MinimizeArgs {
    case: PathBuf,
    /// Where to write the shrunk case.
    #[arg(long)]
    out: PathBuf,
    /// Outcome signature to preserve (default: the case's own).
    #[arg(long)]
    signature: Option<String>,
}

/// `kind|detail` for errors and panics; `ok|flag,...` with positions stripped.
fn signature(outcome: &Outcome) -> String {
    match outcome.kind.as_str() {
        "panic" => format!("panic|{}", outcome.detail),
        "ok" => {
            let flags: BTreeSet<String> = outcome
                .flags
                .iter()
                .map(|flag| {
                    let flag = flag.split('@').next().unwrap_or(flag);
                    flag.split(':').take(2).collect::<Vec<_>>().join(":")
                })
                .collect();
            format!("ok|{}", flags.into_iter().collect::<Vec<_>>().join(","))
        }
        kind => format!("{kind}|{}", variant(&outcome.detail)),
    }
}

/// Greedy shrink of the series that preserves the outcome signature: drop
/// leading and trailing chunks, then round the values.
pub(crate) fn minimize(args: MinimizeArgs) -> Result<()> {
    super::outcome::silence_panics();
    let mut case = load_case(&args.case)?;
    let target = args
        .signature
        .unwrap_or_else(|| signature(&run_case(&case)));
    eprintln!("target signature: {target}");
    let keeps = |candidate: &Case| signature(&run_case(candidate)) == target;
    ensure!(keeps(&case), "case does not reproduce its signature");
    let per = if case.frequency == Frequency::Monthly {
        12usize
    } else {
        4
    };
    // Random specs depend on the series length (dates, regressor rows), so
    // shifting the start is still a valid try.
    let mut progress = true;
    while progress {
        progress = false;
        let n = case.values.len();
        for chunk in [n / 2, n / 4, 4 * per, per, 1] {
            if chunk == 0 || chunk >= n {
                continue;
            }
            let mut trailing = case.clone();
            trailing.values.truncate(n - chunk);
            if keeps(&trailing) {
                case = trailing;
                progress = true;
                break;
            }
            let mut leading = case.clone();
            leading.values.drain(..chunk);
            let date =
                i64::from(leading.start_year) * per as i64 + i64::from(leading.start_subperiod) - 1
                    + chunk as i64;
            leading.start_year = date.div_euclid(per as i64) as i32;
            leading.start_subperiod = (date.rem_euclid(per as i64) + 1) as u8;
            if keeps(&leading) {
                case = leading;
                progress = true;
                break;
            }
        }
    }
    let values = case.values();
    for digits in [3i32, 2, 1] {
        let rounded: Vec<f64> = values
            .iter()
            .map(|&v| {
                if v == 0.0 || !v.is_finite() {
                    v
                } else {
                    let scale = 10f64.powi(digits - 1 - v.abs().log10().floor() as i32);
                    (v * scale).round() / scale
                }
            })
            .collect();
        let mut candidate = case.clone();
        candidate.set_values(&rounded);
        if keeps(&candidate) {
            case = candidate;
            break;
        }
    }
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).with_context(|| parent.display().to_string())?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&case)?)?;
    eprintln!("minimized to n={}", case.values.len());
    Ok(())
}

#[derive(Args)]
pub(crate) struct ConcArgs {
    /// A campaign output directory (its `results.jsonl` supplies the expected results).
    dir: PathBuf,
    /// Threads running cases at once.
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Times each case is run.
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Replay at most this many cases.
    #[arg(long, default_value_t = 2000)]
    limit: usize,
}

#[derive(Deserialize)]
struct Expected {
    seed: u64,
    index: u64,
    first: ExpectedFirst,
}

#[derive(Deserialize)]
struct ExpectedFirst {
    kind: String,
    hash: u64,
}

/// Replays campaign cases on a thread pool, shuffled, and compares each result
/// with the sequential one: the library must be deterministic and independent
/// across concurrent calls, as the parity harness assumes.
pub(crate) fn conc(args: ConcArgs) -> Result<()> {
    super::outcome::silence_panics();
    ensure!(
        args.threads >= 1 && args.reps >= 1,
        "--threads and --reps must be at least 1"
    );
    let path = args.dir.join("results.jsonl");
    let text = fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    let expected: Vec<Expected> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(args.limit)
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let _ = super::generate::context();
    let cases: Vec<Case> = expected.iter().map(|e| generate(e.seed, e.index)).collect();
    let mut jobs: Vec<usize> = (0..cases.len())
        .flat_map(|i| std::iter::repeat_n(i, args.reps))
        .collect();
    let mut rng = Rng::new(0xC0C0);
    for i in (1..jobs.len()).rev() {
        jobs.swap(i, rng.usize(i + 1));
    }
    let next = AtomicUsize::new(0);
    let mismatches = Mutex::new(Vec::new());
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..args.threads {
            scope.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&case_index) = jobs.get(k) else {
                        break;
                    };
                    let outcome = run_case(&cases[case_index]);
                    let want = &expected[case_index];
                    if outcome.hash != want.first.hash || outcome.kind != want.first.kind {
                        mismatches.lock().unwrap().push(format!(
                            "seed {} index {}: {} {:#x} != {} {:#x}",
                            want.seed,
                            want.index,
                            outcome.kind,
                            outcome.hash,
                            want.first.kind,
                            want.first.hash
                        ));
                    }
                }
            });
        }
    });
    let mismatches = mismatches.into_inner().unwrap();
    println!(
        "{} jobs on {} threads in {:.1}s: {} mismatches",
        jobs.len(),
        args.threads,
        started.elapsed().as_secs_f64(),
        mismatches.len()
    );
    for line in mismatches.iter().take(20) {
        println!("  {line}");
    }
    ensure!(
        mismatches.is_empty(),
        "concurrent results differ from sequential ones"
    );
    Ok(())
}
