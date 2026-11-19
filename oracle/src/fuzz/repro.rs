//! `repro`: replay saved case files against Rust and Build 62 and compare their
//! outcome classes.
//!
//! Each Rust run is a child process (`fuzz case`) under the time limit and
//! memory cap, so a hang, an out-of-memory loop or an abort is a classified
//! result, not a stalled or dead replay. Build 62 runs in the supervising
//! process under its own deadline.

use super::build62::{self, Build62};
use super::generate::Case;
use super::outcome::{Outcome, run_case};
use super::supervise::{Clock, End, Limits, watch};
use super::{LimitArgs, output_root};
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Args)]
pub(crate) struct ReproArgs {
    /// A directory of case files (`*.json`), or one case file.
    path: PathBuf,
    /// Only replay cases whose file name contains this text.
    #[arg(long)]
    filter: Option<String>,
    /// Skip the Build 62 side.
    #[arg(long)]
    rust_only: bool,
    /// Where to write `repro.jsonl` and the Build 62 inputs
    /// (default: `target/fuzz/repro`).
    #[arg(long)]
    out: Option<PathBuf>,
    #[command(flatten)]
    limits: LimitArgs,
}

/// Internal: run one case in this process and print its [`Outcome`] as JSON.
#[derive(Args)]
pub(crate) struct CaseArgs {
    #[arg(long)]
    case: PathBuf,
}

pub(crate) fn load_case(path: &Path) -> Result<Case> {
    let text = fs::read_to_string(path).with_context(|| path.display().to_string())?;
    serde_json::from_str(&text).with_context(|| format!("not a case file: {}", path.display()))
}

pub(crate) fn case(args: CaseArgs) -> Result<()> {
    let outcome = run_case(&load_case(&args.case)?);
    println!("{}", serde_json::to_string(&outcome)?);
    Ok(())
}

/// How the Rust side of a replay ended.
#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
enum Class {
    Ok,
    /// A typed error, including a spec or axis that could not be built.
    Error,
    Panic,
    /// No result within the time limit.
    Hang,
    /// Killed at the memory cap.
    Oom,
    /// The process died without a result (abort, stack overflow, signal).
    Crash,
}

impl Class {
    fn is_fault(self) -> bool {
        !matches!(self, Self::Ok | Self::Error)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Panic => "panic",
            Self::Hang => "hang",
            Self::Oom => "oom",
            Self::Crash => "crash",
        }
    }
}

#[derive(Serialize, Debug)]
struct Rust {
    class: Class,
    detail: String,
    seconds: f64,
    peak_rss_mb: u64,
}

#[derive(Serialize)]
struct Report {
    case: String,
    rust: Rust,
    build62: Build62,
    verdict: &'static str,
}

fn classify(outcome: &Outcome) -> (Class, String) {
    match outcome.kind.as_str() {
        "ok" if outcome.flags.is_empty() => (Class::Ok, String::new()),
        "ok" => (Class::Ok, format!("flags: {}", outcome.flags.join(", "))),
        "panic" => {
            let message = outcome.panic.as_ref().map_or("", |p| p.message.as_str());
            (Class::Panic, format!("{}: {message}", outcome.detail))
        }
        _ => (Class::Error, outcome.detail.clone()),
    }
}

fn run_rust(exe: &Path, path: &Path, limits: Limits) -> Rust {
    let mut command = Command::new(exe);
    command
        .args(["case", "--case"])
        .arg(path)
        .env("FUZZ_QUIET", "1");
    let mut last = String::new();
    let watched = watch(command, limits, true, |line| {
        last = line.to_owned();
        Clock::Keep
    });
    let Ok(watched) = watched else {
        return Rust {
            class: Class::Crash,
            detail: "could not start the case process".into(),
            seconds: 0.0,
            peak_rss_mb: 0,
        };
    };
    let (class, detail) = match watched.end {
        End::Timeout => (
            Class::Hang,
            format!("no result in {}s", limits.timeout.as_secs()),
        ),
        End::Memory => (
            Class::Oom,
            format!("passed the {} MB cap", limits.max_rss_kb / 1024),
        ),
        End::Exited(status) => match serde_json::from_str::<Outcome>(&last) {
            Ok(outcome) => classify(&outcome),
            Err(_) => {
                let stderr = watched.stderr.trim();
                (
                    Class::Crash,
                    format!(
                        "{status}: {}",
                        stderr
                            .get(stderr.len().saturating_sub(200)..)
                            .unwrap_or(stderr)
                    ),
                )
            }
        },
    };
    Rust {
        class,
        detail,
        seconds: watched.wall.as_secs_f64(),
        peak_rss_mb: watched.peak_rss_kb / 1024,
    }
}

fn verdict(rust: Class, build62: &Build62) -> &'static str {
    match (rust, build62) {
        (class, _) if class.is_fault() => "FAULT",
        (Class::Ok, Build62::Ok) | (Class::Error, Build62::Rejected(_)) => "agree",
        (Class::Ok, Build62::Rejected(_)) => "rust-accepts",
        (Class::Error, Build62::Ok) => "rust-rejects",
        _ => "-",
    }
}

fn short(text: &str, width: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    if flat.chars().count() <= width {
        flat
    } else {
        let cut: String = flat.chars().take(width).collect();
        format!("{cut}...")
    }
}

fn case_files(path: &Path, filter: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut files = if path.is_dir() {
        let mut files = Vec::new();
        for entry in fs::read_dir(path).with_context(|| path.display().to_string())? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                files.push(path);
            }
        }
        files
    } else {
        vec![path.to_owned()]
    };
    files.retain(|file| {
        filter.is_none_or(|text| {
            file.file_name()
                .is_some_and(|n| n.to_string_lossy().contains(text))
        })
    });
    files.sort();
    ensure!(!files.is_empty(), "no case files under {}", path.display());
    Ok(files)
}

pub(crate) fn repro(args: ReproArgs) -> Result<ExitCode> {
    ensure!(args.limits.workers >= 1, "--workers must be at least 1");
    let files = case_files(&args.path, args.filter.as_deref())?;
    let cases = files
        .iter()
        .map(|file| load_case(file))
        .collect::<Result<Vec<_>>>()?;
    let exe = std::env::current_exe()?;
    let build62_executable = if args.rust_only {
        None
    } else {
        match build62::executable() {
            Ok(path) => Some(path),
            Err(error) => bail!("{error}; set X13AS_ASCII or pass --rust-only"),
        }
    };
    let out = args.out.unwrap_or_else(|| output_root().join("repro"));
    fs::create_dir_all(&out)?;
    let limits = args.limits.limits();
    let next = AtomicUsize::new(0);
    let reports: Mutex<Vec<(usize, Report)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..args.limits.workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(file) = files.get(index) else { break };
                    let stem = file
                        .file_stem()
                        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
                    let rust = run_rust(&exe, file, limits);
                    let build62 = match &build62_executable {
                        Some(executable) => build62::run(
                            &cases[index],
                            executable,
                            limits.timeout.max(Duration::from_secs(1)),
                            &out.join("build62").join(&stem),
                        ),
                        None => Build62::Skipped("--rust-only".into()),
                    };
                    let verdict = verdict(rust.class, &build62);
                    eprintln!(
                        "[{}/{}] {stem}: rust={} build62={} {verdict}",
                        index + 1,
                        files.len(),
                        rust.class.label(),
                        build62.label()
                    );
                    reports.lock().unwrap().push((
                        index,
                        Report {
                            case: stem,
                            rust,
                            build62,
                            verdict,
                        },
                    ));
                }
            });
        }
    });
    let mut reports = reports.into_inner().unwrap();
    reports.sort_by_key(|(index, _)| *index);
    let mut jsonl = String::new();
    let mut faults = 0;
    for (_, report) in &reports {
        let build62 = match &report.build62 {
            Build62::Ok | Build62::Hang => report.build62.label().to_owned(),
            Build62::Rejected(m) | Build62::Skipped(m) | Build62::Failed(m) => {
                format!("{} ({})", report.build62.label(), short(m, 60))
            }
        };
        let detail = if report.rust.detail.is_empty() {
            String::new()
        } else {
            format!(" ({})", short(&report.rust.detail, 90))
        };
        println!(
            "{}\n    rust={}{detail} {:.1}s {} MB\n    build62={build62}\n    verdict={}",
            report.case,
            report.rust.class.label(),
            report.rust.seconds,
            report.rust.peak_rss_mb,
            report.verdict
        );
        faults += usize::from(report.verdict == "FAULT");
        jsonl.push_str(&serde_json::to_string(report)?);
        jsonl.push('\n');
    }
    fs::write(out.join("repro.jsonl"), jsonl)?;
    println!(
        "{} cases, {faults} Rust faults (panic, hang, oom or crash); details in {}",
        reports.len(),
        out.display()
    );
    Ok(if faults == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
