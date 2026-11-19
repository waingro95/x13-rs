//! `run`: a campaign of generated cases over supervised worker processes.
//!
//! The supervisor splits `[start, start + count)` into chunks and hands each to
//! a `fuzz worker` child. A worker runs its cases in order, twice each, and
//! reports over a line protocol: `S <index> <run>` before each run and
//! `R <json>` (a [`Record`]) after both. The supervisor kills a worker whose
//! current run passes the time limit (hang) or memory cap (oom), records an
//! event, and restarts the chunk after the offending case. A worker that dies
//! mid-case (abort, stack overflow) is a crash event.

use super::generate::{context, generate};
use super::outcome::{Record, evaluate};
use super::supervise::{Clock, End, Limits, watch};
use super::{LimitArgs, output_root};
use anyhow::{Context, Result, ensure};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Args)]
pub(crate) struct RunArgs {
    /// Seed of the case stream. Cases are `(seed, index)`, so a finding is
    /// reproduced by `fuzz gen --seed S --index I`.
    #[arg(long)]
    seed: u64,
    /// Number of cases to run.
    #[arg(long)]
    count: u64,
    /// First case index.
    #[arg(long, default_value_t = 0)]
    start: u64,
    /// Cases per worker process before it is restarted.
    #[arg(long, default_value_t = 200)]
    chunk: u64,
    /// Output directory (default: `target/fuzz/seed-<seed>`). Results append.
    #[arg(long)]
    out: Option<PathBuf>,
    #[command(flatten)]
    limits: LimitArgs,
}

/// Internal: run cases `from..to` in this process under the line protocol.
#[derive(Args)]
pub(crate) struct WorkerArgs {
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    from: u64,
    #[arg(long)]
    to: u64,
}

/// A case the supervisor had to cut short, one JSON line in `events.jsonl`.
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Event {
    /// `hang`, `oom`, `crash` or `worker_exit`.
    pub(crate) event: String,
    pub(crate) seed: u64,
    pub(crate) index: Option<u64>,
    pub(crate) run: Option<u32>,
    pub(crate) seconds: f64,
    pub(crate) peak_rss_mb: u64,
    pub(crate) stderr: String,
}

pub(crate) fn worker(args: WorkerArgs) -> Result<()> {
    let _ = context();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for index in args.from..args.to {
        let case = generate(args.seed, index);
        let record = evaluate(args.seed, &case, &mut out);
        writeln!(out, "R {}", serde_json::to_string(&record)?)?;
        out.flush()?;
    }
    Ok(())
}

struct Campaign {
    exe: PathBuf,
    seed: u64,
    limits: Limits,
    out: PathBuf,
    results: Mutex<File>,
    events: Mutex<File>,
    done: AtomicU64,
    started: Instant,
}

fn append(path: &Path) -> Result<Mutex<File>> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| path.display().to_string())?;
    Ok(Mutex::new(file))
}

fn write_line(sink: &Mutex<File>, value: &impl Serialize) {
    let mut line = serde_json::to_string(value).unwrap_or_default();
    line.push('\n');
    let mut file = sink.lock().unwrap();
    let _ = file.write_all(line.as_bytes());
    let _ = file.flush();
}

impl Campaign {
    fn run_range(&self, lo: u64, hi: u64) -> Result<()> {
        let mut lo = lo;
        while lo < hi {
            let mut command = Command::new(&self.exe);
            command
                .args(["worker", "--seed"])
                .arg(self.seed.to_string())
                .arg("--from")
                .arg(lo.to_string())
                .arg("--to")
                .arg(hi.to_string())
                .env("FUZZ_QUIET", "1");
            let mut current: Option<(u64, u32)> = None;
            let mut last_done: Option<u64> = None;
            let watched = watch(command, self.limits, false, |line| {
                if let Some(rest) = line.strip_prefix("S ") {
                    let mut fields = rest.split_whitespace().map(str::parse::<u64>);
                    if let (Some(Ok(index)), Some(Ok(run))) = (fields.next(), fields.next()) {
                        current = Some((index, run as u32));
                    }
                    Clock::Start
                } else if let Some(json) = line.strip_prefix("R ") {
                    match serde_json::from_str::<Record>(json) {
                        Ok(record) => {
                            last_done = Some(record.index);
                            write_line(&self.results, &record);
                            let done = self.done.fetch_add(1, Ordering::Relaxed) + 1;
                            if done.is_multiple_of(2000) {
                                eprintln!(
                                    "{done} cases {:.0}s",
                                    self.started.elapsed().as_secs_f64()
                                );
                            }
                        }
                        Err(error) => eprintln!("unreadable worker record: {error}"),
                    }
                    current = None;
                    Clock::Stop
                } else {
                    Clock::Keep
                }
            })
            .context("could not start a worker process")?;
            let kind = match (&watched.end, current) {
                (End::Timeout, _) => Some("hang"),
                (End::Memory, _) => Some("oom"),
                (End::Exited(_), Some(_)) => Some("crash"),
                (End::Exited(status), None) if !status.success() => Some("worker_exit"),
                (End::Exited(_), None) => None,
            };
            if let Some(kind) = kind {
                self.record_event(kind, current, &watched);
            }
            let next = match current {
                Some((index, _)) => index + 1,
                None => last_done.map_or(lo, |index| index + 1),
            };
            // A worker that fails before finishing any case would loop forever.
            ensure!(next > lo, "worker made no progress at case {lo}");
            lo = next;
        }
        Ok(())
    }

    fn record_event(
        &self,
        kind: &str,
        current: Option<(u64, u32)>,
        watched: &super::supervise::Watched,
    ) {
        let event = Event {
            event: kind.into(),
            seed: self.seed,
            index: current.map(|(index, _)| index),
            run: current.map(|(_, run)| run),
            seconds: watched.wall.as_secs_f64(),
            peak_rss_mb: watched.peak_rss_kb / 1024,
            stderr: watched.stderr.clone(),
        };
        write_line(&self.events, &event);
        eprintln!("{kind} at seed {} index {:?}", self.seed, event.index);
        if let Some((index, _)) = current {
            // Keep the case itself, ready for `fuzz repro <out>/cases`.
            let case = generate(self.seed, index);
            let path = self
                .out
                .join("cases")
                .join(format!("s{}_{index}_{kind}.json", self.seed));
            if let Ok(text) = serde_json::to_string(&case) {
                let _ = fs::write(path, text);
            }
        }
    }
}

pub(crate) fn run(args: RunArgs) -> Result<()> {
    ensure!(args.limits.workers >= 1, "--workers must be at least 1");
    ensure!(args.chunk >= 1, "--chunk must be at least 1");
    let out = args
        .out
        .unwrap_or_else(|| output_root().join(format!("seed-{}", args.seed)));
    fs::create_dir_all(out.join("cases"))?;
    let campaign = Campaign {
        exe: std::env::current_exe()?,
        seed: args.seed,
        limits: args.limits.limits(),
        results: append(&out.join("results.jsonl"))?,
        events: append(&out.join("events.jsonl"))?,
        out: out.clone(),
        done: AtomicU64::new(0),
        started: Instant::now(),
    };
    let end = args.start + args.count;
    let next = AtomicU64::new(args.start);
    let outcomes: Vec<Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.limits.workers)
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    loop {
                        let lo = next.fetch_add(args.chunk, Ordering::Relaxed);
                        if lo >= end {
                            return Ok(());
                        }
                        campaign.run_range(lo, (lo + args.chunk).min(end))?;
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("supervisor thread panicked")))
            })
            .collect()
    });
    for outcome in outcomes {
        outcome?;
    }
    println!(
        "finished {} cases in {:.0}s; results in {}\nsummarize with: fuzz summary {}",
        campaign.done.load(Ordering::Relaxed),
        campaign.started.elapsed().as_secs_f64(),
        out.display(),
        out.display()
    );
    Ok(())
}
