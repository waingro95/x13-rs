//! Robustness fuzzer for the public `x13::adjust` API: a development tool, not
//! a test.
//!
//! # Purpose
//!
//! The parity table shows that Rust matches Build 62 on the corpus and the
//! call-table rows. It cannot reach what lies outside that: odd series lengths
//! and start dates, extreme magnitudes, gaps, nonfinite values, explicit model
//! orders no row uses, and the builder options in unusual combinations. This
//! fuzzer generates those inputs deterministically from `(seed, index)` and
//! looks for what no tolerance can excuse: panics, hangs, runaway memory,
//! aborts, nondeterminism, and `Ok` results carrying nonfinite or wrongly
//! sized components. Findings are then compared with Build 62 by `repro`, so
//! each is classified as a Rust fault, a case Rust wrongly rejects, a case it
//! wrongly accepts, or an agreement.
//!
//! It is a tool, not a test: nothing here runs under `cargo test`, and its
//! output is never committed. A finding becomes a fix by reproducing it with
//! `repro`, guarding it in the library, and rerunning `repro` until it passes;
//! the parity table remains the regression test for anything it can express.
//!
//! # Commands
//!
//! The tool is the `fuzz` example. Run it with the `parity` profile (release
//! optimization plus debug assertions and overflow checks), the profile the
//! parity table runs under:
//!
//! ```text
//! cargo run -p x13-oracle --profile parity --example fuzz -- <command>
//! ```
//!
//! | command | does |
//! |---|---|
//! | `run --seed S --count N` | A campaign: cases `(S, 0..N)` over supervised worker processes. Each case runs twice (a determinism check). Results, events and the cases behind them land in `target/fuzz/seed-S/`. |
//! | `summary <dir>...` | Digest campaign directories: outcome counts, panic sites, flag classes, nondeterminism, events, slow cases. |
//! | `repro <dir or file>` | Replay saved case files against Rust and Build 62 and compare outcome classes. Exits nonzero if any Rust side faults. |
//! | `gen --seed S --index I` | Print the case a campaign generated, to save as a repro. |
//! | `replay <case>` | Run one case in this process with `--spec` and `--dump`; unsupervised. |
//! | `minimize <case> --out <file>` | Shrink a case while keeping its outcome signature. |
//! | `conc <dir>` | Replay campaign cases on a thread pool and compare with the sequential results. |
//!
//! # Caps
//!
//! A hang or a memory blow-up must cost one case, not the machine. `run` and
//! `repro` run every Rust case in a child process (`fuzz worker`, `fuzz case`)
//! and share three limits:
//!
//! - `--workers` child processes at once, default 2;
//! - `--max-rss-mb` resident memory per case, default 2048, killed above it;
//! - `--timeout-secs` per run, default 60, killed above it.
//!
//! A campaign and the full parity table each want every core, so run one at a
//! time.
//!
//! # Replaying a finding
//!
//! A case file is the JSON `gen` prints: a spec source (`Row` for a call-table
//! row, `Random` for a seeded builder spec, `Named` for a hand-built spec, see
//! `named.rs`), a frequency and start, and the series values as `{:?}` strings
//! so NaN and infinity survive. To keep a finding, save its case and replay it:
//!
//! ```text
//! fuzz gen --seed 3 --index 34152 --out target/fuzz/cases/hang.json
//! fuzz repro target/fuzz/cases
//! ```
//!
//! `repro` prints, per case, the Rust class (`ok`, `error`, `panic`, `hang`,
//! `oom`, `crash`), the Build 62 class (`ok`, `rejected`, `hang`, `skipped`,
//! `failed`) and a verdict: `agree`, `rust-accepts` (Rust is `ok`, Build 62
//! rejects), `rust-rejects` (Rust errs, Build 62 succeeds) or `FAULT`. The
//! Build 62 spec, input and error file of every run are kept under
//! `target/fuzz/repro/build62/<case>/`.
//!
//! # Design notes
//!
//! - **Why subprocesses.** `catch_unwind` handles a panic but not an infinite
//!   loop, unbounded allocation or a stack overflow, which are the findings
//!   that matter most. Supervision lives in `supervise.rs`; cases never run in
//!   the supervisor.
//! - **Why `ps` polling for memory.** The crate forbids `unsafe`, so there is
//!   no `setrlimit`; memory is polled through `ps`. A child can overshoot the
//!   cap by about one poll interval of allocation.
//! - **Two runs per case.** Comparing hashes of two in-process runs catches
//!   process-wide mutable numerical state, which CONTRIBUTING.md forbids because the
//!   harness runs cases in parallel. `conc` checks the same property across
//!   threads.
//! - **One grammar, two renderers.** A named spec is parsed once
//!   (`named::parse`) and rendered both to a Rust `Spec` and to a Build 62 spec,
//!   so the two sides of a replay cannot drift. A case marks gaps with NaN, as
//!   `x13::adjust` takes them; Build 62 receives `-99999` at those positions.
//!   An exact `-99999` value in a case is therefore a gap to Build 62 but an
//!   observation to Rust, so such a case disagrees by design. Random builder
//!   specs have no Build 62 rendering, so they report `skipped`; to compare one
//!   with Build 62, state it as a `Named` or `Row` case.
//! - **Build 62 classification is the harness's.** `repro` reuses the parity
//!   harness's `build62_rejection`, so "Build 62 rejected" means the same
//!   thing here as in the table, including the banner check.
//! - **Only the public API.** The fuzzer calls `x13::adjust` and reads public
//!   result accessors, plus `Debug` output for the nonfinite scan.
//! - **Generator mix.** Table rows 65%, random specs 35%, with hand-tuned
//!   length and magnitude tables. It is tuned to find, not to cover; widen it
//!   when a class of bug turns up that it cannot generate.

mod build62;
mod campaign;
mod generate;
mod named;
mod outcome;
mod repro;
mod spec;
mod summary;
mod supervise;
mod tools;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;
use supervise::Limits;

#[derive(Parser)]
#[command(
    name = "fuzz",
    about = "Robustness fuzzer for x13::adjust (development tool)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a campaign of generated cases.
    Run(campaign::RunArgs),
    /// Replay saved cases against Rust and Build 62.
    Repro(repro::ReproArgs),
    /// Summarize campaign output directories.
    Summary(summary::SummaryArgs),
    /// Print the case a campaign generated for (seed, index).
    Gen(tools::GenArgs),
    /// Run one case in this process (unsupervised).
    Replay(tools::ReplayArgs),
    /// Shrink a case while keeping its outcome signature.
    Minimize(tools::MinimizeArgs),
    /// Check concurrent runs against a campaign's sequential results.
    Conc(tools::ConcArgs),
    #[command(hide = true)]
    Worker(campaign::WorkerArgs),
    #[command(hide = true)]
    Case(repro::CaseArgs),
}

/// Limits shared by the commands that supervise child processes.
#[derive(Args, Clone, Copy)]
struct LimitArgs {
    /// Child processes running at once.
    #[arg(long, default_value_t = 2)]
    workers: usize,
    /// Resident-memory cap per case, in megabytes.
    #[arg(long, default_value_t = 2048)]
    max_rss_mb: u64,
    /// Wall-clock limit per run, in seconds.
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,
}

impl LimitArgs {
    fn limits(self) -> Limits {
        Limits {
            timeout: Duration::from_secs(self.timeout_secs),
            max_rss_kb: self.max_rss_mb * 1024,
        }
    }
}

/// `target/fuzz` under the repository (or `CARGO_TARGET_DIR`).
fn output_root() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map_or_else(|| PathBuf::from("target"), |root| root.join("target"))
        })
        .join("fuzz")
}

fn dispatch(command: Command) -> Result<ExitCode> {
    match command {
        Command::Run(args) => campaign::run(args)?,
        Command::Repro(args) => return repro::repro(args),
        Command::Summary(args) => summary::summary(args)?,
        Command::Gen(args) => tools::gen_case(args)?,
        Command::Replay(args) => tools::replay(args)?,
        Command::Minimize(args) => tools::minimize(args)?,
        Command::Conc(args) => tools::conc(args)?,
        Command::Worker(args) => campaign::worker(args)?,
        Command::Case(args) => repro::case(args)?,
    }
    Ok(ExitCode::SUCCESS)
}

/// Entry point of the `fuzz` example.
pub fn main() -> ExitCode {
    outcome::install_hook();
    match dispatch(Cli::parse().command) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("fuzz: {error:#}");
            ExitCode::from(2)
        }
    }
}
