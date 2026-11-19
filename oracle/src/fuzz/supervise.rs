//! Running a child process under a wall-clock limit and a resident-memory cap.
//!
//! The memory cap is enforced by polling `ps`, not by `setrlimit`: the crate
//! forbids `unsafe`, and `ps -o rss=` behaves the same on macOS and Linux. A
//! child can overshoot the cap by whatever it allocates within one poll
//! interval; at 100 ms that stays well inside a gigabyte of slack.
//!
//! # Notes
//!
//! - **Armed clock.** A campaign worker runs many cases in one process, so the
//!   time limit applies to one run, not to the process: the worker's `S` line
//!   starts the clock and its `R` line stops it. Process start-up and loading
//!   the corpus are never charged to a case.
//! - **Polling, not signals.** One reader thread forwards stdout lines over a
//!   channel and the supervisor wakes every 100 ms to check the clock and the
//!   child's memory, so a child that prints nothing is still killed on time.
//! - **The cap is approximate.** `ps` costs a process spawn per poll, and a case
//!   that allocates faster than the poll interval can pass the cap before it is
//!   killed. The cap is a safety net for the machine, not an exact budget.

use std::io::{BufRead, BufReader, Read};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_millis(100);
/// Child stderr kept for diagnostics: the tail of what it printed.
const STDERR_TAIL_BYTES: usize = 4000;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub(crate) timeout: Duration,
    pub(crate) max_rss_kb: u64,
}

/// What the supervisor does with the per-case clock after a stdout line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Clock {
    /// Restart the clock: a case began.
    Start,
    /// Stop the clock: the case finished.
    Stop,
    /// Leave the clock as it was.
    Keep,
}

#[derive(Debug)]
pub(crate) enum End {
    Exited(ExitStatus),
    /// Killed after the armed clock ran past the time limit.
    Timeout,
    /// Killed after resident memory passed the cap.
    Memory,
}

#[derive(Debug)]
pub(crate) struct Watched {
    pub(crate) end: End,
    pub(crate) stderr: String,
    pub(crate) peak_rss_kb: u64,
    pub(crate) wall: Duration,
}

/// Resident set size of a live process in kilobytes, 0 if it is gone.
fn rss_kb(pid: u32) -> u64 {
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .stderr(Stdio::null())
        .output()
        .ok()
        .and_then(|out| String::from_utf8_lossy(&out.stdout).trim().parse().ok())
        .unwrap_or(0)
}

/// Drains `source`, keeping only the last few kilobytes.
fn read_tail(mut source: impl Read) -> String {
    let mut kept: Vec<u8> = Vec::new();
    let mut buffer = [0u8; 8192];
    while let Ok(count) = source.read(&mut buffer) {
        if count == 0 {
            break;
        }
        kept.extend_from_slice(&buffer[..count]);
        if kept.len() > 2 * STDERR_TAIL_BYTES {
            kept.drain(..kept.len() - STDERR_TAIL_BYTES);
        }
    }
    String::from_utf8_lossy(&kept).into_owned()
}

/// Runs `command`, feeding each stdout line to `on_line`, and kills it when
/// the clock (armed from the start iff `armed`) passes `limits.timeout` or its
/// resident memory passes `limits.max_rss_kb`. The memory cap applies for the
/// whole run; the time limit only while the clock is armed.
pub(crate) fn watch(
    mut command: Command,
    limits: Limits,
    mut armed: bool,
    mut on_line: impl FnMut(&str) -> Clock,
) -> std::io::Result<Watched> {
    let started = Instant::now();
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let pid = child.id();
    let (sender, lines) = mpsc::channel::<String>();
    let stdout = child.stdout.take().expect("piped stdout");
    let stdout_reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let stderr = child.stderr.take().expect("piped stderr");
    let stderr_reader = std::thread::spawn(move || read_tail(stderr));
    let mut clock = Instant::now();
    let mut peak_rss_kb = 0;
    let mut killed = None;
    loop {
        match lines.recv_timeout(POLL) {
            Ok(line) => match on_line(&line) {
                Clock::Start => {
                    armed = true;
                    clock = Instant::now();
                }
                Clock::Stop => armed = false,
                Clock::Keep => {}
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // Stdout closed: the process exited or closed its output.
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        let rss = rss_kb(pid);
        peak_rss_kb = peak_rss_kb.max(rss);
        if rss > limits.max_rss_kb {
            killed = Some(End::Memory);
        } else if armed && clock.elapsed() > limits.timeout {
            killed = Some(End::Timeout);
        }
        if killed.is_some() {
            let _ = child.kill();
            break;
        }
    }
    let status = child.wait()?;
    let _ = stdout_reader.join();
    // Lines the reader queued before the process ended still count.
    while let Ok(line) = lines.try_recv() {
        on_line(&line);
    }
    let stderr = stderr_reader.join().unwrap_or_default();
    Ok(Watched {
        end: killed.unwrap_or(End::Exited(status)),
        stderr,
        peak_rss_kb,
        wall: started.elapsed(),
    })
}
