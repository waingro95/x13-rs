//! Finds the Build 62 executable and runs it on one spec file.

use std::{
    env,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Child, Command, Output},
    thread,
    time::{Duration, Instant},
};

use crate::{OracleError, spec_file::INPUT_FILE};

/// The banner line Build 62 prints on standard output.
pub(crate) const VERSION_BANNER: &str = "Version Number 1.1 Build 62";

/// The executable `requested` names, as an absolute path with symlinks
/// resolved. A bare name is looked up on `PATH`.
pub(crate) fn executable(requested: &Path) -> Result<PathBuf, OracleError> {
    let not_found = || OracleError::ExecutableNotFound(requested.display().to_string());
    if requested.components().count() > 1 {
        return requested.canonicalize().map_err(|_| not_found());
    }
    env::var_os("PATH")
        .into_iter()
        .flat_map(|path| env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join(requested))
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| candidate.canonicalize().ok())
        .ok_or_else(not_found)
}

/// One Build 62 run. `directory` must not exist yet: the run creates it,
/// writes `case.spc` and the input file there, and leaves Build 62's
/// `output.*` files beside them.
pub(crate) struct Run<'a> {
    /// The path [`executable`] returned.
    pub(crate) executable: &'a Path,
    pub(crate) directory: &'a Path,
    pub(crate) spec: &'a str,
    pub(crate) input: &'a str,
    /// Passes `-s`, which makes Build 62 write the `.udg` summary file.
    pub(crate) summary: bool,
    pub(crate) timeout: Duration,
}

/// Runs Build 62 and waits for it to exit. A run that outlives `timeout` is
/// killed and reported as [`OracleError::Timeout`].
pub(crate) fn run(run: Run<'_>) -> Result<Output, OracleError> {
    let deadline = Instant::now()
        .checked_add(run.timeout)
        .filter(|_| !run.timeout.is_zero())
        .ok_or_else(|| {
            OracleError::InvalidConfiguration(
                "the Build 62 time limit must be positive and representable".into(),
            )
        })?;
    let directory = run.directory;
    fs::create_dir(directory).map_err(|source| {
        if source.kind() == std::io::ErrorKind::AlreadyExists {
            OracleError::RunDirectoryExists(directory.display().to_string())
        } else {
            OracleError::io(directory, source)
        }
    })?;
    let write = |name: &str, text: &str| {
        let path = directory.join(name);
        fs::write(&path, text).map_err(|source| OracleError::io(&path, source))
    };
    write("case.spc", run.spec)?;
    write(INPUT_FILE, run.input)?;
    let create = |name: &str| {
        let path = directory.join(name);
        File::create(&path).map_err(|source| OracleError::io(&path, source))
    };
    let mut command = Command::new(run.executable);
    command
        .args(["-i", "case", "-o", "output", "-n", "-q"])
        .args(run.summary.then_some("-s"))
        .current_dir(directory)
        .stdout(create("console.stdout")?)
        .stderr(create("console.stderr")?);
    let mut child = command.spawn().map_err(|source| OracleError::Spawn {
        path: run.executable.display().to_string(),
        source,
    })?;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(source) => {
                kill(&mut child, run.executable)?;
                return Err(OracleError::io(run.executable, source));
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            kill(&mut child, run.executable)?;
            return Err(OracleError::Timeout {
                timeout: run.timeout,
            });
        }
        thread::sleep(remaining.min(Duration::from_millis(10)));
    };
    let read = |name: &str| {
        let path = directory.join(name);
        fs::read(&path).map_err(|source| OracleError::io(&path, source))
    };
    Ok(Output {
        status,
        stdout: read("console.stdout")?,
        stderr: read("console.stderr")?,
    })
}

fn kill(child: &mut Child, executable: &Path) -> Result<(), OracleError> {
    // Wait even when the kill fails, so the child is never left a zombie.
    let killed = child.kill();
    let reaped = child.wait();
    killed.map_err(|source| OracleError::io(executable, source))?;
    reaped.map_err(|source| OracleError::io(executable, source))?;
    Ok(())
}
