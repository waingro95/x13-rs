//! Regenerable Build 62 output cache.
//!
//! Keys hash only the spec, input, executable path, summary mode and any
//! unrecorded-column salt, so the cache is reused across runs and worktrees.
//! The salt (currently only `backcasts`) re-runs just the cases whose cached
//! output predates that column; an empty salt leaves older keys unchanged.
//! Build 62 links the host libm, so the same key can produce different bits
//! on another OS, architecture or libm build. A cache-level platform stamp,
//! not a per-key component, keeps platforms apart, so keys do not change with
//! the platform.
//!
//! The stamp is `{arch}-{os} libm:{fingerprint}`. The fingerprint hashes the
//! results of the transcendental functions both implementations call
//! (including acos, asin and atan2), on a fixed SplitMix64 argument set with
//! large and near-π/2 trigonometric arguments, computed in-process: it
//! measures the libm the run actually loads, so an OS update that leaves libm
//! alone keeps the cache, and a CPU-dispatched variant (glibc FMA paths)
//! changes it. The OS version is also written as an informational second line,
//! never compared.
//!
//! The first run on an unstamped cache adopts it for the current platform.
//! Any later mismatch is an error naming both stamps; the developer moves the
//! cache aside rather than mixing entries.
//!
//! The fingerprint uses 8,192 arguments per function across normal and
//! reduction-sensitive ranges, so a libm change that leaves every one
//! bit-identical would go unnoticed.

use std::{fs, io::Write, path::Path, process::Command, time::Duration};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Pair<T> {
    Stable(T),
    Different,
}

// The key decides when an entry is stale. It does not verify the executable
// or the corpus.
pub(crate) fn cache_key(
    spec: &str,
    input: &str,
    executable: &str,
    summary: bool,
    columns: &[&str],
) -> String {
    let mut hash = Sha256::new();
    for text in ["parity-cache-v1", spec, input, executable] {
        hash.update((text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    // `-s` can change what Build 62 computes without changing the spec file,
    // so a run without it has its own key.
    if !summary {
        let mode = "no-summary";
        hash.update((mode.len() as u64).to_le_bytes());
        hash.update(mode.as_bytes());
    }
    // Unrecorded columns (backcasts): their salt re-runs just the cases whose
    // cached output would lack the column, and leaves every other key unchanged.
    for column in columns {
        hash.update((column.len() as u64).to_le_bytes());
        hash.update(column.as_bytes());
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

const PLATFORM_STAMP: &str = "platform-stamp.txt";

/// Arguments per function in the libm fingerprint.
const FINGERPRINT_ARGUMENTS: usize = 8_192;

/// The platform whose libm Build 62 and Rust link.
pub(crate) struct Platform {
    /// `{arch}-{os} libm:{fingerprint}`, the only part a cache is checked against.
    pub(crate) stamp: String,
    /// OS product version and build, recorded for people; never compared.
    pub(crate) info: String,
}

pub(crate) fn platform() -> Platform {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    Platform {
        stamp: format!("{arch}-{os} libm:{}", libm_fingerprint()),
        info: os_version().unwrap_or_else(|| format!("{os} version unknown")),
    }
}

/// First 16 hex digits of SHA-256 over the bit patterns of the libm functions
/// Build 62 and Rust call, on a fixed SplitMix64 argument set.
fn libm_fingerprint() -> String {
    let mut state = 0x7831_335f_6c69_626d_u64;
    let mut uniform =
        || (super::synthetic::splitmix64(&mut state) >> 11) as f64 / (1_u64 << 53) as f64;
    let mut hash = Sha256::new();
    for _ in 0..FINGERPRINT_ARGUMENTS {
        // black_box keeps the compiler from folding calls with the build
        // host's libm.
        let [u, v] = [uniform(), uniform()].map(std::hint::black_box);
        let (wide, hyperbolic) = (u * 1e6 + 1e-6, u * 40.0 - 20.0);
        // Trigonometric arguments span ±1e3 (argument reduction) and a band
        // within 5e-7 of a multiple of π/2 (where tan and the reduction are
        // most sensitive). DACOS, DASIN and DATAN2 back SEATS root allocation
        // and ARIMA root frequencies.
        let trig = u * 2e3 - 1e3;
        let quadrant =
            ((u * 1273.0).floor() - 636.0) * std::f64::consts::FRAC_PI_2 + (v - 0.5) * 1e-6;
        let (unit, other) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
        for value in [
            (u * 100.0 - 50.0).exp(),
            wide.ln(),
            wide.log10(),
            (u * 100.0 + 1e-3).powf(v * 20.0 - 10.0),
            trig.sin(),
            trig.cos(),
            trig.tan(),
            quadrant.sin(),
            quadrant.cos(),
            quadrant.tan(),
            trig.atan(),
            unit.acos(),
            unit.asin(),
            unit.atan2(other),
            hyperbolic.sinh(),
            hyperbolic.cosh(),
            hyperbolic.tanh(),
            (u * 1e6).sqrt(),
        ] {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    hash.finalize()[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn os_version() -> Option<String> {
    let output = |args: &[&str]| -> Option<String> {
        let output = Command::new("sw_vers").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    (std::env::consts::OS == "macos").then_some(())?;
    Some(format!(
        "macOS {} ({})",
        output(&["-productVersion"])?,
        output(&["-buildVersion"])?
    ))
}

/// Refuses a cache whose entries came from another platform; stamps an
/// unstamped cache with `current`, keeping every existing entry. Only the
/// first line is compared; the second records `current.info` at adoption.
pub(crate) fn check_platform(directory: &Path, current: &Platform) -> Result<()> {
    fs::create_dir_all(directory)?;
    let path = directory.join(PLATFORM_STAMP);
    if !path.exists() {
        let mut file = tempfile::NamedTempFile::new_in(directory)?;
        writeln!(file, "{}", current.stamp)?;
        writeln!(file, "# adopted on {} (informational)", current.info)?;
        // A concurrent first run may have stamped it; compare with that.
        let _ = file.persist_noclobber(&path);
    }
    let recorded = fs::read_to_string(&path)
        .with_context(|| format!("reading platform stamp {}", path.display()))?;
    let recorded = recorded.lines().next().unwrap_or_default().trim();
    ensure!(
        recorded == current.stamp,
        "parity cache {} holds Build 62 outputs from platform `{recorded}`, \
         but this platform is `{}` ({}); move the cache aside to regenerate it",
        directory.display(),
        current.stamp,
        current.info
    );
    Ok(())
}

pub(crate) fn cached_pair<T>(
    directory: &Path,
    key: &str,
    timeout: Duration,
    mut run: impl FnMut() -> Result<T>,
) -> Result<Pair<T>>
where
    T: Serialize + DeserializeOwned,
{
    fs::create_dir_all(directory)?;
    let path = directory.join(format!("{key}.json"));
    if let Ok(bytes) = fs::read(&path)
        && let Ok(value) = serde_json::from_slice(&bytes)
    {
        return Ok(value);
    }
    // A completed run is reused whatever the time limit. A timeout says
    // nothing about another limit, so it is cached under its own.
    let timeout_path = directory.join(format!(
        "{key}-timeout-{}s-{}ns.json",
        timeout.as_secs(),
        timeout.subsec_nanos()
    ));
    if let Ok(bytes) = fs::read(&timeout_path)
        && serde_json::from_slice::<Duration>(&bytes).ok() == Some(timeout)
    {
        return Err(crate::OracleError::Timeout { timeout }.into());
    }
    let mut run = || {
        let result = run();
        if let Err(error) = &result
            && matches!(
                error.downcast_ref::<crate::OracleError>(),
                Some(crate::OracleError::Timeout { timeout: observed }) if *observed == timeout
            )
        {
            let mut file = tempfile::NamedTempFile::new_in(directory)?;
            file.write_all(&serde_json::to_vec(&timeout)?)?;
            file.persist(&timeout_path)?;
        }
        result
    };
    let first = run()?;
    let second = run()?;
    // The serialized output holds exact floats and no timestamp.
    let result = if serde_json::to_vec(&first)? == serde_json::to_vec(&second)? {
        Pair::Stable(first)
    } else {
        Pair::Different
    };
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    file.write_all(&serde_json::to_vec(&result)?)?;
    file.persist(path)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(180);

    #[test]
    fn cache_key_has_unambiguous_boundaries_and_invalidates_each_input() {
        let key = cache_key("ab", "c", "/oracle", true, &[]);
        assert_ne!(key, cache_key("a", "bc", "/oracle", true, &[]));
        assert_ne!(key, cache_key("ab!", "c", "/oracle", true, &[]));
        assert_ne!(key, cache_key("ab", "d", "/oracle", true, &[]));
        assert_ne!(key, cache_key("ab", "c", "/other-oracle", true, &[]));
        assert_ne!(key, cache_key("ab", "c", "/oracle", false, &[]));
        assert_ne!(key, cache_key("ab", "c", "/oracle", true, &["backcasts"]));
        assert_ne!(
            cache_key("ab", "c", "/oracle", false, &[]),
            cache_key("ab", "c", "/oracle", false, &["backcasts"])
        );
    }

    #[test]
    fn platform_stamp_adopts_an_unstamped_cache_and_refuses_another_platform() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("existing.json"), b"{}").unwrap();
        let platform = |stamp: &str, info: &str| Platform {
            stamp: stamp.into(),
            info: info.into(),
        };
        let here = platform("aarch64-macos libm:0123456789abcdef", "macOS 1 (A)");
        check_platform(dir.path(), &here).unwrap();
        // An OS update that leaves libm alone keeps the cache.
        check_platform(dir.path(), &platform(&here.stamp, "macOS 2 (B)")).unwrap();
        assert!(dir.path().join("existing.json").exists());
        let recorded = std::fs::read_to_string(dir.path().join(PLATFORM_STAMP)).unwrap();
        assert!(recorded.contains("macOS 1 (A)"), "{recorded}");
        for other in [
            "aarch64-macos libm:fedcba9876543210",
            "x86_64-linux libm:0123456789abcdef",
        ] {
            let message = check_platform(dir.path(), &platform(other, "x"))
                .unwrap_err()
                .to_string();
            assert!(message.contains(&format!("`{}`", here.stamp)), "{message}");
            assert!(message.contains(&format!("`{other}`")), "{message}");
        }
        let current = super::platform();
        assert_eq!(current.stamp, super::platform().stamp);
        let prefix = format!("{}-{} libm:", std::env::consts::ARCH, std::env::consts::OS);
        let fingerprint = current.stamp.strip_prefix(&prefix).unwrap();
        assert_eq!(fingerprint.len(), 16, "{}", current.stamp);
    }

    #[test]
    fn cold_cache_runs_twice_and_warm_cache_runs_neither() {
        let dir = tempfile::tempdir().unwrap();
        let mut calls = 0;
        let mut run = || {
            calls += 1;
            Ok::<_, anyhow::Error>(vec![1_u64, 2])
        };
        let result = cached_pair(dir.path(), "key", TIMEOUT, &mut run).unwrap();
        assert_eq!(result, Pair::Stable(vec![1, 2]));
        let result =
            cached_pair::<Vec<u64>>(dir.path(), "key", TIMEOUT, || panic!("cache miss")).unwrap();
        assert_eq!(result, Pair::Stable(vec![1, 2]));
        assert_eq!(calls, 2);
    }

    #[test]
    fn timeout_is_cached_without_repeating_the_process() {
        let dir = tempfile::tempdir().unwrap();
        let mut calls = 0;
        for _ in 0..2 {
            let error = cached_pair::<u64>(dir.path(), "timeout", TIMEOUT, || {
                calls += 1;
                Err(crate::OracleError::Timeout { timeout: TIMEOUT }.into())
            })
            .unwrap_err();
            assert!(matches!(
                error.downcast_ref::<crate::OracleError>(),
                Some(crate::OracleError::Timeout { timeout })
                    if *timeout == TIMEOUT
            ));
        }
        assert_eq!(calls, 1);
    }

    #[test]
    fn timeout_cache_is_scoped_to_input_and_exact_limit() {
        let dir = tempfile::tempdir().unwrap();
        let mut calls = 0;
        for (key, timeout) in [
            ("input-a", TIMEOUT),
            ("input-b", TIMEOUT),
            ("input-a", Duration::from_secs(181)),
            ("input-a", TIMEOUT + Duration::from_nanos(1)),
        ] {
            for _ in 0..2 {
                let error = cached_pair::<u64>(dir.path(), key, timeout, || {
                    calls += 1;
                    Err(crate::OracleError::Timeout { timeout }.into())
                })
                .unwrap_err();
                assert!(matches!(
                    error.downcast_ref::<crate::OracleError>(),
                    Some(crate::OracleError::Timeout { timeout: observed }) if *observed == timeout
                ));
            }
        }
        assert_eq!(calls, 4);
    }

    #[test]
    fn nondeterminism_is_cached_but_execution_or_parser_errors_are_not() {
        let dir = tempfile::tempdir().unwrap();
        let mut calls = 0_u64;
        let result = cached_pair(dir.path(), "different", TIMEOUT, || {
            calls += 1;
            Ok(calls)
        })
        .unwrap();
        assert_eq!(result, Pair::Different);
        assert!(
            cached_pair::<u64>(dir.path(), "error", TIMEOUT, || anyhow::bail!(
                "parser error"
            ))
            .is_err()
        );
        assert!(!dir.path().join("error.json").exists());
        std::fs::write(dir.path().join("broken.json"), b"{").unwrap();
        assert_eq!(
            cached_pair(dir.path(), "broken", TIMEOUT, || Ok(7_u64)).unwrap(),
            Pair::Stable(7)
        );
    }
}
