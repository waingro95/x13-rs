//! The error of the Build 62 side of a case: rendering its spec file, running
//! the executable and reading the tables it saved.

use std::path::Path;

use thiserror::Error;

use crate::Frequency;

#[derive(Debug, Error)]
pub(crate) enum OracleError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("series {series_id:?} contains non-finite value {value}")]
    NonFiniteValue { series_id: String, value: f64 },
    #[error("invalid X-13 table: {0}")]
    InvalidTable(String),
    #[error("X-11 supports monthly and quarterly data, not {0:?}")]
    UnsupportedX11Frequency(Frequency),
    #[error("series must contain at least one value")]
    EmptySeries,
    #[error("invalid oracle configuration: {0}")]
    InvalidConfiguration(String),
    #[error("X-13 executable {0:?} was not found")]
    ExecutableNotFound(String),
    #[error("run directory already exists: {0}")]
    RunDirectoryExists(String),
    #[error("failed to start X-13 executable {path}: {source}")]
    Spawn {
        path: String,
        #[source]
        source: std::io::Error,
    },
    // `tests/parity/deviations.toml` matches this text.
    #[error("X-13 exceeded its {timeout:?} deadline")]
    Timeout { timeout: std::time::Duration },
}

impl OracleError {
    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.display().to_string(),
            source,
        }
    }
}
