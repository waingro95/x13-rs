//! The map from Build 62 rejection messages to the Rust error kinds that are
//! a sensible rejection of them, loaded from
//! `tests/parity/rejection-map.toml`.
//!
//! See "Rejection map" in the `case` module's docs for why it keys on a
//! message prefix and an [`ErrorKind`], and how enforcement fails a case.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use x13::ErrorKind;

use super::{
    case::{CaseResult, Status},
    compare::Failure,
};

/// Whether a both-reject case off the map fails. With it off, a miss keeps its
/// both-reject status and the report only counts it.
pub(crate) const ENFORCED: bool = true;

/// The [`Failure::output`] of a case that fails on the map, which the report
/// counts as off the map.
pub(crate) const FAILURE_OUTPUT: &str = "rejection map";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MapFile {
    rejections: Vec<EntryFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntryFile {
    prefix: String,
    allowed: Vec<String>,
    routine: String,
    note: String,
}

struct Entry {
    prefix: String,
    allowed: Vec<ErrorKind>,
}

pub(crate) struct RejectionMap {
    entries: Vec<Entry>,
}

/// The map's judgment of one both-reject pair.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Verdict<'a> {
    /// The entry with this prefix allows Rust's kind.
    Allowed(&'a str),
    /// The entry with this prefix does not allow Rust's kind.
    Outside(&'a str),
    /// No entry matches Build 62's message.
    Unmapped,
}

impl Verdict<'_> {
    /// Whether the case would fail once the map is enforced.
    pub(crate) fn misses(&self) -> bool {
        !matches!(self, Self::Allowed(_))
    }

    pub(crate) fn prefix(&self) -> Option<&str> {
        match self {
            Self::Allowed(prefix) | Self::Outside(prefix) => Some(prefix),
            Self::Unmapped => None,
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Allowed(_) => "allowed",
            Self::Outside(_) => "outside",
            Self::Unmapped => "unmapped",
        }
    }
}

impl RejectionMap {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| path.display().to_string())?;
        Self::parse(&text).with_context(|| path.display().to_string())
    }

    /// Parses and validates the map: every entry needs a prefix, a routine, a
    /// note and a nonempty set of distinct known kinds, and no prefix may
    /// start another (so each message matches at most one entry).
    pub(super) fn parse(text: &str) -> Result<Self> {
        let file: MapFile = toml::from_str(text)?;
        let mut entries = Vec::<Entry>::new();
        for entry in file.rejections {
            let prefix = entry.prefix;
            ensure!(
                !prefix.trim().is_empty()
                    && !entry.routine.trim().is_empty()
                    && !entry.note.trim().is_empty(),
                "rejection-map entry {prefix:?} needs a prefix, routine and note"
            );
            ensure!(
                !entry.allowed.is_empty(),
                "rejection-map entry {prefix:?} allows no kind"
            );
            let mut allowed = Vec::new();
            for name in &entry.allowed {
                let kind = parse_kind(name).with_context(|| format!("entry {prefix:?}"))?;
                ensure!(
                    !allowed.contains(&kind),
                    "rejection-map entry {prefix:?} repeats {name}"
                );
                allowed.push(kind);
            }
            if let Some(other) = entries
                .iter()
                .find(|e| e.prefix.starts_with(&prefix) || prefix.starts_with(&e.prefix))
            {
                bail!(
                    "rejection-map prefixes overlap: {:?} and {prefix:?}",
                    other.prefix
                );
            }
            entries.push(Entry { prefix, allowed });
        }
        Ok(Self { entries })
    }

    /// Looks up the first line of Build 62's message, which names its
    /// diagnosis.
    pub(crate) fn check(&self, fortran: &str, kind: ErrorKind) -> Verdict<'_> {
        let first = fortran.lines().next().unwrap_or_default();
        match self.entries.iter().find(|e| first.starts_with(&e.prefix)) {
            Some(entry) if entry.allowed.contains(&kind) => Verdict::Allowed(&entry.prefix),
            Some(entry) => Verdict::Outside(&entry.prefix),
            None => Verdict::Unmapped,
        }
    }
}

/// Fails a both-reject case off the map, but only when `enforced`; otherwise
/// the case keeps its status and the report only counts it.
pub(crate) fn apply(case: &mut CaseResult, map: &RejectionMap, enforced: bool) {
    let Status::BothRejected {
        fortran,
        rust,
        kind,
    } = &case.status
    else {
        return;
    };
    let verdict = map.check(fortran, *kind);
    if enforced && verdict.misses() {
        case.status = Status::Fail(vec![Failure::text(
            FAILURE_OUTPUT,
            format!("rejected: {fortran}"),
            format!("rejected: {rust} ({kind:?}, {})", verdict.label()),
        )]);
    }
}

fn parse_kind(name: &str) -> Result<ErrorKind> {
    Ok(match name {
        "InvalidInput" => ErrorKind::InvalidInput,
        "InvalidSpec" => ErrorKind::InvalidSpec,
        "TooShort" => ErrorKind::TooShort,
        "UnsuitableData" => ErrorKind::UnsuitableData,
        "Numerical" => ErrorKind::Numerical,
        // An unsupported Rust outcome is a failure, not a rejection, and a
        // failed internal check says nothing about the input.
        "Unsupported" | "Internal" => bail!("{name} is never a sensible rejection"),
        _ => bail!("unknown ErrorKind {name:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frequency;

    fn entry(prefix: &str, allowed: &str) -> String {
        format!(
            "[[rejections]]\nprefix = \"{prefix}\"\nallowed = [{allowed}]\n\
             routine = \"r\"\nnote = \"n\"\n"
        )
    }

    #[test]
    fn repository_map_allows_the_expected_kinds() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity/rejection-map.toml");
        let map = RejectionMap::load(&path).unwrap();
        // Row 62's preadjustment conflict is a spec error or a data error (the
        // corpus reaches it only through gaps).
        let pseudo = "ERROR: Multiplicative or log additive seasonal adjustment cannot be \
                      performed when preadjustment factors are derived";
        for kind in [ErrorKind::InvalidSpec, ErrorKind::UnsuitableData] {
            assert!(!map.check(pseudo, kind).misses(), "{kind:?}");
        }
        assert!(map.check(pseudo, ErrorKind::Numerical).misses());
        // Gaps without a model, and a zero residual RMSE, each have a single
        // allowed kind.
        let gaps = "ERROR: Must specify a regARIMA model when the Missing Value procedure is used.";
        assert!(!map.check(gaps, ErrorKind::UnsuitableData).misses());
        assert!(map.check(gaps, ErrorKind::InvalidSpec).misses());
        let scale = "ERROR: Cannot compute outlier t-statistic for outlier backward deletion";
        assert!(!map.check(scale, ErrorKind::Numerical).misses());
        assert!(map.check(scale, ErrorKind::UnsuitableData).misses());
        // The classifier's SEATS search forms are numerical, and its bare
        // errorstop line names no cause.
        for seats in [
            "SEARCH : Model degenerate DET(A) <=0",
            "SEARCH : Model degenerate Diagonal Element too small",
            "CALCFX : Error DET(B) Zero or Negative",
        ] {
            assert!(!map.check(seats, ErrorKind::Numerical).misses(), "{seats}");
            assert!(
                map.check(seats, ErrorKind::UnsuitableData).misses(),
                "{seats}"
            );
        }
        assert_eq!(
            map.check("Build 62 errorstop: yes", ErrorKind::Numerical),
            Verdict::Unmapped
        );
    }

    #[test]
    fn malformed_maps_fail_to_load() {
        let parse = |body: &str| RejectionMap::parse(body);
        assert!(parse(&entry("ERROR: A", "\"Numerical\", \"InvalidSpec\"")).is_ok());
        assert!(parse(&entry("ERROR: A", "\"TooShort\"")).is_ok());
        for bad in [
            entry("ERROR: A", "\"Numeric\""),
            entry("ERROR: A", "\"Unsupported\""),
            entry("ERROR: A", ""),
            entry("ERROR: A", "\"Numerical\", \"Numerical\""),
            entry(" ", "\"Numerical\""),
            entry("ERROR: A", "\"Numerical\"").replace("note = \"n\"", "note = \"\""),
            entry("ERROR: A", "\"Numerical\"").replace("routine = \"r\"\n", ""),
            format!("{}extra = 1\n", entry("ERROR: A", "\"Numerical\"")),
            [
                entry("ERROR: A", "\"Numerical\""),
                entry("ERROR: A", "\"Numerical\""),
            ]
            .concat(),
            [
                entry("ERROR: AB", "\"Numerical\""),
                entry("ERROR: A", "\"Numerical\""),
            ]
            .concat(),
            [
                entry("ERROR: A", "\"Numerical\""),
                entry("ERROR: AB", "\"Numerical\""),
            ]
            .concat(),
        ] {
            assert!(parse(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn first_line_prefix_selects_the_entry_and_kind_the_verdict() {
        let map = RejectionMap::parse(
            &[
                entry("ERROR: Adding ", "\"UnsuitableData\""),
                entry("ERROR: Regression matrix singular", "\"Numerical\""),
            ]
            .concat(),
        )
        .unwrap();
        let adding = "ERROR: Adding MV1977.Jan exceeds the number of regression effects allowed";
        assert_eq!(
            map.check(adding, ErrorKind::UnsuitableData),
            Verdict::Allowed("ERROR: Adding ")
        );
        assert_eq!(
            map.check(adding, ErrorKind::Numerical),
            Verdict::Outside("ERROR: Adding ")
        );
        // Only the first line counts: a later line naming another entry does not.
        let later = "ERROR: Unknown\nERROR: Regression matrix singular because of TC2009.May.";
        assert_eq!(map.check(later, ErrorKind::Numerical), Verdict::Unmapped);
        assert!(Verdict::Unmapped.misses() && Verdict::Outside("p").misses());
        assert!(!Verdict::Allowed("p").misses());
    }

    #[test]
    fn only_an_enforced_map_fails_cases_off_it() {
        let map = RejectionMap::parse(&entry("ERROR: A", "\"Numerical\"")).unwrap();
        let case = |fortran: &str, kind| {
            CaseResult::test(
                "S/row",
                Frequency::Monthly,
                Status::BothRejected {
                    fortran: fortran.into(),
                    rust: "TypedRejection".into(),
                    kind,
                },
            )
        };
        for (fortran, kind) in [
            ("ERROR: B", ErrorKind::Numerical),
            ("ERROR: A", ErrorKind::InvalidSpec),
        ] {
            let mut informational = case(fortran, kind);
            apply(&mut informational, &map, false);
            assert!(matches!(informational.status, Status::BothRejected { .. }));
            let mut enforced = case(fortran, kind);
            apply(&mut enforced, &map, true);
            let Status::Fail(failures) = &enforced.status else {
                panic!("an enforced miss must fail");
            };
            assert_eq!(failures[0].output, FAILURE_OUTPUT);
        }
        let mut allowed = case("ERROR: A detail", ErrorKind::Numerical);
        apply(&mut allowed, &map, true);
        assert!(matches!(allowed.status, Status::BothRejected { .. }));
    }
}
