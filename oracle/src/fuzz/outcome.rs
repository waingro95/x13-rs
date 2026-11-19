//! Running one case in-process: panic capture, result analysis and the record
//! the campaign writes per case.

use super::generate::{Case, SpecSource};
use super::spec::build;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------- execution
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub(crate) struct PanicRecord {
    pub(crate) message: String,
    pub(crate) location: String,
    pub(crate) frames: Vec<String>,
}

/// Set by commands that panic on purpose many times (`minimize`, `conc`).
static QUIET: AtomicBool = AtomicBool::new(false);

/// Stops the panic hook printing each panic to stderr; it still records it.
pub(crate) fn silence_panics() {
    QUIET.store(true, Ordering::Relaxed);
}

thread_local! {
    static LAST_PANIC: RefCell<Option<PanicRecord>> = const { RefCell::new(None) };
}

pub(crate) fn install_hook() {
    std::panic::set_hook(Box::new(|info| {
        let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string payload>".into()
        };
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        let mut frames = Vec::new();
        for line in bt.lines() {
            let t = line.trim();
            if let Some((num, sym)) = t.split_once(": ")
                && num.chars().all(|c| c.is_ascii_digit())
                && (sym.starts_with("x13::") || sym.starts_with("x13_oracle::"))
                && !sym.contains("fuzz::")
            {
                let sym = match sym.rfind("::h") {
                    Some(i) if sym.len() - i == 19 => &sym[..i],
                    _ => sym,
                };
                frames.push(sym.to_string());
            }
            if frames.len() >= 10 {
                break;
            }
        }
        if !QUIET.load(Ordering::Relaxed) && std::env::var_os("FUZZ_QUIET").is_none() {
            eprintln!("panic at {location}: {message}");
        }
        LAST_PANIC.with(|p| {
            *p.borrow_mut() = Some(PanicRecord {
                message,
                location,
                frames,
            })
        });
    }));
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub(crate) struct Outcome {
    /// ok | err | panic | spec_error | period_error
    pub(crate) kind: String,
    pub(crate) detail: String,
    pub(crate) hash: u64,
    pub(crate) flags: Vec<String>,
    pub(crate) panic: Option<PanicRecord>,
    pub(crate) ms: f64,
}

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn is_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn contains_token(s: &str, token: &str) -> bool {
    let b = s.as_bytes();
    let mut from = 0;
    while let Some(i) = s[from..].find(token) {
        let start = from + i;
        let end = start + token.len();
        let before = start == 0 || !is_word_char(b[start - 1]);
        let after = end >= b.len() || !is_word_char(b[end]);
        if before && after {
            return true;
        }
        from = end;
    }
    false
}

/// Field paths in pretty Debug output that hold NaN/inf.
fn nonfinite_paths(adj: &x13::Adjustment) -> Vec<String> {
    let pretty = format!("{adj:#?}");
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut found = std::collections::BTreeSet::new();
    for line in pretty.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim();
        while stack.last().is_some_and(|(i, _)| *i >= indent) {
            stack.pop();
        }
        let name = t.split(':').next().unwrap_or("").trim();
        let opens = t.ends_with('{') || t.ends_with('[') || t.ends_with('(');
        if (contains_token(t, "NaN") || contains_token(t, "inf")) && !t.contains('"') {
            let mut path: Vec<&str> = stack.iter().map(|(_, n)| n.as_str()).collect();
            if !opens && t.contains(": ") {
                path.push(name);
            }
            found.insert(path.join("."));
        }
        if opens {
            let label = if t.contains(": ") {
                name.to_string()
            } else {
                t.trim_end_matches(['{', '[', '(']).trim().to_string()
            };
            stack.push((indent, label));
        }
    }
    found.into_iter().collect()
}

fn analyze(adj: &x13::Adjustment, values: &[f64]) -> (u64, Vec<String>) {
    let debug = format!("{adj:?}");
    let hash = hash_str(&debug);
    let mut flags = Vec::new();
    let n = values.len();
    let seats = adj.diagnostics().seats().is_some();
    let max_in = values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    for (name, comp) in [
        ("seasonal", adj.seasonal()),
        ("seasonally_adjusted", Some(adj.seasonally_adjusted())),
        ("trend_cycle", adj.trend_cycle()),
        ("irregular", Some(adj.irregular())),
    ] {
        // SEATS omits S10 or S12 where Build 62 omits the table; X-11
        // always has both, so an absent X-11 component is a finding.
        let Some(comp) = comp else {
            if !seats {
                flags.push(format!("absent:{name}"));
            }
            continue;
        };
        let comp = comp.values();
        if comp.len() != n {
            flags.push(format!("len:{name}:{}!={n}", comp.len()));
        }
        if let Some(i) = comp.iter().position(|v| !v.is_finite()) {
            flags.push(format!("core_nonfinite:{name}@{i}"));
        }
        if max_in > 0.0 && matches!(name, "seasonally_adjusted" | "trend_cycle") {
            let max_c = comp
                .iter()
                .filter(|v| v.is_finite())
                .fold(0.0f64, |m, v| m.max(v.abs()));
            if max_c / max_in > 1e6 {
                flags.push(format!("magnitude:{name}:{:.3e}", max_c / max_in));
            }
        }
    }
    if let Some(f) = adj.forecasts() {
        for (name, comp) in [
            ("forecast", f.series().values()),
            ("forecast_lo", f.lower_95().values()),
            ("forecast_hi", f.upper_95().values()),
        ] {
            if let Some(i) = comp.iter().position(|v| !v.is_finite()) {
                flags.push(format!("core_nonfinite:{name}@{i}"));
            }
        }
    }
    if contains_token(&debug, "NaN") || contains_token(&debug, "inf") {
        for p in nonfinite_paths(adj) {
            flags.push(format!("nonfinite:{p}"));
        }
    }
    (hash, flags)
}

pub(crate) fn variant(debug: &str) -> String {
    debug
        .split(['{', '(', ' '])
        .next()
        .unwrap_or("")
        .to_string()
}

pub(crate) fn run_case(case: &Case) -> Outcome {
    let started = Instant::now();
    LAST_PANIC.with(|p| *p.borrow_mut() = None);
    let values = case.values();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let (spec, start) = match build(case) {
            Ok(x) => x,
            Err(e) => return Err((e.split(':').next().unwrap_or("").to_string(), e)),
        };
        match x13::adjust(x13::Series::new(start, &values), &spec) {
            Ok(adj) => {
                let (hash, flags) = analyze(&adj, &values);
                Ok((hash, flags))
            }
            Err(e) => Err(("err".to_string(), format!("{e:?}"))),
        }
    }));
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(Ok((hash, flags))) => Outcome {
            kind: "ok".into(),
            detail: String::new(),
            hash,
            flags,
            panic: None,
            ms,
        },
        Ok(Err((kind, detail))) => Outcome {
            kind,
            hash: hash_str(&detail),
            detail,
            flags: vec![],
            panic: None,
            ms,
        },
        Err(_) => {
            let panic = LAST_PANIC.with(|p| p.borrow_mut().take());
            Outcome {
                kind: "panic".into(),
                detail: panic
                    .as_ref()
                    .map(|p| p.location.clone())
                    .unwrap_or_default(),
                hash: 0,
                flags: vec![],
                panic,
                ms,
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Record {
    pub(crate) seed: u64,
    pub(crate) index: u64,
    pub(crate) spec: String,
    pub(crate) class: String,
    pub(crate) freq: String,
    pub(crate) n: usize,
    pub(crate) start: String,
    pub(crate) first: Outcome,
    pub(crate) second_hash: u64,
    pub(crate) second_kind: String,
    pub(crate) deterministic: bool,
    /// Full case for anything non-boring (panic, flags, nondeterminism).
    pub(crate) case: Option<Case>,
}

fn spec_label(s: &SpecSource) -> String {
    match s {
        SpecSource::Row { id } => format!("row{id}"),
        SpecSource::Random { seed } => format!("rand{seed}"),
        SpecSource::Named { name } => format!("named:{name}"),
    }
}

pub(crate) fn evaluate(seed: u64, case: &Case, out: &mut impl Write) -> Record {
    let _ = writeln!(out, "S {} 1", case.index);
    let _ = out.flush();
    let first = run_case(case);
    let _ = writeln!(out, "S {} 2", case.index);
    let _ = out.flush();
    let second = run_case(case);
    let deterministic = first.kind == second.kind && first.hash == second.hash;
    let interesting = first.kind == "panic" || !first.flags.is_empty() || !deterministic;
    Record {
        seed,
        index: case.index,
        spec: spec_label(&case.spec),
        class: case.class.clone(),
        freq: format!("{:?}", case.frequency),
        n: case.values.len(),
        start: format!("{}.{}", case.start_year, case.start_subperiod),
        second_hash: second.hash,
        second_kind: second.kind.clone(),
        deterministic,
        case: interesting.then(|| case.clone()),
        first,
    }
}
