//! `summary`: digest one or more campaign output directories.

use super::campaign::Event;
use super::outcome::{Record, variant};
use anyhow::{Context, Result};
use clap::Args;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

/// Cases slower than this are listed.
const SLOW_MS: f64 = 10_000.0;

#[derive(Args)]
pub(crate) struct SummaryArgs {
    /// Campaign output directories (each holds `results.jsonl`).
    dirs: Vec<PathBuf>,
}

fn read_lines<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<Vec<T>> {
    let text = fs::read_to_string(path).with_context(|| path.display().to_string())?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).with_context(|| path.display().to_string()))
        .collect()
}

/// A flag with its position and magnitude stripped, so like flags group.
fn flag_key(flag: &str) -> String {
    let flag = flag.split('@').next().unwrap_or(flag);
    if flag.starts_with("magnitude") {
        return flag
            .rsplit_once(':')
            .map_or(flag, |(head, _)| head)
            .to_owned();
    }
    if flag.starts_with("len:") {
        return flag.split(':').take(2).collect::<Vec<_>>().join(":");
    }
    flag.to_owned()
}

/// Call-table rows stay separate; every generated spec counts as `random`.
fn spec_group(label: &str) -> String {
    if label.starts_with("row") {
        label.to_owned()
    } else {
        "random".to_owned()
    }
}

fn smallest<'a>(records: &[&'a Record]) -> &'a Record {
    records
        .iter()
        .min_by_key(|r| r.n)
        .copied()
        .expect("non-empty group")
}

fn top_specs(records: &[&Record], limit: usize) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for record in records {
        *counts.entry(spec_group(&record.spec)).or_default() += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts
        .into_iter()
        .take(limit)
        .map(|(spec, count)| format!("{spec}:{count}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn describe(record: &Record) -> String {
    format!(
        "seed={} idx={} {} {} n={} {}",
        record.seed,
        record.index,
        record.spec,
        record.freq,
        record.n,
        record.class.chars().take(60).collect::<String>()
    )
}

pub(crate) fn summary(args: SummaryArgs) -> Result<()> {
    let mut records: Vec<Record> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    for dir in &args.dirs {
        records.extend(read_lines::<Record>(&dir.join("results.jsonl"))?);
        if dir.join("events.jsonl").exists() {
            events.extend(read_lines::<Event>(&dir.join("events.jsonl"))?);
        }
    }
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let mut errors: BTreeMap<String, usize> = BTreeMap::new();
    let mut panics: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    let mut flags: BTreeMap<String, Vec<&Record>> = BTreeMap::new();
    for record in &records {
        *kinds.entry(record.first.kind.as_str()).or_default() += 1;
        match record.first.kind.as_str() {
            "err" => *errors.entry(variant(&record.first.detail)).or_default() += 1,
            "panic" => panics
                .entry(record.first.detail.as_str())
                .or_default()
                .push(record),
            _ => {}
        }
        let mut seen = std::collections::BTreeSet::new();
        for flag in &record.first.flags {
            let key = flag_key(flag);
            if seen.insert(key.clone()) {
                flags.entry(key).or_default().push(record);
            }
        }
    }
    println!("total {}; kinds {kinds:?}", records.len());
    let mut errors: Vec<_> = errors.into_iter().collect();
    errors.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    println!("top errors: {:?}", &errors[..errors.len().min(40)]);

    println!("\nPANICS: {} sites", panics.len());
    let mut sites: Vec<_> = panics.into_iter().collect();
    sites.sort_by_key(|(_, group)| std::cmp::Reverse(group.len()));
    for (location, group) in sites {
        let panic = group[0].first.panic.clone().unwrap_or_default();
        println!(
            "  {:5}  {location}\n         msg={}\n         frames={:?}\n         smallest: {}\n         specs={}",
            group.len(),
            panic.message.chars().take(150).collect::<String>(),
            &panic.frames[..panic.frames.len().min(4)],
            describe(smallest(&group)),
            top_specs(&group, 8)
        );
    }

    println!("\nFLAGS: {} kinds", flags.len());
    let mut flags: Vec<_> = flags.into_iter().collect();
    flags.sort_by_key(|(_, group)| std::cmp::Reverse(group.len()));
    for (key, group) in flags {
        println!(
            "  {:5}  {}  smallest: {}  specs={}",
            group.len(),
            key.chars().take(120).collect::<String>(),
            describe(smallest(&group)),
            top_specs(&group, 5)
        );
    }

    let nondeterministic: Vec<_> = records.iter().filter(|r| !r.deterministic).collect();
    println!("\nNONDETERMINISTIC: {}", nondeterministic.len());
    for record in nondeterministic.iter().take(20) {
        println!(
            "  seed={} idx={} {} {}->{}",
            record.seed, record.index, record.spec, record.first.kind, record.second_kind
        );
    }

    println!("\nEVENTS: {}", events.len());
    for event in events.iter().take(30) {
        println!(
            "  {} seed={} idx={:?} run={:?} {:.0}s peak {} MB",
            event.event, event.seed, event.index, event.run, event.seconds, event.peak_rss_mb
        );
    }

    let mut slow: Vec<_> = records.iter().filter(|r| r.first.ms > SLOW_MS).collect();
    slow.sort_by(|a, b| b.first.ms.total_cmp(&a.first.ms));
    println!("\nSLOW (>{:.0}s): {}", SLOW_MS / 1000.0, slow.len());
    for record in slow.iter().take(20) {
        println!(
            "  {:.1}s {} {}",
            record.first.ms / 1000.0,
            describe(record),
            record.first.kind
        );
    }
    Ok(())
}
