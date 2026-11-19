//! Production-only corpus throughput: `cargo bench -p x13 --bench corpus`.
//! Optional `--config seats-combo --series TOTRESNS --repeat 3 --workers 1`.
//! Without --workers, each configuration runs in fresh one-worker/all-worker
//! processes under macOS `time -l`, so peak RSS is not inherited across runs.
//! CSV loading/spec construction are outside timing; adjustment and result
//! destruction are timed. Rejections are reported, not counted as successes.

mod common;

use common::{CorpusSeries, Result};
use std::{
    collections::BTreeMap,
    hint::black_box,
    path::Path,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use x13::regarima::{OutlierSpec, OutlierTypes};
use x13::{ArimaModel, ArimaOrder, Period, RegArimaSpec, Series, Spec, Transform, adjust, x11};

const CONFIGS: [&str; 5] = [
    "x11",
    "x11-auto",
    "x11-outliers",
    "seats-auto",
    "seats-combo",
];

struct Input {
    id: String,
    start: Period,
    values: Vec<f64>,
}

fn spec(name: &str) -> Result<Spec> {
    let airline = ArimaModel::new(ArimaOrder::new(0, 1, 1)?, ArimaOrder::new(0, 1, 1)?)?;
    let automatic = RegArimaSpec::automatic_model().with_transform(Transform::Log);
    // An explicit additive S3x5 X-11 keeps the X-11 timings comparable across
    // runs.
    let s3x5 = x11::Options::new().with_seasonal_filter(x11::SeasonalFilter::S3x5);
    Ok(match name {
        "x11" => Spec::x11(s3x5.with_mode(x11::Mode::Additive)),
        "x11-auto" => Spec::x11_regarima(automatic, s3x5.with_mode(x11::Mode::Multiplicative)),
        "x11-outliers" => Spec::x11_regarima(
            RegArimaSpec::new(airline)
                .with_transform(Transform::Log)
                .with_outliers(OutlierSpec::new().with_types(OutlierTypes::ALL)),
            s3x5.with_mode(x11::Mode::Multiplicative),
        ),
        "seats-auto" => Spec::seats(automatic),
        "seats-combo" => common::seats_combo()?,
        _ => return Err(format!("unknown configuration: {name}").into()),
    })
}

fn read_series(root: &Path, name: &str, only: Option<&str>) -> Result<Vec<Input>> {
    let mut result = Vec::new();
    for (id, quarterly) in common::corpus_ids(root)? {
        if only.is_some_and(|only| only != id) {
            continue;
        }
        let CorpusSeries {
            quarterly,
            mut start,
            mut values,
        } = common::load(root, &id, quarterly)?;
        // Match the existing corpus bounds for rows 02/38/53/86/87, retaining
        // missing observations. No CSV or corpus manifest is modified.
        let maximum = if quarterly {
            if name.starts_with("seats") {
                328
            } else if name == "x11" {
                340
            } else {
                336
            }
        } else {
            780
        };
        let mut offset = values.len().saturating_sub(maximum);
        if quarterly {
            while values.len() - offset + (usize::from(start.subperiod()) - 1 + offset) % 4
                > maximum
            {
                offset += 1;
            }
        }
        start = start.shift(offset as i64)?;
        values.drain(..offset);
        if !matches!(name, "seats-combo" | "x11") && values.iter().any(|&v| v <= 0.) {
            continue;
        }
        result.push(Input { id, start, values });
    }
    result.sort_by(|a, b| a.id.cmp(&b.id));
    if result.is_empty() {
        return Err("no applicable series".into());
    }
    Ok(result)
}

fn run(name: &str, only: Option<&str>, workers: usize, repeats: usize) -> Result<()> {
    let request = spec(name)?;
    let series = read_series(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/parity"),
        name,
        only,
    )?;
    // Compile-time check that worker threads can share the request and the series.
    fn send_sync<T: Send + Sync>(_: &T) {}
    send_sync(&request);
    send_sync(&series);
    let next = AtomicUsize::new(0);
    let count = series.len() * repeats;
    let started = Instant::now();
    let samples = std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|_| {
                let (series, request, next) = (&series, &request, &next);
                scope.spawn(move || {
                    let mut samples = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        let input = &series[index % series.len()];
                        let start = Instant::now();
                        let result = black_box(adjust(
                            Series::new(input.start, black_box(&input.values)),
                            black_box(request),
                        ));
                        // Dropping Ok destroys the result; retain only an error
                        // so formatting is outside the per-call measurement.
                        let error = result.err();
                        let seconds = start.elapsed().as_secs_f64();
                        samples.push((input.id.clone(), seconds, error.map(|e| format!("{e:?}"))));
                    }
                    samples
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("benchmark worker panicked"))
            .collect::<Vec<_>>()
    });
    let seconds = started.elapsed().as_secs_f64();
    let mut ordered = samples.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
    let quantile = |p: usize| ordered[(count * p).div_ceil(100).saturating_sub(1)].1;
    let mut errors = BTreeMap::new();
    for (_, _, error) in &samples {
        if let Some(error) = error {
            *errors.entry(error).or_insert(0_usize) += 1;
        }
    }
    let successes = samples.iter().filter(|s| s.2.is_none()).count();
    println!(
        "{}",
        serde_json::json!({
            "configuration": name, "workers": workers, "series": series.len(), "calls": count,
            "seconds": seconds, "calls_per_second": count as f64 / seconds,
            "successful_calls": successes, "successful_calls_per_second": successes as f64 / seconds,
            "rejections": errors, "p50_ms": quantile(50) * 1000., "p99_ms": quantile(99) * 1000.,
            "max_ms": ordered[count - 1].1 * 1000., "slowest": ordered[count - 1].0,
            "samples": samples,
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    // cargo test --all-targets must not accidentally run the full benchmark.
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !args.iter().any(|arg| arg == "--bench") {
        return Ok(());
    }
    if cfg!(feature = "__conditioning-scales") {
        return Err(
            "production benchmark requires __conditioning-scales OFF; build with -p x13".into(),
        );
    }
    let option = |key: &str| -> Result<Option<&str>> {
        args.iter()
            .position(|arg| arg == key)
            .map(|i| {
                args.get(i + 1)
                    .map(String::as_str)
                    .ok_or_else(|| format!("{key} needs a value").into())
            })
            .transpose()
    };
    let name = option("--config")?;
    if name.is_some_and(|name| !CONFIGS.contains(&name)) {
        return Err("unknown configuration".into());
    }
    let only = option("--series")?;
    let repeats = option("--repeat")?.unwrap_or("1").parse::<usize>()?;
    if repeats == 0 {
        return Err("repeat must be positive".into());
    }
    if let Some(workers) = option("--workers")? {
        let workers = workers.parse::<usize>()?;
        if workers == 0 {
            return Err("workers must be positive".into());
        }
        return run(
            name.ok_or("--workers requires --config")?,
            only,
            workers,
            repeats,
        );
    }
    let parallelism = std::thread::available_parallelism()?.get();
    for config in CONFIGS
        .into_iter()
        .filter(|config| name.is_none_or(|name| name == *config))
    {
        for workers in [1, parallelism] {
            eprintln!("Benchmark {config}: {workers} workers, {repeats} corpus passes");
            let mut command = Command::new("/usr/bin/time");
            command.args(["-l"]).arg(std::env::current_exe()?).args([
                "--bench",
                "--config",
                config,
                "--workers",
                &workers.to_string(),
                "--repeat",
                &repeats.to_string(),
            ]);
            if let Some(only) = only {
                command.args(["--series", only]);
            }
            let output = command.output()?;
            if !output.status.success() {
                return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
            }
            let mut result: serde_json::Value = serde_json::from_slice(&output.stdout)?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            let peak = stderr
                .lines()
                .find(|line| line.contains("maximum resident set size"))
                .and_then(|line| line.split_whitespace().next())
                .ok_or("time -l did not report peak RSS")?
                .parse::<u64>()?;
            result["peak_rss_bytes"] = peak.into();
            println!("{result}");
        }
    }
    Ok(())
}
