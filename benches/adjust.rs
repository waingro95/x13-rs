//! Runtime regression suite: `cargo bench -p x13 --bench adjust`.
//!
//! Four pipelines, each on one long monthly and one quarterly series from the
//! frozen corpus. Loading and spec construction stay outside the timed region;
//! `adjust` and dropping its result are timed. Save a baseline with
//! `-- --save-baseline <name>` and compare a change against it with
//! `-- --baseline <name>`. Run it on an otherwise idle machine. Under
//! `cargo test` each case runs once, as a smoke test.

mod common;

use common::{CorpusSeries, Result};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::{hint::black_box, path::Path};
use x13::regarima::OutlierSpec;
use x13::{RegArimaSpec, Series, Spec, Transform, adjust, x11};

/// Housing starts (780 months, gap-free) and personal consumption
/// expenditures (318 quarters, gap-free): positive and seasonal, so every
/// pipeline below accepts both.
const SERIES: [(&str, bool); 2] = [("HOUSTSNSA", false), ("NA000349Q", true)];

fn cases() -> Result<[(&'static str, Spec); 4]> {
    let automatic = RegArimaSpec::automatic_model().with_transform(Transform::Log);
    Ok([
        ("x11", Spec::x11(x11::Options::new())),
        (
            "x11-auto-outliers",
            Spec::x11_regarima(
                automatic.clone().with_outliers(OutlierSpec::new()),
                x11::Options::new().with_mode(x11::Mode::Multiplicative),
            ),
        ),
        ("seats-auto", Spec::seats(automatic)),
        ("seats-combo", common::seats_combo()?),
    ])
}

fn adjust_suite(criterion: &mut Criterion) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/parity");
    let series = SERIES.map(|(id, quarterly)| {
        let loaded = common::load(&root, id, quarterly)
            .unwrap_or_else(|error| panic!("cannot load {id}: {error}"));
        (id, loaded)
    });
    let cases = cases().expect("bench specs are valid");
    for (name, spec) in &cases {
        let mut group = criterion.benchmark_group(*name);
        for (id, CorpusSeries { start, values, .. }) in &series {
            group.bench_with_input(BenchmarkId::from_parameter(id), values, |bench, values| {
                bench.iter(|| {
                    // A rejected case would time the error path; fail instead.
                    adjust(Series::new(*start, black_box(values)), black_box(spec))
                        .unwrap_or_else(|error| panic!("{name} rejects {id}: {error}"))
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, adjust_suite);
criterion_main!(benches);
