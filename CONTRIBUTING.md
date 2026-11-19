# Contributing

## What the project promises

`x13` is a pure-Rust X-13ARIMA-SEATS whose results match Census
X-13ARIMA-SEATS Version 1.1 Build 62 within fixed tolerances. The promise is a
table:

- each row of [`docs/parity-call-table.md`](docs/parity-call-table.md) is one
  supported configuration, encoded in `tests/parity/specs.toml`;
- the corpus is the FRED series in `tests/parity/series/`, plus a few generated
  series described in `tests/parity/synthetic.toml`;
- the parity harness runs each row on every series it applies to, through
  Build 62 and through this crate, and compares the outputs.

A case passes when one of these holds:

- Every compared output is within the tolerances in
  `tests/parity/tolerances.toml`, and the discrete choices (transform, model
  orders, outlier types and dates) match exactly.
- Build 62 and this crate both reject the input, and this crate's error is a
  sensible one for Build 62's message (`tests/parity/rejection-map.toml`). The
  messages need not match.
- The case is listed in `tests/parity/deviations.toml`.

When Build 62 succeeds, an error from this crate is a failure, "unsupported"
included. Series with gaps go through every row: where Build 62 estimates the
gaps, the numbers must match; where it refuses, this crate returns an error.

Almost every deviation is a place where Build 62 reaches an invalid or
undefined state: a hang, an out-of-bounds read, a nonfinite or meaningless
result, the log of a nonpositive value, an empty sample. There this crate
returns a typed error that says what went wrong instead of copying the
behaviour. Each deviation is listed with its reason, and a translation bug is
not one.

Correct means within tolerance; bit-identical output is not a goal. A change
may move the numbers, for a speed-up say, if the full table still passes.
Results must be deterministic. Speed is a secondary goal.

Tolerances, call-table rows and deviations change only with a maintainer's
agreement, so raise those in an issue first.

## Reporting a difference from Build 62

A result that differs from Build 62 by more than the tolerances is a bug,
whether or not the call table covers the configuration. Please open an issue
with:

- the series: its values and start period, or its FRED id and date range;
- the request: the Rust `Spec` you built and the `.spc` file you gave Build 62;
- both results for the output that differs (for example Build 62's saved `d11`
  table and this crate's `seasonally_adjusted()`), and the size of the
  difference;
- your platform and the crate version or commit.

Check the reference first. It is Version 1.1 Build 62, whose banner reads
`Version Number 1.1 Build 62`; other builds can differ from it. `README.md`
lists the defaults this crate changes on purpose.

## Proposing a change

Small fixes can go straight to a pull request. Open an issue first for a new
option, a public API change or any large change, so the design is agreed
before the work.

In the pull request, say which checks you ran, on which platform, and whether
any results moved. CI runs formatting, clippy, the `x13` tests on Linux and
macOS, and the documentation build. The parity table is too slow for CI: run
the rows your change affects if you can, and a maintainer runs the full table
on Apple silicon before merging. Keep each pull request to one purpose.

## Repository layout

- `x13` (`src/`) is the published crate: plain Rust with no Fortran,
  subprocesses or spec files, and no dependency on `x13-oracle`.
  `src/x11/` and `src/seats/` do not depend on the ARIMA estimation,
  regression or outlier modules; the pipelines that combine them live in
  `src/engine/`.
- `x13-oracle` (`oracle/`) is development-only and never published. It runs
  Build 62, renders spec files, parses saved tables and compares. The parity
  harness is `oracle/src/parity/`, started from `oracle/tests/parity.rs`.
- `tests/parity/` holds the corpus (`series/`, `series.toml`), the synthetic
  descriptors (`synthetic.toml`), the rows (`specs.toml`), the tolerances, the
  deviations and the rejection map.
- [`docs/provenance/source-map.md`](docs/provenance/source-map.md) maps
  Build 62 routines to Rust modules.
- The hidden `__conditioning-scales` feature exposes two X-11 vectors the
  harness reads. Workspace builds switch it on through the oracle crate;
  `-p x13` builds check the crate without it.

## Development environment

The Nix flake supplies the Rust toolchain and builds Build 62 (`x13as_ascii`)
from the Census source archive:

```sh
nix develop
# if your Nix doesn't enable flakes:
nix --extra-experimental-features 'nix-command flakes' develop
```

Stay in one shell rather than wrapping each command in `nix develop -c`: with
uncommitted changes, every entry copies the repository into the Nix store. Run
`nix flake check` after changing `flake.nix`.

The reference is the executable the flake builds. A Build 62 compiled with
another compiler or other flags can differ in the last bits, so a comparison
against one says nothing about parity.

### Checks

```sh
cargo fmt --all -- --check
cargo clippy -p x13 --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p x13-oracle --test parity --locked -- -D warnings
cargo test -p x13 --all-targets --locked
cargo test -p x13 --lib --features __conditioning-scales --locked
cargo test -p x13 --doc --locked
RUSTDOCFLAGS=-Dwarnings cargo doc -p x13 --no-deps --locked
cargo test -p x13-oracle --lib --locked
cargo test -p x13-oracle --profile parity --test parity
```

The last command is the parity table. Its target is opt-in (`test = false` in
`oracle/Cargo.toml`), so `cargo test --workspace` and `--all-targets` skip it,
which is also why it has its own clippy line. The `parity` profile is release
optimization with debug assertions and overflow checks.

### Running the parity table

The table writes its results to `target/parity-report.md`.

- `PARITY_FILTER='*/87'` selects cases by a `<series>/<row>` glob.
- `PARITY_TIER=quick` runs a few series per row.
- `PARITY_JOBS=<n>` sets how many cases run at once. The default is one per
  core.
- Build 62's output is cached in `.cache/parity/` (`PARITY_CACHE_DIR` moves
  it). The first full run executes Build 62 on every case and takes a long
  time; later runs reuse the cache. The cache is never committed: it can be
  rebuilt from the executable.
- `PARITY_RECORD_RUST=<dir>` saves this crate's results bit for bit, and
  `PARITY_COMPARE_RUST=<dir>` fails on any difference from a saved run. Record
  before a refactor and compare after it to show that no result changed.

Relative paths in these variables resolve against the repository root.
[`oracle/README.md`](oracle/README.md) describes every variable.

To fix a failing row, run it with `PARITY_FILTER`, fix the Rust and rerun
until it passes. The harness case is the regression test for the fix; unit
tests are for behaviour the table does not reach. A new comparison belongs in
`oracle/src/parity/`, not in another test binary or fixture format. A unit
test that fixes exact numbers may be updated when a change moves them and the
full table still passes. Run the affected rows while you work, and the full
table once before asking for a merge.

### Without Nix or Apple silicon

Without Nix, a current stable Rust toolchain runs every check on `x13`:
formatting, the `-p x13` clippy line, its tests, doctests and documentation.
The `x13-oracle` tests and the parity table need Build 62. A few unit tests
pin exact results of the Apple silicon math library; elsewhere they are
ignored.

Parity is verified only on Apple silicon macOS. Both programs take `ln`,
`exp`, `powf` and the trigonometric functions from the platform's math
library, so on another platform a case can fail because of the platform rather
than your change. Say which platform you ran on.

### Benchmarks

```sh
cargo bench -p x13 --bench adjust -- --save-baseline before   # on the base commit
cargo bench -p x13 --bench adjust -- --baseline before        # on your change
cargo bench -p x13 --bench corpus
```

`adjust` times four pipelines on one monthly and one quarterly series with
Criterion. Run it on an otherwise idle machine and mention any case that moves
by more than about 5%. `corpus` reports throughput and peak memory over the
whole corpus; it uses `/usr/bin/time -l`, so it runs on macOS only.

## Translating Fortran

- Keep Build 62's arithmetic wherever it affects results: which `f64`
  operations run, on which values, in what order, plus its constants,
  branching, endpoint rules and state resets. Where gfortran fused a multiply
  and an add, use `mul_add`. At these tolerances a mathematically equivalent
  rewrite usually fails.
- Don't keep its structure. Write idiomatic Rust: names, types, loops and
  control flow change no bits. `PARITY_COMPARE_RUST` over the full table shows
  whether a restyle changed any.
- When you reproduce a Fortran quirk that isn't obvious, say so in a comment
  that names the routine. Keep the source map current when you move code.
- Use `f64` throughout. Repeated identical calls must give identical results
  and must not affect each other, so keep no mutable process-wide numerical
  state.
- `arima::roots::solver` is an independent implementation, not a translation.
  Build 62 finds polynomial roots with RPOLY (ACM TOMS Algorithm 493), which
  carries the ACM software licence, so do not consult RPOLY or ports of it
  when changing the solver.
  [`docs/provenance/clean-room-roots.md`](docs/provenance/clean-room-roots.md)
  says what the solver must do.
- Add a SEATS model family by implementing the family strategy in
  `src/engine/`, not by copying another family's modules.
- Document with ordinary rustdoc, and keep comments for the reason behind
  code that would otherwise look wrong. Git keeps the history.

## Corpus and rows

The corpus is public-domain data from U.S. federal agencies, downloaded from
FRED. The header of `tests/parity/series.toml` describes the format.

- Adding a row or a series, or refetching the corpus, changes what the project
  promises, so open an issue first and say what the table misses. A row is an
  entry in `tests/parity/specs.toml` and a line in the call table. A series is
  a CSV file in `tests/parity/series/` and an entry in `series.toml`.
- Keep the gaps in a series, and use true quarterly series rather than
  aggregated monthly ones.
- Synthetic series (`SYN_` ids) cover inputs FRED lacks. They are generated
  from the descriptors in `synthetic.toml`, not stored, and run only on the
  rows that name them.
- Commit only small inputs: series, specs and tolerances. Build 62's output
  and work directories stay out of git, as do private data and absolute paths
  from your machine.

## Reference executable

Census X-13ARIMA-SEATS ASCII source distribution, Version 1.1 Build 62:

- Archive: `x13as_asciisrc-v1-1-b62.tar.gz`
- URL: `https://www2.census.gov/software/x-13arima-seats/x13as/unix-linux/program-archives/x13as_asciisrc-v1-1-b62.tar.gz`
- SHA-256: `82796c84b54891474df1a6ffb16be7bf7bd4a992e3679158645c888e6cddda12`

The flake builds it from this archive, changing one link line in its makefile.

## License of contributions

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in x13-rs by you shall be dual licensed under the MIT and
Apache-2.0 licenses, without any additional terms.
