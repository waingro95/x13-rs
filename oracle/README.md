# x13-oracle

Development-only tools that check the `x13` crate against the Census
X-13ARIMA-SEATS Build 62 executable. The `x13` crate depends on neither this
crate nor the executable.

The crate holds two tools:

- the **parity harness** (`src/parity/`), which produces the parity table;
- a **fuzz tool** (`src/fuzz/`), which looks for panics, hangs and other
  faults on inputs the table does not cover.

Run both inside the repository's Nix dev shell: it builds `x13as_ascii`, the
Build 62 executable.

The rest of `src/` is what the two tools share:

| Module | Does |
|---|---|
| `configuration` | What a call-table row asks for, as `tests/parity/specs.toml` states it. |
| `spec_file` | Renders a configuration and a series as Build 62's spec and input files. |
| `build62` | Finds the executable and runs it under a time limit. |
| `saved_tables` | Parses the tables Build 62 saves. |
| `rust_spec` | Builds the `x13::Spec` for a configuration and runs `x13::adjust`. |
| `user_regressors` | Generates the user-defined regressors some rows supply. |

## The parity table

```sh
cargo test -p x13-oracle --profile parity --test parity
```

A case is one corpus series (`tests/parity/series/`) under one row of the
[call table](../docs/parity-call-table.md), which `tests/parity/specs.toml`
encodes. For each case the harness writes the row's Build 62 spec file, runs
Build 62 and the Rust library on the same input, and compares every output
the row produces with the tolerances in `tests/parity/tolerances.toml`. It
prints the table (cases passed per row and frequency) and fails when any case
fails.

`cargo test --workspace` does not run the table: name it with `--test parity`.

### Environment

| Variable | Meaning |
|---|---|
| `PARITY_TIER` | `full` (default) or `quick`: the first four applicable series of each frequency for each row. |
| `PARITY_FILTER` | A glob over `<series>/<row>` case ids, for example `'*/04'` or `'IMP7741/*'`. |
| `PARITY_JOBS` | How many cases run at once, a positive integer. Default: one per core. |
| `X13AS_ASCII` | The Build 62 executable, as a path or a name on `PATH`. Default: `x13as_ascii`. |
| `PARITY_CACHE_DIR` | The cache of Build 62 output. Default: `.cache/parity`. |
| `PARITY_RECORD_RUST` | A fresh directory that receives the Rust bit records. |
| `PARITY_COMPARE_RUST` | A recorded directory the Rust bit records must equal byte for byte. |
| `PARITY_COMPARE_RUST_TOLERANT`, `PARITY_COMPARE_RUST_CUMULATIVE` | Recorded directories to measure drift against, given together. |

A relative path resolves against the workspace root.

### Results

The harness writes these to the Cargo target directory (`target/`):

| File | Contents |
|---|---|
| `parity-report.md` | The parity table. |
| `parity-failures.json` | The failures of each failing case, one per output. |
| `parity-headroom.json` | For each row, the compared value that came closest to its tolerance. |
| `parity-both-rejected.json` | The cases both implementations rejected, with both messages. |
| `parity-drift.json` | The drift of every case, when drift baselines are given. |

### The Build 62 cache

Running Build 62 is most of the cost of a run, so the harness caches what it
parsed from each run under `.cache/parity/` (gitignored). An entry's key is a
hash of the spec file, the input file, the executable's path and its flags, so
a case reruns only when one of those changes. A case that misses the cache
runs Build 62 twice and fails as nondeterministic if the two runs differ.

The cache is regenerable: delete it, or point `PARITY_CACHE_DIR` at an empty
directory, and the next run rebuilds what it needs. A full table on an empty
cache is slow. Build 62 links the host's math library, so a cache belongs to
the platform that filled it, and the harness refuses one stamped by another.

### Rust bit records

`PARITY_RECORD_RUST=<dir>` writes one file per case holding every Rust output
exactly. A later run with `PARITY_COMPARE_RUST=<dir>` fails any case whose
record differs by a byte. That is how a refactor shows it moved no number:
record on the commit before it, compare on the commit after.

A change that is meant to move numbers reports how far with the two drift
variables instead: the records of the previous change and those of the
starting point. A case fails when an output moved from either by more than a
tenth of its tolerance.

### Deviations and the rejection map

A case passes in one of three ways:

- every compared output is within tolerance;
- both implementations reject the input, and
  `tests/parity/rejection-map.toml` lists the kind of Rust's error as a
  sensible answer to Build 62's message; or
- the case matches an entry of `tests/parity/deviations.toml`.

A deviation names the cases (a glob over case ids), the output that differs,
and globs for what Build 62 and Rust each produced; all four must match the
failure. Each entry carries its reason. A full run fails when an entry matches
no case, so the list cannot go stale. Adding or changing an entry, a tolerance
or a row is a maintainer decision: see [CONTRIBUTING.md](../CONTRIBUTING.md).

## The fuzz tool

```sh
cargo run -p x13-oracle --profile parity --example fuzz -- <command>
```

The tool generates series and specs outside the corpus (odd lengths and start
dates, extreme magnitudes, gaps, nonfinite values, unusual option
combinations) and runs `x13::adjust` on them in supervised child processes,
looking for panics, hangs, runaway memory, nondeterminism and nonfinite
results. `repro` then runs a saved case through Build 62 as well and says
whether the two agree.

| Command | Does |
|---|---|
| `run --seed S --count N` | Runs a campaign of `N` generated cases; results land in `target/fuzz/seed-S/`. |
| `summary <dir>...` | Summarizes campaign directories. |
| `gen --seed S --index I` | Prints the case a campaign generated, to save as a file. |
| `repro <dir or file>` | Replays saved cases against Rust and Build 62 and compares the outcomes. |
| `replay <case>` | Runs one case in this process, unsupervised. |
| `minimize <case> --out <file>` | Shrinks a case while it keeps its outcome. |
| `conc <dir>` | Replays campaign cases on a thread pool and compares with the sequential results. |

`--help` on any command lists its options. The `fuzz` module's docs describe
the limits on child processes and how to read `repro`'s verdicts. Nothing the
tool writes is committed: a finding becomes a fix in the library, and the
parity table stays the regression test.
