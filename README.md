# STATS 607 – Project 3: High-Performance Simulation Study

This repository contains the code for **STATS 607 Unit 3** (High-Performance Simulation Study), built on top of the **Project 2** exchangeability / Pólya-urn simulation framework.

- Unit 2 codebase (baseline implementation): <https://github.com/apxroad/stats607-project2>

The goal of Unit 3 is to:

- profile the existing Unit‑2 code,
- implement performance optimizations (algorithmic, array programming, parallelisation),
- document and visualise performance improvements,
- and validate that the optimised code is numerically correct and reproducible.

Details of the baseline and optimisations are in:

- `docs/BASELINE.md` – profiling, complexity, and numerical stability for the **Unit‑2 baseline**.
- `docs/OPTIMIZATION.md` – description and evaluation of all **Unit‑3 optimisations**.

---

## 1. Repository structure (Unit 3 view)

### 1.1. Top-level layout

```text
stats607-project3/
├── ADEMPI.md                 # Design notes from Project 2 (ADEMP-style)
├── ANALYSIS.md               # High-level analysis notes
├── Makefile                  # Main orchestration for all parts & Unit 3 tasks
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docs/
│   ├── BASELINE.md           # Unit 2 baseline performance
│   └── OPTIMIZATION.md       # Unit 3 optimisation report
├── src/
│   ├── dgps.py               # Data-generating processes
│   ├── interfaces.py         # Common interfaces / dataclasses
│   ├── methods.py            # Predictive methods
│   ├── metrics.py            # Distance / PIT / diagnostics
│   ├── plotstyle.py          # Shared Matplotlib style
│   ├── polya.py              # Pólya urn / exchangeable sequence core
│   └── simulation.py         # Generic simulation helpers
├── src_cli/
│   ├── analyze.py
│   ├── parta_panels.py                       # Optimised Part A CLI
│   ├── parta_panels_baseline_backup.py       # Baseline Part A CLI (Unit 2)
│   ├── partb_fig_paths.py
│   ├── partb_figures.py
│   ├── partb_log_convergence.py
│   ├── partc_figures_prop26.py
│   ├── partc_log_prop26.py                   # Optimised + parallel Part C CLI
│   └── partc_log_prop26_baseline_backup.py   # Baseline Part C CLI (Unit 2)
├── scripts/
│   ├── benchmark_runtime.py        # Component-wise baseline vs optimised timings
│   ├── complexity_parta.py         # (original) Part A complexity vs n
│   ├── complexity_parta_compare.py # Part A baseline vs optimised complexity
│   ├── complexity_partc_L.py       # Part C runtime vs continuation length L
│   ├── plot_performance.py         # Build performance figures from CSVs
│   └── stability_check.py          # Light numerical stability checks
├── tests/
│   ├── data/                       # Regression/golden data (if any)
│   ├── conftest.py
│   ├── test_dgp.py
│   ├── test_exchangeability.py
│   ├── test_polya_module.py
│   ├── test_repro.py
│   ├── test_smoke_parta.py
│   ├── test_smoke_partb.py
│   ├── test_smoke_partc.py
│   └── test_regression_unit3.py    # Unit 3 regression test for Part C
├── results/
│   ├── figures/                    # Generated figures (PNG + PDF)
│   ├── raw/                        # Raw CSV outputs from simulations
│   └── summary/                    # Summarised / tidy CSVs
└── examples/
    └── polya_panel.py, polya_single_t.py  # Small illustrative scripts
```

This tree is abridged; Python `__pycache__/` directories and temporary files are omitted.

### 1.2. Key components

- `src/` – **core model & methods** (Pólya urn, DGPs, metrics).
- `src_cli/` – **command-line entry points** for each project part.
- `scripts/` – **Unit‑3 utilities** for profiling, complexity, benchmarking, plotting, and stability checks.
- `docs/` – **written report** for Unit‑2 baseline and Unit‑3 optimisations.
- `results/` – **generated outputs** (CSV + figures + profiler stats).
- `tests/` – **pytest suite**, including the Unit‑3 regression test.

---

## 2. Setup

All commands below assume you are in the repository root.

### 2.1. Python environment

```bash
# Create and activate a virtual environment (example)
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

If your environment is already created (e.g. from Project 2), just activate it.

### 2.2. Quick sanity checks

Before running the full pipeline:

```bash
# Run basic tests (smoke tests + unit tests)
make test

# Run the Unit-3 regression test for parallel Part C
make regression
```

Both should pass in a few seconds.

---

## 3. Core pipeline (Unit‑2 functionality, optimised implementation)

These targets reproduce the original Project 2 simulation study (using the **optimised** code where applicable):

```bash
# Clean previous outputs, then run the full pipeline
make all
# equivalent to: clean → simulate → analyze → figures
```

More granular targets:

```bash
# Only generate raw simulation outputs (Part B distances, Part C logs)
make simulate

# Post-process raw outputs into summary CSVs
make analyze

# Generate all figures (Parts A, B, C)
make figures
```

Individual parts:

```bash
# Part A prior panels (n = 0, varying M)
make partA-prior

# Part A posterior panels (n = 100, 500, 1000)
make partA

# Part B (distances + predictive paths + figures)
make partB

# Part C (pooled-Z for Proposition 2.6, sequential 1-worker version)
make partC
```

By default, the main knobs (see `Makefile`) are:

- `BASE = uniform`
- `ALPHA = 5.0`
- `N = 1000`
- `TVALS = 0.25 0.5 0.75`
- `SEED = 2025`

You can override them on the command line, e.g.:

```bash
make partB N=500 ALPHA=1.0
```

---

## 4. Unit‑3 performance study: Makefile targets

The following targets are required by Unit 3 and make the performance study **fully reproducible**.

### 4.1. Profiling

```bash
make profile
```

Runs:

- `profile-parta` – `cProfile` for `src_cli.parta_panels` at a representative configuration.
- `profile-partc` – `cProfile` for `src_cli.partc_log_prop26` (sequential path).

Profiles are written under `results/` (e.g. `results/profile_parta_*.pstats`, `results/profile_partc_*.pstats`) and are summarised in `docs/BASELINE.md`.

### 4.2. Computational complexity

```bash
make complexity
```

Runs two scripts:

1. **Part A complexity vs $n$**  
   `python -m scripts.complexity_parta_compare`

   - Calls both the **baseline** `parta_panels_baseline_backup` and the **optimised** `parta_panels` CLIs.
   - Uses `n ∈ {100, 300, 600, 1000}` (with default `BASE`, `ALPHA`, `TVALS`, `SEED`).
   - Writes `results/complexity_parta_compare.csv`.

2. **Part C complexity vs $L$**  
   `python -m scripts.complexity_partc_L`

   - Runs the sequential Part C CLI for `L ∈ {10000, 30000, 50000}`.
   - Writes `results/complexity_partc_L.csv`.

Both CSVs are used later when building performance figures.

### 4.3. Baseline vs optimised benchmarks

```bash
make benchmark
```

Runs `python -m scripts.benchmark_runtime`, which:

- Times Part A baseline (Unit‑2 CLI).
- Times Part A optimised (Unit‑3 CLI).
- Times Part B (single implementation).
- Times Part C sequential vs parallel (4 workers by default).
- Constructs “All” runtimes by summing the components.

Results are written to:

- `results/benchmark_runtime.csv`

with columns:

- `component ∈ {PartA, PartB, PartC, All}`
- `variant ∈ {baseline, optimized}`
- `runtime_sec`

### 4.4. Parallel execution (Part C)

```bash
make parallel
# equivalent to: make parallel-partc
```

By default, this calls the parallelized Part C CLI via:

```bash
make parallel WORKERS=4    # example: 4 worker processes
```

If `WORKERS` is not set, the default from the `Makefile` is used.

### 4.5. Numerical stability check

```bash
make stability-check
```

Runs:

```bash
python -m scripts.stability_check
```

This script:

- Runs small configurations of Part B and Part C.
- Scans the **numeric columns** of the resulting CSVs for `NaN` or `inf`.
- Prints human-readable messages if any issues are found and exits with non-zero status.

With the current settings, no numerical issues were detected; this is reported in `docs/BASELINE.md` and corroborated by the stability check.

### 4.6. Performance figures

```bash
make perf-figures
```

This target:

1. Runs `make complexity`  
2. Runs `make benchmark`  
3. Calls `python -m scripts.plot_performance`  

and produces:

- `results/figures/perf_parta_runtime_vs_n.{png,pdf}`  
  – Part A runtime vs $n$, baseline vs optimised.
- `results/figures/perf_components_baseline_vs_optimized.{png,pdf}`  
  – Bar chart of runtimes for Part A, Part C, and All (baseline vs optimised).

These are the main figures used in `docs/OPTIMIZATION.md` for the Unit‑3 report.

---

## 5. Tests and regression checks

### 5.1. Test suite

Run all tests with:

```bash
make test
```

This calls `pytest` on the `tests/` directory, including:

- `test_smoke_parta.py` – small sanity run for Part A CLI.
- `test_smoke_partb.py`, `test_smoke_partc.py` – smoke tests for Parts B and C.
- `test_polya_module.py`, `test_dgp.py`, `test_exchangeability.py`, `test_repro.py` – unit tests for underlying stochastic components and reproducibility.

### 5.2. Unit‑3 regression test (Part C)

To verify that the parallelised Part C implementation is **numerically equivalent** to the sequential version (up to floating-point tolerance), run:

```bash
make regression
```

This calls the dedicated regression test:

- `tests/test_regression_unit3.py`  

which:

- Runs a tiny configuration of `src_cli.partc_log_prop26` sequentially and with `--workers=2`.
- Reads the two CSVs, sorts them to remove row-order differences, and checks:
  - integer / categorical columns match exactly,
  - floating-point columns match with tight tolerance (`atol = 1e-12`).

If the parallel code ever produced different results, this test would fail.

We **do not** add a separate numeric regression test for Part A because:

- The main outputs of Part A are figures, not a stable CSV.
- The optimisation only reuses trajectories across thresholds and does not change the model or plotting logic.
- The underlying Pólya urn code is already covered by unit tests, and Part A’s CLI is exercised by `test_smoke_parta.py`.

For a detailed discussion, see Section 6 of `docs/OPTIMIZATION.md`.

---

## 6. Reproducibility notes

- All main entry points take an explicit `--seed` argument, and the `Makefile` exposes a global `SEED` knob (default `2025`).
- The parallel Part C code uses a **deterministic seeding scheme** for each `(n, rep)` pair, so results are reproducible even when changing the number of workers. Only the row order in the CSV may differ, which is handled by the regression test.
- Performance experiments (`make complexity`, `make benchmark`, `make perf-figures`) can all be regenerated from scratch on another machine as long as:
  - dependencies are installed,
  - the same Python version / library versions are reasonably close.

If you want to replicate the exact numbers reported in `docs/OPTIMIZATION.md`, run:

```bash
make clean
make profile
make complexity
make benchmark
make perf-figures
make stability-check
make test
make regression
```

This will regenerate all raw data, figures, and checks used in the Unit‑3 report.

---

## 7. Known warnings

Some Part C figure commands may print Matplotlib warnings like:

> `'created' timestamp seems very low; regarding as unix timestamp`  
> `'modified' timestamp seems very low; regarding as unix timestamp`

These warnings are about PDF metadata timestamps only.  
They do **not** affect any numerical results or plots and are documented in `docs/BASELINE.md`.

---

If you are reviewing this project, the quickest tour is:

1. Read `docs/BASELINE.md` for the baseline performance story.
2. Read `docs/OPTIMIZATION.md` for the Unit‑3 optimisations and results.
3. Run `make perf-figures` to regenerate the main performance plots.
4. Run `make regression` to confirm that the parallel Part C is numerically correct.
