# Optimization Report (Unit 3)

This document describes the performance optimizations we apply to the `stats607-project3` codebase for Unit 3.  
Baseline performance and profiling details are recorded in `BASELINE.md`.

---

## 1. Optimization 1 – Part A (algorithmic / array programming)

### 1.1 Problem (from baseline profiling)

From the baseline profiling of `src_cli.parta_panels` (see `BASELINE.md`):

- **Matplotlib text layout** (`matplotlib.text._get_layout`, text metrics, etc.) dominates the wall-clock time for a single `parta_panels` call at `n = 1000`.  
- Our own Pólya code is also heavy:
  - `PolyaSequenceModel.Pn` is called tens of millions of times.
  - `continue_urn_once` is called thousands of times.
- The implementation used many tiny Python-level operations:
  - Repeated `list.append`, repeated `len`, and a wrapper around the RNG.
- Complexity experiments in the baseline showed that increasing `n` from 100 to 1000 has **very weak effect** on runtime for Part A; the cost is dominated by fixed overhead (imports, plotting, etc.), so the Pólya simulation work is “hidden” behind those costs.

These observations suggested that Part A is a good candidate for:

- **Algorithmic improvements**: remove redundant simulation work.
- **Array programming**: preallocate arrays and reduce Python-level overhead.

### 1.2 Approach

The Part A CLI `src_cli.parta_panels` draws Pólya continuations and, for each cell of the panel, estimates posterior probabilities of events of the form $(-\infty, t]$ for a small grid of thresholds `ts`.

In the **baseline** implementation, for each fixed sample size `n`, the panel was constructed roughly as:

1. Generate a prefix `x_obs` of length `n` using the chosen DGP (e.g. uniform base and Pólya urn with concentration `alpha`).
2. For each threshold `t` in `ts`, for each `alpha` in the list of alphas, and for each replication `r = 1,…,N`:
   - Simulate a **full Pólya continuation path** of length `M` using `continue_urn_once`.
   - Compute `np.mean(traj <= t)` for *that* particular `t` only.
   - Store this posterior estimate into the appropriate panel cell.

This means that, for each `(alpha, r)` pair, we simulated a Pólya trajectory **once for each threshold** `t`. If there are 3 thresholds, each replication’s trajectory is recomputed 3 times, even though a single trajectory suffices to compute the posterior at all thresholds.

In the **optimized** implementation, we reuse each trajectory across *all* thresholds:

- We preallocate a NumPy array `post_all` of shape `(len(ts), len(alphas), N)`, where:
  - axis 0 indexes thresholds `t`,
  - axis 1 indexes alphas,
  - axis 2 indexes replications `r`.
- For each alpha `a`:
  - Set `model.alpha = a`.
  - For each replication `r`:
    - Simulate **one** continuation trajectory `traj` of length `M` starting from the prefix `x_obs`.
    - Convert once to a NumPy array `traj_arr = np.asarray(traj)`.
    - For each threshold index `i` and value `t`:
      - Compute `post_all[i, j, r] = np.mean(traj_arr <= t)`.

Later, in the plotting loop, we simply slice `post_all[i, j, :]` to get the posterior draws for each panel cell – no additional Pólya simulations or repeated `np.asarray` calls are needed.

We also made a minor array-level improvement in the computation of the prefix statistic $k_n$ at each threshold:

- We convert the observed prefix to a NumPy array once, `x_obs_arr = np.asarray(x_obs)`, and compute `np.sum(x_obs_arr <= t)` for each `t`, instead of applying Python loops.

Crucially, we did **not** change:

- The graphical layout, binning, or overlay of Beta densities.
- The definition of the posterior panels.
- The random seed behaviour: we still seed once at the beginning of the CLI.

Thus, the optimized version produces panels that are visually indistinguishable from the baseline, but with considerably less redundant Pólya simulation.

### 1.3 Code changes

The core idea can be summarized as:

**Before (conceptual sketch):**

```python
# For each threshold t, each alpha, and each replication:
for i, t in enumerate(ts):
    for j, a in enumerate(alphas):
        model.alpha = a
        posts = []
        for r in range(N):
            traj = continue_urn_once(x_obs, model, M)
            posts.append(np.mean(np.asarray(traj) <= t))
        # plot histogram of posts for this (t, alpha) cell
```

**After (conceptual sketch):**

```python
post_all = np.empty((len(ts), len(alphas), N), dtype=float)

for j, a in enumerate(alphas):
    model.alpha = a
    for r in range(N):
        traj = continue_urn_once(x_obs, model, M)
        traj_arr = np.asarray(traj)
        for i, t in enumerate(ts):
            post_all[i, j, r] = np.mean(traj_arr <= t)

# later, for plotting
for i, t in enumerate(ts):
    for j, a in enumerate(alphas):
        posts = post_all[i, j, :]
        # plot histogram of posts for this (t, alpha) cell
```

The actual implementation also includes some book-keeping for axes, titles, and file naming; those aspects are unchanged relative to the baseline.

### 1.4 Performance impact – Part A only

We re-ran the **complexity experiment** for Part A using the same command and values of `n` as in the baseline, after swapping in the optimized `parta_panels.py` and the Unit-3 comparison script:

```bash
make complexity        # now calls both Part A & Part C complexity scripts
# Part A numbers come from:
python -m scripts.complexity_parta_compare
```

The script reports:

- `n = 100`: baseline ≈ 62.20 s, optimized ≈ 22.04 s  
- `n = 300`: baseline ≈ 58.71 s, optimized ≈ 21.26 s  
- `n = 600`: baseline ≈ 54.72 s, optimized ≈ 19.82 s  
- `n = 1000`: baseline ≈ 51.22 s, optimized ≈ 17.14 s  

Summarising:

| n     | Baseline runtime (s) | Optimized runtime (s) | Speedup (baseline / optimized) |
|-------|----------------------|------------------------|---------------------------------|
| 100   | 62.20                | 22.04                 | ≈ 2.8×                          |
| 300   | 58.71                | 21.26                 | ≈ 2.8×                          |
| 600   | 54.72                | 19.82                 | ≈ 2.8×                          |
| 1000  | 51.22                | 17.14                 | ≈ 3.0×                          |

Key observations:

- The optimization yields roughly a **2.8–3.0× speedup** across all values of `n` tested.
- The runtime remains largely **flat in `n`**, consistent with the baseline interpretation that plotting and other fixed costs dominate; however, the overall level of the curve is substantially lower.
- Because we changed only how we *reuse* Pólya trajectories across thresholds (and not the model itself), the scientific content of the panels remains the same.

We also re-ran the aggregate Part A make target with the optimized code:

```bash
/usr/bin/time -p make partA
```

The baseline measurement (see `BASELINE.md`) gave:

- `make partA`: real ≈ **166.37 s**

With the optimized implementation, the same command now takes:

- `make partA`: real ≈ **58.77 s**

This corresponds to an overall speedup of about **2.8×** for the full Part A pipeline, consistent with the per-`n` complexity results above.

### 1.5 Trade-offs

The main trade-offs of this optimization are:

- **Slightly more complex control flow** in `src_cli.parta_panels`:
  - Instead of computing each panel cell independently, we now have an explicit 3D array `post_all` that must be indexed carefully.
- **Memory usage**:
  - We allocate a `len(ts) × len(alphas) × N` array of floats. For the default settings (`len(ts)=3`, `len(alphas)=3`, `N=2000`), this is `3×3×2000 = 18,000` doubles, which is negligible relative to typical memory budgets.
- **Reproducibility considerations**:
  - The order of random draws is slightly different from the baseline implementation (because we changed the loop nesting), but we use the same RNG seeding strategy and the same Pólya mechanistic model. The resulting panels are visually indistinguishable and agree to Monte Carlo accuracy.

Overall, this optimization provides a large speedup with minimal conceptual complexity and no change in the intended scientific interpretation of the Part A posterior panels.

---

## 2. Optimization 2 – Part C (parallelisation)

### 2.1 Problem (from baseline profiling)

Baseline profiling of the Part C CLI `src_cli.partc_log_prop26` (see `BASELINE.md`) showed:

- The simulation performs **Pólya urn continuation** to approximate the predictive distribution $\tilde F(t)$ in Proposition 2.6.
- For a typical configuration (`M = 200`, `L = 50,000`, `n = 1000`, `alpha = 5`, uniform base), we observed:
  - ≈ 31.5M function calls in ≈ 13.5 seconds.
  - `draw_polya_next` was called ≈ 10.2M times (roughly `M × L`), accounting for **≈ 9.45 seconds** (~70% of runtime).
- Overheads from Python list appends and `len` calls were small relative to `draw_polya_next`.
- There is essentially **no Matplotlib cost** in this profiling run; the time is spent almost entirely in simulation.

Each replication and each `n` value is **independent**, and the dominant cost is a large number of identical “continuation” draws. This makes Part C a natural candidate for **parallelisation across replications**.

### 2.2 Approach

We refactored the inner logic of `src_cli.partc_log_prop26` into a worker function `_simulate_one_rep` that handles a single `(n, rep)` pair:

- For a given sample size `n` and replication index `rep`, the worker:
  1. Initializes a local RNG with a deterministic seed derived from `(n, rep, global_seed)`.
  2. Generates the prefix $x_1,\dots,x_n$ from the Pólya urn, keeping track of:
     - $K_m(t)$ = number of samples ≤ $t$ up to time $m$,
     - $P_m(t)$ = predictive probabilities,
     - the accumulated variance term $V_{n,t}$ via
       $$
       V_{n,t} = \frac{1}{n} \sum_{m=1}^n m^2 (P_m(t) - P_{m-1}(t))^2.
       $$
  3. Computes $P_n(t)$ for each threshold.
  4. **Continues the same urn** by $L$ steps, accumulating
       $$
       \hat F(t) = \frac{1}{L}\sum_{\ell=1}^L \mathbf{1}\{x_{n+\ell} \le t\}.
       $$
  5. Forms the Wald interval $P_n(t) \pm z\sqrt{V_{n,t}/n}$, and records:
     - coverage indicator (`covered`),
     - interval width,
     - other supporting quantities.

The CLI builds a list of tasks:

```python
tasks = [
    (n, rep, tvals, alpha, args.base, args.L, args.level, args.seed, z)
    for n in nvals
    for rep in range(args.M)
]
```

and then chooses between **sequential** and **parallel** execution:

- If `--workers <= 1`: fall back to a simple Python `for` loop over `tasks`.
- If `--workers > 1`:
  - Determine the number of workers `n_workers` (treat `--workers 0` as “use all cores”).
  - Create a `multiprocessing.Pool(processes=n_workers)` and run:

    ```python
    with mp.Pool(processes=n_workers) as pool:
        for result_rows in pool.imap_unordered(_simulate_one_rep, tasks):
            rows.extend(result_rows)
    ```

The results from all tasks are collected into a single DataFrame and written to the same CSV filename pattern as in the baseline implementation.

This design preserves the original interface while adding a simple `--workers` knob to control the degree of parallelism.

### 2.3 Code changes

Conceptually, the change can be summarised as:

**Before (single-process loop):**

```python
rows = []
for n in nvals:
    for rep in range(M):
        # simulate prefix x1..xn
        # compute K_m(t), P_m(t), V_{n,t}
        # continue urn by L steps to get Fhat(t)
        # form CI and append rows for each t
        rows.extend(rows_for_this_rep_and_n)
df = pd.DataFrame(rows)
df.to_csv(...)
```

**After (task list + optional multiprocessing):**

```python
tasks = [
    (n, rep, tvals, alpha, base, L, level, seed, z)
    for n in nvals
    for rep in range(M)
]

rows = []

if args.workers <= 1:
    # sequential path
    for task in tasks:
        rows.extend(_simulate_one_rep(task))
else:
    n_workers = args.workers
    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    print(f"[parallel] Using {n_workers} worker processes for Part C.")
    with mp.Pool(processes=n_workers) as pool:
        for result_rows in pool.imap_unordered(_simulate_one_rep, tasks):
            rows.extend(result_rows)

df = pd.DataFrame(rows)
df.to_csv(...)
```

The `_simulate_one_rep` worker uses the same Pólya update formula as the baseline code and the same seeding scheme, ensuring that each `(n, rep)` path is reproducible and conceptually identical to the sequential implementation.

### 2.4 Performance impact and speedup

We benchmarked Part C using the Makefile targets:

- Sequential (1 worker):

  ```bash
  /usr/bin/time -p make partC
  ```

- Parallel (2 and 4 workers):

  ```bash
  /usr/bin/time -p make parallel-partc WORKERS=2
  /usr/bin/time -p make parallel-partc WORKERS=4
  ```

with the default configuration:

- `alpha = 5.0`
- `base = uniform`
- `t = 0.25 0.5 0.75`
- `n = 100 500 1000`
- `M = 400`
- `L = 50,000`
- `seed = 2025`

The observed **wall-clock runtimes** (`real`) were:

- `make partC` (workers=1): real ≈ **58.13 s**
- `make parallel-partc WORKERS=2`: real ≈ **30.85 s**
- `make parallel-partc WORKERS=4`: real ≈ **19.24 s**

This yields the following table:

| Workers | Command                               | Runtime (s) | Speedup vs 1 worker |
|---------|---------------------------------------:|------------:|--------------------:|
| 1       | `make partC`                           |      58.13  | 1.00×               |
| 2       | `make parallel-partc WORKERS=2`        |      30.85  | ≈ 1.88×             |
| 4       | `make parallel-partc WORKERS=4`        |      19.24  | ≈ 3.02×             |

Comments:

- The speedup is close to linear when going from 1 to 2 workers, and we still obtain a **~3× speedup** with 4 workers.
- The `user` CPU time increases with the number of workers (e.g., ≈ 71.45 seconds for 4 workers), which is expected: more total CPU time is used, but elapsed wall-clock time decreases because the work is parallelised.
- The parallel runs produce the **same CSV file** (same filename and columns) as the sequential implementation; only the route by which we arrive at the result differs.

### 2.5 Trade-offs

The main trade-offs introduced by this optimization are:

- **Increased code complexity**:
  - We now manage a task list and a multiprocessing pool instead of a simple nested loop.
  - `_simulate_one_rep` must be kept in sync with any future changes to the Part C logic.

- **Reproducibility and seeding**:
  - Each `(n, rep)` uses a deterministic seed of the form `seed + 7919 * rep + 104729 * n`.  
    This ensures that results are reproducible and independent of the number of workers or task scheduling order.
  - The order in which rows appear in the CSV may differ (due to `imap_unordered`), but the set of rows is the same.

- **Resource utilisation**:
  - Using multiple workers increases CPU usage during the Part C run, which is desirable on a dedicated multicore machine but may contend with other processes on a shared laptop.
  - There is minor overhead from process startup and inter-process communication, which prevents perfect linear speedup.

Overall, this optimization substantially reduces wall-clock time for Part C — from about 58 seconds to about 19 seconds with 4 workers — without changing the scientific interpretation of the pooled-$Z$ results.

---

## 3. Complexity and benchmark scripts

To make the performance study **reproducible**, we added several small scripts under `scripts/` and wired them into the `Makefile`:

- `scripts.complexity_parta_compare` – Part A baseline vs optimised:

  ```bash
  make complexity     # runs Part A + Part C complexity
  ```

  - For Part A, this script times the **baseline** CLI
    `src_cli.parta_panels_baseline_backup` and the **optimised** CLI
    `src_cli.parta_panels` for `n ∈ {100, 300, 600, 1000}`.
  - It writes `results/complexity_parta_compare.csv` with columns
    `n, runtime_baseline, runtime_optimized`.

- `scripts.complexity_partc_L` – Part C runtime vs continuation length $L$:

  ```bash
  make complexity     # also calls this script
  ```

  - Runs `src_cli.partc_log_prop26` (sequential, 1 worker) for
    `L ∈ {10,000, 30,000, 50,000}` with the default configuration.
  - Writes `results/complexity_partc_L.csv` with columns `L, runtime_sec`.

- `scripts.benchmark_runtime` – component-wise baseline vs optimised timings:

  ```bash
  make benchmark
  ```

  - Runs:
    - Part A baseline (Unit-2 panels),
    - Part A optimised,
    - Part B (single implementation),
    - Part C sequential (workers=1),
    - Part C parallel (workers=4),
    and constructs “All” runtimes as sums.
  - Writes `results/benchmark_runtime.csv` with columns
    `component, variant, runtime_sec`, where `component ∈ {PartA, PartB, PartC, All}`
    and `variant ∈ {baseline, optimized}` (only baseline exists for PartB).

- `scripts.plot_performance` – builds performance figures from the CSVs:

  ```bash
  make perf-figures   # runs complexity + benchmark + this script
  ```

  - Reads both CSVs above and produces:
    - `results/figures/perf_parta_runtime_vs_n.{png,pdf}`  
      (complexity plot for Part A, baseline vs optimised).
    - `results/figures/perf_components_baseline_vs_optimized.{png,pdf}`  
      (bar chart of runtimes for Part A, Part C, and All).

- `scripts.stability_check` – light numerical-stability sweep:

  ```bash
  make stability-check
  ```

  - Runs a small grid of Part B and Part C configurations and scans the
    produced CSVs for `NaN`/`inf` values.
  - Exits with status 0 if everything is numerically clean and 1 otherwise.

Together with the updated `Makefile` targets (`profile`, `complexity`, `benchmark`, `parallel`, `stability-check`, `perf-figures`, `regression`), these scripts allow anyone to reproduce our performance study with a few simple commands.

---

## 4. Overall timing comparison (baseline vs optimised)

The file `results/benchmark_runtime.csv`, generated by `make benchmark`, summarises the end-to-end runtimes for each component and for the whole pipeline.

Using the default configuration (`BASE=uniform`, `ALPHA=5.0`, `N=1000`, `TVALS=0.25 0.5 0.75`, `SEED=2025`), we obtained:

| Component | Variant    | Runtime (s) |
|-----------|------------|------------:|
| PartA     | baseline   | 164.6       |
| PartA     | optimized  | 59.2        |
| PartB     | baseline   | 1.5         |
| PartC     | baseline   | 57.6        |
| PartC     | optimized  | 21.9        |
| All       | baseline   | 223.8       |
| All       | optimized  | 82.6        |

From these numbers:

- **Part A** achieves a ≈ **2.8×** speedup (164.6 → 59.2 s).
- **Part C** (using 4 workers) achieves a ≈ **2.6×** speedup (57.6 → 21.9 s).
- **The full pipeline (“All”)** is ≈ **2.7×** faster (223.8 → 82.6 s).

Part B has only a single implementation, so it appears only as “baseline” in the table.

These comparisons are visualised in the bar chart

- `results/figures/perf_components_baseline_vs_optimized.{png,pdf}`,

which annotates each bar with its runtime in seconds.

---

## 5. Performance visualisations

We provide two main figures for the Unit-3 report:

1. **Part A runtime vs sample size**  
   (`results/figures/perf_parta_runtime_vs_n.{png,pdf}`)

   - Shows runtime against $n ∈ \{100, 300, 600, 1000\}$ for both the baseline and optimised Part A CLIs.
   - Both curves are roughly flat in $n$, confirming that fixed overhead dominates.
   - The optimised curve lies well below the baseline curve, illustrating the ≈2.8–3× speedup.

2. **Component runtimes: baseline vs optimised**  
   (`results/figures/perf_components_baseline_vs_optimized.{png,pdf}`)

   - Displays a grouped bar chart for:
     - Part A (baseline vs optimised),
     - Part C (baseline vs optimised),
     - All (baseline vs optimised),
     plus the single Part B bar.
   - The chart highlights that the overall speedup is driven by:
     - algorithmic/array improvements in Part A, and
     - parallelisation in Part C.

These figures are built automatically by `scripts.plot_performance` and can be regenerated at any time via:

```bash
make perf-figures
```

---

## 6. Regression testing and correctness

### 6.1 Existing and new tests

We rely on the existing test suite to provide a baseline level of correctness:

- `tests/test_smoke_parta.py` exercises the Part A CLI on small configurations (including the `n=0` prior panel).
- `tests/test_smoke_partb.py` and `tests/test_smoke_partc.py` confirm that Part B and Part C CLIs run successfully and produce output files.
- `tests/test_polya_module.py`, `tests/test_dgp.py`, `tests/test_exchangeability.py`, and `tests/test_repro.py` validate core building blocks such as the Pólya urn implementation, data-generating processes, and reproducibility under fixed seeds.

For Unit 3 we added a **dedicated regression test** for the new parallel Part C implementation:

- `tests/test_regression_unit3.py`  

  - Runs a tiny configuration of `src_cli.partc_log_prop26` **sequentially** and then with `--workers=2`.
  - Reads both CSVs, sorts them to remove any row-order differences, and checks:
    - integer / categorical columns (e.g. `rep`, `n`, `covered`) match exactly, and
    - floating-point columns (e.g. `Pn`, `Vnt`, `Fhat`, `lo`, `hi`, `width`) match to tight tolerance (`atol = 1e-12`).

This regression test can be run on its own via:

```bash
make regression
```

and passes in under two seconds on our machine.

### 6.2 Why we do not add a separate numeric regression test for Part A

We did **not** add a second, “baseline vs optimised” numeric regression test for Part A, for two reasons:

1. **The Part A CLI primarily produces *figures*, not stable numeric CSVs.**  
   The optimisation (trajectory reuse across thresholds) changes only how often we re-simulate Pólya continuations internally; it does **not** change:
   - the Pólya urn model,
   - the definition of the posterior quantities, or
   - the panel layout / plotting logic.

   The most natural outputs to compare would be the PNG/PDF panel images themselves, but tiny numerical differences in Monte Carlo draws (or even Matplotlib version differences) would make “golden image” tests brittle.

2. **The underlying simulation code is already covered by unit tests.**  
   - The Pólya urn dynamics (`PolyaSequenceModel`, `continue_urn_once`, etc.) are tested in `tests/test_polya_module.py`, `tests/test_exchangeability.py`, and `tests/test_repro.py`.
   - `tests/test_smoke_parta.py` ensures that the optimised Part A CLI still runs end-to-end and produces the expected figure files for a small configuration.

Given that:
- the Part A optimisation only reuses trajectories within the CLI,
- the scientific meaning of the panels is unchanged, and
- the core stochastic model is already tested directly,

we judged that an additional golden-file regression for Part A would add complexity without much extra protection. Instead, we concentrated the numeric regression effort on Part C, where the new parallel execution path *could* in principle alter results if implemented incorrectly.

---

## 7. Lessons learned

- The **largest performance gains** came from relatively simple structural changes:
  - Reusing Pólya trajectories across thresholds in Part A (Optimisation 1).
  - Parallelising independent replications in Part C (Optimisation 2).
- Profiling revealed that seemingly “obvious” costs (e.g., increasing `n`) were not the real bottlenecks:
  - In Part A, Matplotlib text layout and fixed overhead dominated; the per-$n$ simulation cost was almost flat over the range we studied.
  - In Part C, the cost was almost entirely in Pólya continuation (`draw_polya_next`) rather than in I/O or plotting.
- If we were designing the simulation from scratch with performance in mind, we would:
  - Separate **simulation** from **plotting** more aggressively (to isolate pure complexity vs $n$).
  - Design for parallelism from the start (e.g., structuring code so that each replication is a naturally independent task).
  - Use profiling early in the project to guide optimization, rather than guessing where the bottlenecks might be.
