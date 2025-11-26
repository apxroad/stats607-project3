# Baseline Performance (Unit 3)

## 1. Baseline setup

We optimize the simulation framework inherited from Project 2 on exchangeability / Pólya project.

**Codebase:** `stats607-project3`  
**Entry points (Make targets):**
- `make partA`
- `make partB`
- `make partC`
- `make all`  (runs the full pipeline)

**Default parameter values (from Makefile):**
- `BASE = uniform`
- `ALPHA = 5.0`
- `N = 1000`
- `TVALS = 0.25 0.5 0.75`
- `SEED = 2025`

We treat these as the *baseline configuration* unless otherwise stated.

## 2. Per-target baseline runtimes

Measured on my laptop using `/usr/bin/time -p`:

| Target | Command      | real time (s) |
|--------|--------------|---------------|
| partA  | `make partA` | 166.37        |
| partB  | `make partB` | 2.16          |
| partC  | `make partC` | 55.90         |
| all    | `make all`   | 309.17        |


## 3. Profiling Part A (posterior panels, n = 1000)

We profiled a single `parta_panels` run using:

```bash
python -m cProfile -o results/profile_parta_n1000_M4000_N2000.pstats \
  -m src_cli.parta_panels \
    --base uniform \
    --t 0.25 0.5 0.75 \
    --alpha 1 5 20 \
    --n 1000 \
    --M 4000 \
    --N 2000 \
    --seed 2025
```

From `python -m pstats results/profile_parta_n1000_M4000_N2000.pstats`, sorting by cumulative time, we obtain (top entries only):

- Matplotlib text layout and metrics:
  - `matplotlib.text._get_layout`: cumtime ≈ 137 s
  - `matplotlib.text._get_text_metrics_with_cache`: cumtime ≈ 136 s
  - `matplotlib.text._get_text_metrics_with_cache_impl`: cumtime ≈ 136 s
  - `matplotlib.backends.backend_agg.get_text_width_height_descent`: cumtime ≈ 68.6 s
  - `matplotlib.text.get_window_extent`: cumtime ≈ 68.6 s
- Core simulation in our code:
  - `src/polya.py:68(continue_urn_once)`: 18,000 calls, cumtime ≈ 65.8 s
  - `src/polya.py:42(Pn)`: 54,000,999 calls, tottime ≈ 52.2 s, cumtime ≈ 55.8 s
- Many tiny Python operations:
  - `{method 'append' of 'list' objects}`: ~54.2M calls, tottime ≈ 2.4 s
  - `src/polya.py:27(_rng)`: ~54.2M calls, tottime ≈ 2.0 s
  - `{built-in method builtins.len}`: ~54.5M calls, tottime ≈ 1.7 s

**Interpretation.** For a single `parta_panels` run at `n = 1000, M = 4000, N = 2000`:

- A substantial amount of cumulative time is spent in **Matplotlib text layout** (titles, tick labels, and bounding boxes). This means plotting cost is non-negligible for Part A.
- The core Pólya simulation functions, especially `Pn` and `continue_urn_once`, are also heavy and account for tens of millions of calls.
- The simulation currently relies on a very large number of tiny Python-level operations (list appends, `len`, RNG wrapper calls), suggesting that **array programming / vectorization** and **algorithmic improvements** in these functions are promising optimization directions.

## 4. Profiling Part C (Prop 2.6 pooled-Z continuation)

For Part C we profile a single continuation experiment using:

```bash
python -m cProfile \
  -o results/profile_partc_baseuniform_alpha5.0_n1000.pstats \
  -m src_cli.partc_log_prop26 \
    --alpha 5.0 \
    --base uniform \
    --t 0.25 0.5 0.75 \
    --n 1000 \
    --M 200 \
    --L 50000 \
    --level 0.95 \
    --seed 2025
```

Inspecting the profile with

```bash
python -m pstats results/profile_partc_baseuniform_alpha5.0_n1000.pstats
# inside the prompt:
sort cumtime
stats 20
```

gives the following summary:

- Overall:
  - ~31.5M function calls (31.5M primitive) in ≈ **13.5 seconds**.
- Dominant function in our code:
  - `src_cli/partc_log_prop26.py:42(draw_polya_next)`:  
    - ≈ **10.2M** calls (roughly `M × L = 200 × 50,000`),  
    - tottime ≈ 9.12 s, cumtime ≈ **9.45 s** (about 70% of total runtime).
- Smaller contributions:
  - `{method 'append' of 'list' objects}`: ≈ 10.2M calls, tottime ≈ 0.53 s.
  - `{built-in method builtins.len}`: ≈ 10.2M calls, tottime ≈ 0.34 s.
  - One-time imports (e.g. `pandas`, `importlib`) account for a few tenths of a second.

**Interpretation.** For the pooled-Z/Prop 2.6 continuation with `M = 200` and `L = 50,000`:

- Runtime is dominated by the **Pólya urn continuation step** `draw_polya_next`, called for each of the `M × L` draws.
- Overheads from list appends, length checks, and imports are comparatively small.
- The effective computational complexity is **O(M × L)**: each of the \(M\) datasets is grown by \(L\) additional Pólya draws.
- Unlike Part A, there is essentially **no Matplotlib cost** in this profiling run; the time is spent almost entirely in simulation.

This profiling motivates our Part C optimization strategy in Unit 3:
- Parallelize across independent replications (and/or across different \(n\) values), since each continuation path is independent.
- Optionally apply micro-optimizations inside `draw_polya_next` to reduce Python-level overhead per draw.

## 5. Computational complexity (Part A vs sample size n)

To study how runtime scales with the sample size \(n\) in the current implementation, we fixed

- `base = uniform`
- `alpha ∈ {1, 5, 20}`
- `M = 4000`
- `N = 2000`
- `seed = 2025`

and ran `src_cli.parta_panels` for a grid of \(n\) values using:

```bash
make complexity-parta
```

which calls:

```bash
python -m scripts.complexity_parta
```

The script times a single `parta_panels` call for each \(n\) and writes
`results/complexity_parta_baseline.csv` (columns: `n`, `runtime_sec`).

Measured runtimes from this script:

- \(n = 100\): 62.365 s
- \(n = 300\): 57.513 s
- \(n = 600\): 53.418 s
- \(n = 1000\): 47.248 s

Surprisingly, the measured runtime **decreases** as \(n\) increases over this range. To check whether this was an artifact of running all values of \(n\) in one script, we also timed individual runs directly from the shell using `/usr/bin/time -p`:

```bash
/usr/bin/time -p python -m src_cli.parta_panels \
  --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 \
  --n 100 --M 4000 --N 2000 --seed 2025

/usr/bin/time -p python -m src_cli.parta_panels \
  --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 \
  --n 1000 --M 4000 --N 2000 --seed 2025
```

and also with the order reversed (first `n = 1000`, then `n = 100`). The representative results were:

- `n = 100` run on its own: real ≈ 61.8 s  
- `n = 1000` run on its own: real ≈ 47.6–48.3 s  

and the same pattern (`n = 1000` faster than `n = 100`) appeared regardless of which we ran first.

This behavior does **not** mean that the algorithm becomes cheaper as \(n\) grows. Instead, combined with the profiling results, it shows that:

- For Part A with these settings, the runtime is dominated by **fixed overhead**:
  - Python interpreter startup and imports,
  - Matplotlib backend initialization,
  - figure creation and text layout (titles, tick labels, bounding boxes).
- The additional work required to increase \(n\) from 100 to 1000 (more Pólya updates and predictive evaluations) is comparatively small relative to these fixed costs and lies within the noise introduced by system load and OS-level caching.

Empirically, over \(n \in \{100, 300, 600, 1000\}\), the wall-clock runtime is therefore approximately **flat in \(n\)** for the baseline Part A implementation, because plotting and other non-\(n\)-dependent work dominate the total cost. A more informative complexity analysis with respect to \(n\) will be possible once we refactor the code in Unit 3 to separate a “simulation-only” path from plotting and other overhead.

---

## 6. Numerical stability

During the baseline experiments, we ran:

- `make partA`
- `make partB`
- `make partC`
- `make all`
- `make profile-parta`
- `make complexity-parta`
- additional `/usr/bin/time -p python -m src_cli.parta_panels ...` runs for selected \(n\)

We monitored console output and logs for numerical issues (e.g., overflow, underflow, invalid operations) affecting the simulation results (distances, predictive paths, and pooled-Z summaries).

Findings:

- We did **not** observe numerical warnings related to floating-point overflow/underflow or invalid arithmetic.
- We did not detect NaN or ∞ values in the recorded distances or predictive paths for the baseline configuration.
- In Part C, Matplotlib emitted a benign warning about PDF metadata timestamps:

  > `'created' timestamp seems very low; regarding as unix timestamp`  
  > `'modified' timestamp seems very low; regarding as unix timestamp`

  This warning concerns figure metadata only and does not affect any numerical quantities in the simulation study.

Overall, the baseline implementation appears **numerically stable** for the configurations tested in Phase A.
