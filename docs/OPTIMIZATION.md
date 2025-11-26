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

The Part A CLI `src_cli.parta_panels` draws Pólya continuations and, for each cell of the panel, estimates posterior probabilities of events of the form \((-\infty, t]\) for a small grid of thresholds `ts`.

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

We also made a minor array-level improvement in the computation of the prefix statistic \(k_n\) at each threshold:

- We convert the observed prefix to a NumPy array once, `x_obs_arr = np.asarray(x_obs)`, and compute `np.sum(x_obs_arr <= t)` for each `t`, instead of applying Python loops.

Crucially, we did **not** change:

- The graphical layout, binning, or overlay of Beta densities.
- The definition of the posterior panels.
- The random seed behavior: we still seed once at the beginning of the CLI.

Thus, the optimized version produces panels that are visually indistinguishable from the baseline, but with considerably less redundant Pólya simulation.

### 1.3 Code changes (before vs after)

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

### 1.4 Performance impact

We re-ran the **complexity experiment** for Part A using the same command and values of `n` as in the baseline, after swapping in the optimized `parta_panels.py`:

```bash
make complexity-parta
# which calls:
python -m scripts.complexity_parta
```

The script reports:

- `n = 100`: time ≈ 21.496 s  
- `n = 300`: time ≈ 20.428 s  
- `n = 600`: time ≈ 19.010 s  
- `n = 1000`: time ≈ 16.844 s  

Comparing to the **baseline** times (from `BASELINE.md`):

- `n = 100`: 62.365 s  
- `n = 300`: 57.513 s  
- `n = 600`: 53.418 s  
- `n = 1000`: 47.248 s  

we obtain the following summary:

| n     | Baseline runtime (s) | Optimized runtime (s) | Speedup (baseline / optimized) |
|-------|----------------------|------------------------|---------------------------------|
| 100   | 62.365               | 21.496                | ≈ 2.9×                          |
| 300   | 57.513               | 20.428                | ≈ 2.8×                          |
| 600   | 53.418               | 19.010                | ≈ 2.8×                          |
| 1000  | 47.248               | 16.844                | ≈ 2.8×                          |

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

## 2. Optimization 2 – Part C (parallelization)

> TODO: Fill in after implementing the Part C parallelization:
> - Problem summary from `BASELINE.md` (Pólya continuation complexity O(M × L), `draw_polya_next` dominating).
> - Description of the parallel design and implementation.
> - Before/after snippets of the continuation loop.
> - Timing table and speedup vs number of cores.
> - Discussion of trade-offs (e.g., seed management, CPU utilization).

---

## 3. (Optional) Optimization 3 – Numerical stability

> TODO (if used):  
> - Describe any clipping or log-transform changes added to improve numerical stability.
> - Explain why they were needed (e.g., avoid log of 0, extreme probabilities).
> - Show that they have negligible effect on the substantive conclusions.

---

## 4. Regression testing and correctness

Describe how we verified that the optimizations preserve the scientific conclusions.

### 4.1 Regression tests

> TODO:
> - Summarize any new tests (e.g., `tests/test_regression_unit3.py`).
> - Explain how the tests compare optimized results against baseline “golden” output or known properties.

### 4.2 Comparison of key summaries

> TODO:
> - Provide a short summary of comparisons (e.g., key distances, pooled-Z statistics).
> - Note if any differences are within Monte Carlo noise.
> - If there are systematic differences, explain and justify them.

---

## 5. Lessons learned

> TODO: Reflect on:
> - Which optimizations gave the biggest gains per unit of effort.
> - Any surprising bottlenecks (e.g., Matplotlib text layout vs actual simulation).
> - How you might design the simulation differently if starting from scratch with performance in mind.
