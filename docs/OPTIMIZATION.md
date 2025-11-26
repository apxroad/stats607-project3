# Optimization Report (Unit 3)

This document describes the performance optimizations we apply to the `stats607-project3` codebase for Unit 3.  
Baseline performance and profiling details are recorded in `BASELINE.md`.

## 1. Optimization 1 – Part A (algorithmic / array programming)

### 1.1 Problem (from baseline profiling)

> TODO: Summarize the key issues for Part A based on `BASELINE.md`:
> - Matplotlib text layout dominating runtime.
> - Heavy use of `PolyaSequenceModel.Pn` and `continue_urn_once`.
> - Many tiny Python-level operations (list appends, `len`, RNG wrapper).

### 1.2 Approach

> TODO: Describe the design of the first optimization, for example:
> - Separate simulation logic from plotting in `src_cli.parta_panels`.
> - Preallocate NumPy arrays instead of repeatedly appending to Python lists.
> - Reduce repeated calls to `Pn` by caching or restructuring loops.
> - Minimize redundant work inside the Pólya update loop.

### 1.3 Code changes (before vs after)

> TODO: Insert short code snippets illustrating the main change.
> - Before: loop with list appends / many `Pn` calls.
> - After: vectorized / preallocated arrays, or refactored `PolyaSequenceModel`.

### 1.4 Performance impact

> TODO: Fill in a small table with timings, for example:
>
> | Scenario                          | Command            | Baseline (s) | Optimized (s) |
> |----------------------------------|--------------------|--------------|---------------|
> | Part A, `n = 1000`               | `make partA`       |      XX.X    |       YY.Y    |
> | Complexity run (all n values)    | `make complexity`  |      XX.X    |       YY.Y    |
>
> Add a brief narrative:
> - How much did runtime improve?
> - Did the complexity vs `n` curve change shape once plotting overhead was separated?

### 1.5 Trade-offs

> TODO: Describe any trade-offs:
> - Slightly more complex code structure.
> - Additional memory for preallocated arrays.
> - Clearer separation between “simulate” and “plot” phases.

---

## 2. Optimization 2 – Part C (parallelization)

### 2.1 Problem (from baseline profiling)

> TODO: Summarize profiling of Part C (pooled Z / Prop 2.6):
> - `draw_polya_next` dominates runtime (~70%).
> - Complexity ~ O(M × L) due to Pólya urn continuation.
> - Each replication / path is independent, so this is a good candidate for parallelization.

### 2.2 Approach

> TODO: Describe the parallel design, for example:
> - Parallelize over replications `M` or over different `n` values.
> - Use `multiprocessing` or `joblib` to run independent simulations in parallel.
> - Control the number of workers via a command-line flag or environment variable.
> - Keep the interface to `src_cli.partc_log_prop26` as close as possible to baseline.

### 2.3 Code changes (before vs after)

> TODO: Provide short “before vs after” snippets:
> - Before: single-process loop over replications.
> - After: helper function for one replication + parallel map across replications.

### 2.4 Performance impact and speedup

> TODO: Fill in a table or bullet list such as:

> | Cores | Command                 | Runtime (s) | Speedup vs 1 core |
> |-------|-------------------------|------------:|------------------:|
> | 1     | `make partC`            |        XX.X |             1.0×  |
> | 2     | `make parallel-partc`   |        YY.Y |          (XX/YY)× |
> | 4     | `make parallel-partc`   |        ZZ.Z |          (XX/ZZ)× |

> Briefly discuss:
> - How close the speedup is to ideal.
> - Diminishing returns as cores increase (overheads, Python GIL boundaries, etc.).

### 2.5 Trade-offs

> TODO: Note trade-offs:
> - Slightly more complex runtime environment (e.g. more CPU usage).
> - Need to ensure reproducibility and seed management across processes.
> - Potential platform-specific quirks (e.g. macOS vs Linux multiprocessing).

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
