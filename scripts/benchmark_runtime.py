"""
scripts/benchmark_runtime.py

Benchmark baseline vs optimized runtimes for key components of the
Pólya / DP simulation study.

This script measures, for a representative configuration:

  * Part A: posterior panels
      - baseline: src_cli.parta_panels_baseline_backup
      - optimized: src_cli.parta_panels

  * Part B: distances + predictive paths
      - (single implementation; timed once)

  * Part C: Proposition 2.6 pooled-Z
      - baseline: sequential run (workers=1)
      - optimized: parallel run (workers=4)

  * Whole pipeline "All":
      - baseline: baseline Part A + Part B + sequential Part C
      - optimized: optimized Part A + Part B + parallel Part C

Timings are written to results/benchmark_runtime.csv with columns:
  component, variant, runtime_sec
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


PY = sys.executable

BASE = "uniform"
TVALS = ["0.25", "0.5", "0.75"]
ALPHAS = ["1", "5", "20"]
N_PARTA_LIST = ["100", "500", "1000"]
N_PARTB = "1000"
ALPHA_PARTB = "5.0"
SEED = "2025"

# Part C config (roughly matches project defaults)
ALPHA_PARTC = "5.0"
N_PARTC_LIST = ["100", "500", "1000"]
M_PARTC = "400"
L_PARTC = "50000"
LEVEL = "0.95"


def _time_cmd(cmd: list[str]) -> float:
    """Run a subprocess and return wall-clock runtime in seconds."""
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    end = time.perf_counter()
    return end - start


def run_parta(module: str) -> float:
    """Run Part A (panels) for all n in N_PARTA_LIST using the given module.

    Parameters
    ----------
    module : str
        Dotted module path (e.g. 'src_cli.parta_panels').
    """
    total = 0.0
    for n in N_PARTA_LIST:
        cmd = [
            PY,
            "-m",
            module,
            "--base",
            BASE,
            "--t",
            *TVALS,
            "--alpha",
            *ALPHAS,
            "--n",
            n,
            "--M",
            "4000",
            "--N",
            "2000",
            "--seed",
            SEED,
        ]
        total += _time_cmd(cmd)
    return total


def run_partb() -> float:
    """Run Part B log + figures once and return combined runtime."""
    stem = f"partB_n{N_PARTB}_a{ALPHA_PARTB}_seed{SEED}_{BASE}"
    total = 0.0

    # Log convergence + predictive paths
    cmd_log = [
        PY,
        "-m",
        "src_cli.partb_log_convergence",
        "--n",
        N_PARTB,
        "--alpha",
        ALPHA_PARTB,
        "--t",
        *TVALS,
        "--seed",
        SEED,
        "--base",
        BASE,
    ]
    total += _time_cmd(cmd_log)

    # Figures
    cmd_fig = [
        PY,
        "-m",
        "src_cli.partb_figures",
        "--stem",
        stem,
        "--title",
        f"n={N_PARTB}, alpha={ALPHA_PARTB}, base={BASE}",
    ]
    total += _time_cmd(cmd_fig)

    return total


def run_partc(workers: int | None) -> float:
    """Run Part C (Prop 2.6).

    If workers is None or 1, we omit the --workers flag to use the sequential
    code path. For workers>1 we pass --workers=<workers> to use parallel mode.
    """
    base_cmd = [
        PY,
        "-m",
        "src_cli.partc_log_prop26",
        "--alpha",
        ALPHA_PARTC,
        "--base",
        BASE,
        "--t",
        *[str(t) for t in [0.25, 0.5, 0.75]],
        "--n",
        *N_PARTC_LIST,
        "--M",
        M_PARTC,
        "--L",
        L_PARTC,
        "--level",
        LEVEL,
        "--seed",
        SEED,
    ]
    if workers is not None and workers > 1:
        base_cmd.extend(["--workers", str(workers)])

    t_log = _time_cmd(base_cmd)

    # Figures (same CSV regardless of workers)
    stem = f"prop26_M{M_PARTC}_L{L_PARTC}_a{ALPHA_PARTC}_seed{SEED}_{BASE}.csv"
    cmd_fig = [
        PY,
        "-m",
        "src_cli.partc_figures_prop26",
        "--csv",
        f"results/raw/{stem}",
        "--title",
        f"Proposition 2.6: alpha={ALPHA_PARTC}, base={BASE}",
    ]
    t_fig = _time_cmd(cmd_fig)

    return t_log + t_fig


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "benchmark_runtime.csv"

    rows: list[dict[str, object]] = []

    # Part A baseline vs optimized
    print("[benchmark] Part A (baseline: parta_panels_baseline_backup)...")
    t_parta_base = run_parta("src_cli.parta_panels_baseline_backup")
    print(f"  baseline Part A: {t_parta_base:.3f} s")
    rows.append({"component": "PartA", "variant": "baseline", "runtime_sec": t_parta_base})

    print("[benchmark] Part A (optimized: parta_panels)...")
    t_parta_opt = run_parta("src_cli.parta_panels")
    print(f"  optimized Part A: {t_parta_opt:.3f} s")
    rows.append({"component": "PartA", "variant": "optimized", "runtime_sec": t_parta_opt})

    # Part B (single implementation)
    print("[benchmark] Part B (single implementation)...")
    t_partb = run_partb()
    print(f"  Part B total: {t_partb:.3f} s")
    rows.append({"component": "PartB", "variant": "baseline", "runtime_sec": t_partb})

    # Part C baseline (sequential) vs optimized (parallel)
    print("[benchmark] Part C (baseline: sequential workers=1)...")
    t_partc_base = run_partc(workers=1)
    print(f"  baseline Part C: {t_partc_base:.3f} s")
    rows.append({"component": "PartC", "variant": "baseline", "runtime_sec": t_partc_base})

    print("[benchmark] Part C (optimized: parallel workers=4)...")
    t_partc_opt = run_partc(workers=4)
    print(f"  optimized Part C: {t_partc_opt:.3f} s")
    rows.append({"component": "PartC", "variant": "optimized", "runtime_sec": t_partc_opt})

    # Aggregate "All" baseline vs optimized
    t_all_base = t_parta_base + t_partb + t_partc_base
    t_all_opt = t_parta_opt + t_partb + t_partc_opt
    print(f"[benchmark] All (baseline): {t_all_base:.3f} s")
    print(f"[benchmark] All (optimized): {t_all_opt:.3f} s")

    rows.append({"component": "All", "variant": "baseline", "runtime_sec": t_all_base})
    rows.append({"component": "All", "variant": "optimized", "runtime_sec": t_all_opt})

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "variant", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
