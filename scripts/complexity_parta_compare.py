"""
scripts/complexity_parta_compare.py

Compare runtime as a function of n for the baseline vs optimized
Part A panel implementations.

Baseline:   src_cli.parta_panels_baseline_backup
Optimized:  src_cli.parta_panels

For each n in N_VALUES we time *both* implementations and write
results/complexity_parta_compare.csv with columns:
  n, runtime_baseline, runtime_optimized
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable

N_VALUES = [100, 300, 600, 1000]
BASE = "uniform"
TVALS = ["0.25", "0.5", "0.75"]
ALPHAS = ["1", "5", "20"]
M = "4000"
N_REPS = "2000"
SEED = "2025"


def _time_cmd(module: str, n: int) -> float:
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
        str(n),
        "--M",
        M,
        "--N",
        N_REPS,
        "--seed",
        SEED,
    ]
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    end = time.perf_counter()
    return end - start


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "complexity_parta_compare.csv"

    rows: list[dict[str, object]] = []

    for n in N_VALUES:
        print(f"[complexity] n={n} (baseline)...")
        t_base = _time_cmd("src_cli.parta_panels_baseline_backup", n)
        print(f"  baseline: {t_base:.3f} s")

        print(f"[complexity] n={n} (optimized)...")
        t_opt = _time_cmd("src_cli.parta_panels", n)
        print(f"  optimized: {t_opt:.3f} s")

        rows.append(
            {
                "n": n,
                "runtime_baseline": t_base,
                "runtime_optimized": t_opt,
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n", "runtime_baseline", "runtime_optimized"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
