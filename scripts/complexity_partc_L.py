"""
scripts/complexity_partc_L.py

Empirical complexity analysis for Part C (Prop 2.6) as a function of L,
the number of continuation draws.

For each L in L_VALUES, this script runs the sequential Part C CLI
(src_cli.partc_log_prop26 without --workers) and records the wall-clock
runtime in seconds.

Output: results/complexity_partc_L.csv with columns:
    L, runtime_sec
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable

# Grid of L values to study. These are chosen to keep runtimes reasonable
# for classroom machines while still showing the scaling with L.
L_VALUES = [10000, 30000, 50000]

BASE = "uniform"
ALPHA = "5.0"
N_LIST = ["100", "500", "1000"]
M = "400"
LEVEL = "0.95"
SEED = "2025"
TVALS = ["0.25", "0.5", "0.75"]


def _time_for_L(L: int) -> float:
    cmd = [
        PY,
        "-m",
        "src_cli.partc_log_prop26",
        "--alpha",
        ALPHA,
        "--base",
        BASE,
        "--t",
        *TVALS,
        "--n",
        *N_LIST,
        "--M",
        M,
        "--L",
        str(L),
        "--level",
        LEVEL,
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
    out_csv = outdir / "complexity_partc_L.csv"

    rows: list[dict[str, float]] = []
    for L in L_VALUES:
        print(f"[complexity_partc] L={L} ...")
        t = _time_for_L(L)
        print(f"  runtime: {t:.3f} s")
        rows.append({"L": L, "runtime_sec": t})

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
