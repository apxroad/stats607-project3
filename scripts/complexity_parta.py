import subprocess
import sys
import time
import csv
from pathlib import Path

# Values of n to test
N_VALUES = [100, 300, 600, 1000]

BASE = "uniform"
TVALS = ["0.25", "0.5", "0.75"]
ALPHAS = ["1", "5", "20"]
M = "4000"
N_REPS = "2000"
SEED = "2025"


def run_for_n(n: int) -> float:
    """Run parta_panels once for a given n and return wall-clock seconds."""
    cmd = [
        sys.executable,
        "-m",
        "src_cli.parta_panels",
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


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "complexity_parta_baseline.csv"

    rows = []
    for n in N_VALUES:
        print(f"Running parta_panels for n={n} ...")
        t = run_for_n(n)
        print(f"n={n}, time={t:.3f} s")
        rows.append({"n": n, "runtime_sec": t})

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
