"""
scripts/stability_check.py

Light-weight numerical stability check for the Unit 3 simulation.

This script runs a small grid of scenarios for Parts B and C and inspects
the resulting CSVs for NaN or infinite values.

It writes a brief text report to stdout and returns exit code 0 if no
issues are found, or 1 if any potential numerical problems are detected.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PY = sys.executable


def _run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _check_csv(path: Path, label: str) -> list[str]:
    """
    Load a CSV, look for NaN/inf in numeric columns only, and return
    a list of warning messages (empty if everything is fine).

    - Non-numeric columns (strings, categories) are ignored.
    - If the file is missing, we return a message so the caller
      can decide whether to treat that as an error.
    """
    messages: list[str] = []

    if not path.exists():
        messages.append(f"[{label}] MISSING: {path}")
        return messages

    df = pd.read_csv(path)

    if df.empty:
        messages.append(f"[{label}] WARNING: {path} is empty")
        return messages

    # Only check numeric columns (floats / ints)
    num = df.select_dtypes(include=[np.number])

    if num.empty:
        # Nothing numeric to check; this is not an error.
        messages.append(f"[{label}] NOTE: {path} has no numeric columns; skipping numeric checks")
        return messages

    arr = num.to_numpy()
    mask = ~np.isfinite(arr)

    if mask.any():
        bad_rows, bad_cols = np.where(mask)
        col_names = num.columns.to_list()
        messages.append(
            f"[{label}] FOUND non-finite values in {path}: "
            f"{len(bad_rows)} entries across {len(col_names)} numeric columns"
        )
        # Show a few examples to help debugging
        for i in range(min(5, len(bad_rows))):
            r = bad_rows[i]
            c = bad_cols[i]
            messages.append(
                f"  row {r}, column '{col_names[c]}' = {arr[r, c]!r}"
            )

    return messages
    if not path.exists():
        return [f"[stability] WARNING: expected file {path} for {label} not found."]
    df = pd.read_csv(path)
    msgs: list[str] = []
    if df.isna().any().any():
        msgs.append(f"[stability] NaN detected in {label} ({path}).")
    arr = df.to_numpy()
    if not np.isfinite(arr).all():
        msgs.append(f"[stability] non-finite values (inf/-inf) in {label} ({path}).")
    if not msgs:
        msgs.append(f"[stability] OK: {label} ({path}).")
    return msgs


def main() -> None:
    outdir_raw = Path("results/raw")
    outdir_raw.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []

    # ---- Part B stability: one moderate configuration ----
    print("[stability] Checking Part B...")
    base = "uniform"
    alpha = "5.0"
    n = "500"
    seed = "2025"
    tvals = ["0.25", "0.5", "0.75"]

    stem = f"partB_n{n}_a{alpha}_seed{seed}_{base}"

    _run_cmd(
        [
            PY,
            "-m",
            "src_cli.partb_log_convergence",
            "--n",
            n,
            "--alpha",
            alpha,
            "--t",
            *tvals,
            "--seed",
            seed,
            "--base",
            base,
        ]
    )

    messages += _check_csv(outdir_raw / f"distances_{stem}.csv", "Part B distances")
    messages += _check_csv(outdir_raw / f"Pm_paths_{stem}.csv", "Part B predictive paths")

    # ---- Part C stability: one moderate configuration ----
    print("[stability] Checking Part C...")
    base_c = "uniform"
    alpha_c = "5.0"
    n_list = ["100", "500"]
    M = "100"
    L = "5000"
    level = "0.95"
    seed_c = "2025"
    tvals_c = ["0.25", "0.5", "0.75"]

    cmd_c = [
        PY,
        "-m",
        "src_cli.partc_log_prop26",
        "--alpha",
        alpha_c,
        "--base",
        base_c,
        "--t",
        *tvals_c,
        "--n",
        *n_list,
        "--M",
        M,
        "--L",
        L,
        "--level",
        level,
        "--seed",
        seed_c,
        "--workers",
        "2",
    ]
    _run_cmd(cmd_c)

    stem_c = f"prop26_M{M}_L{L}_a{alpha_c}_seed{seed_c}_{base_c}.csv"
    messages += _check_csv(outdir_raw / stem_c, "Part C prop26 summary")

    # ---- Report ----
    print("\n".join(messages))

    # Non-zero exit code if any serious issue detected
    bad = any("NaN" in m or "non-finite" in m for m in messages)
    if bad:
        raise SystemExit(1)
    else:
        print("[stability] No numerical issues detected in checked scenarios.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
