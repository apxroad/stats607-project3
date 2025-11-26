"""
Unit 3 regression test.

Goal: verify that the optimized *parallel* Part C implementation
produces the same numerical results as the original sequential version,
for a small, fast configuration.
"""

import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd


def _run_partc(M: int, L: int, seed: int, workers: int | None = None) -> pathlib.Path:
    """
    Run the Part C CLI with a small configuration and return the CSV path.

    If workers is None or <=1, we use the sequential path.
    Otherwise we pass --workers=workers to enable parallelism.
    """
    raw_dir = pathlib.Path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        "python -m src_cli.partc_log_prop26 "
        "--alpha 5 --t 0.5 "
        "--n 100 "
        f"--M {M} --L {L} "
        f"--seed {seed} --base uniform"
    )
    if workers is not None and workers > 1:
        cmd += f" --workers {workers}"

    subprocess.run(cmd, shell=True, check=True)

    stem = f"prop26_M{M}_L{L}_a5.0_seed{seed}_uniform.csv"
    csv_path = raw_dir / stem
    assert csv_path.exists(), f"expected CSV {csv_path}"
    return csv_path


def test_partc_parallel_matches_sequential(tmp_path):
    """
    Regression check for Part C (Prop 2.6 pooled-Z).

    We run a *sequential* run and a *parallel* run with the same
    small configuration and seed. After sorting the rows, all
    numeric columns should match to tight tolerance, and the
    integer columns should match exactly.
    """
    M, L, seed = 40, 2000, 123

    # Sequential baseline (no workers arg)
    csv_seq = _run_partc(M, L, seed, workers=None)
    baseline = tmp_path / "partc_seq_baseline.csv"
    shutil.copy(csv_seq, baseline)

    # Parallel run with 2 workers
    csv_par = _run_partc(M, L, seed, workers=2)

    df_seq = pd.read_csv(baseline)
    df_par = pd.read_csv(csv_par)

    # Sort to ignore any row-order differences from parallel scheduling
    sort_cols = ["rep", "n", "t"]
    df_seq = df_seq.sort_values(sort_cols).reset_index(drop=True)
    df_par = df_par.sort_values(sort_cols).reset_index(drop=True)

    assert df_seq.shape == df_par.shape

    # Integer / categorical columns: must match exactly
    for col in ["rep", "n", "alpha", "L", "covered"]:
        assert (df_seq[col].values == df_par[col].values).all(), f"mismatch in {col}"

    # Floating-point columns: match to tight tolerance
    float_cols = ["Pn", "Vnt", "Fhat", "lo", "hi", "width"]
    for col in float_cols:
        assert np.allclose(
            df_seq[col].values,
            df_par[col].values,
            rtol=0,
            atol=1e-12,
        ), f"mismatch in {col}"
