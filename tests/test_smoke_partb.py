from pathlib import Path
import pandas as pd, subprocess, sys

def test_partb_smoke(tmp_path: Path):
    # Smoke test for Part B logging CLI with a tiny configuration.
    # Ensures the script runs and writes the expected CSVs.
    cmd = [sys.executable, "-m", "src_cli.partb_log_convergence",
           "--n", "50", "--alpha", "5", "--t", "0.5", "--seed", "123", "--base", "uniform"]
    subprocess.run(cmd, check=True)

    raw = Path("results/raw")
    # Expect two files: distances_*.csv and Pm_paths_*.csv
    dfs = list(raw.glob("distances_partB_n50_a5.0_seed123_uniform.csv")) +           list(raw.glob("Pm_paths_partB_n50_a5.0_seed123_uniform.csv"))
    assert dfs, "Expected small CSVs in results/raw"

    # Basic schema check for the distances file.
    d = pd.read_csv(raw / "distances_partB_n50_a5.0_seed123_uniform.csv")
    assert {"i","d_infty","d_rmse"} <= set(d.columns)
