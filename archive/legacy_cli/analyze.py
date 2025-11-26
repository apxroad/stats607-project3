from __future__ import annotations
import glob, os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def summarize_pit(pit_vals: np.ndarray) -> dict:
    pit = pit_vals[~np.isnan(pit_vals)]
    if pit.size == 0:
        return {"pit_n": 0, "pit_mean": np.nan, "pit_std": np.nan,
                "pit_ks_stat": np.nan, "pit_ks_p": np.nan}
    ks = stats.kstest(pit, "uniform")
    return {"pit_n": pit.size, "pit_mean": float(np.mean(pit)),
            "pit_std": float(np.std(pit, ddof=0)),
            "pit_ks_stat": float(ks.statistic), "pit_ks_p": float(ks.pvalue)}

def main():
    ap = argparse.ArgumentParser(description="Aggregate raw simulation outputs")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, "r"))
    raw_dir = cfg["io"]["raw_dir"]
    out_dir = "results"
    Path(out_dir).mkdir(exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        print("[analyze] no raw files found — run simulate first.")
        return

    rows = []
    pit_rows = []
    for f in files:
        df = pd.read_parquet(f)
        # final recorded distances (last non-NaN)
        d_inf_final  = df["d_infty"].dropna()
        d_rmse_final = df["d_rmse"].dropna()
        d_inf_final  = float(d_inf_final.iloc[-1])  if not d_inf_final.empty  else np.nan
        d_rmse_final = float(d_rmse_final.iloc[-1]) if not d_rmse_final.empty else np.nan
        method = df["method"].iloc[0]
        n      = int(df["n"].iloc[0])
        seed   = int(df["seed"].iloc[0])
        rows.append({"file": f, "method": method, "n": n,
                     "seed": seed, "d_infty_final": d_inf_final,
                     "d_rmse_final": d_rmse_final})
        pit_rows.append(pd.DataFrame({
            "method": method, "n": n, "pit": df["pit"].values
        }))

    summary = pd.DataFrame(rows)
    summary.sort_values(["method","n","seed"], inplace=True)
    summary.to_csv(os.path.join(out_dir, "summary_baseline.csv"), index=False)

    # PIT summaries per (method, n)
    pit_all = pd.concat(pit_rows, ignore_index=True)
    pit_stats = (pit_all
                .groupby(["method","n"])["pit"]
                .apply(lambda s: summarize_pit(s.to_numpy()))
                .apply(pd.Series)
                .reset_index())
    pit_stats.to_csv(os.path.join(out_dir, "pit_summary.csv"), index=False)

    print("[analyze] wrote:",
          os.path.join(out_dir, "summary_baseline.csv"),
          "and", os.path.join(out_dir, "pit_summary.csv"))

if __name__ == "__main__":
    main()
