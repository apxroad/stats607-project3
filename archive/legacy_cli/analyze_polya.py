from __future__ import annotations
import os, glob, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
import yaml
from src.dgps import NormalTruth, UniformTruth

def main():
    ap = argparse.ArgumentParser(description="Pólya DP theory checks")
    ap.add_argument("--config", required=True)
    ap.add_argument("--t", nargs="+", type=float, default=[0.25, 0.5, 0.75],
                    help="thresholds t to evaluate")
    ap.add_argument("--level", type=float, default=0.95, help="credible level")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    raw_dir = cfg["io"]["raw_dir"]

    # read method params to know alpha and base
    mp   = cfg.get("method_params", {}).get("polya_dp", {})
    alpha = float(mp.get("alpha", 5.0))
    base  = mp.get("base", "uniform")

    files = sorted(glob.glob(os.path.join(raw_dir, "polya_dp_*.parquet")))
    if not files:
        print("[analyze_polya] no polya_dp raw files found — run simulate first.")
        return

    # base CDF
    G0 = UniformTruth(0.0, 1.0) if base == "uniform" else NormalTruth()
    ts = np.array(args.t, dtype=float)
    g0 = G0.cdf_truth(ts)

    rows = []
    for f in files:
        df = pd.read_parquet(f, engine="fastparquet")
        xs = df["x_i"].to_numpy()
        n  = int(df["n"].iloc[0])
        # K_n(t): vectorized counts
        K  = (xs[:, None] <= ts[None, :]).sum(axis=0)
        # posterior mean at t (random across reps through K)
        post_mean = (alpha * g0 + K) / (alpha + n)
        # Beta(a,b) posterior for P((−inf,t]) given data
        a = alpha * g0 + K
        b = alpha * (1.0 - g0) + (n - K)
        lo = beta.ppf((1-args.level)/2, a, b)
        hi = beta.ppf(1 - (1-args.level)/2, a, b)
        covered = (g0 >= lo) & (g0 <= hi)
        for j, tj in enumerate(ts):
            rows.append({
                "file": f, "n": n, "alpha": alpha, "base": base, "t": float(tj),
                "g0": float(g0[j]), "K_n": int(K[j]),
                "post_mean": float(post_mean[j]),
                "ci_lo": float(lo[j]), "ci_hi": float(hi[j]),
                "covered": bool(covered[j]),
            })

    per_rep = pd.DataFrame(rows)

    # Aggregate: empirical mean/var of post_mean across reps, and coverage rate
    agg = (per_rep
           .groupby(["base","n","alpha","t","g0"], as_index=False)
           .agg(emp_mean=("post_mean","mean"),
                emp_var =("post_mean","var"),
                cov_rate=("covered","mean"),
                reps    =("post_mean","size")))

    # Theoretical mean/variance across reps (data ~ G0)
    agg["theory_mean"] = agg["g0"]
    agg["theory_var"]  = (agg["n"] * agg["g0"] * (1.0 - agg["g0"])) / (agg["alpha"] + agg["n"])**2

    Path("results").mkdir(exist_ok=True)
    out = "results/polya_checks.csv"
    agg.to_csv(out, index=False)
    print("[analyze_polya] wrote", out)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(agg)

if __name__ == "__main__":
    main()
