from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import yaml

def load_runs(raw_dir: str):
    runs = {}  # (method,n) -> df
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.parquet"))):
        df = pd.read_parquet(f, engine="fastparquet")
        key = (str(df["method"].iloc[0]), int(df["n"].iloc[0]))
        runs.setdefault(key, df)
    return runs

def plot_overview_for_method(method: str, by_n: dict[int, pd.DataFrame], outdir: str):
    if not by_n: return
    ns = sorted(by_n.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    ax_dinf, ax_rmse, ax_pit = axes

    # left: d∞
    for n in ns:
        df = by_n[n]
        m = df["d_infty"].notna()
        ax_dinf.plot(df.loc[m, "i"].values, df.loc[m, "d_infty"].values, label=f"n={n}")
    ax_dinf.set_xlabel("step i"); ax_dinf.set_ylabel(r"$d^{(\infty)}$")
    ax_dinf.set_title("Convergence: $d^{(\\infty)}$")
    ax_dinf.legend(title="sample size", frameon=False, loc="upper right")

    # middle: RMSE
    for n in ns:
        df = by_n[n]
        m = df["d_rmse"].notna()
        ax_rmse.plot(df.loc[m, "i"].values, df.loc[m, "d_rmse"].values, label=f"n={n}")
    ax_rmse.set_xlabel("step i"); ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Convergence: RMSE")

    # right: PIT (overlaid histograms per n)
    bins = np.linspace(0, 1, 21)
    for n in ns:
        u = by_n[n]["pit"].dropna().values
        ax_pit.hist(u, bins=bins, density=True, alpha=0.35, edgecolor="black", label=f"n={n}")
    ax_pit.set_xlabel("PIT"); ax_pit.set_ylabel("Density")
    ax_pit.set_title("PIT")
    ax_pit.legend(frameon=False, loc="upper right")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Overview: {method}", y=1.02, fontsize=13)
    fig.savefig(os.path.join(outdir, f"overview_{method}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Overview panel with d∞, RMSE, PIT in subplots")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    raw_dir = cfg.get("io", {}).get("raw_dir", "results/raw")
    outdir  = cfg.get("io", {}).get("fig_dir", "results/figures")

    runs = load_runs(raw_dir)
    # regroup by method
    per_method: dict[str, dict[int, pd.DataFrame]] = {}
    for (method, n), df in runs.items():
        per_method.setdefault(method, {})[n] = df

    for method, by_n in per_method.items():
        plot_overview_for_method(method, by_n, outdir)

    print("[figures] wrote overview panels to", outdir)

if __name__ == "__main__":
    main()
