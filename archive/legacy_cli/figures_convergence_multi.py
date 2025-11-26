from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import yaml

def load_examples(raw_dir: str):
    ex = {}  # (method,n) -> df
    for f in sorted(glob.glob(os.path.join(raw_dir, "*.parquet"))):
        df = pd.read_parquet(f, engine="fastparquet")
        key = (str(df["method"].iloc[0]), int(df["n"].iloc[0]))
        ex.setdefault(key, df)
    return ex

def plot_metric_multi(method, by_n, outdir, metric, ylabel, title_prefix):
    if not by_n: return
    fig = plt.figure()
    for n in sorted(by_n):
        df = by_n[n]
        m = df[metric].notna()
        plt.plot(df.loc[m, "i"].values, df.loc[m, metric].values, label=f"n={n}")
    plt.xlabel("step i"); plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: {method} (multi-n)")
    plt.legend(title="sample size", frameon=False, loc="upper right")
    plt.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{metric}_{method}_multi.png"), dpi=150)
    plt.close(fig)

def plot_pit_multi(method, by_n, outdir):
    if not by_n: return
    fig = plt.figure()
    bins = np.linspace(0,1,21)
    for n in sorted(by_n):
        df = by_n[n]
        u = df["pit"].dropna().values
        plt.hist(u, bins=bins, density=True, alpha=0.35, edgecolor="black", label=f"n={n}")
    plt.xlabel("PIT"); plt.ylabel("Density"); plt.title(f"PIT: {method} (multi-n)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"pit_{method}_multi.png"), dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Merged convergence & PIT plots across n")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    raw_dir = cfg.get("io", {}).get("raw_dir", "results/raw")
    outdir  = cfg.get("io", {}).get("fig_dir", "results/figures")
    examples = load_examples(raw_dir)

    per_method: dict[str, dict[int, pd.DataFrame]] = {}
    for (method, n), df in examples.items():
        per_method.setdefault(method, {})[n] = df

    for method, by_n in per_method.items():
        plot_metric_multi(method, by_n, outdir, "d_infty", r"$d^{(\infty)}$", "Convergence")
        plot_metric_multi(method, by_n, outdir, "d_rmse",  "RMSE",            "Convergence")
        plot_pit_multi(method, by_n, outdir)

    print("[figures] wrote merged d_inf/d_rmse/PIT plots to", outdir)

if __name__ == "__main__":
    main()
