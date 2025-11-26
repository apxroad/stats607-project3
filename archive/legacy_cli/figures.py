from __future__ import annotations
import glob, os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def plot_convergence(raw_files: list[str], outdir: str):
    """
    Plot d_infty and d_rmse vs i for each method (one example file per (method, n)).
    We drop NaNs (because distances are recorded every k steps), and add markers.
    """
    examples: dict[tuple[str, int], pd.DataFrame] = {}

    for f in raw_files:
        df = pd.read_parquet(f, engine="fastparquet")
        key = (str(df["method"].iloc[0]), int(df["n"].iloc[0]))
        # keep the first file we see for each (method, n)
        if key not in examples:
            examples[key] = df

    for (method, n), df in examples.items():
        # ---- d_infty ----
        dfi = df.dropna(subset=["d_infty"])
        if not dfi.empty:
            plt.figure()
            plt.plot(dfi["i"], dfi["d_infty"], marker="o", markersize=2, linewidth=1, label=r"$d^{(\infty)}$")
            plt.xlabel("step i")
            plt.ylabel(r"$d^{(\infty)}$")
            plt.title(f"Convergence: {method}, n={n}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"conv_dinf_{method}_n{n}.png"))
            plt.close()

        # ---- d_rmse ----
        dfm = df.dropna(subset=["d_rmse"])
        if not dfm.empty:
            plt.figure()
            plt.plot(dfm["i"], dfm["d_rmse"], marker="o", markersize=2, linewidth=1, label="RMSE")
            plt.xlabel("step i")
            plt.ylabel("RMSE")
            plt.title(f"Convergence: {method}, n={n}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"conv_rmse_{method}_n{n}.png"))
            plt.close()

def plot_pit_hist(raw_files: list[str], outdir: str):
    """Combine PIT across reps per (method, n) and draw histograms."""
    frames = []
    for f in raw_files:
        df = pd.read_parquet(f, engine="fastparquet")[["method", "n", "pit"]].dropna()
        if not df.empty:
            frames.append(df)
    if not frames:
        return

    allpit = pd.concat(frames, ignore_index=True)
    for (method, n), grp in allpit.groupby(["method", "n"]):
        plt.figure()
        plt.hist(grp["pit"].values, bins=20, density=True, edgecolor="black")
        plt.xlabel("PIT")
        plt.ylabel("Density")
        plt.title(f"PIT histogram: {method}, n={int(n)}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"pit_{method}_n{int(n)}.png"))
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Make baseline figures")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    raw_dir = cfg["io"]["raw_dir"]
    fig_dir = cfg["io"]["fig_dir"]
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        print("[figures] no raw files found — run simulate first.")
        return

    plot_convergence(files, fig_dir)
    plot_pit_hist(files, fig_dir)
    print("[figures] wrote figures to", fig_dir)

if __name__ == "__main__":
    main()
