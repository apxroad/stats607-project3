import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    """Plot predictive paths m ↦ P_m(t) from a CSV.

    Expected CSV schema (rows are time points for various thresholds t):
      - m   : int, step index (0,1,2,...)
      - t   : float, threshold value for the CDF evaluation
      - Pm  : float, predictive probability at step m for (−∞, t]

    The plot shows, for each unique t, a line of P_m(t) versus m, with a dashed
    reference line at y=t (the Unif(0,1) baseline).
    """
    ap = argparse.ArgumentParser(description="Plot predictive paths m↦Pm(t) from a CSV produced by log_predictive_paths.py")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--title", type=str, default="Predictive paths m ↦ Pm(t)")
    args = ap.parse_args()

    # Load and identify all thresholds to create one subplot per t
    df = pd.read_csv(args.csv)
    ts = sorted(df["t"].unique())
    fig, axes = plt.subplots(len(ts), 1, figsize=(7, 2.6*len(ts)), sharex=True)
    if len(ts)==1: axes=[axes]  # normalize to a list for uniform iteration

    for ax, t in zip(axes, ts):
        # Filter series for this t and sort by step m
        sub = df[df["t"]==t].sort_values("m")
        ax.plot(sub["m"], sub["Pm"], lw=1.6)
        ax.axhline(t, ls="--", color="k", lw=1)  # baseline for Unif(0,1)
        ax.set_ylabel(f"P_m({t})")
        ax.set_ylim(0,1)
        ax.grid(alpha=.25, linestyle=":", linewidth=.8)

    # Common x-label and figure title
    axes[-1].set_xlabel("m (step)")
    fig.suptitle(args.title)
    fig.tight_layout(rect=[0,0,1,0.95])

    # Save to results/figures using a stem derived from the CSV filename
    outdir = Path("results/figures"); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.csv).stem.replace("predictive_path_", "predictive_paths_")
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[ok] wrote {png}")

if __name__ == "__main__":
    main()
