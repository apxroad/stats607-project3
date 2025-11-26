from __future__ import annotations
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path

def sci(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    fmt.set_scientific(True)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

def main():
    ap = argparse.ArgumentParser(description="Plot Pólya DP summary (emp vs theory + coverage).")
    ap.add_argument("--csv", default="results/polya_checks.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # --- 1) Mean: emp vs theory
    ts = sorted(df["t"].unique()); ns = sorted(df["n"].unique())
    fig1, axes1 = plt.subplots(1, len(ts), figsize=(5*len(ts), 4), squeeze=False)
    for j, t in enumerate(ts):
        ax = axes1[0, j]
        sub = df[df["t"] == t]
        for n in ns:
            s2 = sub[sub["n"] == n]
            ax.scatter(s2["theory_mean"], s2["emp_mean"], label=f"n={n}", alpha=0.9)
        ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"Mean: t={t}"); ax.set_xlabel("Theory mean"); ax.set_ylabel("Empirical mean")
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02,1))
    fig1.tight_layout(rect=[0,0,0.88,1])
    fig1.savefig("results/figures/polya_mean_emp_vs_theory.png", dpi=150); plt.close(fig1)

    # --- 2) Variance: emp vs theory
    fig2, axes2 = plt.subplots(1, len(ts), figsize=(6*len(ts), 4.2), squeeze=False)
    for j, t in enumerate(ts):
        ax = axes2[0, j]
        sub = df[df["t"] == t]
        for n in ns:
            s2 = sub[sub["n"] == n]
            ax.scatter(s2["theory_var"], s2["emp_var"], label=f"n={n}", alpha=0.9)
        lo, hi = 0.0, max(sub["theory_var"].max(), sub["emp_var"].max())*1.05
        ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); sci(ax)
        ax.set_title(f"Variance: t={t}"); ax.set_xlabel("Theory var"); ax.set_ylabel("Empirical var")
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02,1))
    fig2.tight_layout(rect=[0,0,0.88,1])
    fig2.savefig("results/figures/polya_var_emp_vs_theory.png", dpi=200); plt.close(fig2)

    # --- 3) Coverage bars (single blue hue, legend outside)
    piv = df.groupby(["t","n"], as_index=False)["cov_rate"].mean()
    fig3, ax3 = plt.subplots(figsize=(6.5,4))
    tvals = sorted(piv["t"].unique()); nvals = sorted(piv["n"].unique())
    x = np.arange(len(tvals)); width = 0.8 / max(1,len(nvals))
    cmap = plt.cm.Blues; cols = [cmap(v) for v in np.linspace(0.45,0.9,len(nvals))]
    for k,(n,c) in enumerate(zip(nvals, cols)):
        y = [piv[(piv.t==t)&(piv.n==n)]["cov_rate"].values[0] if not piv[(piv.t==t)&(piv.n==n)].empty else np.nan
             for t in tvals]
        ax3.bar(x + k*width - 0.4 + width/2, y, width, label=f"n={n}",
                color=c, edgecolor="black", linewidth=0.5)
    ax3.axhline(0.95, ls="--", color="k", lw=1)
    ax3.set_xticks(x); ax3.set_xticklabels([f"t={t}" for t in tvals])
    ax3.set_ylim(0,1); ax3.set_ylabel("Coverage rate"); ax3.set_title("Beta CI coverage by t and n")
    ax3.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02,1))
    fig3.tight_layout(rect=[0,0,0.88,1])
    fig3.savefig("results/figures/polya_coverage.png", dpi=150); plt.close(fig3)

    print("Wrote figures in results/figures/: polya_mean_emp_vs_theory.png, polya_var_emp_vs_theory.png, polya_coverage.png")

if __name__ == "__main__":
    main()
