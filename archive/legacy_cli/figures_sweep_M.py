from __future__ import annotations
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot coverage and bias vs Monte Carlo reps (M).")
    ap.add_argument("--glob", default="results/polya_checks_M*.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob), key=lambda s: int(re.findall(r"M(\d+)", s)[0]))
    if not files:
        print("[sweep_M] no polya_checks_M*.csv found—run sweep_M.py first.")
        return

    parts = []
    for f in files:
        M = int(re.findall(r"M(\d+)", f)[0])
        df = pd.read_csv(f)
        df["M"] = M
        df["abs_bias"] = (df["emp_mean"] - df["theory_mean"]).abs()
        parts.append(df)
    all_ = pd.concat(parts, ignore_index=True)

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    tvals = sorted(all_["t"].unique())
    nvals = sorted(all_["n"].unique())
    Ms    = sorted(all_["M"].unique())

    # Coverage vs M
    fig1, axes1 = plt.subplots(len(tvals), 1, figsize=(6.5, 3.2*len(tvals)), sharex=True)
    if len(tvals) == 1: axes1 = [axes1]
    for ax, t in zip(axes1, tvals):
        sub = all_[all_["t"] == t]
        for n in nvals:
            s2 = sub[sub["n"] == n].sort_values("M")
            ax.plot(s2["M"], s2["cov_rate"], marker="o", label=f"n={n}")
        ax.axhline(0.95, ls="--", color="k", lw=1)
        ax.set_title(f"Coverage vs M (t={t})"); ax.set_ylabel("Coverage")
        ax.set_ylim(0.7, 1)
        ax.legend(frameon=False, fontsize=9, loc="best")
    axes1[-1].set_xlabel("M (Monte Carlo reps)")
    fig1.tight_layout()
    fig1.savefig("results/figures/polya_coverage_vs_M.png", dpi=150); plt.close(fig1)

    # Absolute bias vs M (lower is better)
    fig2, axes2 = plt.subplots(len(tvals), 1, figsize=(6.5, 3.2*len(tvals)), sharex=True)
    if len(tvals) == 1: axes2 = [axes2]
    for ax, t in zip(axes2, tvals):
        sub = all_[all_["t"] == t]
        for n in nvals:
            s2 = sub[sub["n"] == n].sort_values("M")
            ax.plot(s2["M"], s2["abs_bias"], marker="o", label=f"n={n}")
        ax.set_title(f"|emp_mean - theory_mean| vs M (t={t})"); ax.set_ylabel("Abs bias")
        ax.legend(frameon=False, fontsize=9, loc="best")
    axes2[-1].set_xlabel("M (Monte Carlo reps)")
    fig2.tight_layout()
    fig2.savefig("results/figures/polya_bias_vs_M.png", dpi=150); plt.close(fig2)

    print("[figures] wrote results/figures/polya_coverage_vs_M.png and polya_bias_vs_M.png")

if __name__ == "__main__":
    main()
