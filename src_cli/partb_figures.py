from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.plotstyle import apply_plot_style

def _read_csv(p: Path):
    """Read a CSV if it exists; otherwise return None (silent skip helper)."""
    return pd.read_csv(p) if p.exists() else None

def main():
    """Part B figure generator: convergence plot + predictive paths.

    Inputs
    ------
    --stem : common filename stem used by the data-producing scripts, e.g.
             'partB_n1000_a5.0_seed2025_uniform'.
    --title : optional string appended to plot titles.

    Expected raw files (searched under results/raw/):
      - distances_{stem}.csv           → convergence distances over i
      - Pm_paths_{stem}.csv            → predictive paths (preferred)
      - predictive_path_{stem}.csv     → fallback name for predictive paths

    Output
    ------
    Two figures saved to results/figures/ as both PNG and PDF:
      - partB_convergence_{stem}.(png|pdf)
      - partB_paths_{stem}.(png|pdf)
    """
    ap = argparse.ArgumentParser(
        description="Part B — figures: convergence + predictive paths (no PIT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--stem", required=True, help="common stem, e.g. partB_n1000_a5.0_seed2025_uniform")
    ap.add_argument("--title", default="", help="optional title")
    args = ap.parse_args()
    apply_plot_style()  # ensure consistent styling across figures

    raw = Path("results/raw")
    out = Path("results/figures"); out.mkdir(parents=True, exist_ok=True)

    # Candidate input files
    path_dist = raw / f"distances_{args.stem}.csv"
    path_pm   = raw / f"Pm_paths_{args.stem}.csv"
    path_alt  = raw / f"predictive_path_{args.stem}.csv"  # fallback

    # Load data if present
    dfD = _read_csv(path_dist)
    dfP = _read_csv(path_pm)
    if dfP is None:
        dfP = _read_csv(path_alt)

    # --- Figure 1: convergence (distances vs i) ---
    if dfD is not None:
        # Basic schema validation
        need = {"i","d_infty","d_rmse"}
        if not need.issubset(dfD.columns):
            raise ValueError(f"Distances CSV missing {need}; got {set(dfD.columns)}.")
        plt.figure(figsize=(6.5, 4.0))
        plt.plot(dfD["i"], dfD["d_infty"], label=r"$d^{(\infty)}$")
        plt.plot(dfD["i"], dfD["d_rmse"],  label="RMSE")
        plt.xlabel("Step $i$ (number of observations)"); plt.ylabel("Distance between $\\tilde P_i$ and $\\tilde F$")
        plt.title(f"Convergence of Predictive CDF $\\tilde F$ — {args.title}" if args.title else "Convergence")
        plt.legend(frameon=False)
        plt.tight_layout()
        out1 = out / f"partB_convergence_{args.stem}.png"
        # Save both raster and vector formats
        plt.savefig(out1, dpi=150); plt.savefig(out1.with_suffix('.pdf')); plt.close()
        print(f"[ok] wrote {out1}")
    else:
        print(f"[skip] No distances CSV found at {path_dist}; skipped convergence figure.")

    # --- Figure 2: predictive paths P_m(t) vs m ---
    if dfP is not None:
        # Basic schema validation
        need = {"m","t","Pm"}
        if not need.issubset(dfP.columns):
            raise ValueError(f"Predictive paths CSV missing {need}; got {set(dfP.columns)}.")
        tvals = sorted(pd.unique(dfP["t"]))
        plt.figure(figsize=(7.0, 4.2))
        for t in tvals:
            # Plot each threshold's trajectory; dashed y=t reference (U(0,1) baseline)
            sub = dfP[dfP["t"] == t].sort_values("m")
            plt.plot(sub["m"], sub["Pm"], label=f"t={t}")
            try:
                plt.axhline(float(t), lw=1, ls="--", color="k")
            except Exception:
                # Gracefully skip if 't' is not castable to float
                pass
        plt.xlabel("Step $m$"); plt.ylabel(r"Predictive CDF value")
        plt.title(f"Predictive Probability Trajectories $m \\mapsto P_m((−∞, t])$" if args.title else "Predictive path trajectories")
        plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.tight_layout()
        out2 = out / f"partB_paths_{args.stem}.png"
        plt.savefig(out2, dpi=150); plt.savefig(out2.with_suffix('.pdf')); plt.close()
        print(f"[ok] wrote {out2}")
    else:
        print(f"[skip] No predictive paths CSV found at {path_pm} or {path_alt}; skipped paths figure.")

if __name__ == "__main__":
    main()
