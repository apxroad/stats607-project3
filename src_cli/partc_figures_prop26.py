from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotstyle import apply_plot_style

def normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard Normal PDF φ(x) evaluated elementwise."""
    return np.exp(-0.5 * x**2) / np.sqrt(2*np.pi)

def extract_Z(df: pd.DataFrame) -> np.ndarray:
    """Extract pooled Z from a dataframe, or compute it from components.

    Preferred:
        Use column 'Z' directly if present (compat with older logs).

    Otherwise (computed path):
        Z = (Pn − Fhat) / sqrt(Vnt / n),
        where
          - Pn   : predictive probability at threshold t,
          - Fhat : empirical fraction at t (or oracle truth on t),
          - Vnt  : variance term for the Bernoulli indicators at step n,t,
          - n    : sample size at which quantities are evaluated.

    Notes
    -----
    - Lowercase 'z' (if present) is a CI *critical value* (≈1.96), not the
      statistic. We intentionally ignore it to avoid confusion.
    - Raises ValueError if neither 'Z' nor all of {Pn, Fhat, Vnt, n} exist.
    """
    cols = set(df.columns)
    # If 'Z' already exists (older logs), use it.
    if "Z" in cols:
        return df["Z"].to_numpy(dtype=float)
    # Do NOT use lowercase 'z' — that is the CI critical value (≈1.96), not the statistic.
    need = {"Pn", "Fhat", "Vnt", "n"}
    if need.issubset(cols):
        Pn   = df["Pn"].to_numpy(float)
        Fhat = df["Fhat"].to_numpy(float)
        Vnt  = df["Vnt"].to_numpy(float)
        n    = df["n"].to_numpy(float)
        return (Pn - Fhat) / np.sqrt(Vnt / n)
    raise ValueError(f"CSV must contain 'Z' or computable fields {need}. Got: {sorted(cols)}")

def main():
    """Plot pooled Z histogram with N(0,1) overlay for Part C.

    Inputs
    ------
    --csv   : CSV path produced by partc_log_prop26.py (or compatible).
    --title : optional title prefix.
    --bins  : histogram bins.

    Output
    ------
    Saves PNG and PDF under results/figures/ with a stem derived from input CSV.

    Figure
    ------
    - Histogram of pooled Z values.
    - Overlaid standard Normal density for a visual GOF check.
    - Title includes empirical mean and sd of Z for quick diagnostics.
    """
    ap = argparse.ArgumentParser(
        description="Part C — pooled Z only, with N(0,1) overlay (computed from Pn,Fhat,Vnt,n)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="CSV from partc_log_prop26.py")
    ap.add_argument("--title", default="", help="Custom title (optional)")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins")
    args = ap.parse_args()

    apply_plot_style()

    df = pd.read_csv(args.csv)
    Z = extract_Z(df)

    # Title bits from CSV (if present)
    alpha = df.get("alpha", pd.Series([None])).iloc[0]
    base  = df.get("base",  pd.Series([None])).iloc[0]

    # Basic diagnostics for the subtitle
    z_mean = float(np.mean(Z))
    z_sd   = float(np.std(Z, ddof=1)) if len(Z) > 1 else 0.0

    # X-range for overlay - use a reasonable range based on the data
    # Use ±3 standard deviations or ±4, whichever is smaller, for better visual balance
    data_range = max(abs(np.min(Z)), abs(np.max(Z)))
    z_range = min(max(4.0, 3.0 * z_sd), data_range + 1.0)
    xs = np.linspace(-z_range, z_range, 1200)

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Increase number of bins for smoother histogram appearance
    n_bins = min(args.bins, max(30, int(len(Z) / 20)))
    ax.hist(Z, bins=n_bins, density=True, alpha=0.65, edgecolor="black", linewidth=0.5, color='steelblue')
    ax.plot(xs, normal_pdf(xs), linewidth=2.5, label=r"Standard Normal $\mathcal{N}(0,1)$", color="tab:orange")

    # Set symmetric x-axis limits for better visual balance
    ax.set_xlim(-z_range, z_range)
    
    # Set y-axis to show the peak of Normal distribution nicely (0.4 is the peak of N(0,1))
    y_max = max(0.42, np.max(normal_pdf(xs)) * 1.05)
    ax.set_ylim(0, y_max)

    ax.set_xlabel("Z statistic")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    
    # Main title and subtitle
    fig.suptitle(r"Distribution of Pooled $Z$ Statistic", fontsize=14, y=0.98)
    subtitle = rf"$\alpha = {alpha}$, Base = {base} | mean = {z_mean:.2f}, sd = {z_sd:.2f}"
    fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=10, style='italic')
    
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    outdir = Path("results/figures"); outdir.mkdir(parents=True, exist_ok=True)
    stem = f"prop26_zcheck_{Path(args.csv).stem}"
    out_png = outdir / f"{stem}.png"
    out_pdf = outdir / f"{stem}.pdf"

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_pdf}")

if __name__ == "__main__":
    main()