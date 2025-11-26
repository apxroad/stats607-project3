from __future__ import annotations
from pathlib import Path
import argparse, numpy as np
import matplotlib.pyplot as plt
from src.plotstyle import apply_plot_style
from scipy.stats import beta
from src.polya import PolyaSequenceModel, build_prefix, continue_urn_once
FIG_DIR = Path("results") / "figures"

def panel_for_n(n: int, ts: list[float], alphas: list[float], M: int, N: int, base: str, seed: int):
    """Render a grid of panels showing distributions of P((−∞, t]) via Pólya continuation.

    Parameters
    ----------
    n : int
        Length of the observed prefix x_{1:n}. If n==0 the figure shows *prior* panels.
    ts : list[float]
        Row-wise thresholds t at which we evaluate probability mass.
    alphas : list[float]
        Column-wise concentration parameters α of the Pólya/DP prior.
    M : int
        Continuation length per trajectory (final length of each simulated path).
    N : int
        Number of Monte Carlo continuations per cell (panel).
    base : {"uniform","normal"}
        Base distribution G0 for the urn (U(0,1) or N(0,1)).
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    # Initialize model with first α (will be reassigned inside the loop).
    model = PolyaSequenceModel(alpha=alphas[0], base=base, rng=rng)

    # Observed prefix x_{1:n} used for all panels (same dataset across the grid).
    x_obs = build_prefix(n, model)
    x_obs_arr = np.asarray(x_obs, dtype=float) if len(x_obs) > 0 else np.empty(0, dtype=float)

    # Precompute posterior/prior draws of mass P((−∞, t]) for all (t, α) cells.
    # Key optimization: for each α and Monte Carlo replicate we simulate ONE Pólya path
    # and reuse it across all thresholds ts, instead of resimulating per (t, α) cell.
    R, C = len(ts), len(alphas)
    post_all = np.empty((R, C, N), dtype=float)

    for j, a in enumerate(alphas):
        # Update α for this column (reuse same model/RNG).
        model.alpha = a
        for r in range(N):
            # Continue the same prefix once to length M.
            traj = continue_urn_once(x_obs, model, M)
            traj_arr = np.asarray(traj, dtype=float)

            # For this trajectory, compute empirical mass at EACH t.
            # This reuses the same simulated path for all thresholds.
            for i, t in enumerate(ts):
                post_all[i, j, r] = np.mean(traj_arr <= t)

    # Figure layout: rows correspond to thresholds t, columns correspond to α.
    fig, axes = plt.subplots(R, C, figsize=(12, 8), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)  # normalize shape for R=C=1 case

    for i, t in enumerate(ts):
        # Sufficient statistic at threshold t: K_n(t) = #{x_i ≤ t} over the prefix.
        if x_obs_arr.size:
            k_n = np.sum(x_obs_arr <= t)
        else:
            k_n = 0

        for j, a in enumerate(alphas):
            # Posterior/prior draws for this (t, α) cell.
            post = post_all[i, j, :]

            ax = axes[i, j]
            x = np.linspace(0, 1, 600)
            # Conjugate Beta overlay parameters for indicators 1{x ≤ t}.
            a_post = a * t + k_n
            b_post = a * (1 - t) + (n - k_n)

            # Histogram (posterior/prior draws of mass at t) + Beta overlay + reference line t.
            ax.hist(post, bins=50, density=True, alpha=0.8, edgecolor="none")
            ax.plot(x, beta.pdf(x, a_post, b_post), lw=1.0, color="tab:orange")
            ax.axvline(t, ls="--", lw=1.0, color="tab:blue")

            # Keep x-axis in [0,1]; add a small vertical margin for readability.
            ax.set_xlim(0, 1)
            ax.margins(y=0.05)

            # Column headers only on the top row: label the α used in this column.
            if i == 0:
                ax.set_title(f"α = {a}", pad=6, fontweight="regular")

        # Row headers once per row (left margin): label the t used in this row.
        axes[i, 0].text(-0.12, 0.5, f"t = {t}",
                        transform=axes[i, 0].transAxes,
                        va="center", ha="right", fontsize=11)

    # Shared axis labels for the whole figure (avoid repeating per-panel labels).
    label = "Prior" if n == 0 else "Posterior"
    fig.supxlabel(r"Probability mass $\tilde P((-\infty, t])$", x=0.55, y=0.08)
    fig.supylabel(f"{label} density", x=1)

    # Figure title and configuration subtitle.
    title = rf"{label} draws of $\tilde P((-\infty, t])$ under Pólya sequence"
    fig.suptitle(title, fontsize=13, fontweight="regular", x=0.6, y=0.94)
    subtitle = f"Base: {base}  |  n = {n},  α ∈ {alphas}  |  M = {M}, N = {N}"
    fig.text(0.6, 0.9, subtitle, ha="center", va="top", fontsize=10, style='italic')

    # Layout: reserve margins for row/column headers and the two-line header.
    fig.tight_layout(rect=[0.08, 0.08, 1.0, 0.90])

    # Save PNG + PDF variants; mirror existing naming convention.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    out = FIG_DIR / f"post_panels_cont_n{n}_M{M}_N{N}_{base}.png"
    fig.savefig(out, dpi=140)
    fig.savefig(Path(out).with_suffix('.pdf'))
    plt.close(fig)
    print(f"[ok] wrote {out}")
def main():
    """CLI entry point: build posterior panels for a grid of (t, α)."""
    ap = argparse.ArgumentParser(description="Panels of posterior via Pólya continuation.")
    ap.add_argument("--base", choices=["uniform","normal"], default="uniform")
    ap.add_argument("--t", dest="ts", type=float, nargs="+", default=[0.25,0.5,0.75])
    ap.add_argument("--alpha", dest="alphas", type=float, nargs="+", default=[1,5,20])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20250101)
    args = ap.parse_args()
    apply_plot_style()  # apply global rcParams for consistent styling

    # Run the panel generator with parsed CLI arguments.
    panel_for_n(args.n, args.ts, args.alphas, args.M, args.N, args.base, args.seed)

if __name__ == "__main__":
    main()
