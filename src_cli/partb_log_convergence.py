from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.methods import PolyaPredictive
from src.dgps import UniformTruth, NormalTruth
from src.metrics import make_grid, d_infty, d_rmse

def main():
    """Log convergence metrics and predictive paths for Part B (no PIT).

    For a chosen base truth (Uniform or Normal), this script:
      1) Simulates an i.i.d. data stream X_1,...,X_n from the oracle truth.
      2) Runs an online predictive method (Pólya DP) and, BEFORE each update:
         - evaluates the estimated CDF on a grid to compute d_∞ and RMSE
         - records P_m(t) for each requested threshold t
      3) Writes two tidy CSVs under results/raw/:
         - distances_{stem}.csv with columns [i, d_infty, d_rmse]
         - Pm_paths_{stem}.csv with columns [m, t, Pm]

    Notes
    -----
    - Evaluation is prequential (one-step-ahead): metrics at time i use data up
      to i−1 only. The first step (i=0) has no metrics recorded.
    - Grid endpoints default to [0,1] for Uniform and [-4,4] for Normal;
      you can override via --tmin/--tmax when base=normal.
    """
    ap = argparse.ArgumentParser(
        description="Part B — log convergence metrics and P_m(t) paths (no PIT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--t", type=float, nargs="+", required=True, help="one or more thresholds t")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--base", choices=["uniform","normal"], default="uniform")
    ap.add_argument("--J", type=int, default=100)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=1.0)
    args = ap.parse_args()

    # Truth and evaluation range
    # Match the DGP to the method's base to maintain a clean exchangeable setup.
    if args.base == "uniform":
        truth = UniformTruth(0.0, 1.0)
        tmin, tmax = 0.0, 1.0
    else:
        truth = NormalTruth(0.0, 1.0)
        tmin, tmax = -4.0, 4.0

    # Data stream (fixed by seed for reproducibility).
    x = truth.sample(n=args.n, seed=args.seed)

    # Predictive model (Pólya/DP) and its internal state.
    pred = PolyaPredictive(alpha=args.alpha, base=args.base)
    state = pred.init_state()

    # Grid for distance metrics.
    # For Normal base, allow user overrides via --tmin/--tmax; else use defaults above.
    grid = make_grid(args.J, args.tmin if args.base == "normal" else tmin,
                            args.tmax if args.base == "normal" else tmax)
    c_true_grid = truth.cdf_truth(grid)

    rec_dist = []  # rows: (i, d_infty, d_rmse)   — convergence diagnostics
    rec_Pm   = []  # rows: (m, t, Pm)             — predictive path trajectories

    for i in range(args.n):
        xi = float(x[i])

        # Evaluate BEFORE update (prequential).
        if i > 0:
            c_est = np.asarray(pred.cdf_est(state, grid), dtype=float)
            rec_dist.append((i, float(d_infty(c_est, c_true_grid)),
                                float(d_rmse (c_est, c_true_grid))))
            for t in args.t:
                pm_t = float(pred.cdf_est(state, float(t)))
                rec_Pm.append((i, float(t), pm_t))

        # Bayesian update with the new observation.
        state = pred.update(state, xi)

    # Write CSVs to results/raw with a descriptive stem.
    outdir = Path("results/raw"); outdir.mkdir(parents=True, exist_ok=True)
    stem = f"partB_n{args.n}_a{args.alpha}_seed{args.seed}_{args.base}"

    pd.DataFrame(rec_dist, columns=["i","d_infty","d_rmse"])\
      .to_csv(outdir / f"distances_{stem}.csv", index=False)
    pd.DataFrame(rec_Pm,   columns=["m","t","Pm"])\
      .to_csv(outdir / f"Pm_paths_{stem}.csv",   index=False)

    print(f"[ok] wrote {outdir / ('distances_' + stem + '.csv')}")
    print(f"[ok] wrote {outdir / ('Pm_paths_'   + stem + '.csv')}")

if __name__ == "__main__":
    main()
