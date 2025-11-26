import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

# ---- base CDF and base sampler ----
def G0_cdf(t, base="uniform"):
    """Base CDF G0(t) for the chosen prior base.

    Parameters
    ----------
    t : float
        Threshold.
    base : {"uniform","normal"}
        Name of the base distribution.

    Returns
    -------
    float
        G0(t). Uses exact formulas; Φ(t) via `erf` for Normal.
    """
    if base == "uniform":
        if t <= 0.0: return 0.0
        if t >= 1.0: return 1.0
        return float(t)
    elif base == "normal":
        # Φ(t) via erf
        return 0.5*(1.0 + math.erf(t / math.sqrt(2.0)))
    else:
        raise ValueError(f"unknown base: {base}")

def sample_from_base(rng, base="uniform"):
    """Draw a single sample from the base G0 using the provided RNG."""
    if base == "uniform":
        return float(rng.random())
    elif base == "normal":
        return float(rng.normal())
    else:
        raise ValueError(f"unknown base: {base}")

# ---- draw next X under the Pólya urn given current list xs  ----
def draw_polya_next(xs, alpha, rng, base="uniform"):
    """One-step Blackwell–MacQueen Pólya update.

    With probability α/(α+m) draw a fresh atom from G0; otherwise pick
    uniformly among the existing atoms in `xs`.
    """
    m = len(xs)                       # current size
    p_new = alpha / (alpha + m)       # prob of a fresh draw from G0
    if rng.random() < p_new:
        return sample_from_base(rng, base)
    else:
        j = rng.integers(0, m)        # pick an existing atom uniformly
        return float(xs[j])

def main():
    """Compute Proposition 2.6 predictive CIs for \tilde F(t) via continuation.

    Outline
    -------
    For each n in --n and each replication:
      1) Generate x_1..x_n from the SAME Pólya urn with base G0 and α.
      2) Track K_m(t) and P_m(t) iteratively for each t; accumulate
         V_{n,t} = (1/n) * Σ_{m=1}^n m^2 (P_m − P_{m−1})^2.
      3) Compute P_n(t) at the end of the prefix.
      4) CONTINUE THE SAME URN by L extra draws to estimate \tilde F(t) as
         F̂(t) = (1/L) * Σ 1{x_{n+ℓ} ≤ t}.
      5) Form Wald CI: P_n(t) ± z * sqrt(V_{n,t}/n), and record coverage of F̂(t).

    Notes
    -----
    - We fix z for 95% CIs without SciPy; other levels warn and still use z≈1.95996.
    - RNG seeding uses a hash of (rep, n) to keep replicates independent.
    - The continuation uses the SAME urn (same `xs` list), as required by Prop 2.6.
    """
    ap = argparse.ArgumentParser(
        description="Prop 2.6 predictive CIs for F~(t), with target via continuation on the SAME urn."
    )
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--base",  choices=["uniform","normal"], default="uniform")
    ap.add_argument("--t",     nargs="+", type=float, required=True, help="thresholds t (one or more)")
    ap.add_argument("--n",     nargs="+", type=int,   required=True, help="sample sizes n (one or more)")
    ap.add_argument("--M",     type=int, default=200, help="number of datasets (MC reps)")
    ap.add_argument("--L",     type=int, default=50000, help="tail length for continuation")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed",  type=int, default=123)
    args = ap.parse_args()

    # z critical: avoid SciPy; exact for 0.95, warn otherwise
    if abs(args.level - 0.95) < 1e-12:
        z = 1.959963984540054
    else:
        # normal-approx inverse via rational approx would be overkill here; lock to 0.95
        print(f"[warn] level {args.level} not 0.95; using z≈1.95996 anyway.")
        z = 1.959963984540054

    outdir = Path("results/raw"); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    tvals = list(map(float, args.t))
    nvals = list(map(int,   args.n))
    alpha = float(args.alpha)

    for n in nvals:
        for rep in range(args.M):
            # Independent per-(rep,n) seed to avoid path reuse across settings.
            rng = np.random.default_rng(args.seed + 7919*rep + 104729*n)

            # --- generate prefix x1..xn from the Pólya urn
            xs = []
            # Book-keeping for each t: K_m(t), previous P_{m-1}, running sum for V_{n,t}
            Km = {t: 0 for t in tvals}
            P_prev = {t: G0_cdf(t, args.base) for t in tvals}  # P0(t) = G0(t)
            Vnt = {t: 0.0 for t in tvals}

            for m in range(1, n+1):
                # draw x_m using the same urn
                x_m = draw_polya_next(xs, alpha, rng, base=args.base)
                xs.append(x_m)

                # update counts and P_m, accumulate m^2 (P_m - P_{m-1})^2
                for t in tvals:
                    if x_m <= t:
                        Km[t] += 1
                    Pm = (alpha*G0_cdf(t, args.base) + Km[t]) / (alpha + m)
                    Vnt[t] += (m**2) * (Pm - P_prev[t])**2
                    P_prev[t] = Pm     # becomes P_m for next step

            # finalize V_{n,t}
            for t in tvals:
                Vnt[t] /= n

            # P_n(t) for each t
            Pn = {t: (alpha*G0_cdf(t, args.base) + Km[t]) / (alpha + n) for t in tvals}

            # --- continuation: extend the SAME urn by L steps and estimate F~(t)
            tail_leq = {t: 0 for t in tvals}
            for j in range(args.L):
                x_next = draw_polya_next(xs, alpha, rng, base=args.base)  # continues same xs/urn
                xs.append(x_next)
                for t in tvals:
                    if x_next <= t:
                        tail_leq[t] += 1
            Fhat = {t: tail_leq[t] / float(args.L) for t in tvals}

            # rows: record CI, coverage, width, and supporting quantities
            for t in tvals:
                se = math.sqrt(max(Vnt[t], 1e-12) / n)
                lo = Pn[t] - z*se
                hi = Pn[t] + z*se
                covered = int(lo <= Fhat[t] <= hi)
                rows.append({
                    "rep": rep, "n": n, "alpha": alpha, "base": args.base, "t": t,
                    "Pn": Pn[t], "Vnt": Vnt[t], "level": args.level, "z": z,
                    "L": args.L, "Fhat": Fhat[t], "lo": lo, "hi": hi,
                    "covered": covered, "width": 2*z*se
                })

    # Persist results
    df = pd.DataFrame(rows)
    stem = f"prop26_M{args.M}_L{args.L}_a{alpha}_seed{args.seed}_{args.base}.csv"
    out = outdir / stem
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    main()
