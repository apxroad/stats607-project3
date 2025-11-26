from __future__ import annotations
import argparse, numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from src.polya import PolyaSequenceModel, build_prefix, continue_urn_once

def main():
    ap = argparse.ArgumentParser(description="Posterior for P((−∞,t]) via Pólya continuation (single figure).")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=150, help="observed prefix length")
    ap.add_argument("--M", type=int, default=1000, help="final continuation length")
    ap.add_argument("--N", type=int, default=3000, help="MC repetitions")
    ap.add_argument("--base", choices=["uniform","normal"], default="uniform")
    ap.add_argument("--seed", type=int, default=20250101)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = PolyaSequenceModel(alpha=args.alpha, base=args.base, rng=rng)

    # observed data (prefix)
    x_obs = build_prefix(args.n, model)
    k_n = sum(1 for x in x_obs if x <= args.t)

    # posterior samples via continuation
    post = np.empty(args.N, dtype=float)
    for r in range(args.N):
        traj = continue_urn_once(x_obs, model, args.M)
        post[r] = np.mean(np.asarray(traj) <= args.t)

    a_post = args.alpha*args.t + k_n
    b_post = args.alpha*(1-args.t) + (args.n - k_n)

    # plot
    x = np.linspace(0,1,600)
    plt.figure(figsize=(6.4,4.6))
    plt.hist(post, bins=60, density=True, alpha=0.55, edgecolor="black", label="MC posterior")
    plt.plot(x, beta.pdf(x, a_post, b_post), lw=2.2, label=f"Beta(a={a_post:.1f}, b={b_post:.1f})")
    plt.axvline(args.t if args.base=="uniform" else 0.5, ls="--", lw=1.2, label=f"G0(t)={args.t if args.base=='uniform' else 'Φ(t)'}")
    plt.title(f"Posterior P((−∞,{args.t}]) | n={args.n}, α={args.alpha}, base={args.base}")
    plt.xlabel("P((−∞, t])"); plt.ylabel("Density"); plt.legend()
    plt.tight_layout()
    out = f"results/figures/post_cont_t{args.t:+.2f}_a{args.alpha}_n{args.n}_M{args.M}_{args.base}.png"
    plt.savefig(out, dpi=140)
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    main()
