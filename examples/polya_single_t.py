from __future__ import annotations
import argparse, numpy as np, matplotlib.pyplot as plt
from scipy.stats import beta
from src.dgps import UniformTruth, NormalTruth

def main():
    ap = argparse.ArgumentParser(description="Single-t prior/posterior Beta overlays for DP/Polya")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--base", type=str, default="uniform", choices=["uniform","normal"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    G0  = UniformTruth(0.0, 1.0) if args.base == "uniform" else NormalTruth()
    p0  = float(G0.cdf_truth(args.t))

    # ----- Prior: P((−inf,t]) ~ Beta(alpha*p0, alpha*(1-p0))
    a0, b0 = args.alpha * p0, args.alpha * (1.0 - p0)
    prior_samples = beta.rvs(a0, b0, size=args.reps, random_state=rng)
    x = np.linspace(0, 1, 400)
    prior_pdf = beta.pdf(x, a0, b0)

    plt.figure()
    plt.hist(prior_samples, bins=40, density=True, alpha=0.5, edgecolor="black", label="MC (prior)")
    plt.plot(x, prior_pdf, linewidth=2, label=f"Beta(a={a0:.2f}, b={b0:.2f})")
    plt.axvline(p0, linestyle="--", label=f"G0(t)={p0:.3f}")
    plt.title(f"Prior: P((−∞,{args.t}]) under DP(α={args.alpha}, base={args.base})")
    plt.xlabel("P((−∞, t])"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/figures/prior_beta_t{args.t:+.2f}_a{args.alpha}_{args.base}.png"); plt.close()

    # ----- Posterior: indicators urn for K_n(t), Beta(a_post, b_post)
    post_draws = []
    for _ in range(args.reps):
        K = 0
        for i in range(1, args.n+1):
            p_i = (args.alpha * p0 + K) / (args.alpha + i - 1)
            K += rng.random() < p_i
        a_post = args.alpha * p0 + K
        b_post = args.alpha * (1.0 - p0) + (args.n - K)
        post_draws.append(beta.rvs(a_post, b_post, random_state=rng))
    post_draws = np.asarray(post_draws)

    # plot posterior histogram + reference Beta near average K
    Kbar = np.mean([sum((rng.random(args.n) < (args.alpha * p0 + np.arange(args.n)) / (args.alpha + np.arange(args.n)))) for _ in range(200)])
    a_bar = args.alpha * p0 + Kbar
    b_bar = args.alpha * (1.0 - p0) + (args.n - Kbar)
    post_pdf = beta.pdf(x, a_bar, b_bar)

    plt.figure()
    plt.hist(post_draws, bins=40, density=True, alpha=0.5, edgecolor="black", label="MC (posterior)")
    plt.plot(x, post_pdf, linewidth=2, label=f"Beta(~a={a_bar:.1f}, ~b={b_bar:.1f})")
    plt.axvline(p0, linestyle="--", label=f"G0(t)={p0:.3f}")
    plt.title(f"Posterior: P((−∞,{args.t}]) | n={args.n}, α={args.alpha}, base={args.base}")
    plt.xlabel("P((−∞, t])"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/figures/post_beta_t{args.t:+.2f}_a{args.alpha}_n{args.n}_{args.base}.png"); plt.close()

    print("Saved prior/posterior single-t figures in results/figures/")
if __name__ == "__main__":
    main()
