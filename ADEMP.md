# ADEMP: Prediction-Based Uncertainty Quantification for Exchangeable Sequences  
*(Reproduction and extension of Fortini & Petrone, 2023)*

---

## **Aims**

We aim to reproduce and extend the simulation experiments from **Fortini & Petrone (2023)** and **Fortini & Petrone (2024)** to illustrate how *Bayesian uncertainty quantification* can be expressed entirely through *predictive rules* under exchangeability.

Specifically, we investigate:

1. How the predictive distributions \( P_m \) in a Pólya sequence evolve as data accumulate.  
2. Whether the empirical distributions \( P_m(t) \) approximate the random limit \( \tilde{P}(t) \) (the “directing random measure”).  
3. How well the **asymptotic Gaussian approximation** (Proposition 2.6 in **Fortini & Petrone (2024)**)
   \[
   \tilde{P}(A) \mid x_{1:n} \approx \mathcal{N}\!\left(P_n(A), \frac{V_{n,A}}{n}\right)
   \]
   captures posterior uncertainty.  

---

## **Data-Generating Mechanisms**

We simulate data from the **Blackwell–MacQueen Pólya urn**, equivalently a **Dirichlet Process (DP) prior** \( \mathrm{DP}(\alpha, G_0) \) on the random distribution \( F \):

\[
X_{m+1} \mid X_{1:m} \sim \frac{\alpha}{\alpha + m} G_0 + \frac{1}{\alpha + m}\sum_{i=1}^m \delta_{X_i},
\]
where \( G_0 = \mathrm{Unif}(0,1) \).

At each step \( m \), we record predictive probabilities:
\[
P_m(t) = \Pr(X_{m+1} \le t \mid X_{1:m}) = \frac{\alpha t + K_m(t)}{\alpha + m},
\]
where \( K_m(t) = \sum_{i=1}^m \mathbf{1}\{X_i \le t\} \).

**Parameter settings**

| Parameter | Description | Values |
|------------|-------------|---------|
| \( \alpha \) | DP concentration | 5 |
| \( G_0 \) | Base measure | Uniform(0, 1) |
| \( n \) | Observed sample size | {100, 500, 1000} |
| \( t \) | Thresholds for predictive CDF | {0.25, 0.5, 0.75} |
| \( M \) | Total urn length (for MC replication) | up to several hundred |
| \( L \) | Max path length for diagnostics | varies (≈ 1000) |

We optionally swap \( G_0 \) for Normal(0, 1) in robustness checks.

---

## **Estimands / Targets**

For each threshold \( t \), the target of inference is the **random mass**
\[
\tilde{F}(t) = \tilde{P}((−\infty, t]) \sim \mathrm{Beta}(\alpha G_0(t),\, \alpha(1 - G_0(t))).
\]
We study:
- The convergence \( P_m(t) \to \tilde{F}(t) \) as \( m \) grows.
- The asymptotic distribution of \( \sqrt{n}(\tilde{F}(t) - P_n(t)) \).

---

## **Methods**

We implement the **prediction-based Monte Carlo algorithm** (Algorithm 1 of Fortini–Petrone):

1. **Prior simulation:**  
   - Generate predictive sequence \( X_{1:M} \) from the rule above.  
   - For each threshold \( t \), record  
     \(\widehat{P}_M(t) = \frac{1}{M}\sum_{i=1}^M \mathbf{1}\{X_i \le t\}\).  
   - Repeat over many Monte Carlo replicates to approximate the prior density of \( \tilde{F}(t) \).

2. **Posterior continuation:**  
   - Fix an observed prefix \( x_{1:n} \).  
   - Continue the urn to length \( M \), re-recording \( \widehat{P}_M(t) \).  
   - Compare histogram to theoretical \( \mathrm{Beta}(\alpha t + k_n,\, \alpha(1 - t) + (n - k_n)) \).

3. **Predictive learning diagnostics:**  
   - Track \( P_m(t) \) vs. \( m \) (“predictive path”) and compute sup-norm or RMSE distances to the limit \( \tilde{F}(t) \).  
   - Plot convergence curves and predictive trajectories to visualize learning stability.

4. **Asymptotic normality check:**  
   - Compute
     \[
     Z = \frac{P_n(t) - \widehat{F}(t)}{\sqrt{V_{n,t}/n}},
     \]
     where \( V_{n,t} = \frac{1}{n}\sum_{k=1}^n k^2\Delta_{t,k}^2 \).  
   - Examine pooled \( Z \)-histograms for approximate standard normality.

---

## **Performance Measures**

| Measure | Description | Purpose |
|----------|-------------|----------|
| **Predictive convergence** | Sup-norm / RMSE between \( P_m \) and \( \tilde{F} \) | Learning rate |
| **Histogram alignment** | Visual | Prior/posterior validation |
| **Path flatness** | Visual | Shrinking predictive uncertainty |
| **Normality of Z** | Visual | Support for asymptotic theorem |

---

## **Simulation Design Summary**

### Global Parameters

| Parameter | Values | Purpose |
|-----------|--------|---------|
| **α** (concentration) | 1, 5, 20 | Controls prior strength: smaller α = more flexible; larger α = closer to base measure |
| **n** (observations) | 0, 100, 500, 1000 | Sample size: n=0 shows pure prior; larger n shows posterior learning |
| **t** (thresholds) | 0.25, 0.5, 0.75 | CDF evaluation points for predictive probability P̃((-∞, t]) |
| **Base measure G₀** | Uniform(0,1) | Prior centering distribution (Normal not used in current implementation) |
| **Random seed** | 2025 | Ensures full reproducibility across all experiments |

### Experiment-Specific Configurations

#### **Part A: Prior/Posterior Panels**
*Goal: Visualize how posterior predictive distributions evolve with data*

| Setting | Configuration | Purpose |
|---------|---------------|---------|
| **Prior panels (n=0)** | M ∈ {10, 100, 1000, 4000} | Show convergence of prior predictive as continuation length increases |
| **Posterior panels** | n ∈ {100, 500, 1000}, M = 4000 | Demonstrate posterior learning with different sample sizes |
| **MC replicates (N)** | 2000 | Monte Carlo trajectories per (α, t, n, M) combination for histogram |

*Each panel compares empirical histogram vs. analytical Beta density*

---

#### **Part B: Convergence Diagnostics**
*Goal: Assess how fast predictive probabilities converge to posterior*

| Setting | Value | Purpose |
|---------|-------|---------|
| **Sample size** | n = 1000 | Fixed observation count |
| **Concentration** | α = 5.0 | Moderate prior strength |
| **Metrics tracked** | d^(∞) (sup-norm), RMSE | Measure distance between predictive and limit distribution |
| **Predictive paths** | P_m(t) for m = 1, ..., M | Track convergence trajectory over continuation steps |

*Generates convergence plots and path visualizations*

---

#### **Part C: Asymptotic Validation** 
*Goal: Verify theoretical asymptotic normality (Proposition 2.6)*

| Setting | Value | Purpose |
|---------|-------|---------|
| **Sequences per config** | M = 400 | Independent Pólya sequences |
| **Total observations** | L = 50,000 | Pooled across all sequences for Z statistic |
| **Sample sizes tested** | n ∈ {100, 500, 1000} | Check asymptotic behavior at different scales |
| **Statistics computed** | Pooled Z | Test if Z ~ N(0,1)|

*Outputs histogram of pooled Z overlaid with N(0,1) density for visual goodness-of-fit*

---

### Summary Table (Quick Reference)

| Part | n values | α values | M values | Output |
|------|----------|----------|----------|--------|
| A (Prior) | 0 | 1, 5, 20 | 10, 100, 1000, 4000 | 4 panel grids (3×3 subplots each) |
| A (Posterior) | 100, 500, 1000 | 1, 5, 20 | 4000 | 3 panel grids (3×3 subplots each) |
| B | 1000 | 5.0 | Variable | Convergence curves + predictive paths |
| C | 100, 500, 1000 | 5.0 | 400 | Pooled Z histogram + summary statistics |

---

## **Expected Findings**

- Predictive histograms align with theoretical Beta densities for both prior and posterior phases.  
- Predictive paths \( P_m(t) \) stabilize around random limits \( \tilde{F}(t) \).  
- The standardized differences \( Z \) follow an approximate \( \mathcal{N}(0,1) \), confirming Theorem 3.1’s asymptotic normality.  
- Increasing \( \alpha \) or \( n \) tightens predictive uncertainty (narrower credible intervals).  


---

## **Reproducibility Notes**

- Random seeds fixed in all scripts (`numpy.random.default_rng(seed=123)`).  
- All results are generated via a Makefile pipeline:  
  ```bash
  make simulate   # Run Monte Carlo replicates
  make analyze    # Compute metrics, summarize
  make figures    # Generate plots A–C
  make all
  ```
- Outputs cached under `results/raw/` and figures in `results/figures/`.  
