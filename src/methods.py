from __future__ import annotations

import numpy as np
from typing import Any
from .interfaces import PredictiveMethod, PredictiveState, Array
from .dgps import NormalTruth, UniformTruth

class PolyaPredictive(PredictiveMethod):
    r"""
    Dirichlet–process (Pólya sequence) one-step predictive for CDFs.

    For any threshold t and observed data x_{1:n}, with base G0 and concentration α>0,
        \tilde P_n((−∞, t]) = (α G0(t) + K_n(t)) / (α + n),
    where K_n(t) = \sum_{i=1}^n 1{x_i ≤ t}.

    Notes
    -----
    - This class estimates the **CDF**; `pdf_est` returns NaNs because the DP
      predictive has discrete atoms (not suitable for log-density scoring).
    - `base` controls the prior base CDF G0:
        * "normal"  → NormalTruth(mean=0, sd=1)
        * "uniform" → UniformTruth(a=0, b=1) (default)

    Implementation details
    ----------------------
    - State stores only the observed values (exchangeability ⇒ order irrelevant).
    - CDF evaluation is vectorized in `t` (supports scalar or array thresholds).
    - When n=0 (no data), the predictive reduces to the base CDF G0(t).
    - Complexity per `cdf_est` call is O(n * |t|) due to pairwise comparisons.
    """

    def __init__(self, alpha: float = 5.0, base: str = "normal"):
        # α > 0 ensures a proper DP prior; larger α ⇒ stronger pull to G0.
        assert alpha > 0, "alpha must be positive"
        self.alpha = float(alpha)
        # Choose the oracle base distribution used for G0(t).
        if base == "normal":
            self.base = NormalTruth()
        elif base == "uniform":
            self.base = UniformTruth(0.0, 1.0)
        else:
            raise ValueError(f"Unknown base '{base}'. Use 'normal' or 'uniform'.")

    # ---- PredictiveMethod API ----
    def init_state(self, **kwargs: Any) -> PredictiveState:
        # Minimal sufficient state for the DP predictive: the sample x_{1:n}.
        # Stored as a list for cheap appends; converted to ndarray on demand.
        return {"xs": []}

    def update(self, state: dict, x: float) -> dict:
        # Online update: append the new observation; maintains exchangeability.
        state["xs"].append(float(x))
        return state

    def cdf_est(self, state: dict, t: Array) -> Array:
        # Compute \tilde P_n((−∞, t]) for scalar or vector 't'.
        xs = np.asarray(state["xs"], dtype=float)
        n = xs.size
        t_arr = np.asarray(t, dtype=float)

        # counts K_n(t): number of observed x ≤ t
        # Vectorized branch when t is an array; scalar branch otherwise.
        if n == 0:
            counts = 0.0    # with no data, K_n(t)=0 for all t
        else:
            counts = (
                (xs[:, None] <= t_arr[None, :]).sum(axis=0)
                if t_arr.ndim
                else (xs <= t_arr).sum()
            )

        # Base CDF G0(t) from the chosen oracle truth.
        g0 = self.base.cdf_truth(t_arr)
        # Posterior predictive CDF: convex combination of G0 and empirical CDF.
        return (self.alpha * g0 + counts) / (self.alpha + n)

    def pdf_est(self, state: dict, x: Array) -> Array:
        # DP predictive is a mixture with point masses at observed xs;
        # density is not well-defined as a smooth function → return NaNs.
        x_arr = np.asarray(x, dtype=float)
        return np.full_like(x_arr, np.nan, dtype=float) if getattr(x_arr, "ndim", 0) else np.nan
