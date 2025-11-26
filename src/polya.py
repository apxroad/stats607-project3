from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal, Sequence

# Restrict allowable base names at type-check time.
BaseName = Literal["uniform","normal"]

@dataclass
class PolyaSequenceModel:
    """Blackwell–MacQueen Pólya urn, base G0 ∈ {Uniform(0,1), Normal(0,1)}.

    Parameters
    ----------
    alpha : float
        Concentration parameter (α > 0). Larger α ⇒ stronger pull toward G0.
    base : {"uniform","normal"}
        Choice of base distribution G0. "uniform" uses U(0,1); "normal" uses N(0,1).
    rng : np.random.Generator | None
        Optional external RNG for reproducibility and stream control. If None,
        a fresh default_rng() is created on demand (not reproducible across calls).
    """
    alpha: float = 5.0
    base: BaseName = "uniform"
    rng: np.random.Generator | None = None

    def _rng(self) -> np.random.Generator:
        """Return the active RNG (use provided one if available)."""
        return self.rng or np.random.default_rng()

    # base draw
    def P0(self) -> float:
        """Draw P0 from the base distribution G0 (prior)."""
        r = self._rng()
        if self.base == "uniform":
            return float(r.random())
        elif self.base == "normal":
            return float(r.standard_normal())
        raise ValueError(f"unknown base={self.base}")

    # one-step predictive given past x_{1:n}
    def Pn(self, n: int, history: Sequence[float]) -> float:
        """Draw X_{n+1} from the DP predictive given history x_{1:n}.

        Mixture form:
            with prob α/(α+n): draw from G0;
            otherwise: pick a past value uniformly from {x1,...,xn}.
        """
        r = self._rng()
        if n != len(history):
            raise ValueError("n must equal len(history)")
        w_base = self.alpha / (self.alpha + n)
        if r.random() < w_base:
            return self.P0()
        # sample a past value uniformly
        j = r.integers(0, n)
        return float(history[j])

# ----- small helpers used by both CLIs -----

def build_prefix(n_obs: int, model: PolyaSequenceModel) -> list[float]:
    """Generate x_{1:n_obs} from the urn (sequentially using predictive)."""
    x = [model.P0()]              # start with a base draw for X1
    for m in range(1, n_obs):
        x.append(model.Pn(m, x))  # append X_{m+1} | x_{1:m}
    return x

def continue_urn_once(prefix: list[float], model: PolyaSequenceModel, M: int) -> list[float]:
    """Fix x_{1:n} and continue to length M using the predictive.

    Parameters
    ----------
    prefix : list[float]
        Existing sequence (treated as observed history).
    model : PolyaSequenceModel
        Pólya urn model providing predictive draws.
    M : int
        Target total length after continuation (M ≥ len(prefix)).

    Returns
    -------
    list[float]
        Extended sequence of length M.
    """
    x = list(prefix)
    n = len(x)
    for m in range(n, M):
        x.append(model.Pn(m, x))
    return x

def sample_prior_once(M: int, model: PolyaSequenceModel) -> list[float]:
    """Unconditional Pólya sequence of length M (from prior).

    Starts with a base draw, then proceeds with predictive updates.
    """
    x = [model.P0()]
    for m in range(1, M):
        x.append(model.Pn(m, x))
    return x
