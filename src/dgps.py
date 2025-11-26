from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Iterable, overload
import numpy as np
from scipy.special import erf  

# Accept either NumPy arrays or Python floats in public APIs.
# Returned values mirror the input type whenever reasonable.
ArrayLike = Union[np.ndarray, float]

# Re-exported symbols to keep the public surface minimal/clean.
# External modules should import from here instead of internal helpers.
__all__ = [
    "NormalTruth",
    "UniformTruth",
    "sample_truth",
    "cdf_truth",
    "pdf_truth",
]

# ----------------------------
# Standard Normal helpers
# ----------------------------

def _phi(x: ArrayLike) -> ArrayLike:
    """Standard normal PDF.

    Parameters
    ----------
    x : ArrayLike
        Scalar or array-like input. Internally cast to float array.

    Returns
    -------
    ArrayLike
        φ(x) with the same scalar/array flavor as input.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

def _Phi(x: ArrayLike) -> ArrayLike:
    """Standard normal CDF.

    Notes
    -----
    Uses ``erf`` for numerical stability vs direct integration.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

# ----------------------------
# Truth classes
# ----------------------------

@dataclass
class NormalTruth:
    """
    Oracle truth F for a Normal(mean, sd) distribution.
    Defaults to N(0,1) (the baseline in other experiments
    that we will explore in further projects).

    Attributes
    ----------
    mean : float
        Location parameter μ.
    sd : float
        Standard deviation σ (> 0 expected; not enforced here).
    """
    mean: float = 0.0
    sd: float = 1.0

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw n iid samples from N(mean, sd^2) with optional seeding.

        Parameters
        ----------
        n : int
            Sample size (will be cast to int).
        seed : int | None
            If provided, used to seed a fresh Generator for reproducibility.

        Returns
        -------
        np.ndarray
            Shape (n,) float array.
        """
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(int(n))
        return self.mean + self.sd * z

    def cdf_truth(self, t: ArrayLike) -> ArrayLike:
        """Evaluate the true CDF F(t) for the configured Normal.

        Fast path for standard normal to avoid extra ops.
        """
        t = np.asarray(t, dtype=float)
        if self.sd == 1.0 and self.mean == 0.0:
            return _Phi(t)
        z = (t - self.mean) / self.sd
        return _Phi(z)

    def pdf_truth(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the true PDF f(x) for the configured Normal.

        Notes
        -----
        Applies change-of-variables scaling by 1/sd when not standard.
        """
        x = np.asarray(x, dtype=float)
        if self.sd == 1.0 and self.mean == 0.0:
            return _phi(x)
        z = (x - self.mean) / self.sd
        return _phi(z) / self.sd


@dataclass
class UniformTruth:
    """
    Oracle truth F for a Uniform(a, b) distribution.
    Defaults to U(0,1) to match classic DP/Pólya examples, the main concern
    in this project.

    Attributes
    ----------
    a : float
        Lower bound (can equal b for a degenerate case; not enforced here).
    b : float
        Upper bound (must satisfy b > a in typical use).
    """
    a: float = 0.0
    b: float = 1.0

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Draw n iid samples from Uniform(a, b)."""
        rng = np.random.default_rng(seed)
        return rng.uniform(self.a, self.b, size=int(n))

    def cdf_truth(self, t: ArrayLike) -> ArrayLike:
        """Evaluate the true CDF on [a,b].

        Implementation
        --------------
        Uses the closed form (t-a)/(b-a) with clipping outside [a,b].
        """
        t = np.asarray(t, dtype=float)
        a, b = float(self.a), float(self.b)
        out = (t - a) / (b - a)
        return np.clip(out, 0.0, 1.0)

    def pdf_truth(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the true PDF: 1/(b-a) on [a,b], 0 elsewhere.

        Returns
        -------
        ArrayLike
            Matches array shape; returns a scalar float if the input was scalar.
        """
        x = np.asarray(x, dtype=float)
        a, b = float(self.a), float(self.b)
        inside = (x >= a) & (x <= b)
        val = np.zeros_like(x, dtype=float)
        val[inside] = 1.0 / (b - a)
        # Return scalar if input was scalar
        if np.ndim(x) == 0:
            return float(val)
        return val

# ----------------------------
# Convenience functions (Normal by default)
# ----------------------------

def sample_truth(n: int, seed: int | None = None) -> np.ndarray:
    """
    Backward-compatible helper used in earlier steps.
    Draws from Normal(0,1). For Uniform base, instantiate UniformTruth() directly.

    Rationale
    ---------
    Keeps older code working while the project transitioned to explicit
    Truth objects. Prefer explicit objects in new code for clarity.
    """
    return NormalTruth().sample(n=n, seed=seed)

def cdf_truth(t: ArrayLike) -> ArrayLike:
    """
    Backward-compatible helper: CDF of Normal(0,1).
    For Uniform base, use UniformTruth().cdf_truth(t).

    See also
    --------
    NormalTruth.cdf_truth
    """
    return NormalTruth().cdf_truth(t)

def pdf_truth(x: ArrayLike) -> ArrayLike:
    """
    Backward-compatible helper: PDF of Normal(0,1).
    For Uniform base, use UniformTruth().pdf_truth(x).

    See also
    --------
    NormalTruth.pdf_truth
    """
    return NormalTruth().pdf_truth(x)
