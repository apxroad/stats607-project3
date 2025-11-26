from __future__ import annotations
import numpy as np
from typing import Tuple

def d_infty(c_est: np.ndarray, c_true: np.ndarray) -> float:
    """Sup norm on the grid.

    Parameters
    ----------
    c_est : np.ndarray
        Estimated CDF values evaluated on a common grid (shape (J,)).
    c_true : np.ndarray
        True CDF values on the same grid (shape (J,)).

    Returns
    -------
    float
        ||c_est - c_true||_∞ computed as max absolute deviation across the grid.

    Notes
    -----
    Assumes `c_est` and `c_true` have identical shape and are aligned pointwise.
    """
    return float(np.max(np.abs(c_est - c_true)))

def d_rmse(c_est: np.ndarray, c_true: np.ndarray) -> float:
    """Root-mean-square error on the grid.

    Parameters
    ----------
    c_est : np.ndarray
        Estimated CDF values on a common grid (shape (J,)).
    c_true : np.ndarray
        True CDF values on the same grid (shape (J,)).

    Returns
    -------
    float
        sqrt(mean((c_est - c_true)^2)) across grid points.

    Notes
    -----
    RMSE is scale-aware and penalizes larger deviations more heavily than L1.
    """
    return float(np.sqrt(np.mean((c_est - c_true) ** 2)))

def make_grid(J: int, tmin: float, tmax: float) -> np.ndarray:
    """Create an evaluation grid of length J on [tmin, tmax].

    Parameters
    ----------
    J : int
        Number of grid points (cast to int).
    tmin : float
        Left endpoint of the interval.
    tmax : float
        Right endpoint of the interval.

    Returns
    -------
    np.ndarray
        1D array of shape (J,) with equally spaced points from tmin to tmax.

    Notes
    -----
    The grid includes both endpoints (NumPy `linspace` default, inclusive).
    """
    return np.linspace(float(tmin), float(tmax), int(J))
