# src/simulation.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from .dgps import NormalTruth, UniformTruth
from .metrics import d_infty, d_rmse, make_grid


def _build_method(method_name: str, n: int, **params):
    """Factory for predictive methods used in the stream simulation.

    Parameters
    ----------
    method_name : str
        Identifier of the predictive method (currently only "polya_dp").
    n : int
        Sample size (forwarded to methods that might need it).
    **params
        Additional keyword args forwarded to the method constructor.

    Returns
    -------
    (method, name) : tuple
        Instantiated method object and a short method label for output.
    """
    if method_name == "polya_dp":
        from .methods import PolyaPredictive
        return PolyaPredictive(**params), "polya_dp"
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_stream(
    method_name: str,
    n: int,
    J: int,
    tmin: float,
    tmax: float,
    record_every: int,
    seed: int,
    out_path: str,
    **params,
) -> str:
    """
    Simulate an online predictive-evaluation stream and save results.

    Workflow
    --------
    1) Draw i.i.d. data X_1..X_n from the *truth* (Uniform or Normal), chosen to
       match the method's base (keeps exchangeable benchmarking consistent).
    2) For each time i:
         - BEFORE seeing X_i, evaluate the method's one-step predictive CDF.
         - Record the PIT at X_i and distance metrics on a fixed grid (thinned).
         - Update the method with X_i.
    3) Save a row per i to a parquet file.

    Parameters
    ----------
    method_name : str
        Predictive method identifier (e.g., "polya_dp").
    n : int
        Length of the stream.
    J : int
        Number of grid points for distance evaluation.
    tmin, tmax : float
        Grid interval endpoints for CDF comparison.
    record_every : int
        Thinning interval for recording distances (smaller → more frequent).
    seed : int
        RNG seed for reproducibility of the data stream.
    out_path : str
        Destination parquet path.
    **params
        Extra parameters passed to the method constructor
        (e.g., alpha=..., base="uniform"/"normal").

    Returns
    -------
    str
        The `out_path` that was written.
    """
    rng = np.random.default_rng(seed)

    # Pick the truth to match the method's base (exchangeable setup).
    # This ensures G0 in the method aligns with the DGP for clean comparisons.
    base_name = params.get("base", "uniform")
    if base_name == "uniform":
        truth = UniformTruth(0.0, 1.0)
    else:
        truth = NormalTruth()

    # i.i.d. data from the oracle truth
    x = truth.sample(n=n, seed=seed)

    # Fixed evaluation grid and oracle CDF values on that grid
    t_grid = make_grid(J, tmin, tmax)
    c_true_grid = truth.cdf_truth(t_grid)

    # Build method (e.g., Pólya DP) and initialize its internal state
    method, mname = _build_method(method_name, n, **params)
    try:
        state = method.init_state(max_n=n)  # some methods may accept this kwarg
    except TypeError:
        state = method.init_state()

    recs = []
    for i in range(n):
        x_i = x[i]

        # Evaluate BEFORE observing x_i (one-step predictive, proper online eval)
        if i == 0:
            # No predictive available before any data; mark as NaN
            pit_i = np.nan
            d_inf_i = np.nan
            d_rmse_i = np.nan
        else:
            # PIT at the realized x_i using the current state (pre-update)
            pit_i = float(method.cdf_est(state, x_i))

            # Distances on grid, thinned by `record_every` and at the final step
            if (i % record_every) == 0 or i == n - 1:
                c_est_grid = np.asarray(method.cdf_est(state, t_grid), dtype=float)
                d_inf_i = d_infty(c_est_grid, c_true_grid)
                d_rmse_i = d_rmse(c_est_grid, c_true_grid)
            else:
                d_inf_i = np.nan
                d_rmse_i = np.nan

        # Append a record for this time step
        recs.append((i, mname, x_i, pit_i, d_inf_i, d_rmse_i, seed, n))

        # Online update with the new observation
        state = method.update(state, float(x_i))

    # Materialize the log as a tidy DataFrame
    df = pd.DataFrame(
        recs,
        columns=["i", "method", "x_i", "pit", "d_infty", "d_rmse", "seed", "n"],
    )

    # Ensure output directory exists, then write parquet (fastparquet engine)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="fastparquet")
    return out_path
