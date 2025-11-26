from __future__ import annotations
from typing import Protocol, runtime_checkable, Any
import numpy as np

# Lightweight alias for public array-like numeric inputs/outputs used by methods.
# Keep this narrow (np.ndarray or float) so return types are predictable in plots.
Array = np.ndarray | float

@runtime_checkable
class Truth(Protocol):
    # Protocol for an oracle data-generating distribution F.
    # Any class conforming to this interface can serve as the “true” model
    # in simulations (e.g., NormalTruth, UniformTruth).
    def sample(self, n: int, seed: int | None = None) -> np.ndarray: ...
    def cdf_truth(self, t: Array) -> Array: ...
    def pdf_truth(self, x: Array) -> Array: ...

@runtime_checkable
class PredictiveState(Protocol):
    # Marker protocol for method-specific internal state (sufficient statistics,
    # posterior hyperparameters, cached arrays, RNG, etc.). We keep it open-ended
    # so each PredictiveMethod can choose its own state structure.
    ...

@runtime_checkable
class PredictiveMethod(Protocol):
    # Interface for online/sequential predictive estimators (e.g., DP/Polya,
    # kernel density with online updates, parametric conjugate updates, etc.).
    # The simulator will call these methods in order:
    #   state = init_state(**kwargs)
    #   for x in stream: state = update(state, x)
    #   cdf_est(state, t) / pdf_est(state, x) for evaluation and plotting.
    def init_state(self, **kwargs: Any) -> PredictiveState: ...
    def update(self, state: PredictiveState, x: float) -> PredictiveState: ...
    def cdf_est(self, state: PredictiveState, t: Array) -> Array: ...
    def pdf_est(self, state: PredictiveState, x: Array) -> Array: ...
