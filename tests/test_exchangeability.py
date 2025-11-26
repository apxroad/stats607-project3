import numpy as np
from src.methods import PolyaPredictive
from src.metrics import make_grid
from src.dgps import UniformTruth

def test_polya_order_invariance():
    # Exchangeability check: the DP/Pólya predictive should be invariant
    # to the order of observations (depends only on the multiset of data).
    n = 200
    G0 = UniformTruth(0.0, 1.0)
    x  = G0.sample(n, seed=42)
    grid = make_grid(J=50, tmin=0.0, tmax=1.0)

    m = PolyaPredictive(alpha=5.0, base="uniform")
    
    # Forward order
    st1 = m.init_state()
    for xi in x:
        st1 = m.update(st1, float(xi))
    c1 = m.cdf_est(st1, grid)

    # Reverse order
    st2 = m.init_state()
    for xi in x[::-1]:
        st2 = m.update(st2, float(xi))
    c2 = m.cdf_est(st2, grid)

    # Exact equality should hold since updates are deterministic given the same set.
    assert np.array_equal(c1, c2)
