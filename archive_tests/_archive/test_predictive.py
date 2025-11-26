import numpy as np
from src.methods import EmpiricalPredictive

def test_ecdf_monotone_and_bounds():
    m = EmpiricalPredictive(max_n=10)
    st = m.init_state()
    for z in [0.0, 1.0, -1.0, 2.0, 0.5]:
        st = m.update(st, z)
    grid = np.array([-2,-1,0,0.5,1,2,3], dtype=float)
    c = m.cdf_est(st, grid)
    assert np.all((0.0 <= c) & (c <= 1.0))
    assert np.all(np.diff(c) >= -1e-12)
    assert c[0] == 0.0 and c[-1] == 1.0
