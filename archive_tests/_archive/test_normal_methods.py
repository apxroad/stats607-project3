import numpy as np
from src.methods import NormalPluginPredictive, StudentTPredictive

def _feed(sym_values):
    # helper: feed a symmetric sample around 0
    m1 = NormalPluginPredictive()
    s1 = m1.init_state()
    m2 = StudentTPredictive()
    s2 = m2.init_state()
    for z in sym_values:
        s1 = m1.update(s1, z)
        s2 = m2.update(s2, z)
    return (m1, s1), (m2, s2)

def test_normal_plugin_basic_behavior():
    (m, st), _ = _feed([-2,-1,0,1,2])
    # around zero, predictive CDF should be near 0.5
    c0 = m.cdf_est(st, 0.0)
    assert 0.3 < c0 < 0.7
    # pdf positive and finite
    assert np.isfinite(m.pdf_est(st, 0.0))
    # vectorized cdf works and is monotone
    grid = np.array([-3,-1,0,1,3], float)
    c = m.cdf_est(st, grid)
    assert np.all(np.diff(c) >= -1e-12)

def test_student_t_basic_behavior():
    _, (m, st) = _feed([-2,-1,0,1,2])
    c0 = m.cdf_est(st, 0.0)
    assert 0.3 < c0 < 0.7
    assert np.isfinite(m.pdf_est(st, 0.0))
    grid = np.array([-3,-1,0,1,3], float)
    c = m.cdf_est(st, grid)
    assert np.all(np.diff(c) >= -1e-12)
