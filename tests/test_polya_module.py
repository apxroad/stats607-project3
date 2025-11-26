import numpy as np
from src.polya import PolyaSequenceModel, build_prefix, continue_urn_once, sample_prior_once

def test_shapes_and_bounds():
    # Basic sanity checks for the Pólya sequence helpers with Uniform(0,1) base.
    model = PolyaSequenceModel(alpha=3.0, base="uniform", rng=np.random.default_rng(0))
    
    # Prefix generation x_{1:20} from the urn
    pref = build_prefix(20, model)
    assert len(pref) == 20 and all(0<=x<=1 for x in pref)
    
    # Continuation to length 200 (same urn/history)
    cont = continue_urn_once(pref, model, 200)
    assert len(cont) == 200 and min(cont) >= 0 and max(cont) <= 1
    
    # Prior (unconditional) sequence of length 50
    prior = sample_prior_once(50, model)
    assert len(prior) == 50
