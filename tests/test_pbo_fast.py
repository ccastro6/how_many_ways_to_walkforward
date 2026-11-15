import numpy as np

from wfo.stats import pbo_cscv
from wfo.pbo_fast import pbo_cscv_fast


def test_fast_pbo_matches_reference():
    rng = np.random.default_rng(11)
    M = rng.standard_normal((1203, 9)) * 0.01 + rng.standard_normal(9) * 1e-3
    a = pbo_cscv(M, S=8)
    b = pbo_cscv_fast(M, S=8)
    # population (ddof=0) std in both implementations
    assert np.allclose(np.sort(a["logits"]), np.sort(b["logits"]))
    assert a["pbo"] == b["pbo"]
