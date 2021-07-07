"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest

from jax import numpy as jnp
from scipy import stats as st
from scipy.integrate import quad

from ploo.transforms import LogTransform


class TestTransforms(unittest.TestCase):
    def test_log_transform(self):
        log_t = LogTransform()

        def lpdf(sig_sq):
            return st.gamma(2.0, 2.0).logpdf(sig_sq)

        # pdf defined on half real line
        def pdf(sig_sq):
            return jnp.exp(lpdf(sig_sq))

        # pdf defined on full real line
        def tfm_pdf(tau):
            sig_sq = log_t.to_constrained(tau)
            ldet = log_t.log_det(sig_sq)
            return jnp.exp(lpdf(sig_sq) + ldet)

        vol, err = quad(pdf, -jnp.inf, jnp.inf)
        self.assertAlmostEqual(1.0, vol, places=6)
        vol, err = quad(pdf, 0, jnp.inf)
        self.assertAlmostEqual(1.0, vol, places=6)
        vol, err = quad(tfm_pdf, -5, 5)
        self.assertAlmostEqual(1.0, vol, places=6)


if __name__ == "__main__":
    unittest.main()
