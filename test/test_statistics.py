"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest
from test.util import TestCase, fixture

import arviz as az
from jax import numpy as jnp

from ploo.statistics import ess, split_rhat


class TestStatistics(TestCase):
    def setUp(self) -> None:
        self.gaussian_post = az.from_netcdf(fixture("gaussian.nc"))
        self.mu = jnp.array(self.gaussian_post.posterior.mu)
        self.sigma = jnp.array(self.gaussian_post.posterior.sigma)

    def test_split_Rhat(self):
        # ordinary Gelman et al (2013) split Rhat
        # compare to 6dp because we're using 32 bit arithmetic
        split = az.rhat(self.gaussian_post, method="split")
        sr_mu = split_rhat(self.mu)
        self.assertAlmostEqual(float(sr_mu), float(split["mu"]), places=6)
        sr_sigma = split_rhat(self.sigma)
        self.assertAlmostEqual(float(sr_sigma), float(split["sigma"]), places=6)

    def test_ess(self):
        # ess as described in Vehtari et al 2021, Geyer 2011
        # compare to within 10% because we're using a slightly
        # different, and a somewhat sketchy, algorithm
        azess = az.ess(self.gaussian_post, method="mean")
        sr_mu = ess(self.mu)
        self.assertClose(float(sr_mu), float(azess["mu"]), rtol=0.1)
        sr_sigma = ess(self.sigma)
        self.assertClose(float(sr_sigma), float(azess["sigma"]), rtol=0.1)


if __name__ == "__main__":
    unittest.main()
