"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module tests the classic radon model from posteriordb.
"""
import unittest
from test.util import TestCase

from jax import numpy as jnp

from ploo.model import _Posterior
from ploo.models.radon import RadonCountyIntercept
from ploo.util import DummyProgress


class TestRadonModel(TestCase):
    """Test definition and inference by MCMC of radon model"""

    def test_inference(self):
        model = RadonCountyIntercept()
        # sanity checks
        ipot = model.potential(model.initial_value())
        self.assertTrue(jnp.isfinite(ipot))
        # run mcmc
        post = model.inference(draws=100, warmup_steps=100, out=DummyProgress())
        self.assertIsInstance(post, _Posterior)
        self.assertEqual(post.total_divergences, 0)


if __name__ == "__main__":
    unittest.main()
