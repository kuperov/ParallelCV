"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module tests the classic radon model from posteriordb.
"""
import unittest
from test.util import TestCase

import chex
from jax import numpy as jnp
from jax import random

from diag import KFold
from diag.model import CrossValidation, _Posterior
from diag.models.radon import RadonCountyIntercept


class TestRadonModel(TestCase):
    """Test definition and inference by MCMC of radon model"""

    def test_inference_and_cv(self):
        """Check a few iterations of MCMC. This seems way slower than it should be."""
        chex.clear_trace_counter()
        model = RadonCountyIntercept()
        # sanity checks
        ipot = model.cv_potential(model.initial_value(), 1.0)
        self.assertTrue(jnp.isfinite(ipot))
        # run mcmc
        post = model.inference(draws=100, warmup_steps=100)
        self.assertIsInstance(post, _Posterior)
        self.assertEqual(post.total_divergences, 0)
        # cross-validate by k-fold
        rng_key = random.PRNGKey(seed=42)
        kfold = KFold(shape=model.log_radon.shape, k=5, rng_key=rng_key)
        cross_validation = post.cross_validate(scheme=kfold)
        self.assertIsInstance(cross_validation, CrossValidation)


if __name__ == "__main__":
    unittest.main()
