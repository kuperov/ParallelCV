"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

This module tests the classic radon model from posteriordb.
"""
import unittest
from test.util import TestCase

from ploo.model import _Posterior
from ploo.models.radon import RadonCountyIntercept


class TestRadonModel(TestCase):
    """Test definition and inference by MCMC of radon model"""

    def test_inference(self):
        model = RadonCountyIntercept()
        post = model.inference()
        self.assertIsInstance(post, _Posterior)
        self.assertEqual(post.num_divergences, 0)


if __name__ == "__main__":
    unittest.main()
