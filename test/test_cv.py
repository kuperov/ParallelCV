"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""

import unittest

import numpy as np
from jax import random

from ploo import LFO, LOO, KFold

from test.util import TestCase


class TestCrossValidation(TestCase):

    def test_loo_linear(self):
        l0 = LOO(100)
        self.assertEqual(l0.shape, (100,))
        self.assertEqual(list(range(100)), list(l0))
        l0_mask0 = l0.mask_for(next(iter(l0)))
        self.assertEqual(l0_mask0.shape, (100,))
        self.assertEqual(l0_mask0[0], 0.0)
        self.assertEqual(l0_mask0[1], 1.0)
        l1 = LOO((50,))
        self.assertEqual(l1.shape, (50,))
        self.assertEqual(list(l1), list(range(50)))

    def test_lfo(self):
        lf0 = LFO(shape=20, margin=10)
        self.assertEqual(list(range(10)), list(lf0))
        self.assertEqual(10, lf0.folds())
        lf1 = LFO(shape=(20,), margin=15)
        self.assertEqual(list(range(5)), list(lf1))
        self.assertEqual(5, lf1.folds())
        lf1_mask0 = lf1.mask_for(next(iter(lf1)))
        self.assertEqual(lf1_mask0.shape, (20,))
        self.assertEqual(lf1_mask0[15], 0.0)
        for i in [0, 14, 16, 19]:
            self.assertEqual(lf1_mask0[i], 1.0)

    def test_kfold_linear(self):
        rng_key = random.PRNGKey(seed=42)
        k0 = KFold(100, 5, rng_key=rng_key)
        self.assertEqual(k0.folds(), 5)
        self.assertEqual(list(k0), list(range(5)))
        mask0 = k0.mask_for(0)
        self.assertEqual(mask0.shape, (100,))
        self.assertEqual(np.sum(mask0), 80)


if __name__ == "__main__":
    unittest.main()
