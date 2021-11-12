"""diag is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""

import unittest
from test.util import TestCase

import numpy as np
from jax import random

from diag import LFO, LOO, KFold
from diag.schemes import LGO


class TestCrossValidationSchemes(TestCase):
    """Basic checks on cross-validation subclasses"""

    def test_loo_linear(self):
        """Check folds and shape of linear LOO scheme"""
        linear_1 = LOO(100)
        self.assertEqual(linear_1.shape, (100,))
        self.assertEqual(linear_1.folds, 100)
        self.assertEqual(list(range(100)), list(linear_1))
        linear_1_coords = linear_1.pred_indexes()
        self.assertEqual(linear_1_coords.coords.shape, (100, 1, 1))
        self.assertClose(linear_1_coords.lengths, np.array([1] * 100))
        linear_1_mask0 = linear_1.mask_for(next(iter(linear_1)))
        self.assertEqual(linear_1_mask0.shape, (100,))
        self.assertEqual(linear_1_mask0[0], 0.0)
        self.assertEqual(linear_1_mask0[1], 1.0)
        linear_1_masks = linear_1.mask_array()
        self.assertEqual(linear_1_masks.shape, (100, 100))
        self.assertClose(np.ones((100, 100)) - np.eye(100), linear_1_masks)
        linear_1_summary = linear_1.summary_array()
        self.assertEqual(linear_1_summary.shape, (100, 100))
        self.assertClose(np.ones((100, 100)) - 2 * np.eye(100), linear_1_summary)
        linear_2 = LOO((50,))
        self.assertEqual(linear_2.shape, (50,))
        self.assertEqual(list(linear_2), list(range(50)))

    def test_loo_multidim(self):
        """Check folds and shape of multidimensional LOO scheme"""
        multi_1 = LOO((20, 30))
        self.assertEqual(multi_1.shape, (20, 30))
        self.assertEqual(multi_1.folds, 20 * 30)
        self.assertEqual(len(list(multi_1)), multi_1.folds)
        multi_1_coords = multi_1.pred_indexes()
        self.assertEqual(
            multi_1_coords.coords.shape, (600, 1, 2)
        )  # folds*npred*data dim
        self.assertClose(multi_1_coords.lengths, [1] * 600)
        multi_1_fold0 = next(iter(multi_1))
        multi_1_mask0 = multi_1.mask_for(multi_1_fold0)
        self.assertEqual(multi_1_mask0.shape, (20, 30))
        self.assertEqual(np.sum(multi_1_mask0), multi_1.folds - 1)
        multi_1_summary = multi_1.mask_array()
        self.assertEqual(multi_1_summary.shape, (600, 20, 30))
        self.assertEqual(np.sum(multi_1_summary), 20 * 30 * (20 * 30 - 1))

    def test_lfo(self):
        """Check folds and shape of linear LFO scheme"""
        lfo_1 = LFO(shape=20, margin=10)
        self.assertEqual(list(range(10)), list(lfo_1))
        self.assertEqual(10, lfo_1.folds)
        lfo_2 = LFO(shape=(20,), margin=15)
        self.assertEqual(list(range(5)), list(lfo_2))
        self.assertEqual(5, lfo_2.folds)
        lfo_2_mask0 = lfo_2.mask_for(next(iter(lfo_2)))
        self.assertEqual(lfo_2_mask0.shape, (20,))
        for i in [0, 14]:
            self.assertEqual(lfo_2_mask0[i], 1.0)
        for i in [15, 19]:
            self.assertEqual(lfo_2_mask0[i], 0.0)

    def test_kfold_linear(self):
        """Check folds and shape of linear k-fold scheme"""
        rng_key = random.PRNGKey(seed=42)
        kfold_1 = KFold(100, 5, rng_key=rng_key)
        self.assertEqual(kfold_1.folds, 5)
        self.assertEqual(list(kfold_1), list(range(5)))
        kfold_1_mask0 = kfold_1.mask_for(0)
        self.assertEqual(kfold_1_mask0.shape, (100,))
        self.assertEqual(np.sum(kfold_1_mask0), 80)
        kfold_1_coords = kfold_1.pred_indexes()
        self.assertClose(kfold_1_coords.lengths, [20] * 5)
        self.assertEqual(kfold_1_coords.coords.shape, (5, 20, 1))
        self.assertClose(
            [len(cs) for cs in kfold_1_coords.coords], kfold_1_coords.lengths
        )

    def test_kfold_multidim(self):
        """Check folds and shape of multidimensional K-fold scheme"""
        rng_key = random.PRNGKey(seed=42)
        kfold_1 = KFold((20, 10), 5, rng_key=rng_key)
        self.assertEqual(kfold_1.folds, 5)
        self.assertEqual(list(kfold_1), list(range(5)))
        kfold_1_fold0 = next(iter(kfold_1))
        kfold_1_fold0_mask = kfold_1.mask_for(kfold_1_fold0)
        self.assertEqual(kfold_1_fold0_mask.shape, (20, 10))
        self.assertEqual(np.sum(kfold_1_fold0_mask), 4 * 20 * 10 / 5)

    def test_lgo(self):
        """Test leave-group-out with made-up group identifiers `grps`"""
        grps = [1] * 5 + [2] * 3 + [3] * 2
        lgo_1 = LGO(10, grps)
        self.assertEqual(lgo_1.shape, (10,))
        self.assertEqual(lgo_1.folds, 3)
        lgo_1_summ = lgo_1.mask_array()
        self.assertClose(lgo_1_summ[0][0:3], [0] * 3)
        self.assertEqual(np.sum(lgo_1_summ), 3 * 10 - 10)
        lgo_1_coords = lgo_1.pred_indexes()
        self.assertEqual(lgo_1_coords.coords.shape, (3, 5, 1))
        self.assertEqual(lgo_1_coords.lengths.shape, (3,))


if __name__ == "__main__":
    unittest.main()
