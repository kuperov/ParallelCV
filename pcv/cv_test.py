import unittest

import cv
from chex import assert_equal, assert_shape, assert_tree_all_close
from jax import numpy as jnp
from tree import assert_same_structure


class TestCVSchemes(unittest.TestCase):
    def test_loo(self):
        loo = cv.LOOCVScheme(120)
        assert_equal(loo.n_folds(), 120)
        tstm = jnp.concatenate((jnp.array([False, True]), jnp.repeat(False, 118)))
        assert_tree_all_close(loo.test_mask(1), tstm)
        assert_tree_all_close(loo.train_mask(1), jnp.logical_not(tstm))
        Stest1 = loo.S_test(1)
        I = jnp.eye(120)
        assert_same_structure(Stest1, I[tstm, :])
        assert_tree_all_close(Stest1, I[tstm, :])
        Strain1 = loo.S_train(1)
        assert_same_structure(Strain1, I[jnp.logical_not(tstm), :])
        assert_tree_all_close(Strain1, I[jnp.logical_not(tstm), :])

    def test_kfold(self):
        fivefold = cv.KFoldCVScheme(120, 5)  # 24 cells each
        assert_equal(fivefold.n_folds(), 5)
        tstm = jnp.concatenate((jnp.repeat(False, 24), jnp.repeat(True, 96)))
        assert_tree_all_close(fivefold.train_mask(0), tstm)
        assert_tree_all_close(fivefold.test_mask(0), jnp.logical_not(tstm))
        assert_shape(fivefold.S_test(0), (24, 120))
        assert_shape(fivefold.S_train(0), (96, 120))
        assert_shape(fivefold.S_test(1), (24, 120))
        assert_shape(fivefold.S_train(1), (96, 120))

    def test_hvblock(self):
        hvblock = cv.HVBlockCVScheme(120, h=4, v=3)
        assert_equal(hvblock.n_folds(), 120)
        # test 20th fold (i=19)
        tstm = jnp.concatenate(
            [
                jnp.repeat(False, 12 + 4),
                jnp.repeat(True, 1 + 2 * 3),
                jnp.repeat(False, 120 - 12 - 4 - 1 - 2 * 3),
            ]
        )
        tram = jnp.concatenate(
            [
                jnp.repeat(True, 12),
                jnp.repeat(False, 2 * 4 + 2 * 3 + 1),
                jnp.repeat(True, 120 - 12 - 2 * 4 - 2 * 3 - 1),
            ]
        )
        assert_tree_all_close(hvblock.test_mask(19), tstm)
        assert_tree_all_close(hvblock.train_mask(19), tram)
