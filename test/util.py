"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import os
import unittest

from jax import numpy as jnp


def fixture(fname):
    return os.path.join(os.path.dirname(__file__), fname)


class TestCase(unittest.TestCase):
    def assertClose(self, array1, array2, msg=None, **kwargs) -> None:
        """Assert all elements of array1 and array2 are close in value.

        Keyword args:
            array1: first array for comparison
            array2: second array for comparison
            msg: message to display on error
            rtol: relative tolerance
            atol: absolute tolerance
        """
        self.assertTrue(jnp.allclose(array1, array2, **kwargs), msg)
