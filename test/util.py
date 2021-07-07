"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import os
import unittest

from jax import numpy as jnp


def fixture(fname: str) -> str:
    """Get path of test fixture

    Keyword arguments
        fname: filename without path

    Returns
        relative path to test fixture
    """
    return os.path.join(os.path.dirname(__file__), "fixtures", fname)


class TestCase(unittest.TestCase):
    """TestCase with a few convenience test methods."""

    # pylint: disable=invalid-name
    def assertClose(self, array1, array2, msg=None, **kwargs) -> None:
        """Assert all elements of array1 and array2 are close in value.

        Args:
            array1: first array for comparison
            array2: second array for comparison
            msg: message to display on error
            rtol: relative tolerance
            atol: absolute tolerance
        """
        self.assertTrue(jnp.allclose(array1, array2, **kwargs), msg)
