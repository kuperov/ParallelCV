"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import unittest

from ploo.util import Timer


class TestTimer(unittest.TestSuite):
    def test_timer(self):
        t = Timer()
        elapsed = t.elapsed_sec()
        self.assertTrue(elapsed > 0 and elapsed < 1e-4)
        self.assertEqual(str(t), "0.0 sec")
