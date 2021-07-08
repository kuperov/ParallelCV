"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
from test.util import TestCase

from ploo.util import Timer


class TestTimer(TestCase):
    """Check utility classes behave as expected"""

    def test_timer(self):
        """We don't apply any delay, but python is slow enough for measureable
        time to elapse between creating and evaluating the timer
        """
        t = Timer()
        elapsed = t.sec
        self.assertGreater(elapsed, 0.0)
        self.assertLess(elapsed, 1e-4)
        self.assertEqual(str(t), "0.0 sec")  # by rounding
