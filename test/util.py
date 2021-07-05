"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
import os


def fixture(fname):
    return os.path.join(os.path.dirname(__file__), fname)
