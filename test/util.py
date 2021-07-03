import os


def fixture(fname):
    return os.path.join(os.path.dirname(__file__), fname)
