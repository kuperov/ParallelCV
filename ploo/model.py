from functools import partial
from jax import random, numpy as jnp, lax, jit


class CVModel(object):
    """Cross-validated model: encapsulates data, prior, likelihood, and
    predictive, and details of the cross-validation scheme.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def initial_value(self):
        """A deterministic starting value for sampling.

        TODO: Deal with multiple independent chains.
              Make this a function of the chain number?
              Supply vectors of starting positions?
        """
        raise NotImplementedError()

    def cv_potential(self, param, cv_fold):
        """Potential for the given CV fold.

        This version of the potential function leaves out a likelihood
        contribution, according to the value of cv_fold.

        Keyword arguments:
            param: transformed model parameters
            cv_fold: an integer in the range 0..(self.cv_folds)
        """
        return -self.log_joint(cv_fold=cv_fold, **param)

    def potential(self, param):
        """Potential for the full-data model.

        This version of the potential function includes all data, i.e. does
        not omit any CV folds.

        Keyword arguments:
            param: transformed model parameters
        """
        return self.cv_potential(param, cv_fold=-1)

    def log_joint(self, cv_fold=None, **kwargs):
        """Log joint: log p(args) + log p(data | args), leaving out the
        specified cv_fold.
        """
        raise NotImplementedError()

    def log_pred(self, y_tilde, param):
        """Log joint predictive.

        param is a map but contains vectors
        """
        raise NotImplementedError()

    @property
    def cv_folds(self):
        """Number of cross-validation folds."""
        raise NotImplementedError()

    @property
    def parameters(self):
        """Names of parameters"""
        return list(self.initial_value.keys())

    @classmethod
    def generate(cls, random_key):
        """Generate a dataset corresponding to the specified random key."""
        raise NotImplementedError()
