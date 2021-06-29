from functools import partial
from typing import Dict
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

        We use index=-1 to indicate full-data likelihood (ie no CV folds dropped). If index >= 0, then
        the potential function should leave out a likelihood contribution identified by the value of cv_fold.

        Keyword arguments:
            param: transformed model parameters
            cv_fold: an integer in the range 0..(self.cv_folds)
        """
        return -self.log_joint(cv_fold=cv_fold, **param)

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


class TransformedCVModel(CVModel):
    """Transformed cross-validated model
    
    Instances of this class wrap a CVModel object and apply transformations
    between the constrained model space and unconstrained sampler space.
    """

    def __init__(self, original_model: CVModel) -> None:
        self.original_model = original_model
    
    def to_unconstrained_coordinates(self, params: Dict[str, jnp.DeviceArray]) -> Dict[str, jnp.DeviceArray]:
        """Convert params to unconstrained (sampler) parameter space

        The argument params is expressed in constrained (model) coordinate
        space.

        Keyword arguments:
            params: dictionary of parameters in constrained (model) parameter
                    space, keyed by name
        
        Returns:
            dictionary of parameters with same structure as params, but
            in unconstrained (sampler) parameter space.
        """
        raise NotImplementedError()

    def to_constrained_coordinates(self, params: Dict[str, jnp.DeviceArray]) -> Dict[str, jnp.DeviceArray]:
        """Convert params to constrained (model) parameter space
        
        Keyword arguments:
            params: dictionary of parameters in unconstrained (sampler) parameter
                    space, keyed by name
        
        Returns:
            dictionary of parameters with same structure as params, but
            in constrained (model) parameter space.
        """
        raise NotImplementedError()

    def log_det(self, params: Dict[str, jnp.DeviceArray]) -> jnp.DeviceArray:
        """Return total log determinant of transformation to constrained parameters
        """
        raise NotImplementedError()

    def log_joint(self, cv_fold, **mcmc_params):
        model_params = self.to_constrained_coordinates(mcmc_params)
        lj = self.original_model.log_joint(cv_fold=cv_fold, **model_params)
        ldet = self.log_det(model_params)
        return lj + ldet

    def log_pred(self, y_tilde, mcmc_params):
        model_params = self.to_constrained_coordinates(mcmc_params)
        return self.original_model.log_pred(y_tilde, model_params)        

    @property
    def initial_value(self):
        iv = self.original_model.initial_value
        return self.to_unconstrained_coordinates(iv)
