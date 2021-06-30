from typing import Dict
from jax import numpy as jnp


# model parameters are in a constrained coordinate space
ModelParams = Dict[str, jnp.DeviceArray]

# inference parameters
InfParams = Dict[str, jnp.DeviceArray]


class CVModel(object):
    """Cross-validated model: encapsulates both data and model spec.

    There are two sets of parameters referenced in this class, both of
    which are expressed as dicts keyed by variable name:

      * Model parameters (type annotation ModelParams) are in the
        coordinate system used by the model, which may be constrained
        (e.g. variance parameters might take values on positive half-line)

      * Inference parameters (type annotation InfParams) are in an
        unconstrained real coordinate system, i.e. Θ = ℝᵈ for some d ∈ ℕ.

    Transformations are up to the user, but should probably be done using
    instances of the Transform class.
    """

    name = "Unnamed model"

    def log_likelihood(self, model_params: ModelParams, cv_fold=-1):
        """Log likelihood

        JAX needs to be able to trace this function.

        Note: future versions of this function won't take cv_fold as a
        parameter. But we haven't yet built the CV abstraction. Future
        version will return log likelihood contributions contributions
        { log p(yᵢ|θ): i=1,2,⋯,n }, as a 1- or 2-dimensional array.
        The shape of the array corresponds to the shape of the model's
        dependence structure, or a 1-dimensional array if data are iid.

        Keyword args:
            params:  dict of model parameters, in constrained (model)
                     parameter space
            cv_fold: cross-validation fold to evaluate

        Returns:
            log likelihood at the given parameters for the given CV fold
        """
        raise NotImplementedError()

    def log_prior(self, model_params: ModelParams):
        """Compute log prior log p(θ)

        JAX needs to be able to trace this function.

        Keyword args:
            params: dict of model parameters in constrained (model)
                    parameter space
        """
        raise NotImplementedError()

    def log_cond_pred(self, cv_fold, model_params):
        """Computes log conditional predictive ordinate, log p(ỹ|θˢ).

        FIXME: needs some kind of index to identify the conditioning values

        Keyword arguments:
            cv_fold: index of point at which to evaluate predictive density
            params:  a dict of parameters (constrained (model) parameter
                     space) that potentially contains vectors
        """
        raise NotImplementedError()

    def initial_value(self) -> ModelParams:
        """A deterministic starting value in model parameter space.

        FIXME: Deal with multiple independent chains.
               Make this a function of the chain number?
               Supply vectors of starting positions?
        """
        raise NotImplementedError()

    def initial_value_unconstrained(self) -> InfParams:
        """Deterministic starting value, transformed to unconstrained inference space"""
        return self.to_inference_params(self.initial_value())

    def cv_potential(self, inf_params: InfParams, cv_fold: int) -> jnp.DeviceArray:
        """Potential for the given CV fold.

        We use index=-1 to indicate full-data likelihood (ie no CV folds dropped). If index >= 0, then
        the potential function should leave out a likelihood contribution identified by the value of cv_fold.

        Keyword arguments:
            inf_params: model parameters in inference (unconstrained) space
            cv_fold:    an integer corresponding to a CV fold
        """
        model_params = self.to_model_params(inf_params)
        llik = self.log_likelihood(cv_fold=cv_fold, model_params=model_params)
        lprior = self.log_prior(model_params=model_params)
        ldet = self.log_det(model_params=model_params)
        return -llik - lprior - ldet

    def cv_folds(self):
        """Number of cross-validation folds."""
        raise NotImplementedError()

    def parameters(self):
        """Names of parameters"""
        return list(self.initial_value().keys())

    @classmethod
    def generate(
        cls, random_key: jnp.DeviceArray, model_params: ModelParams
    ) -> jnp.DeviceArray:
        """Generate a dataset corresponding to the specified random key."""
        raise NotImplementedError()

    def to_inference_params(self, model_params: ModelParams) -> InfParams:
        """Convert constrained (model) params to unconstrained (sampler) parameter space

        The argument model_params is expressed in constrained (model) coordinate
        space.

        Keyword arguments:
            model_params: dictionary of parameters in constrained (model) parameter
                          space, keyed by name

        Returns:
            dictionary of parameters with same structure as params, but
            in unconstrained (sampler) parameter space.
        """
        return model_params

    def to_model_params(self, inf_params: InfParams) -> ModelParams:
        """Convert unconstrained (inference) params to constrained (model) parameter space

        Keyword arguments:
            inf_params: dictionary of parameters in unconstrained (sampler) parameter
                        space, keyed by name

        Returns:
            dictionary of parameters with same structure as inf_params, but
            in constrained (model) parameter space.
        """
        return inf_params

    def log_det(self, model_params: ModelParams) -> jnp.DeviceArray:
        """Return total log determinant of transformation to constrained parameters

        Keyword arguments:
            model_params: dictionary of parameters in constrained (model) parameter space

        Returns:
            dictionary of parameters with same structure as model_params
        """
        return 0

    def log_pred(self, cv_fold, mcmc_params):
        model_params = self.to_constrained_coordinates(mcmc_params)
        y_tilde = self.y[cv_fold]
        return self.original_model.log_pred(y_tilde, model_params)
