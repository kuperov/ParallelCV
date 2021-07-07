"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

HMC is happiest when potentials have unbounded real support.

Transform objects map from a bounded parameter to an unbounded one.
All functions should be JAX jitable. These objects transform a single
variable.
"""

import chex
from jax import numpy as jnp


class Transform:
    """Tranformation from one parameter space to another."""

    def to_unconstrained(self, model_param: chex.ArrayDevice) -> chex.ArrayDevice:
        """Translate param to unconstrained (inference) parameters.

        Args:
            model_param: constrained model parameters

        Returns:
            unconstrained (inference) parameters
        """
        raise NotImplementedError()

    def to_constrained(self, inf_param: chex.ArrayDevice) -> chex.ArrayDevice:
        """Translate parameters to constrained (model) parameter space.

        Args:
            inf_param: unconstrained inference parameters

        Returns
            dict of constrained model parameters
        """
        raise NotImplementedError()

    def log_det(self, inf_param: chex.ArrayDevice) -> chex.ArrayDevice:
        """Log determinant of the to_constrained transformation."""
        raise NotImplementedError()


class LogTransform(Transform):
    """Transforms a univariate positive parameter to the real line using log()"""

    def to_unconstrained(self, model_param: chex.ArrayDevice) -> chex.ArrayDevice:
        return jnp.log(model_param)

    def to_constrained(self, inf_param: chex.ArrayDevice) -> chex.ArrayDevice:
        return jnp.exp(inf_param)

    def log_det(self, inf_param: chex.ArrayDevice) -> chex.ArrayDevice:
        return jnp.log(inf_param)


class IntervalTransform(Transform):
    """Transformation [a,b] ⟶ ℝ using generalized odds ratio."""

    def __init__(self, a: float, b: float) -> None:
        self.a, self.b = a, b

    def to_unconstrained(self, model_param: chex.ArrayDevice) -> chex.ArrayDevice:
        prob = (model_param - self.b) / (self.b - self.a)
        odds = prob / (1 - prob)
        return odds

    def to_constrained(self, inf_param: chex.ArrayDevice) -> chex.ArrayDevice:
        prob = inf_param / (1 + inf_param)
        return self.a + prob * (self.b - self.a)

    def log_det(self, inf_param: chex.ArrayDevice) -> chex.ArrayDevice:
        return 0.0  # FIXME
