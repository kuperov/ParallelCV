"""HMC is happiest when potentials have unbounded real support

Transform objects map from a bounded parameter to an unbounded one.
All functions should be JAX jitable

These objects transform a single variable. Use TransformedCVModel to
transform a model's parameter space for MCMC.
"""

from jax import numpy as jnp


class Transform(object):
    """Tranformation from one parameter space to another."""

    def to_unconstrained(self, param):
        raise NotImplementedError()

    def to_constrained(self, param):
        raise NotImplementedError()

    def __call__(self, param):
        return self.to_unconstrained(param)

    def log_det(self, param):
        """Log determinant of the to_constrained transformation."""
        raise NotImplementedError()


class LogTransform(Transform):
    """Transforms a univariate positive parameter to the real line using log()"""

    def to_unconstrained(self, param):
        return jnp.log(param)

    def to_constrained(self, param):
        return jnp.exp(param)

    def log_det(self, param):
        return param


class IntervalTransform(Transform):
    """Transformation [a,b] ⟶ ℝ using generalized odds ratio."""

    def __init__(self, a, b) -> None:
        self.a, self.b = a, b

    def to_unconstrained(self, param):
        p = (param - self.b) / (self.b - self.a)
        odds = p / (1 - p)
        return odds

    def to_constrained(self, param):
        p = param / (1 + param)
        return self.a + p * (self.b - self.a)

    def log_det(self, param):
        return 0.0  # FIXME
