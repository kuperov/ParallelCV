# HMC is happiest when potentials have unbounded real support
#
# BoundTransform objects map from a bounded parameter to an unbounded one.
# All functions should be JAX jitable


class Constraint(object):
    """Tranformation from one parameter space to another."""

    def to_unconstrained(self, param):
        raise NotImplementedError()

    def to_constrained(self, param):
        raise NotImplementedError()

    def __call__(self, param):
        return self.to_constrained(param)

    def log_det(self, param):
        """Log determinant of the to_constrained transformation."""
        raise NotImplementedError()


class PositiveConstraint(Constraint):
    def __init__(self) -> None:
        super().__init__()

    def to_unbounded(self, param):
        return super().to_unconstrained(param)

    def to_bounded(self, param):
        return super().to_constrained(param)

    def log_det(self, param):
        return super().log_det(param)


class IntervalConstraint(Constraint):
    """Transformation from [a,b] --> R using generalized odds ratio transform."""

    def __init__(self, a, b) -> None:
        self.a, self.b = a, b

    def apply(self, param):
        p = (param - self.b) / (self.b - self.a)
        odds = p / (1 - p)
        return odds

    def apply(self, param):
        p = param / (1 + param)
        return self.a + p * (self.b - self.a)

    def log_det(self, param):
        return 0.0
