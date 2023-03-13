from typing import NamedTuple, Callable


class Model(NamedTuple):
    """Container for cross-validatable model functions."""
    num_folds: int
    num_models: int
    logjoint_density: Callable
    log_pred: Callable
    make_initial_pos: Callable
    to_constrained: Callable
