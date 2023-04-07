from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import optax


class Model(NamedTuple):
    """Container for cross-validatable model functions."""
    num_folds: int
    num_models: int
    logjoint_density: Callable
    log_pred: Callable
    make_initial_pos: Callable
    to_constrained: Callable


def minimize_adam(
        loss: Callable,
        x0,
        verbose: bool=False,
        niter: int=101,
        learning_rate = 1e-1):
    """Use adam to find the approximate minimum of a multivariate function.
    
    Args:
        loss: loss function
        x0: PyTree, starting point
        verbose: Print loss every 10 iterations.
        niter: Number of iterations.
        learning_rate: Learning rate for Adam.

    Returns:
        PyTree of initial parameters.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x0)
    params = x0
    def update_param(i, state):
        params, opt_state = state
        grads = jax.grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    params, _ = jax.lax.fori_loop(0, niter, update_param, (params, opt_state))
    return params


def get_mode(
        model: Model,
        key=jax.random.PRNGKey(0),
        verbose: bool=False,
        niter: int=101,
        learning_rate = 1e-1):
    """Use adam to find the approximate mode of the joint log density.
    
    Args:
        model: Model object.
        key: JAX PRNG key.
        verbose: Print loss every 10 iterations.
        niter: Number of iterations.
        learning_rate: Learning rate for Adam.

    Returns:
        PyTree of initial parameters.
    """
    params = model.make_initial_pos(key)
    loss = lambda theta: -model.logjoint_density(theta, fold_id=-1, model_id=0, prior_only=False)
    return minimize_adam(loss, params, verbose, niter, learning_rate)
