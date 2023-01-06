from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp


class WelfordState(NamedTuple):
    """Welford state object for univariate data."""
    K: chex.Array  # central estimate of data
    Ex: chex.Array  # sum of deviations from K
    Eax: chex.Array  # sum of absolute deviations from K
    Ex2: chex.Array  # sum of squared deviations from K
    n: chex.Array  # number of data points


def welford_init(K: chex.Array) -> WelfordState:
    """Initialize new welford algorithm state.

    Args:
      K: estimated mean value of data. Same shape as data.
    """
    return WelfordState(K=K * 1.0, Ex=K * 0.0, Eax=K * 0.0, Ex2=K * 0.0, n=K * 0)


def welford_add(x: chex.Array, state: WelfordState) -> WelfordState:
    return WelfordState(
        K=state.K,
        Ex=state.Ex + x - state.K,
        Eax=state.Eax + jnp.abs(x - state.K),
        Ex2=state.Ex2 + (x - state.K) ** 2,
        n=state.n + 1,
    )


def welford_mean(state: WelfordState):
    return state.K + state.Ex / state.n


def welford_mad(state: WelfordState):
    return state.Eax / state.n


def welford_var(state: WelfordState, ddof=1):
    return (state.Ex2 - state.Ex**2 / state.n) / (state.n - ddof)


class BatchWelfordState(NamedTuple):
    """Welford state object for batch means of univariate data."""
    batch_size: int
    current: WelfordState
    batches: WelfordState


def batch_welford_init(K: chex.Array, batch_size: int) -> BatchWelfordState:
    return BatchWelfordState(
        batch_size=batch_size,
        current=welford_init(K=K),
        batches=welford_init(K=K),
    )


def batch_welford_add(x: chex.Array, state: BatchWelfordState) -> BatchWelfordState:
    upd_current = welford_add(x, state.current)
    def incr_batch():
        return BatchWelfordState(
            batch_size=state.batch_size,
            current=welford_init(K=state.current.K),
            batches=welford_add(welford_mean(upd_current), state.batches))
    def incr_current():
        return BatchWelfordState(
            batch_size=state.batch_size,
            current=upd_current,
            batches=state.batches)
    return jax.lax.cond(upd_current.n == state.batch_size, incr_batch, incr_current)


def batch_welford_mean(state: BatchWelfordState):
    def whole_mean():  # total is even multiple of batch size
        return welford_mean(state.batches)
    def resid_mean():  # include current batch
        return welford_mean(welford_add(welford_mean(state.current), state.batches))
    return jax.lax.cond(state.current.n == 0, whole_mean, resid_mean)


def batch_welford_var(state: BatchWelfordState, ddof=1):
    def whole_var():  # total is even multiple of batch size
        return welford_var(state.batches)
    def resid_var():  # include current batch
        return welford_var(welford_add(welford_mean(state.current), state.batches), ddof=ddof)
    return jax.lax.cond(state.current.n == 0, whole_var, resid_var)


class VectorWelfordState(NamedTuple):
    K: jax.Array  # central estimate of data
    Ex: jax.Array  # sum of deviations from K
    Ex2: jax.Array  # sum of squared deviations from K
    n: jax.Array  # number of data points


def vector_welford_init(K: jax.Array) -> VectorWelfordState:
    """Initialize new welford algorithm state.

    Args:
      K: estimated mean vector of data. Vector.
    """
    chex.assert_rank(K, 1)
    return VectorWelfordState(K=K * 1.0, Ex=K * 0.0, Ex2=jnp.zeros((K.shape[0],K.shape[0])), n=0)


def vector_welford_add(x: jax.Array, state: VectorWelfordState) -> VectorWelfordState:
    return VectorWelfordState(
        K=state.K,
        Ex=state.Ex + x - state.K,
        Ex2=state.Ex2 + jnp.outer(x - state.K, x - state.K),
        n=state.n + 1,
    )


def vector_welford_mean(state: VectorWelfordState):
    return state.K + state.Ex / state.n


def vector_welford_cov(state: VectorWelfordState, ddof=1):
    """Covariance matrix for data"""
    return (state.Ex2 - jnp.outer(state.Ex, state.Ex) / state.n) / (state.n - ddof)


class BatchVectorWelfordState(NamedTuple):
    batch_size: int
    current: VectorWelfordState
    batches: VectorWelfordState


def batch_vector_welford_init(K: jax.Array, batch_size: int) -> BatchVectorWelfordState:
    return BatchVectorWelfordState(
        batch_size=batch_size,
        current=vector_welford_init(K=K),
        batches=vector_welford_init(K=K),
    )


def batch_vector_welford_add(x: jax.Array, state: BatchVectorWelfordState) -> BatchVectorWelfordState:
    upd_current = vector_welford_add(x, state.current)
    def incr_batch():
        return BatchVectorWelfordState(
            batch_size=state.batch_size,
            current=vector_welford_init(K=state.current.K),
            batches=vector_welford_add(vector_welford_mean(upd_current), state.batches))
    def incr_current():
        return BatchVectorWelfordState(
            batch_size=state.batch_size,
            current=upd_current,
            batches=state.batches)
    return jax.lax.cond(upd_current.n == state.batch_size, incr_batch, incr_current)


def batch_vector_welford_mean(state: BatchVectorWelfordState):
    def whole_mean():  # total is even multiple of batch size
        return vector_welford_mean(state.batches)
    def resid_mean():  # include current batch
        return vector_welford_mean(vector_welford_add(vector_welford_mean(state.current), state.batches))
    return jax.lax.cond(state.current.n == 0, whole_mean, resid_mean)


def batch_vector_welford_cov(state: BatchVectorWelfordState, ddof=1):
    def whole_cov():  # total is even multiple of batch size
        return vector_welford_cov(state.batches)
    def resid_cov():  # include current batch
        return vector_welford_cov(vector_welford_add(vector_welford_mean(state.current), state.batches), ddof=ddof)
    return jax.lax.cond(state.current.n == 0, whole_cov, resid_cov)
