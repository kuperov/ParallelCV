from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


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


def welford_var_combine(state: WelfordState, ddof=1, comb_axis=(1,), out_axis=0):
    """Univariate variance for data
    
    Axis should be the axis over which the data is combined.
    """
    ex2 = state.Ex2.sum(axis=comb_axis)
    ex = state.Ex.sum(axis=comb_axis)
    n = state.n.sum(axis=comb_axis)
    K = jnp.reshape(state.K, (-1))[0]  # there has to be a better way than this
    # TODO: check K is the same for all elements of the batch
    def f(i):
        return (ex2[i] - ex[i]**2 / n[i]) / (n[i] - ddof)
    return jax.vmap(f)(jnp.arange(ex2.shape[out_axis]))


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
        return welford_var(state.batches, ddof=ddof)
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
    return VectorWelfordState(
        K=K,
        Ex=jnp.zeros_like(K),
        Ex2=jnp.zeros((K.shape[0],K.shape[0])),
        n=jnp.array(0)
    )


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

def vector_welford_cov_combine(state:VectorWelfordState, ddof=1, comb_axis=(1,), out_axis=0):
    """Covariance matrix for data
    
    Args:
        comb_axis: axes over which the data is combined.
    """
    ex2 = state.Ex2.sum(axis=comb_axis)
    ex = state.Ex.sum(axis=comb_axis)
    n = state.n.sum(axis=comb_axis)
    def f(i):
        return (ex2[i] - jnp.outer(ex[i], ex[i]) / n[i]) / (n[i] - ddof)
    return jax.vmap(f)(jnp.arange(ex2.shape[out_axis]))


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
        return vector_welford_cov(state.batches, ddof=ddof)
    def resid_cov():  # include current batch
        return vector_welford_cov(vector_welford_add(vector_welford_mean(state.current), state.batches), ddof=ddof)
    return jax.lax.cond(state.current.n == 0, whole_cov, resid_cov)


class LogWelfordState(NamedTuple):
    """Welford state object for data expressed in logs.
    
    This can only handle positive values (of course) and the reference point
    is zero. It is intended for densities.
    """
    logEx: jax.Array  # log of sum of values
    logEx2: jax.Array  # log of sum of squared values
    n: jax.Array  # number of data points


def log_welford_init(shape: tuple) -> LogWelfordState:
    """Initialize new welford algorithm state.
    """
    return LogWelfordState(
        logEx=-jnp.inf*jnp.ones(shape=shape),
        logEx2=-jnp.inf*jnp.ones(shape=shape),
        n=jnp.zeros(shape=shape, dtype=int)
    )


def log_welford_add(logx: jax.Array, state: LogWelfordState) -> LogWelfordState:
    return LogWelfordState(
        logEx=jnp.logaddexp(state.logEx, logx),
        logEx2=jnp.logaddexp(state.logEx2, 2*logx),
        n=state.n + 1,
    )


def log_welford_mean(state: LogWelfordState):
    # NB: Not valid if n == 0
    return state.logEx - jnp.log(state.n)


def log_welford_var(state: LogWelfordState, ddof=0):
    # NB: Not valid if n <= 1
    return (
        - jnp.log(state.n - ddof)
        + state.logEx2
        + jnp.log1p(-jnp.exp(2*state.logEx - jnp.log(state.n) - state.logEx2))
    )


def log_welford_var_combine(state: LogWelfordState, ddof:int=0, comb_axis:int=1, out_axis:int=0):
    """Univariate variance for data from multiple welford states
    
    Axis should be the axis over which the data is combined.

    Args:
        state: log welford state, for measures in logs
        ddof (int): degrees of freedom
        comb_axis (tuple): axis over which to combine data
        out_axis (int): axis over which to return results
    
    Return:
        array of log variances of length of out_axis
    """
    log_ex2 = logsumexp(state.logEx2, axis=comb_axis)
    log_ex = logsumexp(state.logEx, axis=comb_axis)
    n = state.n.sum(axis=comb_axis)
    def f(i):
        return (
            log_ex2[i]
            + jnp.log1p(- jnp.exp(2*log_ex[i] - log_ex2[i] - jnp.log(n[i])))
            - jnp.log(n[i] - ddof)
        )
    return jax.vmap(f)(jnp.arange(log_ex2.shape[out_axis]))


def log_welford_log_var_combine(state: LogWelfordState, ddof:int=0, comb_axis:int=1, out_axis:int=0):
    """Univariate log variance of log mean of data, combining multiple welford states
    
    This is a delta method estimate, var(log(mean)) = var/(mean^2*n) + approximation error.
    Result is returned in log space to avoid underflow. Should work fine using fp32 values.

    Parameter comb_axis should be the axis over which the data is combined, usually 1 (chains within a fold).

    Args:
        state: log welford state, for measures in logs
        ddof (int): degrees of freedom
        comb_axis (tuple): axis over which to combine data
        out_axis (int): axis over which to return results
    
    Return:
        array of log variances of length of out_axis
    """
    log_ex2 = logsumexp(state.logEx2, axis=comb_axis)
    log_ex = logsumexp(state.logEx, axis=comb_axis)
    n = state.n.sum(axis=comb_axis)
    def f(i):
        log_var = (
            log_ex2[i]
            + jnp.log1p(- jnp.exp(2*log_ex[i] - log_ex2[i] - jnp.log(n[i])))
            - jnp.log(n[i] - ddof)
        )
        return log_var - 2*log_ex[i] - jnp.log(n[i]) # log(var/(n*mean**2))
    return jax.vmap(f)(jnp.arange(log_ex2.shape[out_axis]))


class BatchLogWelfordState(NamedTuple):
    """Welford state object for batch means of univariate data."""
    batch_size: int
    current: LogWelfordState
    batches: LogWelfordState


def batch_log_welford_init(shape: tuple, batch_size: int) -> BatchLogWelfordState:
    return BatchLogWelfordState(
        batch_size=batch_size,
        current=log_welford_init(shape=shape),
        batches=log_welford_init(shape=shape),
    )


def batch_log_welford_add(x: chex.Array, state: BatchLogWelfordState) -> BatchLogWelfordState:
    upd_current = log_welford_add(x, state.current)
    def incr_batch():
        return BatchLogWelfordState(
            batch_size=state.batch_size,
            current=log_welford_init(shape=state.current.logEx.shape),
            batches=log_welford_add(log_welford_mean(upd_current), state.batches))
    def incr_current():
        return BatchLogWelfordState(
            batch_size=state.batch_size,
            current=upd_current,
            batches=state.batches)
    return jax.lax.cond(upd_current.n == state.batch_size, incr_batch, incr_current)


def batch_log_welford_mean(state: BatchLogWelfordState):
    def whole_mean():  # total is even multiple of batch size
        return log_welford_mean(state.batches)
    def resid_mean():  # include current batch
        return log_welford_mean(log_welford_add(log_welford_mean(state.current), state.batches))
    return jax.lax.cond(state.current.n == 0, whole_mean, resid_mean)


def batch_log_welford_var(state: BatchLogWelfordState, ddof=0):
    def whole_var():  # total is even multiple of batch size
        return log_welford_var(state.batches, ddof=ddof)
    def resid_var():  # include current batch
        return log_welford_var(log_welford_add(log_welford_mean(state.current), state.batches), ddof=ddof)
    return jax.lax.cond(state.current.n == 0, whole_var, resid_var)

