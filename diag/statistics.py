
import chex
from jax import lax
from jax import numpy as jnp
from scipy.fftpack import next_fast_len
from typing import Callable, Dict, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp

import blackjax.adaptation as adaptation
import blackjax.mcmc as mcmc
import blackjax.sgmcmc as sgmcmc
import blackjax.smc as smc
import blackjax.vi as vi
from blackjax.base import AdaptationAlgorithm, MCMCSamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import Array, PRNGKey, PyTree
from blackjax.kernels import ghmc

import blackjax
import jax
import chex
import jax.numpy as jnp
import arviz as az
from tensorflow_probability.substrates import jax as tfp
from collections import namedtuple
import matplotlib.pyplot as plt
from typing import NamedTuple
from jax.tree_util import tree_map, tree_structure, tree_flatten, tree_unflatten
from jax.scipy.special import logsumexp
import pandas as pd


def estimate_elpd(extended_state: ExtendedState):
    """Estimate the expected log pointwise predictive density from welford state.

    The resulting elpd is in sum scale, that is we average over (half)
    chains and sum over folds.
    """
    # AVERAGE over (half) chains (chain dim is axis 1, chain half dim is axis 2)
    nchains, nhalfs = extended_state.log_pred_mean.shape[1:3]
    fold_means = logsumexp(extended_state.log_pred_mean, axis=(1,2)) - jnp.log(nchains * nhalfs)
    fold_means = fold_means.squeeze()
    # SUM over folds
    elpd = jnp.sum(fold_means)
    return float(elpd)


def ess(x: chex.ArrayDevice, relative=False):
    r"""Effective sample size as described in Vehtari et al 2021 and Geyer 2011.

    Adapted for JAX from pyro implementation, see:
    https://github.com/pyro-ppl/numpyro/blob/048d2c80d9f4087aa9614225568bb88e1f74d669/numpyro/diagnostics.py#L148
    Some parts also adapted from ArviZ, see:
    https://github.com/arviz-devs/arviz/blob/8115c7a1b8046797229b654c8389b7c26769aa82/arviz/stats/diagnostics.py#L65

    :param samples: 2D array of samples
    :param relative: if true return relative measure
    """  # noqa: B950
    assert x.ndim >= 2
    assert x.shape[1] >= 2

    # find autocovariance for each chain at lag k
    N = x.shape[1]
    M = next_fast_len(N)
    # transpose axis with -1 for Fourier transform
    acov_x = jnp.swapaxes(x, 1, -1)
    # centering x
    centered_signal = acov_x - acov_x.mean(axis=-1, keepdims=True)
    freqvec = jnp.fft.rfft(centered_signal, n=2 * M, axis=-1)
    autocorr = jnp.fft.irfft(freqvec * jnp.conjugate(freqvec), n=2 * M, axis=-1)
    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / jnp.arange(N, 0.0, -1)
    autocorr = autocorr / autocorr[..., :1]
    autocorr = jnp.swapaxes(autocorr, 1, -1)
    gamma_k_c = autocorr * x.var(axis=1, keepdims=True)
    # find autocorrelation at lag k (from Stan reference)
    _, var_within, var_estimator = chain_variance(x)
    rho_k = jnp.concatenate(
        [
            jnp.array([1.0]),
            jnp.array(1.0 - (var_within - gamma_k_c.mean(axis=0)) / var_estimator)[1:],
        ]
    )
    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k[:-1:2, ...] + rho_k[1::2, ...]
    # initial monotone (decreasing) sequence
    Rho_k_pos = jnp.clip(Rho_k[1:, ...], a_min=0, a_max=None)
    _, init_mon_seq = lax.scan(
        lambda c, a: (jnp.minimum(c, a), jnp.minimum(c, a)), jnp.inf, Rho_k_pos
    )
    Rho_k = jnp.concatenate([Rho_k[:1], init_mon_seq])
    tau = -1.0 + 2.0 * jnp.sum(Rho_k, axis=0)
    n_eff = jnp.prod(jnp.array(x.shape[:2])) / tau
    return n_eff
