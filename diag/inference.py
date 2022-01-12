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


def fold_posterior(prng_key, inference_loop, logjoint_density, log_p, make_initial_pos, num_chains, num_samples, warmup_iter):
    """Compute posterior for a single fold.
    
    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        inference_loop: function to use for inference loop, which may or may not retain draws
        logjoint_density: callable, log joint density function
        log_p: callable, log density for this fold
        make_initial_pos: callable, function to make initial position for each chain
        num_chains: int, number of chains to run
        num_samples: int, number of samples to draw
        warmup_iter: int, number of warmup iterations to run
    
    Returns:
        state: ExtendedState, final state of the inference loop
        trace: trace of posterior draws if the offline inference loop was used, otherwise None
    """
    warmup_key, sampling_key, init_key = jax.random.split(prng_key, 3)
    # warmup adaption
    init_chain_keys = jax.random.split(init_key, num_chains)
    init_states = jax.vmap(make_initial_pos)(init_chain_keys)
    final_warmup_state, parameters = run_meads(
        logjoint_density_fn=logjoint_density,
        num_chains=num_chains,
        prng_key=warmup_key,
        positions=init_states,
        num_steps=warmup_iter)
    # central points for estimating folded rhat
    centers = tree_map(lambda x: jnp.median(x, axis=0), final_warmup_state.position)
    # construct GHMC kernel
    step_fn = ghmc.kernel()
    def kernel(rng_key, state):
        return step_fn(
            rng_key,
            state,
            logjoint_density,
            **parameters,
        )
    # run chain
    sampling_keys = jax.random.split(sampling_key, num_chains)
    results = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None))(
        sampling_keys, kernel, final_warmup_state, num_samples, log_p, centers)   
    return results