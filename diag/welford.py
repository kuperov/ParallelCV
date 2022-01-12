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


# stack arrays in pytrees
def tree_stack(trees):
    return tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)

# stack arrays in pytrees
def tree_concat(trees):
    return tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


class WelfordState(NamedTuple):
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
  return WelfordState(K=K*1., Ex=K*0., Eax=K*0., Ex2=K*0., n=K*0)

def welford_add(x: chex.Array, state: WelfordState) -> WelfordState:
  return WelfordState(
    K=state.K,
    Ex=state.Ex + x - state.K,
    Eax=state.Eax + jnp.abs(x - state.K),
    Ex2=state.Ex2 + (x - state.K)**2,
    n=state.n + 1)

def welford_mean(state: WelfordState):
  return state.K + state.Ex / state.n

def welford_mad(state: WelfordState):
  return state.Eax / state.n

def welford_var(state: WelfordState):
  return (state.Ex2 - state.Ex**2 / state.n) / (state.n - 1)


class ExtendedState(NamedTuple):
    state: blackjax.mcmc.ghmc.GHMCState  # current HMC state
    rng_key: chex.Array  # current random seed
    pred_ws: WelfordState  # accumulator for log predictive
    log_pred_mean: float  # log of mean predictive
    param_ws: WelfordState  # accumulator for parameters
    divergences: chex.Array  # divergence counts (int array)

    def diagnostics(self) -> None:
        """Summarize the state of this object."""
        # TODO: add mean and s.e. of the parameters
        rhats = [(n, rhat(v)) for n, v in zip(self.param_ws._fields, self.param_ws)]
        predrh, predsrh = rhat(self.pred_ws)
        status = [
            f'       Summary: {int(jnp.sum(self.pred_ws.n[0,:]))} draws * {self.state.position[0].shape[0]} chains',
        ]
        param = [f'{n: >9} Rhat: {v} ({desc})' for n, (rh, frh) in rhats for v, desc in [(rh, 'regular'), (frh, 'tail')]]
        lines = status + param + [
            f"    pred. Rhat: {predrh:.4f}  tail: {predsrh:.4f}",
            f"   divergences: {int(jnp.sum(self.divergences))}"
        ]
        print('\n'.join(lines))


# single chain inference loop we will run in parallel using vmap
def offline_inference_loop(rng_key, kernel, initial_state, num_samples, log_pred, theta_center):
    log_half_samp = jnp.log(0.5 * num_samples + 1)  # +1 for initialization
    def one_mcmc_step(ext_state, idx):
        i_key, carry_key = jax.random.split(ext_state.rng_key)
        chain_state, chain_info = kernel(i_key, ext_state.state)
        elpd_contrib = log_pred(chain_state.position) - log_half_samp  # contrib to mean log predictive
        carry_log_pred_mean = ext_state.log_pred_mean + jnp.log1p(jnp.exp(elpd_contrib - ext_state.log_pred_mean))
        div_count = ext_state.divergences +  1.0 * chain_info.is_divergent
        carry_pred_ws = welford_add(elpd_contrib, ext_state.pred_ws)
        carry_param_ws = tree_map(welford_add, chain_state.position, ext_state.param_ws)
        carry_state = ExtendedState(
          state=chain_state,
          rng_key=carry_key,
          pred_ws=carry_pred_ws,
          log_pred_mean=carry_log_pred_mean,
          param_ws=carry_param_ws,
          divergences=div_count)
        return carry_state, chain_state
    # first half of chain
    initial_state_1h = ExtendedState(
        initial_state,
        rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean = log_pred(theta_center) - log_half_samp,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0)
    carry_state_1h, states_1h = jax.lax.scan(one_mcmc_step, initial_state_1h, jnp.arange(0, num_samples//2))
    # second half of chain - continue at same point but accumulate into new welford states
    initial_state_2h = ExtendedState(
        carry_state_1h.state,
        carry_state_1h.rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean = log_pred(theta_center) - log_half_samp,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0)
    carry_state_2h, states_2h = jax.lax.scan(one_mcmc_step, initial_state_2h, jnp.arange(num_samples//2, num_samples))
    return tree_stack((carry_state_1h, carry_state_2h,)), tree_concat((states_1h, states_2h,))

# single chain inference loop we will run in parallel using vmap
def online_inference_loop(rng_key, kernel, initial_state, num_samples, log_pred, theta_center):
    log_half_samp = jnp.log(0.5 * num_samples + 1)  # +1 for initialization
    def one_mcmc_step(ext_state, idx):
        i_key, carry_key = jax.random.split(ext_state.rng_key)
        chain_state, chain_info = kernel(i_key, ext_state.state)
        elpd_contrib = log_pred(chain_state.position) - log_half_samp  # contrib to mean log predictive
        carry_log_pred_mean = ext_state.log_pred_mean + jnp.log1p(jnp.exp(elpd_contrib - ext_state.log_pred_mean))
        div_count = ext_state.divergences +  1.0 * chain_info.is_divergent
        carry_pred_ws = welford_add(elpd_contrib, ext_state.pred_ws)
        carry_param_ws = tree_map(welford_add, chain_state.position, ext_state.param_ws)
        carry_state = ExtendedState(
          state=chain_state,
          rng_key=carry_key,
          pred_ws=carry_pred_ws,
          log_pred_mean=carry_log_pred_mean,
          param_ws=carry_param_ws,
          divergences=div_count)
        return carry_state, None  # don't retain chain trace
    # first half of chain
    initial_state_1h = ExtendedState(
        initial_state,
        rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean = log_pred(theta_center) - log_half_samp,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0)
    carry_state_1h, _ = jax.lax.scan(one_mcmc_step, initial_state_1h, jnp.arange(0, num_samples//2))
    # second half of chain - continue at same point but accumulate into new welford states
    initial_state_2h = ExtendedState(
        carry_state_1h.state,
        carry_state_1h.rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean = log_pred(theta_center) - log_half_samp,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0)
    carry_state_2h, _ = jax.lax.scan(one_mcmc_step, initial_state_2h, jnp.arange(num_samples//2, num_samples))
    return tree_stack((carry_state_1h, carry_state_2h,))