from typing import Callable, Dict, NamedTuple, Tuple, Union

import blackjax
import blackjax.adaptation as adaptation
import jax
import jax.numpy as jnp
import pandas as pd
from blackjax.kernels import ghmc
from blackjax.types import PRNGKey, PyTree
from jax import lax
from jax import numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from .welford import *


def run_meads(
    logjoint_density_fn: Callable,
    num_chains: int,
    prng_key: PRNGKey,
    positions: PyTree,
    num_steps: int = 1000,
) -> Tuple[PyTree, Dict]:
    """Adapt the parameters of the Generalized HMC algorithm.

    See docco at https://github.com/blackjax-devs/blackjax/blob/bab42d809b48492f2cbc06471497cefbbf8a90f8/blackjax/kernels.py#L750

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Number of chains used for cross-chain warm-up training.
    rng_key: PRNGKey
        Random number generator key.
    positions: PyTree
        Initial position of the chains.
    num_steps: int
        Number of steps to run the adaptation for.

    Returns
    -------
    Last states of the chains and the parameters of the Generalized HMC algorithm.

    """
    step_fn = ghmc.kernel()
    init, update = adaptation.meads.base()
    batch_init = jax.vmap(lambda r, p: ghmc.init(r, p, logjoint_density_fn))

    def one_step_online(carry, rng_key):
        states, adaptation_state = carry

        def kernel(rng_key, state):
            return step_fn(
                rng_key,
                state,
                logjoint_density_fn,
                adaptation_state.step_size,
                adaptation_state.position_sigma,
                adaptation_state.alpha,
                adaptation_state.delta,
            )

        keys = jax.random.split(rng_key, num_chains)
        new_states, info = jax.vmap(kernel)(keys, states)
        new_adaptation_state = update(
            adaptation_state, new_states.position, new_states.potential_energy_grad
        )

        return (new_states, new_adaptation_state), None

    key_init, key_adapt = jax.random.split(prng_key)

    rng_keys = jax.random.split(key_init, num_chains)
    init_states = batch_init(rng_keys, positions)
    init_adaptation_state = init(positions, init_states.potential_energy_grad)

    keys = jax.random.split(key_adapt, num_steps)
    (last_states, last_adaptation_state), _ = jax.lax.scan(
        one_step_online, (init_states, init_adaptation_state), keys
    )

    parameters = {
        "step_size": last_adaptation_state.step_size,
        "momentum_inverse_scale": last_adaptation_state.position_sigma,
        "alpha": last_adaptation_state.alpha,
        "delta": last_adaptation_state.delta
    }

    return last_states, parameters


class ExtendedState(NamedTuple):
    state: blackjax.mcmc.ghmc.GHMCState  # current HMC state
    rng_key: jax.Array  # current random seed
    pred_ws: WelfordState  # accumulator for log predictive
    log_pred_mean: float  # log of mean predictive
    param_ws: WelfordState  # accumulator for parameters
    divergences: jax.Array  # divergence counts (int array)


class TfmExtendedState(NamedTuple):
    """Transformed extended state--also includes accumulators for transformed variables
    """
    state: blackjax.mcmc.ghmc.GHMCState  # current HMC state
    rng_key: jax.Array  # current random seed
    pred_ws: WelfordState  # accumulator for log predictive
    pred_bws: BatchWelfordState  # batch accumulator for log predictive
    pred_tfm_ws: WelfordState # accumulator for transformed log predictive
    log_pred_mean: float  # log of mean predictive
    param_ws: WelfordState  # accumulator for parameters
    param_tfm_ws: WelfordState  # accumulator for transformed parameters
    param_vws: VectorWelfordState  # vector accumulator for parameters
    param_bws: BatchVectorWelfordState  # vector batch accumulator for parameters
    divergences: jax.Array  # divergence counts (int array)


def state_diagnostics(state) -> None:
    """Summarize the state of a state object."""
    # TODO: add mean and s.e. of the parameters
    rhats = [(n, rhat(v)) for n, v in zip(state.param_ws._fields, state.param_ws)]
    predrh, predsrh = rhat(state.pred_ws)
    status = [
        f"       Summary: {int(jnp.sum(state.pred_ws.n[0,:]))} draws * {state.state.position[0].shape[0]} chains",
    ]
    param = [
        f"{n: >9} Rhat: {v} ({desc})"
        for n, (rh, frh) in rhats
        for v, desc in [(rh, "regular"), (frh, "tail")]
    ]
    lines = (
        status
        + param
        + [
            f"    pred. Rhat: {predrh:.4f}  tail: {predsrh:.4f}",
            f"   divergences: {int(jnp.sum(state.divergences))}",
        ]
    )
    print("\n".join(lines))


def estimate_elpd(extended_state: Union[ExtendedState, TfmExtendedState]):
    """Estimate the expected log pointwise predictive density from welford state.

    The resulting elpd is in sum scale, that is we average over (half)
    chains and sum over folds.
    """
    # AVERAGE over (half) chains (chain dim is axis 1, chain half dim is axis 2)
    nchains, nhalfs = extended_state.log_pred_mean.shape[1:3]
    fold_means = logsumexp(extended_state.log_pred_mean, axis=(1, 2)) - jnp.log(
        nchains * nhalfs
    )
    fold_means = fold_means.squeeze()
    # SUM over folds
    elpd = jnp.sum(fold_means)
    return float(elpd)


def mv_ess_folds(extended_state: TfmExtendedState):
    """Multivariate ESS for parameters
    """
    Sigmas = vector_welford_cov_combine(extended_state.param_bws.batches)
    Lambdas = vector_welford_cov_combine(extended_state.param_vws)
    ns = extended_state.param_vws.n.sum(axis=(1,2))
    nfolds, p = Sigmas.shape[0], Sigmas.shape[1]
    def f(i):
        return ns[i] * (jnp.linalg.det(Sigmas[i]) / jnp.linalg.det(Lambdas[i])) ** (1/p)
    return jax.vmap(f)(jnp.arange(nfolds))


def pred_ess_folds(extended_state: TfmExtendedState):
    """Univariate ESS for predictives
    """
    Sigmas = welford_var_combine(extended_state.pred_bws.batches)
    Lambdas = welford_var_combine(extended_state.pred_ws)
    ns = extended_state.pred_ws.n.sum(axis=(1,2))
    return ns * Sigmas / Lambdas


# stack arrays in pytrees
def tree_stack(trees):
    return tree_map(lambda *xs: jnp.stack(xs, axis=0), *trees)


# stack arrays in pytrees
def tree_concat(trees):
    return tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


class SigmoidParam(NamedTuple):
    mean: jax.Array
    std: jax.Array


def sigmoid_transform_param(state: WelfordState, axis: Tuple = 0) -> SigmoidParam:
    """Return sigmoid transform parameters estimated from welford state.
    
    This method combines multiple means and variances by averaging over the specified axis.
    For now it is just the mean and sd but it could change.

    Params:
        state: Welford state with leaves in the shape of (num_chains, param_dim1, ...)
        axis: axis to average over (default 0)
    
    Returns:
        A callable that takes a (possibly vector) input and returns it transformed to [0, 1]
    """
    mean = welford_mean(state).mean(axis=axis)
    std = jnp.sqrt(welford_var(state).mean(axis=axis))
    return SigmoidParam(mean=mean, std=std)


def apply_sigmoid(param: SigmoidParam, x: jax.Array) -> jax.Array:
    """Apply sigmoid transform to input.

    Shape of param and x should match.

    Params:
        param: SigmoidParam
        x: input to transform
    """
    return stats.norm.cdf(x, loc=param.mean, scale=param.std)


def apply_sigmoid_tree(param: PyTree, x: PyTree) -> PyTree:
    """Apply sigmoid transform to input.

    Shape of param and x should match.

    Params:
        param: PyTree of SigmoidParams
        x: PyTree of arrays, input to transform
    """
    return tree_map(apply_sigmoid, param, x, is_leaf=lambda x: isinstance(x, SigmoidParam))


# single chain inference loop we will run in parallel using vmap
def offline_inference_loop(
    rng_key, kernel, initial_state, num_samples, log_pred, _param_sigp, _pred_sigp, theta_center
) -> ExtendedState:
    """Offline version of inference loop.
    
    Because this is offline, we'll need to use the ExtendedState to keep track of the
    transformed version of the parameter/predictive accumulators as well as the regular
    accumulators.

    Params:
        rng_key: random key for the inference loop
        kernel: kernel to use for inference
        initial_state: initial state for inference
        num_samples: number of samples to draw
        log_pred: log predictive density function
        _param_sigp: not used
        _pred_sigp: not used
        theta_center: center of the parameter distribution (used for initialization)
    
    Returns:
        TfmExtendedState for two chain halves
    """
    log_half_samp = jnp.log(0.5 * num_samples)
    init_pred = log_pred(theta_center) - log_half_samp # initial guess for predictive density (for numerical stability)

    def one_mcmc_step(ext_state, _idx):
        i_key, carry_key = jax.random.split(ext_state.rng_key)
        chain_state, chain_info = kernel(i_key, ext_state.state)
        elpd_contrib = (
            log_pred(chain_state.position) - log_half_samp
        )  # contrib to mean log predictive
        carry_log_pred_mean = ext_state.log_pred_mean + jnp.log1p(
            jnp.exp(elpd_contrib - ext_state.log_pred_mean)
        )
        div_count = ext_state.divergences + 1.0 * chain_info.is_divergent
        carry_pred_ws = welford_add(elpd_contrib, ext_state.pred_ws)
        carry_param_ws = tree_map(welford_add, chain_state.position, ext_state.param_ws)
        carry_state = ExtendedState(
            state=chain_state,
            rng_key=carry_key,
            pred_ws=carry_pred_ws,
            log_pred_mean=carry_log_pred_mean,
            param_ws=carry_param_ws,
            divergences=div_count,
        )
        return carry_state, chain_state

    def remove_init_pred(state):
        # remove initial guess for predictive density
        return ExtendedState(
            state=state.state,
            rng_key=state.rng_key,
            pred_ws=state.pred_ws,
            log_pred_mean=state.log_pred_mean + jnp.log1p(
                -jnp.exp(init_pred - state.log_pred_mean)
            ),
            param_ws=state.param_ws,
            divergences=state.divergences,
        )

    # first half of chain
    initial_state_1h = ExtendedState(
        initial_state,
        rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean=init_pred,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0,
    )
    carry_state_1h, states_1h = jax.lax.scan(
        one_mcmc_step, initial_state_1h, jnp.arange(0, num_samples // 2)
    )
    carry_state_1h = remove_init_pred(carry_state_1h)
    # second half of chain - continue at same point but accumulate into new welford states
    initial_state_2h = ExtendedState(
        carry_state_1h.state,
        carry_state_1h.rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean=init_pred,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0,
    )
    carry_state_2h, states_2h = jax.lax.scan(
        one_mcmc_step, initial_state_2h, jnp.arange(num_samples // 2, num_samples)
    )
    carry_state_2h = remove_init_pred(carry_state_2h)
    final_state = tree_stack((carry_state_1h, carry_state_2h,))
    trace = tree_concat((states_1h, states_2h,))
    return final_state, trace


def online_inference_loop(
    rng_key, kernel, initial_state, num_samples, log_pred, param_sigp, pred_sigp, theta_center
) -> TfmExtendedState:
    """Online version of inference loop.
    
    Because this is online, we'll need to use the TfmExtendedState to keep track of the
    transformed version of the parameter/predictive accumulators as well as the regular
    accumulators.

    Params:
        rng_key: random key for the inference loop
        kernel: kernel to use for inference
        initial_state: initial state for inference
        num_samples: number of samples to draw
        log_pred: log predictive density function
        param_sigp: PyTree of transformation parameters of same structure as Theta
        pred_sigp: transformation parameters for predictive density
        theta_center: center of the parameter distribution (used for initialization)
    
    Returns:
        TfmExtendedState for two chain halves
    """
    log_half_samp = jnp.log(0.5 * num_samples)
    init_pred = log_pred(theta_center) - log_half_samp # initial guess for predictive density (for numerical stability)
    batch_size = jnp.floor(num_samples ** 0.5).astype(jnp.int32)  # batch size for computing batch mean, variance
    vec_theta_center = jnp.hstack(jax.tree_util.tree_flatten(theta_center)[0])

    def one_mcmc_step(ext_state: TfmExtendedState, _idx):
        i_key, carry_key = jax.random.split(ext_state.rng_key)
        chain_state, chain_info = kernel(i_key, ext_state.state)
        elpd_contrib = (
            log_pred(chain_state.position) - log_half_samp
        )  # contrib to mean log predictive
        carry_log_pred_mean = ext_state.log_pred_mean + jnp.log1p(
            jnp.exp(elpd_contrib - ext_state.log_pred_mean)
        )
        div_count = ext_state.divergences + 1.0 * chain_info.is_divergent
        carry_pred_ws = welford_add(elpd_contrib, ext_state.pred_ws)
        carry_pred_bws = batch_welford_add(elpd_contrib, ext_state.pred_bws)
        carry_pred_tfm_ws = welford_add(apply_sigmoid(pred_sigp, elpd_contrib), ext_state.pred_tfm_ws)
        carry_param_ws = tree_map(welford_add, chain_state.position, ext_state.param_ws)
        tfm_pos = apply_sigmoid_tree(param_sigp, chain_state.position)
        carry_param_tfm_ws = tree_map(welford_add, tfm_pos, ext_state.param_tfm_ws)
        vec_pos = jnp.hstack(jax.tree_util.tree_flatten(chain_state.position)[0])
        param_bws = batch_vector_welford_add(vec_pos, ext_state.param_bws)
        param_vws = vector_welford_add(vec_pos, ext_state.param_vws)
        carry_state = TfmExtendedState(
            state=chain_state,
            rng_key=carry_key,
            pred_ws=carry_pred_ws,
            pred_tfm_ws=carry_pred_tfm_ws,
            pred_bws=carry_pred_bws,
            param_tfm_ws=carry_param_tfm_ws,
            log_pred_mean=carry_log_pred_mean,
            param_ws=carry_param_ws,
            param_bws=param_bws,
            param_vws=param_vws,
            divergences=div_count,
        )
        return carry_state, None  # don't retain chain trace

    def remove_init_pred(state):
        # remove initial guess for predictive density
        return TfmExtendedState(
            state=state.state,
            rng_key=state.rng_key,
            pred_ws=state.pred_ws,
            pred_tfm_ws=state.pred_tfm_ws,
            pred_bws=state.pred_bws,
            param_tfm_ws=state.param_tfm_ws,
            log_pred_mean=state.log_pred_mean + jnp.log1p(
                -jnp.exp(init_pred - state.log_pred_mean)
            ),
            param_ws=state.param_ws,
            param_bws=state.param_bws,
            param_vws=state.param_vws,
            divergences=state.divergences,
        )

    # first half of chain
    initial_state_1h = TfmExtendedState(
        initial_state,
        rng_key,
        pred_ws=welford_init(init_pred),
        pred_bws=batch_welford_init(init_pred, batch_size),
        pred_tfm_ws=welford_init(0.),
        log_pred_mean=init_pred,
        param_ws=tree_map(welford_init, theta_center),
        param_tfm_ws=tree_map(welford_init, tree_map(jnp.zeros_like, theta_center)),
        param_bws=batch_vector_welford_init(vec_theta_center, batch_size),
        param_vws=vector_welford_init(vec_theta_center),
        divergences=0,
    )
    carry_state_1h, _ = jax.lax.scan(
        one_mcmc_step, initial_state_1h, jnp.arange(0, num_samples // 2)
    )
    carry_state_1h = remove_init_pred(carry_state_1h)
    # second half of chain - continue at same point but feed new accumulator
    initial_state_2h = TfmExtendedState(
        state=carry_state_1h.state,
        rng_key=carry_state_1h.rng_key,
        pred_ws=welford_init(init_pred),
        pred_bws=batch_welford_init(init_pred, batch_size),
        pred_tfm_ws=welford_init(0.),
        log_pred_mean=init_pred,
        param_ws=tree_map(welford_init, theta_center),
        param_tfm_ws=tree_map(welford_init, tree_map(jnp.zeros_like, theta_center)),
        param_bws=batch_vector_welford_init(vec_theta_center, batch_size),
        param_vws=vector_welford_init(vec_theta_center),
        divergences=0,
    )
    carry_state_2h, _ = jax.lax.scan(
        one_mcmc_step, initial_state_2h, jnp.arange(num_samples // 2, num_samples)
    )
    carry_state_2h = remove_init_pred(carry_state_2h)
    return tree_stack((carry_state_1h, carry_state_2h,))


def transform_inference_loop(
    rng_key, kernel, initial_state, num_samples, log_pred, theta_center
):
    """Inference loop for estimating transformation parameters

    This loop is similar to the online inference loop, but does not split chains
    and returns only the information required to estimate transformation parameters.

    Params:
        rng_key: random number generator key
        kernel: MCMC kernel
        initial_state: initial state of the chain
        num_samples: number of samples to draw
        log_pred: log predictive density function
        theta_center: rough guess at center of the distribution
    """
    log_half_samp = jnp.log(0.5 * num_samples + 1)  # +1 for initialization

    def one_mcmc_step(ext_state, _idx):
        i_key, carry_key = jax.random.split(ext_state.rng_key)
        chain_state, chain_info = kernel(i_key, ext_state.state)
        elpd_contrib = (
            log_pred(chain_state.position) - log_half_samp
        )  # contrib to mean log predictive
        carry_log_pred_mean = ext_state.log_pred_mean + jnp.log1p(
            jnp.exp(elpd_contrib - ext_state.log_pred_mean)
        )
        div_count = ext_state.divergences + 1.0 * chain_info.is_divergent
        carry_pred_ws = welford_add(elpd_contrib, ext_state.pred_ws)
        carry_param_ws = tree_map(welford_add, chain_state.position, ext_state.param_ws)
        carry_state = ExtendedState(
            state=chain_state,
            rng_key=carry_key,
            pred_ws=carry_pred_ws,
            log_pred_mean=carry_log_pred_mean,
            param_ws=carry_param_ws,
            divergences=div_count,
        )
        return carry_state, None  # don't retain chain trace

    initial_state = ExtendedState(
        initial_state,
        rng_key,
        pred_ws=welford_init(log_pred(theta_center)),
        log_pred_mean=log_pred(theta_center) - log_half_samp,
        param_ws=tree_map(welford_init, theta_center),
        divergences=0,
    )
    carry_state, _ = jax.lax.scan(
        one_mcmc_step, initial_state, jnp.arange(0, num_samples // 2)
    )
    return carry_state


def fold_posterior(
    prng_key,
    inference_loop,
    logjoint_density,
    log_p,
    make_initial_pos,
    num_chains,
    num_samples,
    warmup_iter,
):
    """Compute posterior for a single fold using parallel chains.

    This function is the building block for parallel CV.

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
    # warmup - GHMC adaption via MEADS
    init_chain_keys = jax.random.split(init_key, num_chains)
    init_states = jax.vmap(make_initial_pos)(init_chain_keys)
    final_warmup_state, parameters = run_meads(
        logjoint_density_fn=logjoint_density,
        num_chains=num_chains,
        prng_key=warmup_key,
        positions=init_states,
        num_steps=warmup_iter,
    )

    # construct GHMC kernel by incorporating warmup parameters
    step_fn = ghmc.kernel()
    def kernel(rng_key, state):
        return step_fn(
            rng_key,
            state,
            logjoint_density,
            **parameters,
        )

    sampling_keys = jax.random.split(sampling_key, num_chains)
    
    # central points for estimating folded rhat
    centers = tree_map(lambda x: jnp.median(x, axis=0), final_warmup_state.position)

    # now that we have adaption parameters, estimate rough mean and variance of parameters and log predictive
    # in order to construct the sigmoid transformation
    tfm_states = jax.vmap(transform_inference_loop, in_axes=(0, None, 0, None, None, None))(
        sampling_keys, kernel, final_warmup_state, warmup_iter, log_p, centers
    )
    param_tfmp: PyTree = tree_map(sigmoid_transform_param, tfm_states.param_ws, is_leaf=lambda x: isinstance(x, WelfordState))
    pred_tfmp: SigmoidParam = sigmoid_transform_param(tfm_states.pred_ws)

    # run chain
    results = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None, None, None))(
        sampling_keys, kernel, final_warmup_state, num_samples, log_p, param_tfmp, pred_tfmp, centers
    )
    return results


def split_rhat(means: jax.Array, vars: jax.Array, n: int) -> float:
    """Compute a single split Rhat from summary statistics of split chains.

    Args:
        means: means of split chains
        vars:  variances of split chains
        n:     number of draws per split chain (ie half draws in an original chain)
    """
    W = jnp.mean(vars, axis=1)
    # m = means.shape[1]  # number of split chains
    B = n * jnp.var(means, ddof=1, axis=1)
    varplus = (n - 1) / n * W + B / n
    Rhat = jnp.sqrt(varplus / W)
    return Rhat


def split_rhat_welford(ws: WelfordState) -> float:
    """Compute split Rhat from Welford state of split chains.

    Args:
        ws: Welford state of split chains

    Returns:
        split Rhat: array of split Rhats
    """
    means = jax.vmap(welford_mean)(ws)
    vars = jax.vmap(welford_var)(ws)
    n = ws.n[:, 0, ...]  # we aggregate over chain dim, axis=1
    return split_rhat(means, vars, n)


def folded_split_rhat_welford(ws: WelfordState) -> float:
    """Compute folded split Rhat from Welford state of split chains.

    Args:
        ws: Welford state of split chains

    Returns:
        folded split Rhat: array of folded split Rhats
    """
    mads = jax.vmap(welford_mad)(ws)
    vars = jax.vmap(welford_var)(ws)
    n = ws.n[:, 0, ...]
    return split_rhat(mads, vars, n)


def rhat(welford_tree):
    """Compute split Rhat and folded split Rhat from welford states of split chains.

    This version assumes there are multiple posteriors, so that the states have dimension
    (cv_fold #, chain #, half #, ...).

    Args:
        welford_tree: pytree of Welford states for split chains

    Returns:
        split Rhat: pytree pytree of split Rhats
        folded split Rhat: pytree of folded split Rhats
    """
    # collapse axis 1 (chain #) and axis 2 (half #) to a single dimension
    com_chains = tree_map(
        lambda x: jnp.reshape(x, (x.shape[0], -1, *x.shape[3:])), welford_tree
    )
    sr = tree_map(
        split_rhat_welford, com_chains, is_leaf=lambda x: isinstance(x, WelfordState)
    )
    fsr = tree_map(
        folded_split_rhat_welford,
        com_chains,
        is_leaf=lambda x: isinstance(x, WelfordState),
    )
    return sr, fsr


def rhat_summary(fold_states):
    """Compute split Rhat and folded split Rhat from welford states of split chains.

    This version assumes there are multiple posteriors, so that the states have dimension
    (cv_fold #, chain #, half #, ...).

    Args:
        fold_states: pytree of Welford states for split chains

    Returns:
        pandas data frame summarizing rhats
    """
    par_rh, par_frh = rhat(fold_states.param_ws)
    par_rh_tfm, par_frh_tfm = rhat(fold_states.param_tfm_ws)
    pred_rh, pred_frh = rhat(fold_states.pred_ws)
    pred_rh_tfm, pred_frh_tfm = rhat(fold_states.pred_tfm_ws)
    K = pred_rh.shape[0]
    rows = []
    max_row = None
    for i in range(K):
        for (par, pred, meas) in [
            (par_rh, pred_rh, "Split Rhat"),
            (par_rh_tfm, pred_rh_tfm, "Split Rhat (transformed)"),
            (par_frh, pred_frh, "Folded Split Rhat"),
            (par_frh_tfm, pred_frh_tfm, "Folded Split Rhat (transformed)"),
        ]:
            row = {"fold": f"Fold {i}", "measure": meas}
            for j, parname in enumerate(par._fields):
                if jnp.ndim(par[j]) > 1:
                    # vector parameter, add a column for each element
                    for k in range(par[j].shape[1]):
                        row[f"{parname}[{k}]"] = float(par[j][i][k])
                else:
                    row[parname] = float(par[j][i])
            row["log p"] = float(pred[i])
            rows.append(row)
            if max_row:
                max_row = {
                    k: max_row[k]
                    if isinstance(max_row[k], str)
                    else max(max_row[k], row[k])
                    for k in max_row
                }
            else:
                max_row = row.copy()
                max_row.update({"fold": "All folds", "measure": "Max"})
    rows.append(max_row)
    return pd.DataFrame(rows)
