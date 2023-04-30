from typing import Callable, Dict, NamedTuple, Tuple, Union
import time

import blackjax.adaptation as adaptation
from blackjax.adaptation.meads import MEADSAdaptationState
import jax
import jax.numpy as jnp
import pandas as pd
from blackjax.kernels import ghmc
from blackjax.types import PyTree
from blackjax.mcmc.ghmc import GHMCState
from jax import lax
from jax import numpy as jnp
from jax.scipy import stats
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from pcv.welford import *
from pcv.util import logmean, logvar, tree_stack, tree_concat
from pcv.model import Model, get_mode, minimize_adam
from pcv.rules import CONTINUE, STOP_INCONCLUSIVE, STOP_CONCLUSIVE


def run_meads(
    logjoint_density_fn: Callable,
    num_chains: int,
    prng_key: jax.random.KeyArray,
    positions: PyTree,
    num_steps: int = 1000,
) -> Tuple[GHMCState, Dict]:
    """Adapt the parameters of the Generalized HMC algorithm.

    See docco at https://github.com/blackjax-devs/blackjax/blob/bab42d809b48492f2cbc06471497cefbbf8a90f8/blackjax/kernels.py#L750

    Parameters
    ----------
    logdensity_fn
        The log density to sample from.
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
    init_states = jax.vmap(lambda r, p: ghmc.init(r, p, logjoint_density_fn))(rng_keys, positions)
    init_adaptation_state = init(positions, init_states.potential_energy_grad)
    keys = jax.random.split(key_adapt, num_steps)
    (last_states, last_adaptation_state), _ = jax.lax.scan(
        one_step_online, (init_states, init_adaptation_state), keys
    )
    parameters = {
        "step_size": last_adaptation_state.step_size,
        "momentum_inverse_scale": last_adaptation_state.position_sigma,
        "alpha": last_adaptation_state.alpha,
        "delta": last_adaptation_state.delta,
    }
    return last_states, parameters


def rerun_meads(
    logjoint_density_fn: Callable,
    num_chains: int,
    prng_key: jax.random.KeyArray,
    states: GHMCState,
    adaptation_state: MEADSAdaptationState,
    num_steps: int = 1000,
) -> Tuple[GHMCState, Dict]:
    """Adapt GHMC parameters, starting with already adapted parameters.

    Scope: 1 fold, num_chains chains.

    The purpose of this function is to adapt parameters for individual folds,
    after having already adapted parameters for the entire dataset.
    See docco at https://github.com/blackjax-devs/blackjax/blob/bab42d809b48492f2cbc06471497cefbbf8a90f8/blackjax/kernels.py#L750

    Parameters
    ----------
    logdensity_fn
        The log density to sample from
    num_chains
        Number of chains used for cross-chain warm-up training.
    rng_key: PRNGKey
        Random number generator key.
    states:
        Current GHMC state of the chains
    adaption_state:
        Adaption state to continue working with
    num_steps: int
        Number of steps to run the adaptation for.

    Returns
    -------
    Last states of the chains and the parameters of the Generalized HMC algorithm.

    """
    step_fn = ghmc.kernel()
    _, update = adaptation.meads.base()

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

    keys = jax.random.split(prng_key, num_steps)
    (last_states, last_adaptation_state), _ = jax.lax.scan(
        one_step_online, (states, adaptation_state), keys
    )
    parameters = {
        "step_size": last_adaptation_state.step_size,
        "momentum_inverse_scale": last_adaptation_state.position_sigma,
        "alpha": last_adaptation_state.alpha,
        "delta": last_adaptation_state.delta,
    }
    return last_states, parameters


class ModelWarmupResults(NamedTuple):
    """Warmup results for full-data model."""
    num_chains: int
    num_models: int
    num_iter: int
    states: GHMCState
    parameters: Dict


class ExtendedState(NamedTuple):
    """MCMC state--extends regular GHMC state variable--also includes batch welford accumulators"""
    state: GHMCState  # current HMC state
    rng_key: jax.random.KeyArray  # current random seed
    pred_ws: LogWelfordState  # accumulator for log predictive
    pred_bws: BatchLogWelfordState  # batch accumulator for log predictive, size b
    pred_bws3: BatchLogWelfordState  # batch accumulator for log predictive, size floor(b/3)
    divergences: jax.Array  # divergence counts (int array)


class FoldWarmupResults(NamedTuple):
    """Warmup results for a multi-model CV procedure."""
    num_chains: int
    num_models: int
    num_folds: int
    model_warmup_iter: int
    model_states: ExtendedState
    model_parameters: Dict  # GHMC hyper-parameters
    fold_warmup_iter: int
    fold_states: ExtendedState
    fold_parameters: Dict  # GHMC hyper-parameters

    def plot_dist(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        for m in range(self.num_models):
            model_fold_idxs = jnp.arange(m*self.num_folds, (m+1)*self.num_folds)
            for i, key in enumerate(['step_size', 'alpha', 'delta']):
                axes[i][m].hist(self.fold_parameters[key][model_fold_idxs])
                axes[i][m].axvline(self.model_parameters[key][m])
                axes[i][m].set_title(f"Model {m} {key}")
        fig.tight_layout()


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


def estimate_elpd_diff(
    ext_state: ExtendedState, model_A_folds: jax.Array, model_B_folds: jax.Array
):
    """Estimate the expected log pointwise predictive density from welford state.

    The resulting elpd is in sum scale, that is we average over (half)
    chains and sum over folds.
    """
    # AVERAGE over chains (chain dim is axis 1, chain half dim is axis 2)
    fmeans = logmean(log_welford_mean(ext_state.pred_ws), axis=1)
    fmeans = fmeans.squeeze()
    # SUM over folds
    fold_indexes = jnp.arange(fmeans.shape[0])
    model_A = float(((fold_indexes == model_A_folds) * fmeans).sum())
    model_B = float(((fold_indexes == model_B_folds) * fmeans).sum())
    elpd_diff = model_A - model_B
    return elpd_diff, model_A, model_B


def pred_ess_folds(extended_state: ExtendedState):
    """Univariate ESS for predictives"""
    log_Sigmas = log_welford_var_combine(extended_state.pred_bws.batches, comb_axis=1)
    log_Lambdas = log_welford_var_combine(extended_state.pred_ws, comb_axis=1)
    ns = extended_state.pred_ws.n.sum(axis=(1,))
    return ns * jnp.exp(log_Sigmas - log_Lambdas)


def inference_loop(
    rng_key, kernel, initial_state, num_samples, log_pred, online=True
) -> Tuple[ExtendedState, ExtendedState]:
    """Optionally online inference loop.

    This inference loop can be either online or offline, governed by the
    online flag. State is kept in the ExtendedState, which is a wrapper
    around the Blackjax HMCState object.

    We use regular python control flow to switch between online and offline
    versions of the inference loop. This is optimized out by JAX, so there
    is (hopefully) no performance penalty.

    Params:
        rng_key: random key for the inference loop
        kernel: kernel to use for inference
        initial_state: initial state for inference
        num_samples: number of samples to draw
        log_pred: log predictive density function
        online: (default True) if true, don't retain the MCMC trace

    Returns:
        2-Tuple containing ExtendedState for two chain halves, and the MCMC
        trace if online if False.
    """
    log_samp = jnp.log(num_samples)
    batch_size: int = int(
        jnp.floor(num_samples**0.5)
    )  # batch size for computing batch mean, variance

    def one_mcmc_step(ext_state: ExtendedState, _idx):
        i_key, carry_key = jax.random.split(ext_state.rng_key)
        chain_state, chain_info = kernel(i_key, ext_state.state)
        elpd_contrib = log_pred(chain_state.position)
        carry_state = ExtendedState(
            state=chain_state,
            rng_key=carry_key,
            pred_ws=log_welford_add(elpd_contrib, ext_state.pred_ws),
            pred_bws=batch_log_welford_add(elpd_contrib, ext_state.pred_bws),
            pred_bws3=batch_log_welford_add(elpd_contrib, ext_state.pred_bws3),
            divergences=ext_state.divergences + chain_info.is_divergent,
        )
        if online:
            return carry_state, None  # don't retain chain trace
        else:
            return carry_state, chain_state

    init_state = ExtendedState(
        state=initial_state,
        rng_key=rng_key,
        pred_ws=log_welford_init(shape=tuple()),
        pred_bws=batch_log_welford_init(shape=tuple(), batch_size=batch_size),
        pred_bws3=batch_log_welford_init(shape=tuple(), batch_size=batch_size//3),
        divergences=jnp.array(0),
    )
    state, trace = jax.lax.scan(one_mcmc_step, init_state, jnp.arange(0, num_samples))
    return state, trace


def fold_posterior(
    prng_key: jax.random.KeyArray,
    logjoint_density: Callable,
    log_p: Callable,
    make_initial_pos: Callable,
    num_chains: int,
    num_samples: int,
    warmup_iter: int,
    online: bool,
):
    """Compute posterior for a single fold using parallel chains.

    This function is the building block for parallel CV.

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        logjoint_density: callable, log joint density function
        log_p: callable, log density for this fold
        make_initial_pos: callable, function to make initial position for each chain
        num_chains: int, number of chains to run
        num_samples: int, number of samples to draw
        warmup_iter: int, number of warmup iterations to run
        online: bool (default True) if true, don't retain the MCMC trace

    Returns:
        2-tuple of:
            ExtendedState, final state of the inference loop
            trace of posterior draws if the offline inference loop was used, otherwise None
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
    # run chain
    results = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None))(
        sampling_keys, kernel, final_warmup_state, num_samples, log_p, online
    )
    return results


def init_batch_inference_state(
    rng_key: jax.random.KeyArray,
    num_chains: int,
    make_initial_pos: Callable,
    logjoint_density: Callable,
    batch_size: int,
    warmup_iter: int,
) -> Tuple[ExtendedState, Dict]:
    """Initialize batched inference loop.

    Scope: one fold, many chains.

    Params:
        rng_key: random key for the inference loop
        initial_state: initial state for inference
        batch_size: number of samples to draw per batch

    Returns:
        2-Tuple containing ExtendedState for two chain halves, and the MCMC
        trace if online if False.
    """
    init_key, warmup_key, sampling_key = jax.random.split(rng_key, 3)
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
    sampling_keys = jax.random.split(sampling_key, num_chains)
    # initialize all chains for fold
    def create_state(chain_init_state, chain_rng_key):
        return ExtendedState(
            state=chain_init_state,
            rng_key=chain_rng_key,
            pred_ws=log_welford_init(shape=tuple()),
            pred_bws=batch_log_welford_init(shape=tuple(), batch_size=batch_size),
            pred_bws3=batch_log_welford_init(shape=tuple(), batch_size=batch_size//3),
            divergences=jnp.array(0),
        )
    states = jax.vmap(create_state, in_axes=(0,0,),)(final_warmup_state, sampling_keys)
    return states, parameters


def fold_batched_inference_loop(
    kernel_parameters, logjoint_density, ext_states, batch_size, log_pred
) -> ExtendedState:
    """Batched online inference loop.

    Scope: one fold, many chains.

    This one is always online, and performs a single batch at a time.
    It requires a fully initialized ExtendedState, made by init_inference_loop().

    Params:
        kernel_parameters: kernel parameters to use for inference
        ext_states: starting state for inference, all chains
        batch_size: number of samples to draw
        log_pred: log predictive density function

    Returns:
        ExtendedState for all chains
    """
    # construct GHMC kernel by incorporating warmup parameters
    step_fn = ghmc.kernel()

    def kernel(rng_key, state):
        return step_fn(
            rng_key,
            state,
            logjoint_density,
            **kernel_parameters,
        )

    def one_chain_inference_loop(state):
        """Single chain inference loop."""

        def one_mcmc_step(ext_state: ExtendedState, _idx):
            """Single chain, single MCMC step."""
            iter_key, carry_key = jax.random.split(ext_state.rng_key)
            chain_state, chain_info = kernel(iter_key, ext_state.state)
            elpd_contrib = log_pred(chain_state.position)
            carry_state = ExtendedState(
                state=chain_state,
                rng_key=carry_key,
                pred_ws=log_welford_add(elpd_contrib, ext_state.pred_ws),
                pred_bws=batch_log_welford_add(elpd_contrib, ext_state.pred_bws),
                pred_bws3=batch_log_welford_add(elpd_contrib, ext_state.pred_bws3),
                divergences=ext_state.divergences + chain_info.is_divergent,
            )
            return carry_state, None  # don't retain chain trace

        next_state, _ = jax.lax.scan(one_mcmc_step, state, jnp.arange(0, batch_size))
        return next_state

    # run all chains for this fold in parallel
    next_state = jax.vmap(one_chain_inference_loop, in_axes=(0,))(ext_states)
    return next_state


def one_model_inference(
    prng_key: jax.random.KeyArray,
    model: Model,
    model_id: int,
    num_chains: int,
    num_samples: int,
    warmup_iter: int,
    prior_only: bool = False,
    adam_steps: int = 100,
    adam_lr: float = 0.1
):
    """Compute posterior for a single fold using parallel chains.

    This function is the building block for parallel CV.

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        logjoint_density: callable, log joint density function, signature (theta, fold_id, model_id, prior_only)
        log_p: callable, log density, signature (theta, fold_id, model_id)
        make_initial_pos: callable, function to make initial position for each chain
        model_id: int, model index as understood by the likelihood and prior functions
        num_chains: int, number of chains to run
        warmup_iter: int, number of warmup iterations to run
        online: bool (default True) if true, don't retain the MCMC trace
        num_batches: num batches

    Returns:
        Arviz InferenceData object
        Final state of the MCMC chain
    """
    import arviz as az
    def model_ldens(theta):
        return model.logjoint_density(theta, fold_id=-1, model_id=model_id, prior_only=prior_only)
    def model_log_p(theta):
        return model.log_pred(theta, fold_id=-1, model_id=model_id)
    warmup_key, sampling_key, init_key = jax.random.split(prng_key, 3)
    # warmup - GHMC adaption via MEADS
    init_chain_keys = jax.random.split(init_key, num_chains)
    init_states = jax.vmap(lambda key: get_mode(model, key, niter=adam_steps, learning_rate=adam_lr))(init_chain_keys)
    final_warmup_state, parameters = run_meads(
        logjoint_density_fn=model_ldens,
        num_chains=num_chains,
        prng_key=warmup_key,
        positions=init_states,
        num_steps=warmup_iter,
    )
    print(f"GHMC alpha: {parameters['alpha']:.4f}, delta: {parameters['delta']:.4f}, step size: {parameters['step_size']:.4f}")
    # construct GHMC kernel by incorporating warmup parameters
    step_fn = ghmc.kernel()
    def kernel(rng_key, state):
        return step_fn(
            rng_key,
            state,
            model_ldens,
            **parameters,
        )
    sampling_keys = jax.random.split(sampling_key, num_chains)
    # run chain
    online = False
    state, trace = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None))(
        sampling_keys, kernel, final_warmup_state, num_samples, model_log_p, online
    )
    # apply transformations to get back to constrained parameter space
    constrained_trace = jax.vmap(model.to_constrained)(trace.position)
    idata = az.convert_to_inference_data({
        v : constrained_trace[i] for i, v in enumerate(trace.position._fields)})
    return idata, state


def one_model_inference_mtune(
    prng_key: jax.random.KeyArray,
    model: Model,
    mcmc_step_fn: Callable,
    init_state_fn: Callable,
    model_id: int,
    num_chains: int,
    num_samples: int,
    burnin_iter: int,
    prior_only: bool = False,
):
    """Compute posterior for a single model using parallel chains, manually tuning kernel

    This function is the building block for parallel CV.

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        model: model object containing logjoint_density, log_pred, make_initial_pos
        mcmc_step_fn: Callable, MCMC kernel step function
        init_state_fn: Callable, function to create new, random parameter objects
        parameters: Dict, parameters for mcmc_kernel
        model_id: int, model index as understood by the likelihood and prior functions
        num_chains: int, number of chains to run
        warmup_iter: int, number of warmup iterations to run
        online: bool (default True) if true, don't retain the MCMC trace
        num_batches: num batches

    Returns:
        Arviz InferenceData object
        Final state of the MCMC chain
    """
    import arviz as az
    def model_ldens(theta):
        return model.logjoint_density(theta, fold_id=-1, model_id=model_id, prior_only=prior_only)
    def model_log_p(theta):
        return model.log_pred(theta, fold_id=-1, model_id=model_id)
    burnin_key, sampling_key, init_key = jax.random.split(prng_key, 3)
    # warmup - GHMC adaption via MEADS
    init_chain_keys = jax.random.split(init_key, num_chains)
    initial_positions = jax.vmap(get_mode, in_axes=(None, 0))(model, init_chain_keys)
    initial_state = jax.vmap(init_state_fn)(initial_positions)
    # We'll call this initial sample "burn in" to avoid confusion with proper warmup, we just
    # want to get the chains to a reasonable starting point
    online = True
    burnin_keys = jax.random.split(burnin_key, num_chains)
    initial_estate, _ = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None))(
        burnin_keys, mcmc_step_fn, initial_state, burnin_iter, model_log_p, online
    )
    sampling_keys = jax.random.split(sampling_key, num_chains)
    # run chain
    online = False
    state, trace = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None))(
        sampling_keys, mcmc_step_fn, initial_estate.state, num_samples, model_log_p, online
    )
    # apply transformations to get back to constrained parameter space
    constrained_trace = jax.vmap(model.to_constrained)(trace.position)
    idata = az.convert_to_inference_data({
        v : constrained_trace[i] for i, v in enumerate(trace.position._fields)})
    return idata, state


def cv_adaptation(
    prng_key: jax.random.KeyArray,
    model: Model,
    num_chains: int,
    model_warmup_iter: int,
    fold_warmup_iter: int,
    batch_size: int
) -> FoldWarmupResults:
    """Adaptation procedure for full-data model(s) and all CV folds.

    Runs MEADS adaptation procedure for full-data model(s), followed by MEADS
    for each CV fold of each model. The CV folds begin with the same initial
    conditions.

    Models and folds must be numbered sequentially from zero. Fold -1 means
    all data.

    Scope: multi-model * multi-fold * multi-chain

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        model: Model object
        num_chains: int, number of chains to run
        model_warmup_iter: int, number of warmup iterations to run
        fold_warmup_iter: int, number of warmup iterations to run
        batch_size: int, number of samples to draw
    """
    init_key, model_warmup_key, fold_warmup_key, sampling_key = jax.random.split(prng_key, 4)
    step_fn = ghmc.kernel()
    init_a_s, update_a_s = adaptation.meads.base()

    def to_params(last_adaptation_state):
        return {
            "step_size": last_adaptation_state.step_size,
            "momentum_inverse_scale": last_adaptation_state.position_sigma,
            "alpha": last_adaptation_state.alpha,
            "delta": last_adaptation_state.delta,
        }
    
    def create_state(chain_init_state, chain_rng_key):
        return ExtendedState(
            state=chain_init_state,
            rng_key=chain_rng_key,
            pred_ws=log_welford_init(shape=tuple()),
            pred_bws=batch_log_welford_init(shape=tuple(), batch_size=batch_size),
            pred_bws3=batch_log_welford_init(shape=tuple(), batch_size=batch_size//3),
            divergences=jnp.array(0),
        )

    def full_data_warmup(model_id):
        # curry all but the parameter for the density fn
        def model_density(theta):
            return model.logjoint_density(theta, fold_id=-1, model_id=model_id, prior_only=False)

        def one_adaptation_step(carry, rng_key):
            states, adaptation_state = carry
            def kernel(rng_key, state):
                return step_fn(
                    rng_key,
                    state,
                    model_density,
                    adaptation_state.step_size,
                    adaptation_state.position_sigma,
                    adaptation_state.alpha,
                    adaptation_state.delta,
                )
            keys = jax.random.split(rng_key, num_chains)
            new_states, info = jax.vmap(kernel)(keys, states)
            new_adaptation_state = update_a_s(
                adaptation_state, new_states.position, new_states.potential_energy_grad
            )
            return (new_states, new_adaptation_state), None  # None means online adaptation

        key_init, key_adapt = jax.random.split(model_warmup_key)
        init_state_keys = jax.random.split(key_init, num_chains)
        init_positions = jax.vmap(lambda key: get_mode(model, key))(jax.random.split(init_key, num_chains))
        init_states: GHMCState = jax.vmap(lambda r, p: ghmc.init(r, p, model_density))(init_state_keys, init_positions)
        init_adaptation_state: MEADSAdaptationState = init_a_s(init_positions, init_states.potential_energy_grad)
        # run adaptation
        (last_states, last_adaptation_state), _ = jax.lax.scan(
            one_adaptation_step, (init_states, init_adaptation_state), jax.random.split(key_adapt, model_warmup_iter)
        )
        model_extended_states = jax.vmap(create_state)(last_states, jax.random.split(sampling_key, num_chains))
        return last_states, last_adaptation_state, model_extended_states

    print(f"MEADS warmup for {model.num_models} model(s) ({model.num_models*num_chains} chains)...")
    started_at = time.time()
    model_warmup_states, model_adaptation_states, model_extended_states = jax.vmap(full_data_warmup)(jnp.arange(model.num_models))
    print(f"Meads warmup done in {time.time() - started_at:.2f} seconds. ")
    print(f"Step size: {model_adaptation_states.step_size} "
        f"Alpha: {model_adaptation_states.alpha} "
        f"Delta: {model_adaptation_states.delta}")
    def folds_warmup(model_id, fold_id):
        # warmup - GHMC adaption via MEADS
        def fold_density(theta):
            return model.logjoint_density(
                theta=theta,
                fold_id=fold_id,
                model_id=model_id,
                prior_only=False)
        def one_step_online(carry, rng_key):
            states, adaptation_state = carry

            def kernel(rng_key, state):
                return step_fn(
                    rng_key,
                    state,
                    fold_density,
                    adaptation_state.step_size,
                    adaptation_state.position_sigma,
                    adaptation_state.alpha,
                    adaptation_state.delta,
                )
            keys = jax.random.split(rng_key, num_chains)
            new_states, info = jax.vmap(kernel)(keys, states)
            new_adaptation_state = update_a_s(
                adaptation_state, new_states.position, new_states.potential_energy_grad
            )
            return (new_states, new_adaptation_state), None

        keys = jax.random.split(fold_warmup_key, fold_warmup_iter)
        warmup_state = jax.tree_map(lambda x: x[model_id], model_warmup_states)
        adaptation_state = jax.tree_map(lambda x: x[model_id], model_adaptation_states)
        (last_states, last_adaptation_state), _ = jax.lax.scan(
            one_step_online, (warmup_state, adaptation_state), keys
        )
        parameters = to_params(last_adaptation_state)
        sampling_keys = jax.random.split(sampling_key, num_chains)
        # initialize all chains for fold
        states = jax.vmap(create_state, in_axes=(0,0,),)(last_states, sampling_keys)
        return states, parameters
    print(f"MEADS warmup for {model.num_folds} folds per model ({model.num_folds*num_chains*model.num_models} chains)...")
    start_at = time.time()
    fold_ids = jnp.tile(jnp.arange(model.num_folds), model.num_models)
    model_ids = jnp.repeat(jnp.arange(model.num_models), model.num_folds)
    warmup_states, warmup_parameters = jax.vmap(folds_warmup)(model_ids, fold_ids)
    total_sec = time.time() - start_at
    min, sec = int(total_sec) // 60, total_sec % 60
    print(f"MEADS warmup took {min} min {sec:.1f} sec")
    model_parameters = to_params(model_adaptation_states)
    return FoldWarmupResults(
        num_models=model.num_models,
        num_folds=model.num_folds,
        num_chains=num_chains,
        model_warmup_iter=model_warmup_iter,
        model_states=model_extended_states,
        model_parameters=model_parameters,
        fold_warmup_iter=fold_warmup_iter,
        fold_states=warmup_states,
        fold_parameters=warmup_parameters)


def simple_cv_adaptation(
    prng_key: jax.random.KeyArray,
    model: Model,
    warmup_model_id: int,
    num_chains: int,
    adaptation_iter: int,
    burnin_iter: int,
    batch_size: int
) -> FoldWarmupResults:
    """Adaptation procedure for full-data model(s) and all CV folds.

    Runs MEADS adaptation procedure for full-data model(s), followed by MEADS
    for each CV fold of each model. The CV folds begin with the same initial
    conditions.

    Models and folds must be numbered sequentially from zero. Fold -1 means
    all data.

    Scope: multi-model * multi-fold * multi-chain

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        model: Model object
        warmup_model_id: index of model to warm up on
        num_chains: int, number of chains to run
        model_warmup_iter: int, number of warmup iterations to run
        fold_warmup_iter: int, number of warmup iterations to run
        batch_size: int, number of samples to draw
    """
    init_key, model_warmup_key, fold_warmup_key, sampling_key, burnin_key = jax.random.split(prng_key, 5)
    step_fn = ghmc.kernel()
    init_a_s, update_a_s = adaptation.meads.base()
    def create_state(chain_init_state, chain_rng_key):
        return ExtendedState(
            state=chain_init_state,
            rng_key=chain_rng_key,
            pred_ws=log_welford_init(shape=tuple()),
            pred_bws=batch_log_welford_init(shape=tuple(), batch_size=batch_size),
            pred_bws3=batch_log_welford_init(shape=tuple(), batch_size=batch_size//3),
            divergences=jnp.array(0),
        )
    # step 1: MEADS adaptation for the chosen model
    # curry all but the parameter for the density fn
    def model_density(theta):
        return model.logjoint_density(theta, fold_id=-1, model_id=warmup_model_id, prior_only=False)
    def one_adaptation_step(carry, rng_key):
        states, adaptation_state = carry
        def kernel(rng_key, state):
            return step_fn(
                rng_key,
                state,
                model_density,
                adaptation_state.step_size,
                adaptation_state.position_sigma,
                adaptation_state.alpha,
                adaptation_state.delta,
            )
        keys = jax.random.split(rng_key, num_chains)
        new_states, info = jax.vmap(kernel)(keys, states)
        new_adaptation_state = update_a_s(
            adaptation_state, new_states.position, new_states.potential_energy_grad
        )
        return (new_states, new_adaptation_state), None  # None means online adaptation
    started_at = time.time()
    print(f"MEADS warmup for {model.num_models} model(s) ({model.num_models*num_chains} chains)...")
    key_init, key_adapt = jax.random.split(model_warmup_key)
    init_state_keys = jax.random.split(key_init, num_chains)
    init_positions = jax.vmap(lambda key: get_mode(model, key))(jax.random.split(init_key, num_chains))
    init_states: GHMCState = jax.vmap(lambda r, p: ghmc.init(r, p, model_density))(init_state_keys, init_positions)
    init_adaptation_state: MEADSAdaptationState = init_a_s(init_positions, init_states.potential_energy_grad)
    # run adaptation
    (last_states, last_adaptation_state), _ = jax.lax.scan(
        one_adaptation_step, (init_states, init_adaptation_state), jax.random.split(key_adapt, adaptation_iter)
    )
    model_extended_states = jax.vmap(create_state)(last_states, jax.random.split(sampling_key, num_chains))
    print(f"Meads warmup done in {time.time() - started_at:.2f} seconds. ")
    print(f"Step size: {last_adaptation_state.step_size:.4f} "
        f"Alpha: {last_adaptation_state.alpha:.4f} "
        f"Delta: {last_adaptation_state.delta:.4f}")
    # step 2: find starting point for each fold using stochastic optimization (adam)
    start_at = time.time()
    fold_ids = jnp.tile(jnp.arange(model.num_folds), model.num_models)
    model_ids = jnp.repeat(jnp.arange(model.num_models), model.num_folds)
    def find_fold_mode(key, model_id, fold_id):
        return minimize_adam(
            loss = lambda x: -model.logjoint_density(x, fold_id, model_id),
            x0 = model.make_initial_pos(key)
        )
    init_key = jax.random.PRNGKey(0)
    fold_modes = jax.vmap(find_fold_mode, in_axes=(None,0,0))(init_key, model_ids, fold_ids)
    total_sec = time.time() - start_at
    min, sec = int(total_sec) // 60, total_sec % 60
    print(f"Approximated {model.num_models*model.num_folds} modes in {min} min {sec:.1f} sec")
    # step 3: short burn-in for each chain to obtain starting point
    step_fn = ghmc.kernel()
    kernel_parameters = {
        "step_size": last_adaptation_state.step_size,
        "momentum_inverse_scale": last_adaptation_state.position_sigma,
        "alpha": last_adaptation_state.alpha,
        "delta": last_adaptation_state.delta,
    }
    def fold_burn_in(key, model_id, fold_id, initial_pos):
        log_dens = lambda theta: model.logjoint_density(theta, model_id=model_id, fold_id=fold_id)
        log_pred = lambda theta: model.log_pred(theta, model_id=model_id, fold_id=fold_id)
        kernel = lambda rng_key, state: step_fn(rng_key, state, log_dens, **kernel_parameters)
        bkey, ikey = jax.random.split(key)
        burnin_keys = jax.random.split(bkey, num_chains)
        init_chains_keys = jax.random.split(ikey, num_chains)
        initial_states = jax.vmap(ghmc.init, in_axes=(0, None, None))(init_chains_keys, initial_pos, log_dens)
        state, _ = jax.vmap(inference_loop, in_axes=(0, None, 0, None, None, None))(
            burnin_keys, kernel, initial_states, burnin_iter, log_pred, True)
        return state
    burnin_keys = jax.random.split(burnin_key, model.num_models*model.num_folds)
    fold_states = jax.vmap(fold_burn_in)(burnin_keys, model_ids, fold_ids, fold_modes)
    # broadcast fold parameters for use by main cv proceudre
    fold_parameters = jax.tree_map(lambda x: jnp.broadcast_to(x, shape=(model.num_models*model.num_folds,)+x.shape), kernel_parameters)
    total_sec = time.time() - start_at
    min, sec = int(total_sec) // 60, total_sec % 60
    print(f"Burn-in for {num_chains*model.num_folds*model.num_models} chains run took {min} min {sec:.1f} sec")
    if jnp.sum(fold_states.divergences) > 0:
        print(f"**** WARNING: {jnp.sum(fold_states.divergences)} divergences during burn-in. ****")
    return FoldWarmupResults(
        num_models=model.num_models,
        num_folds=model.num_folds,
        num_chains=num_chains,
        model_warmup_iter=adaptation_iter,
        model_states=model_extended_states,
        model_parameters=kernel_parameters,
        fold_warmup_iter=0,
        fold_states=fold_states,
        fold_parameters=fold_parameters)


def run_cv(
    prng_key: jax.random.KeyArray,
    logjoint_density: Callable,
    log_p: Callable,
    make_initial_pos: Callable,
    num_folds: int,
    num_chains: int,
    batch_size: int,
    warmup_iter: int,
    max_batches: int,
):
    """Compute posterior for a single fold using parallel chains.

    Scope: one chain, one fold.

    This function is the building block for parallel CV.

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        logjoint_density: callable, log joint density function, signature (theta, fold_id)
        log_p: callable, log density, signature (theta, fold_id)
        make_initial_pos: callable, function to make initial position for each chain
        num_chains: int, number of chains to run
        batch_size: int, iterations per batch
        warmup_iter: int, number of warmup iterations to run
        online: bool (default True) if true, don't retain the MCMC trace
        max_batches: int, maximum number of batches to run

    Returns:
        2-tuple of:
            ExtendedState, final state of the inference loop
    """

    def fold_warmup(fold_id):
        return init_batch_inference_state(
            rng_key=prng_key,
            num_chains=num_chains,
            make_initial_pos=make_initial_pos,
            logjoint_density=lambda theta: logjoint_density(theta, fold_id),
            batch_size=batch_size,
            warmup_iter=warmup_iter,
        )

    def run_batch(fold_id, fold_kernel_params, fold_state):
        return fold_batched_inference_loop(
            kernel_parameters=fold_kernel_params,
            batch_size=batch_size,
            ext_states=fold_state,
            logjoint_density=lambda theta: logjoint_density(theta, fold_id),
            log_pred=lambda theta: log_p(theta, fold_id),
        )

    print(f"Warmup: {warmup_iter} iterations, {num_folds} folds at {num_chains} chains per fold...")
    # parallel warmup for each fold, initialize mcmc state
    fold_ids = jnp.arange(num_folds)
    states, kernel_params = jax.vmap(fold_warmup, in_axes=(0,))(fold_ids)
    print(f"Running {max_batches} batches of {batch_size} iterations on {num_folds*num_chains} chains...")
    # use python flow control for now but jax can actually do this too
    # https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.cond
    esss, rhatss, elpdss, drawss = [], [], [], []
    mcses = []
    for i in range(max_batches):
        states = jax.vmap(run_batch)(fold_ids, kernel_params, states)
        ess = pred_ess_folds(states)
        esss.append(ess)
        rhats = rhat_log(states.pred_ws)
        rhatss.append(rhats)
        elpd_contribs = log_welford_mean(states.pred_ws)
        elpd = logmean(elpd_contribs, axis=1)
        elpdss.append(elpd)
        drawss.append((i + 1) * batch_size * num_chains)
        log_mcvar = log_welford_log_var_combine(states.pred_bws.batches, ddof=1)
        mcses.append(jnp.exp(0.5 * log_mcvar))
        print(f"Batch {i+1}: {jnp.sum(ess)} total ess")
    else:
        print(f"Warning: max batches ({max_batches}) reached")
    ess, rhats, elpds, draws = (
        jnp.stack(esss),
        jnp.stack(rhatss),
        jnp.stack(elpdss),
        jnp.stack(drawss),
    )
    mcse = jnp.stack(mcses)
    total_elpd = jnp.sum(elpds, axis=1)
    total_mcse = jnp.sqrt(jnp.sum(mcse**2, axis=1))
    elpd_var = jnp.var(elpds, axis=1)
    total_cvse = jnp.sqrt(elpd_var)
    total_se = jnp.sqrt(total_mcse**2 + elpd_var)
    return {
        "ess": ess,
        "rhat": rhats,
        "elpd": elpds,
        "draws": draws,
        "mcse": mcse,
        "total_elpd": total_elpd,
        "total_mcse": total_mcse,
        "total_cvse": total_cvse,
        "total_se": total_se,
    }


def run_cv_sel(
    prng_key: jax.random.KeyArray,
    model: Model,
    stoprule: Callable,
    warmup_results: FoldWarmupResults,
    batch_size: int,
    max_batches: int,
    ignore_stoprule: bool = False,
    prior_only: bool = False
):
    """Compute cross-validation comparison of two models.

    Scope: two models, multiple folds, multiple chains.

    Args:
        prng_key: jax.random.PRNGKey, random number generator state
        logjoint_density: callable, log joint density function, signature (theta, fold_id, model_id)
        log_p: callable, log density, signature (theta, fold_id, model_id)
        num_folds: int, number of folds
        make_initial_pos: callable, function to make initial position for each chain
        stoprule: callable, function to determine if inference should stop
        num_chains: int, number of chains to run
        batch_size: int, number of iterations to run per batch
        warmup_iter: int, number of warmup iterations to run
        max_batches: int, maximum number of batches to run
        ignore_stoprule: (bool) if true, ignore the stoprule and run for max_batches
        prior_only: (bool) if true, only compute the prior

    Returns:
        2-tuple of:
            ExtendedState, final state of the inference loop
    """

    def run_batch(fold_id, model_id, fold_kernel_params, fold_state):
        results = fold_batched_inference_loop(
            kernel_parameters=fold_kernel_params,
            batch_size=batch_size,
            ext_states=fold_state,
            logjoint_density=lambda theta: model.logjoint_density(theta, fold_id, model_id, prior_only),
            log_pred=lambda theta: model.log_pred(theta, fold_id, model_id),
        )
        return results

    num_folds, num_chains = warmup_results.num_folds, warmup_results.num_chains
    print(
        f"Starting cross-validation with {num_folds*num_chains*2} parallel GHMC chains..."
    )
    start_at = time.time()
    # use python flow control for now but jax can actually do this too
    # at the cost of potentially informative output during the loop
    # https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.cond
    fold_drawss, fold_esss, fold_rhatss, fold_elpds, fold_mcses, fold_elpd_diffss, fold_divs = [],[],[],[],[],[],[]
    diff_mcses, diff_elpd, diff_cvses, diff_ses = [],[],[],[]
    model_esss, model_elpdss, model_mcses, model_cvses, model_ses, model_max_rhats = [],[],[],[],[],[]
    stoprules = []
    fold_Rhat_score, model_Rhat_score = [], []
    model_totals_M = jnp.vstack([
            jnp.repeat(jnp.array([1.0, 0.0]), num_folds),
            jnp.repeat(jnp.array([0.0, 1.0]), num_folds),
        ]).T
    model_diffs_M = jnp.repeat(jnp.array([1.0, -1.0]), num_folds)  # A - B
    fold_diffs_M = jnp.vstack([jnp.eye(num_folds), -jnp.eye(num_folds)])
    has_not_stopped = True
    i = 0
    divergences = jnp.zeros((2*num_folds,))
    states = warmup_results.fold_states
    fold_ids = jnp.tile(jnp.arange(num_folds), 2)
    model_ids = jnp.repeat(jnp.array([0, 1]), num_folds)
    for i in range(max_batches):
        fold_draws = (i + 1) * batch_size * num_chains
        fold_drawss.append(fold_draws)
        states: ExtendedState = jax.vmap(run_batch)(
            fold_ids, model_ids, warmup_results.fold_parameters, states)
        fold_ess = pred_ess_folds(states)
        # per-fold statistics
        fold_esss.append(fold_ess)
        fold_rhats = rhat_log(states.pred_ws)
        fold_rhatss.append(fold_rhats)
        fold_mcvars = jnp.exp(log_welford_log_var_combine(states.pred_bws.batches, ddof=1))
        fold_mcses.append(jnp.sqrt(fold_mcvars))
        fold_elpd = logmean(log_welford_mean(states.pred_ws), axis=1)
        fold_elpds.append(fold_elpd)
        fold_elpd_diffs = fold_elpd @ fold_diffs_M
        fold_elpd_diffss.append(fold_elpd_diffs)
        fold_div_count = jnp.sum(states.divergences, axis=(1,))
        fold_divs.append(fold_div_count)
        # chain rhats using delta method estimate of var(log(mean))
        lmeans = log_welford_mean(states.pred_ws)
        lvars = log_welford_log_var_combine(states.pred_ws, comb_axis=1, ddof=1)
        # log var log mean(X) = log (var X / mean(X)^2 / M)
        #                     = log var X - 2 log mean X - log M
        logW = log_welford_var_combine(states.pred_ws, comb_axis=1, ddof=1) - 2*(logsumexp(log_welford_mean(states.pred_ws), axis=1) - jnp.log(num_chains))
        logB = jnp.log(jnp.var(lmeans, axis=1, ddof=1)) + jnp.log(fold_draws)
        log_varplus = jnp.logaddexp(jnp.log(fold_draws - 1) + logW, logB) - jnp.log(fold_draws)
        log_Rhat = 0.5 * (log_varplus - logW)
        fold_Rhat_score.append(jnp.exp(log_Rhat))
        # combined Rhat model
        comb = lambda x: logsumexp(jnp.reshape(x, (model.num_models, num_folds)), axis=1)
        logWt, logBt = comb(logW), comb(logB)
        log_varplust = jnp.logaddexp(jnp.log(fold_draws - 1) + logWt, logBt) - jnp.log(fold_draws)
        log_Rhatt = 0.5 * (log_varplust - logWt)
        model_Rhat_score.append(jnp.exp(log_Rhatt))
        # lugsail rhat estimators
        lvars = log_welford_var_combine(states.pred_ws, comb_axis=1, ddof=1)
        lvarsb = log_welford_var_combine(states.pred_bws.batches, comb_axis=1, ddof=1)
        lvarsb3 = log_welford_var_combine(states.pred_bws3.batches, comb_axis=1, ddof=1)
        lvarsl = jnp.log(2) + lvarsb + jnp.log1p(-jnp.exp(lvarsb3 - lvarsb - jnp.log(2)))
        coef = jnp.maximum(jnp.log(2) + lvarsb, lvarsb3)
        lvarsl = coef + jnp.log(2 * jnp.exp(lvarsb - coef) - jnp.exp(lvarsb3 - coef))
        # per-model statistics
        model_elpdss.append(fold_elpd @ model_totals_M)
        model_ess = fold_ess @ model_totals_M
        model_esss.append(model_ess)
        model_mcvars = fold_mcvars @ model_totals_M
        model_mcse = jnp.sqrt(model_mcvars)
        model_mcses.append(model_mcse)
        model_cvvar = num_folds * jnp.var(jnp.reshape(fold_elpd, (2, num_folds)), ddof=1, axis=1)
        model_cvses.append(jnp.sqrt(model_cvvar))
        model_se = jnp.sqrt(fold_mcvars @ model_totals_M + model_cvvar)
        model_ses.append(model_se)
        model_max_rhat = jnp.nanmax(jnp.reshape(fold_rhats, (2, num_folds)), axis=1)
        model_max_rhats.append(model_max_rhat)
        # difference statistics (elpd(A) - elpd(B))
        diff, diff_cvse = fold_elpd @ model_diffs_M, jnp.sqrt(num_folds) * jnp.std(fold_elpd_diffs, ddof=1)
        diff_se = jnp.sqrt(num_folds * jnp.var(fold_elpd_diffs, ddof=1) + jnp.sum(model_mcvars))
        diff_elpd.append(diff)
        diff_cvses.append(diff_cvse)
        # contributions to diff_mcses and diff_ses are independent, so add them
        diff_mcses.append(jnp.sqrt(jnp.sum(model_mcvars)))
        diff_ses.append(diff_se)
        ruleoc = stoprule(diff, diff_cvse, model_mcse, model_ess, num_folds, (i + 1) * batch_size, model_max_rhat)
        stoprules.append(ruleoc)
        if (i % 10 == 0) or (i == max_batches-1):
            print(f"{i: 4d}. "
                f" Model A: {model_elpdss[-1][0]:.2f} ±{model_ses[-1][0]:.2f} ess {model_ess[0]:.0f},"
                f" Model B: {model_elpdss[-1][1]:.2f} ±{model_ses[-1][1]:.2f} ess {model_ess[1]:.0f}")
            print(f"       Diff: {diff_elpd[-1]:.2f} ±{diff_ses[-1]:.2f},"
                f" Rhat < {jnp.max(fold_rhats):.4f} "
                + (" stop" if ruleoc else " continue"))
            div_incr_fold = jnp.sum(fold_div_count > divergences)
            if div_incr_fold > 0:
                print(f"       Warning: new divergences in {div_incr_fold}/{2*num_folds} folds")
                divergences = fold_div_count
        if ruleoc != CONTINUE and not ignore_stoprule:
            print(f"       Stopping after {i+1} batches")
            break
        elif ruleoc != CONTINUE and has_not_stopped:
            print(f"       Triggered stoprule after {i+1} batches in {time.time() - start_at:.0f} seconds")
            has_not_stopped = False
    else:
        if not ignore_stoprule:
            print(f"Warning: max batches ({max_batches}) reached")
    iter = num_folds * num_chains * 2 * batch_size * (i + 1)
    total_sec = time.time() - start_at
    min, sec = int(total_sec) // 60, total_sec % 60
    print(f"Drew {iter} samples in {min:.0f} min {sec:.0f} sec ({iter/total_sec:.0f} per sec)")
    return {
        "fold_ess": jnp.stack(fold_esss),
        "fold_rhat": jnp.stack(fold_rhatss),
        "fold_elpd": jnp.stack(fold_elpds),
        "fold_draws": jnp.stack(fold_drawss),
        "fold_mcse": jnp.stack(fold_mcses),
        "fold_elpd_diff": jnp.stack(fold_elpd_diffss),
        "fold_divergences": jnp.stack(fold_divs),
        "fold_rhat_score": jnp.stack(fold_Rhat_score),
        "model_ess": jnp.stack(model_esss),
        "model_elpd": jnp.stack(model_elpdss),
        "model_mcse": jnp.stack(model_mcses),
        "model_cvse": jnp.stack(model_cvses),
        "model_se": jnp.stack(model_ses),
        "model_max_rhat": jnp.stack(model_max_rhats),
        "model_rhat_score": jnp.stack(model_Rhat_score),
        "diff_elpd": jnp.stack(diff_elpd),
        "diff_cvse": jnp.stack(diff_cvses),
        "diff_mcse": jnp.stack(diff_mcses),
        "diff_se": jnp.stack(diff_ses),
        "num_folds": num_folds,
        "num_chains": num_chains,
        "stop": jnp.stack(stoprules),
        "states": states
    }


def base_rhat(
    means: jax.Array, vars: jax.Array, n: jax.Array, axis: int = 1
) -> jax.Array:
    """Compute a single Rhat from summary statistics.

    Args:
        means: means of chains
        vars:  variances of chains
        n:     number of draws per chain
    """
    W = jnp.mean(vars, axis=axis)
    B = n * jnp.var(means, ddof=1, axis=axis)
    varplus = (n - 1) / n * W + B / n
    Rhat = jnp.sqrt(varplus / W)
    return Rhat


def base_rhat_log(
    log_means: jax.Array, log_vars: jax.Array, n: jax.Array, axis: int = 1
) -> jax.Array:
    """Compute a single Rhat from summary statistics.

    Args:
        log_means: log means of chains
        log_vars:  log variances of chains
        n:         number of draws per chain

    Return:
        Rhat in *levels* (not logged)
    """
    logW = logmean(log_vars, axis=axis)
    logB = jnp.log(n) + logvar(log_means, ddof=1, axis=axis)
    log_varplus = jnp.logaddexp(jnp.log(n - 1) + logW, logB) - jnp.log(n)
    log_Rhat = 0.5 * (log_varplus - logW)
    return jnp.exp(log_Rhat)


def rhat_welford(ws: WelfordState) -> jax.Array:
    """Compute Rhat from Welford state of chains.

    Args:
        ws: Welford state

    Returns:
        array of Rhats
    """
    means = jax.vmap(welford_mean)(ws)
    vars = jax.vmap(welford_var)(ws)
    n = ws.n[:, 0, ...]  # we aggregate over chain dim, axis=1
    return base_rhat(means, vars, n)


def folded_rhat_welford(ws: WelfordState) -> jax.Array:
    """Compute folded Rhat from Welford states.

    Args:
        ws: Welford state

    Returns:
        folded Rhat: array of folded Rhats
    """
    mads = jax.vmap(welford_mad)(ws)
    vars = jax.vmap(welford_var)(ws)
    n = ws.n[:, 0, ...]
    return base_rhat(mads, vars, n)


def rhat(welford_tree):
    """Compute Rhat and folded Rhat from welford states.

    This version assumes there are multiple posteriors, so that the states have dimension
    (cv_fold #, chain #, ...).

    Args:
        welford_tree: pytree of Welford states for split chains

    Returns:
        split Rhat: pytree pytree of split Rhats
        folded split Rhat: pytree of folded split Rhats
    """
    sr = tree_map(
        rhat_welford, welford_tree, is_leaf=lambda x: isinstance(x, WelfordState)
    )
    fsr = tree_map(
        folded_rhat_welford,
        welford_tree,
        is_leaf=lambda x: isinstance(x, WelfordState),
    )
    return sr, fsr


def rhat_welford_log(lws: LogWelfordState) -> jax.Array:
    """Compute Rhat from Welford state of chains.

    Args:
        ws: log Welford state (ie quantities in logs)

    Returns:
        array of Rhats (in levels)
    """
    means = jax.vmap(log_welford_mean)(lws)
    vars = jax.vmap(log_welford_var)(lws)
    n = lws.n[:, 0, ...]  # we aggregate over chain dim, axis=1
    return base_rhat_log(means, vars, n)


def rhat_log(log_welford_tree):
    """Compute Rhat and folded Rhat from welford states.

    This version assumes there are multiple posteriors, so that the states have dimension
    (cv_fold #, chain #, ...).

    Args:
        log_welford_tree: pytree of Welford states for split chains (quantities in logs)

    Returns:
        Rhat: pytree pytree of Rhats
    """
    return tree_map(
        rhat_welford_log,
        log_welford_tree,
        is_leaf=lambda x: isinstance(x, LogWelfordState),
    )
