from typing import List
import jax.numpy as jnp
from jax import random, lax, vmap
from blackjax import nuts, hmc, stan_warmup
from datetime import datetime

from .model import CVModel
from .progress import Progress
from .hmc import cv_kernel, new_cv_state
from .cv_posterior import CVPosterior


def run_hmc(
    model: CVModel,
    draws: int = 2000,
    warmup_steps: int = 500,
    chains: int = 40,
    cv_chains_per_fold: int = 4,
    seed: int = 42,
    out: Progress = None,
) -> CVPosterior:
    """Run HMC after using Stan warmup with NUTS.

    Keyword arguments:
        model: model, including data, to run inference on
        draws: number of draws per chain
        warmup_steps: number of Stan warmup steps to run
        chains: number of chains for main inference step
        cv_chains_per_fold: number of chains to run per cv fold
        seed: random seed
        out: progress indicator
    """
    key = random.PRNGKey(seed)
    print = (out or Progress()).print

    print("The Cross-Validatory Sledgehammer")
    print("=================================\n")

    print(f"Step 1/3. Starting Stan warmup using NUTS...")
    start = datetime.now()
    step_size, mass_matrix, hmc_warmup_state, int_steps = warmup(
        model, warmup_steps, key
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(
        f"          {warmup_steps} warmup draws took {elapsed:.1f} sec"
        f" ({warmup_steps/elapsed:.1f} iter/sec)."
    )

    hmc_kernel = hmc.kernel(model.potential, step_size, mass_matrix, int_steps)

    print(f"Step 2/3. Running main inference with {chains} chains...")
    start = datetime.now()
    key, states = full_data_inference(
        model, draws, chains, key, hmc_warmup_state, hmc_kernel
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(
        f"          {chains*draws:,} HMC draws took {elapsed:.1f} sec"
        f" ({chains*draws/elapsed:,.0f} iter/sec)."
    )

    start = datetime.now()
    cv_chains = cv_chains_per_fold * model.cv_folds
    print(
        f"Step 3/3. Cross-validation with {model.cv_folds:,} folds "
        f"using {cv_chains:,} chains..."
    )
    cv_states = cross_validate(
        model,
        draws,
        cv_chains_per_fold,
        key,
        step_size,
        mass_matrix,
        hmc_warmup_state,
        int_steps,
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(
        f"          {cv_chains*draws:,} HMC draws took {elapsed:.1f} sec"
        f" ({cv_chains*draws/elapsed:,.0f} iter/sec)."
    )

    return CVPosterior(model, states, cv_states, seed)


def warmup(model, warmup_steps, key):
    """Run Stan warmup
    
    Keyword args:
        model: model to work with
        warmup_steps: number of warmup iterations
        key: random generator state
    """
    assert jnp.isfinite(model.potential(model.initial_value)), "Invalid initial value"
    initial_state = nuts.new_state(model.initial_value, model.potential)
    kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
        model.potential, step_size, inverse_mass_matrix
    )
    state, (step_size, mass_matrix), adapt_chain = stan_warmup.run(
        key,
        kernel_factory,
        initial_state,
        num_steps=warmup_steps,
        is_mass_matrix_diagonal=True,
        initial_step_size=1e-3,
    )
    hmc_warmup_state, stan_warmup_state, nuts_info = adapt_chain
    assert jnp.isfinite(step_size), "Woops, step size is not finite."

    # FIXME: eventually we want median of NUTS draws from actual inference,
    #        because we're actually capturing different warmup stages
    int_steps = int(jnp.median(nuts_info.integration_steps[(warmup_steps // 2) :]))
    return step_size, mass_matrix, hmc_warmup_state, int_steps


def full_data_inference(model, draws, chains, key, hmc_warmup_state, hmc_kernel):
    """Full-data inference on model with no CV folds dropped.
    
    Keyword args:
        model: model to perform inference on
        draws: number of posterior draws per chain
        chains: number of chains
        key: random generator state
        hmc_warmup_state: output of warmup
        hmc_kernel: kernel to use for sampling
    """

    def inference_loop(rng_key, kernel, initial_state, num_samples, num_chains):
        def one_step(states, rng_key):
            keys = random.split(rng_key, num_chains)
            states, _ = vmap(kernel)(keys, states)
            return states, states

        keys = random.split(rng_key, num_samples)
        _, states = lax.scan(one_step, initial_state, keys)
        return states

    # sample initial positions from second half of warmup
    key, subkey = random.split(key)
    varname = next(iter(hmc_warmup_state.position))
    warmup_steps = hmc_warmup_state.position[varname].shape[0]
    start_idxs = random.choice(
        subkey,
        a=jnp.arange(warmup_steps // 2, warmup_steps),
        shape=(chains,),
        replace=True,
    )
    initial_positions = {
        k: hmc_warmup_state.position[k][start_idxs] for k in model.initial_value
    }
    initial_states = vmap(hmc.new_state, in_axes=(0, None))(
        initial_positions, model.potential
    )

    states = inference_loop(
        key, hmc_kernel, initial_states, num_samples=draws, num_chains=chains
    )

    return key, states


def cross_validate(
    model,
    draws,
    cv_chains_per_fold,
    key,
    step_size,
    mass_matrix,
    hmc_warmup_state,
    int_steps,
):
    """Cross validation step.

    Runs inference acros all CV folds, using cross-validated version of model potential.

    Keyword args:
        model: model instance
        draws: number of draws per chain
        key: random generator state
        step_size: static HMC step size
        mass_matrix: HMC mass matrix - can be vector if diagonal
        hmc_warmup_state: results from warmup step
        int_steps: number of integration steps
    """
    cv_hmc_kernel = cv_kernel(model.cv_potential, step_size, mass_matrix, int_steps)
    cv_chains = model.cv_folds * cv_chains_per_fold
    cv_folds = jnp.repeat(jnp.arange(1, cv_chains_per_fold + 1), model.cv_folds)
    varname = next(iter(hmc_warmup_state.position))
    warmup_steps = hmc_warmup_state.position[varname].shape[0]
    key, subkey = random.split(key)
    cv_start_idxs = random.choice(
        subkey,
        a=jnp.arange(warmup_steps // 2, warmup_steps),
        shape=(cv_chains,),
        replace=True,
    )
    cv_initial_positions = {
        k: hmc_warmup_state.position[k][cv_start_idxs] for k in model.initial_value
    }
    cv_initial_states = vmap(new_cv_state, in_axes=(0, None, 0))(
        cv_initial_positions, model.cv_potential, cv_folds
    )

    def cv_inference_loop(rng_key, kernel, initial_state, num_samples, num_chains):
        def one_step(states, rng_key):
            keys = random.split(rng_key, num_chains)
            states, _ = vmap(kernel)(keys, states)
            return states, states

        keys = random.split(rng_key, num_samples)
        _, states = lax.scan(one_step, initial_state, keys)
        return states

    cv_states = cv_inference_loop(
        key, cv_hmc_kernel, cv_initial_states, num_samples=draws, num_chains=cv_chains
    )

    return cv_states
