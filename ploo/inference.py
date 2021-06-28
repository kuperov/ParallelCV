from typing import List
import jax.numpy as jnp
import jax.scipy.stats as st
from jax import random, lax, vmap
import blackjax as bj
from blackjax import nuts, hmc, stan_warmup
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate

from .model import CVModel
from .progress import Progress
from .cv_kernel import cv_kernel, new_cv_state
from .cv_posterior import CVPosterior


def run_hmc(
    model: CVModel, draws=2000, warmup_steps=500, chains=40, cv_chains_per_fold=4, seed=42, out: Progress = None
) -> CVPosterior:
    """Run HMC after using Stan warmup with NUTS."""
    key = random.PRNGKey(seed)
    print = (out or Progress()).print

    print("Alex's Cross-Validatory Sledgehammer")
    print("====================================\n")

    assert jnp.isfinite(model.potential(model.initial_value)), "Invalid initial value"
    initial_state = nuts.new_state(model.initial_value, model.potential)
    kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
        model.potential, step_size, inverse_mass_matrix
    )
    print(f"Step 1/3. Starting Stan warmup using NUTS...")
    start = datetime.now()
    state, (step_size, mass_matrix), adapt_chain = stan_warmup.run(
        key,
        kernel_factory,
        initial_state,
        num_steps=warmup_steps,
        is_mass_matrix_diagonal=True,
        initial_step_size=1e-3,
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(
        f"          {warmup_steps} warmup draws took {elapsed:.1f} sec ({warmup_steps/elapsed:.1f} iter/sec)."
    )
    hmc_warmup_state, stan_warmup_state, nuts_info = adapt_chain
    assert jnp.isfinite(step_size), "Woops, step size is not finite."

    # FIXME: eventually we want median of NUTS draws from actual inference,
    #        because we're actually capturing different warmup stages
    int_steps = int(jnp.median(nuts_info.integration_steps[(warmup_steps // 2) :]))
    hmc_kernel = hmc.kernel(model.potential, step_size, mass_matrix, int_steps)

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

    print(f"Step 2/3. Running main inference with {chains} chains...")
    start = datetime.now()
    states = inference_loop(
        key, hmc_kernel, initial_states, num_samples=draws, num_chains=chains
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(
        f"          {chains*draws:,} HMC draws took {elapsed:.1f} sec"
        f" ({chains*draws/elapsed:,.0f} iter/sec)."
    )

    cv_hmc_kernel = cv_kernel(model.cv_potential, step_size, mass_matrix, int_steps)
    cv_chains = model.cv_folds * cv_chains_per_fold
    cv_folds = jnp.repeat(jnp.arange(1,cv_chains_per_fold+1), model.cv_folds)
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

    print(f"Step 3/3. Cross-validation with {model.cv_folds} folds using {cv_chains} chains...")
    start = datetime.now()
    loo_states = cv_inference_loop(
        key, cv_hmc_kernel, cv_initial_states, num_samples=draws, num_chains=cv_chains
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(
        f"          {cv_chains*draws:,} HMC draws took {elapsed:.1f} sec"
        f" ({cv_chains*draws/elapsed:,.0f} iter/sec)."
    )

    return CVPosterior(model, states, loo_states, seed)
