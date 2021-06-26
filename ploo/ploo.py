import jax.numpy as jnp
import jax.scipy.stats as st
from jax import random, lax, vmap
import blackjax as bj
from blackjax import nuts, hmc, stan_warmup
import matplotlib.pyplot as plt
from datetime import datetime

def run_hmc(potential_fn, initial_position, draws=2000, warmup_steps=500, chains=40):
    """Run HMC after using Stan warmup with NUTS."""
    potential = lambda x: potential_fn(**x)
    assert jnp.isfinite(potential(initial_position))
    initial_state = nuts.new_state(initial_position, potential)

    # conduct Stan warmup
    kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(potential, step_size, inverse_mass_matrix)
    print(f'Step 1/3. Starting Stan warmup using NUTS...')
    start = datetime.now()
    state, (step_size, mass_matrix), adapt_chain = stan_warmup.run(
        key,
        kernel_factory,
        initial_state,
        num_steps=warmup_steps,
        is_mass_matrix_diagonal=True,
        initial_step_size=1e-3
    )
    elapsed = (datetime.now() - start).total_seconds()
    print(f'          {warmup_steps} warmup draws took {elapsed:.1f} sec ({warmup_steps/elapsed:.1f} iter/sec).')
    hmc_warmup_state, stan_warmup_state, nuts_info = adapt_chain
    assert jnp.isfinite(step_size), "Woops, step size is not finite."

    def inference_loop(rng_key, kernel, initial_state, num_samples, num_chains):
        def one_step(states, rng_key):
            keys = random.split(rng_key, num_chains)
            states, _ = vmap(kernel)(keys, states)
            return states, states
        keys = random.split(rng_key, num_samples)
        _, states = lax.scan(one_step, initial_state, keys)
        return states
    # FIXME: eventually we want median of NUTS draws from actual inference,
    #        because we're actually capturing different warmup stages
    int_steps = int(jnp.median(nuts_info.integration_steps[(warmup_steps//2):]))
    hmc_kernel = hmc.kernel(potential, step_size, mass_matrix, int_steps)
    
    # sample initial positions from second half of warmup
    start_idxs = random.choice(subkey, a=jnp.arange(warmup_steps//2, warmup_steps), shape=(chains,), replace=True)
    initial_positions = {k: hmc_warmup_state.position[k][start_idxs] for k in initial_position}
    initial_states = vmap(hmc.new_state, in_axes=(0, None))(initial_positions, potential)
    
    print(f'Step 2/3. Running main inference with {chains} chains...')
    start = datetime.now()
    states = inference_loop(key, hmc_kernel, initial_states, num_samples=draws, num_chains=chains)
    elapsed = (datetime.now() - start).total_seconds()
    print(f'          {chains*draws:,} HMC draws took {elapsed:.1f} sec ({chains*draws/elapsed:,.0f} iter/sec).')

    print(f'Step 3/3. Running LOO-CV with {chains} chains...')
    start = datetime.now()
    states = inference_loop(key, hmc_kernel, initial_states, num_samples=draws, num_chains=chains)
    elapsed = (datetime.now() - start).total_seconds()
    print(f'          {chains*draws:,} HMC draws took {elapsed:.1f} sec ({chains*draws/elapsed:,.0f} iter/sec).')

    return states

