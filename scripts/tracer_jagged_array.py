#!./venv/bin/python3

# Experiment to validate method for simulating jagged arrays
# for unblanced groups. We also try out chex data classes as
# a potential model data structure.

from typing import List, Tuple
import chex
import jax
from jax import numpy as jnp


T = 5
M = 100


@chex.dataclass
class MarkovState:
    """Markov chain state"""

    position: chex.ArrayDevice
    cumsum: chex.ArrayDevice


@chex.dataclass
class JaggedArray:
    params: chex.ArrayDevice
    mask: chex.ArrayDevice


initial_state = MarkovState(
    position=jnp.arange(T),
    cumsum=jnp.zeros(T),
)

jagged_array = JaggedArray(
    params=jnp.stack([jnp.arange(T)] * T),
    mask=jnp.stack([1.0 * (jnp.arange(T) <= arr_len) for arr_len in range(T)])
)


def kernel(curr_state: MarkovState, jag_array: JaggedArray) -> MarkovState:
    """Markov kernel that drives all these shenanigans

    Args:
        curr_state: state for a *single* position

    Returns:
        updated position
    """
    new_pos = curr_state.position + 1
    increment = jnp.sum(new_pos * jag_array.params * jag_array.mask)
    new_state = MarkovState(position=new_pos, cumsum=increment + curr_state.cumsum)
    return new_state


def run_markov_chain(m):
    def one_step(states, _rng_key):
        new_states = jax.vmap(kernel)(states, jagged_array)
        return new_states, new_states.position

    # we don't actually use these random keys, but scan needs a sequence to map over
    rng_key = jax.random.PRNGKey(seed=42)
    rng_keys = jax.random.split(rng_key, m)

    final_state, mc_chain = jax.lax.scan(one_step, initial_state, rng_keys)
    return final_state, mc_chain


print('starting jit')
run_markov_chain_j = jax.jit(run_markov_chain, static_argnames='m')
print('done with jit')
state, chain = run_markov_chain_j(M)

assert jnp.allclose(chain[:,0], jnp.arange(1,101))
assert state.position[0] == 100
assert state.cumsum[4] == 54500
