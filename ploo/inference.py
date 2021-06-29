from typing import Dict, List, NamedTuple
from jax.interpreters.xla import DeviceArray
import jax.numpy as jnp
from jax import random, lax, vmap
from blackjax import nuts, stan_warmup

from .model import CVModel
from .util import Progress
from .hmc import cv_kernel, new_cv_state
from .cv_posterior import CVPosterior
from .util import Timer


class WarmupResults(NamedTuple):
    """Results of the warmup procedure

    These parameters are used to configure future HMC runs.
    """

    step_size: float
    mass_matrix: DeviceArray
    starting_values: List[Dict]
    int_steps: int

    @property
    def code(self) -> str:
        """Python code for recreating this warmup output.
        
        Use this to create reproducible tests that don't take too ong to run.
        """
        da2c = lambda a: (
            str(a).replace("DeviceArray", "jnp.array").replace(", dtype=float32", "")
        )
        sv_code = da2c(self.starting_values)
        mm_code = da2c(jnp.array(self.mass_matrix))
        py = f"""WarmupResults(
    step_size={self.step_size},
    mass_matrix=jnp.array({mm_code}),
    starting_values={sv_code},
    int_steps={self.int_steps})"""
        return py


def run_hmc(
    model: CVModel,
    draws: int = 2000,
    warmup_steps: int = 500,
    chains: int = 8,
    seed: int = 42,
    out: Progress = None,
    warmup_results=None,
) -> CVPosterior:
    """Run HMC after using Stan warmup with NUTS.

    Keyword arguments:
        model: model, including data, to run inference on
        draws: number of draws per chain
        warmup_steps: number of Stan warmup steps to run
        chains: number of chains for main inference step
        seed: random seed
        out: progress indicator
    """
    print = (out or Progress()).print
    rng_key = random.PRNGKey(seed)
    warmup_key, inference_key, cv_key = random.split(rng_key, 3)

    print("The Cross-Validatory Sledgehammer™")
    print("==================================\n")

    if warmup_results:
        print("Step 1/3. Skipping warmup")
    else:
        print(f"Step 1/3. Starting Stan warmup using NUTS...")
        timer = Timer()
        warmup_results = warmup(model, warmup_steps, chains, warmup_key)
        print(
            f"          {warmup_steps} warmup draws took {timer}"
            f" ({warmup_steps/timer.sec:.1f} iter/sec)."
        )

    print(f"Step 2/3. Running main inference with {chains} chains...")
    timer = Timer()
    states = full_data_inference(model, warmup_results, draws, chains, inference_key)
    print(
        f"          {chains*draws:,} HMC draws took {timer}"
        f" ({chains*draws/timer.sec:,.0f} iter/sec)."
    )

    timer = Timer()
    cv_chains = chains * model.cv_folds
    print(
        f"Step 3/3. Cross-validation with {model.cv_folds:,} folds "
        f"using {cv_chains:,} chains..."
    )
    cv_states = cross_validate(model, warmup_results, draws, chains, cv_key)
    print(
        f"          {cv_chains*draws:,} HMC draws took {timer}"
        f" ({cv_chains*draws/timer.sec:,.0f} iter/sec)."
    )

    return CVPosterior(model, states, cv_states, seed, chains)


def warmup(model, warmup_steps, num_start_pos, rng_key) -> WarmupResults:
    """Run Stan warmup

    We sample initial positions from second half of the Stan warmup, running
    NUTS. Yes I know this is awful, awful, awful. Please don't judge, we'll
    replace it with a shiny new warmup scheme that will surely never have any
    problems ever.

    Keyword args:
        model: model to work with
        warmup_steps: number of warmup iterations
        num_start_pos: number of starting positions to extract
        key: random generator state

    Returns:
        WarmupResults object containing step size, mass matrix, initial positions,
        and integration steps.
    """
    assert jnp.isfinite(
        model.cv_potential(model.initial_value, -1)
    ), "Invalid initial value"
    warmup_key, start_val_key = random.split(rng_key)
    potential = lambda p: model.cv_potential(p, cv_fold=-1)  # full-data potential
    initial_state = nuts.new_state(model.initial_value, potential)
    kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
        potential, step_size, inverse_mass_matrix
    )
    state, (step_size, mass_matrix), adapt_chain = stan_warmup.run(
        warmup_key,
        kernel_factory,
        initial_state,
        num_steps=warmup_steps,
        is_mass_matrix_diagonal=True,
        initial_step_size=1e-3,
    )
    hmc_warmup_state, stan_warmup_state, nuts_info = adapt_chain
    assert jnp.isfinite(step_size), "Woops, step size is not finite."

    # Sample the initial values uniformly from the second half of the
    # warmup chain
    varname = next(iter(hmc_warmup_state.position))
    warmup_steps = hmc_warmup_state.position[varname].shape[0]
    start_idxs = random.choice(
        start_val_key,
        a=jnp.arange(warmup_steps // 2, warmup_steps),
        shape=(num_start_pos,),
        replace=True,
    )
    initial_values = {
        k: hmc_warmup_state.position[k][start_idxs] for k in model.initial_value
    }

    # take median of NUTS integration steps for static path length
    int_steps = int(jnp.median(nuts_info.integration_steps[(warmup_steps // 2) :]))

    return WarmupResults(step_size, mass_matrix, initial_values, int_steps)


def full_data_inference(
    model: CVModel, warmup: WarmupResults, draws: int, chains: int, rng_key: DeviceArray
):
    """Full-data inference on model with no CV folds dropped.

    Keyword args:
        model: model to perform inference on
        warmup: results from warmup procedure
        draws: number of posterior draws per chain
        chains: number of chains
        rng_key: random generator state
    """
    # NB: the special CV fold index of -1 indicates full-data inference
    # one initial state per chain
    initial_states = vmap(new_cv_state, in_axes=(0, None, None))(
        warmup.starting_values, model.cv_potential, -1
    )
    kernel = cv_kernel(
        model.cv_potential, warmup.step_size, warmup.mass_matrix, warmup.int_steps
    )

    def one_step(states, iter_key):
        keys = random.split(iter_key, chains)
        states, _ = vmap(kernel)(keys, states)
        return states, states

    draw_keys = random.split(rng_key, draws)
    _, states = lax.scan(one_step, initial_states, draw_keys)

    return states


def cross_validate(
    model: CVModel,
    warmup: WarmupResults,
    draws: int,
    chains: int,
    rng_key: DeviceArray,
):
    """Cross validation step.

    Runs inference acros all CV folds, using cross-validated version of model potential.

    For now, we will collect all the MCMC samples. In the future we'll replace
    the inference loop with something that calculates the objective functions
    (and diagnostics) we want using online estimators.

    Keyword args:
        model: model instance
        warmup: results from warmup procedure
        draws: number of draws per chain
        chains: number of chains per fold
        key: random generator state
    """
    cv_folds = model.cv_folds
    # chain_indexes = [0, 1, 2, 3, 0, 1, 2, 3, 0, ...]
    chain_indexes = jnp.resize(jnp.arange(chains), chains * cv_folds)
    # fold_indexes  = [0, 0, 0, 0, 1, 1, 1, 1, 2, ...]
    fold_indexes = jnp.repeat(jnp.arange(cv_folds), chains)
    assert chain_indexes.shape == fold_indexes.shape
    chain_starting_values = {
        k: sv[chain_indexes] for (k, sv) in warmup.starting_values.items()
    }
    cv_initial_states = vmap(new_cv_state, (0, None, 0))(
        chain_starting_values, model.cv_potential, fold_indexes
    )
    kernel = cv_kernel(
        model.cv_potential, warmup.step_size, warmup.mass_matrix, warmup.int_steps
    )
    # each step operates vector of states (representing a cross-section across chains)
    # and vector of rng keys, one per draw
    def one_step(states, rng_subkey):
        keys = random.split(rng_subkey, chains * cv_folds)
        states, _ = vmap(kernel)(keys, states)
        # TODO: untransform position and evaluate lpd
        return states, states

    draw_keys = random.split(rng_key, draws)
    _, states = lax.scan(one_step, cv_initial_states, draw_keys)

    return states
