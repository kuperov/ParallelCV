"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

Implementation of HMC borrowed from blackjax and modified for use as a driver for
cross-validation. Kernels are instatiated with a CrossValidationScheme object
which defines the CV folds and deletion patterns.
"""

from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import chex
import jax
import numpy as np
from blackjax import nuts, stan_warmup
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import scipy as jscipy
from jax import vmap
from jax.flatten_util import ravel_pytree

Array = Union[np.ndarray, chex.ArrayDevice]
PyTree = Union[Dict, List, Tuple]


# inference parameter type annotation
InfParams = Dict[str, chex.ArrayDevice]


class IntegratorState(NamedTuple):
    """State information for a single step of the integrator"""

    position: PyTree
    momentum: PyTree
    potential_energy: float
    potential_energy_grad: PyTree


class HMCInfo(NamedTuple):
    """Detailed information about the current HMC state.

    This information is not stored for each trace, but some elements (like divergences)
    are counted for diagnostics.
    """

    momentum: PyTree
    acceptance_probability: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: IntegratorState


class CVHMCState(NamedTuple):
    """HMC state. This is kept for inference."""

    position: PyTree
    potential_energy: float
    potential_energy_grad: PyTree
    cv_fold: int


class Proposal(NamedTuple):
    """An HMC proposal generated by the integrator."""

    state: IntegratorState
    energy: float
    weight: float  # log sum canonical densities of eah state e^{-H(z)} along trajectory
    sum_log_p_accept: float  # sum of MH acceptance probs along trajectory


class CrossValidationState(NamedTuple):
    """State information for the CV HMC kernel."""

    divergence_count: int
    accepted_count: int
    sum_log_pred_dens: Array
    hmc_state: CVHMCState


class WarmupResults(NamedTuple):
    """Results of the warmup procedure.
    These parameters are used to configure future HMC runs.
    """

    step_size: float
    mass_matrix: chex.ArrayDevice
    starting_values: List[Dict]
    int_steps: int

    @property
    def code(self) -> str:
        """Python code for recreating this warmup output.

        Use this to create reproducible tests that don't take too ong to run.
        """

        def da2c(array):
            return (
                str(array)
                .replace("DeviceArray", "jnp.array")
                .replace(", dtype=float32", "")
            )

        sv_code = da2c(self.starting_values)
        mm_code = da2c(jnp.array(self.mass_matrix))
        python_code = (
            "WarmupResults(\n"
            f"    step_size={self.step_size},\n"
            f"    mass_matrix=jnp.array({mm_code}),\n"
            f"    starting_values={sv_code},\n"
            f"    int_steps={self.int_steps})"
        )
        return python_code


# pylint: disable=too-many-locals
def warmup(
    potential: Callable,
    initial_value: InfParams,
    warmup_steps: int,
    num_start_pos: int,
    rng_key: chex.ArrayDevice,
) -> WarmupResults:
    """Run Stan warmup.
    We sample initial positions from second half of the Stan warmup, running
    NUTS. Yes I know this is awful, awful, awful. Please don't judge, we'll
    replace it with a shiny new warmup scheme that will surely never have any
    problems ever.

    :param potential: potential function, takes model parameter and cross-validation
        fold
    :param initial_value: an initial value, expressed in inference (unconstrained)
        parameters
    :param warmup_steps: number of warmup iterations
    :param num_start_pos: number of starting positions to extract
    :param rng_key: random generator state
    :return: WarmupResults object containing step size, mass matrix, initial positions,
        and integration steps.
    """
    # pass cv_fold = -1 even though should not be necessary
    assert jnp.isfinite(potential(initial_value)), "Invalid initial value"
    warmup_key, start_val_key = random.split(rng_key)

    def kernel_factory(step_size, inverse_mass_matrix):
        return nuts.kernel(potential, step_size, inverse_mass_matrix)

    initial_state = nuts.new_state(initial_value, potential)
    _, (step_size, mass_matrix), adapt_chain = stan_warmup.run(
        warmup_key,
        kernel_factory,
        initial_state,
        num_steps=warmup_steps,
        is_mass_matrix_diagonal=True,
        initial_step_size=1e-3,
    )
    hmc_warmup_state, _, nuts_info = adapt_chain
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
        k: hmc_warmup_state.position[k][start_idxs] for k in initial_value
    }

    # take median of NUTS integration steps for static path length
    int_steps = int(jnp.median(nuts_info.integration_steps[(warmup_steps // 2) :]))

    return WarmupResults(step_size, mass_matrix, initial_values, int_steps)


def full_data_inference(
    potential: Callable,
    warmup_res: WarmupResults,
    draws: int,
    chains: int,
    rng_key: chex.ArrayDevice,
) -> Tuple[CrossValidationState, CVHMCState]:
    """Full-data inference on model (i.e. no CV folds dropped)

    :param potential: potential function for HMC
    :param warmup_res: results from warmup procedure
    :param draws: number of posterior draws per chain
    :param chains: number of chains
    :param rng_key: random generator state
    :return: tuple of CrossValidationState, representing the accumulated state
        over the entire MCMC chain, and CVHMCState object containing all draws
        for all MCMC iterations
    """
    # NB: the special CV fold index of -1 indicates full-data inference.
    # Create one initial state per chain.
    initial_states = vmap(new_cv_state, in_axes=(0, None, None))(
        warmup_res.starting_values, potential, -1
    )
    initial_accumulator = CrossValidationState(
        divergence_count=jnp.zeros((chains,)),
        accepted_count=jnp.zeros((chains,)),
        sum_log_pred_dens=jnp.zeros((chains,)),
        hmc_state=initial_states,
    )
    kernel = cv_kernel(
        potential, warmup_res.step_size, warmup_res.mass_matrix, warmup_res.int_steps
    )

    def one_step(state: CrossValidationState, iter_key):
        keys = random.split(iter_key, chains)
        hmc_state, hmc_info = vmap(kernel)(keys, state.hmc_state)
        divs = state.divergence_count + jnp.where(hmc_info.is_divergent, 1, 0)
        accepted = state.accepted_count + jnp.where(hmc_info.is_accepted, 1, 0)
        updated_state = CrossValidationState(
            divergence_count=divs,
            accepted_count=accepted,
            sum_log_pred_dens=state.sum_log_pred_dens,
            hmc_state=hmc_state,
        )
        return updated_state, hmc_state

    draw_keys = random.split(rng_key, draws)
    accumulator, states = lax.scan(one_step, initial_accumulator, draw_keys)

    return accumulator, states


# pylint: disable=too-many-arguments
def cross_validate(
    cv_potential: Callable,
    cv_cond_pred: Callable,
    warmup_res: WarmupResults,
    cv_folds: int,
    draws: int,
    chains: int,
    rng_key: chex.ArrayDevice,
    retain_draws: bool = False,
) -> Tuple[CrossValidationState, CVHMCState]:
    """Cross validation step.

    Runs inference across all CV folds, using cross-validated version of model potential.

    :param cv_potential: cross-validation model potential, function of
        (inference parameters, cv_fold)
    :param cv_cond_pred: cross-validation conditional predictive density, function of
        (inference_parameters, cv_fold)
    :param warmup_res: results from warmup procedure
    :param cv_folds: number of cross-validation folds
    :param draws: number of draws per chain
    :param chains: number of chains per fold
    :param rng_key: random generator state
    :param retain_draws: if True, retain MCMC draws from all chains
    :return: Tuple of :class`CrosssValidationState` (representing the accumulated state
        across all MCMC draws), and (if `retain_draws==True`) a :class:`CVHMCState`
        object containing conditional predictive
    """
    # assuming 4 chains and a 1-dimensional cross-validation structure,
    # chain_indexes = [0, 1, 2, 3, 0, 1, 2, 3, 0, ...]
    chain_indexes = jnp.concatenate([jnp.arange(chains)] * cv_folds)
    # fold_indexes  = [0, 0, 0, 0, 1, 1, 1, 1, 2, ...]
    fold_indexes = jnp.repeat(jnp.arange(cv_folds), chains)
    assert chain_indexes.shape == fold_indexes.shape
    chain_starting_values = {
        k: sv[chain_indexes] for (k, sv) in warmup_res.starting_values.items()
    }
    cv_initial_states = vmap(new_cv_state, (0, None, 0))(
        chain_starting_values, cv_potential, fold_indexes
    )
    cv_initial_accumulator = CrossValidationState(
        divergence_count=jnp.zeros(fold_indexes.shape),
        accepted_count=jnp.zeros(fold_indexes.shape),
        sum_log_pred_dens=jnp.zeros(fold_indexes.shape),
        hmc_state=cv_initial_states,
    )
    kernel = cv_kernel(
        cv_potential, warmup_res.step_size, warmup_res.mass_matrix, warmup_res.int_steps
    )

    if retain_draws:

        def do_cv():
            # each step operates vector of states (representing a cross-section
            # across chains) and vector of rng keys, one per draw
            def one_step(
                cv_state: CrossValidationState, rng_subkey: chex.ArrayDevice
            ) -> Tuple[CrossValidationState, CVHMCState]:
                keys = random.split(rng_subkey, chains * cv_folds)
                hmc_state, hmc_info = vmap(kernel)(keys, cv_state.hmc_state)
                cond_pred = vmap(cv_cond_pred)(hmc_state.position, hmc_state.cv_fold)
                div_count = cv_state.divergence_count + jnp.where(
                    hmc_info.is_divergent, 1, 0
                )
                accept_count = cv_state.accepted_count + jnp.where(
                    hmc_info.is_accepted, 1, 0
                )
                updated_state = CrossValidationState(
                    divergence_count=div_count,
                    accepted_count=accept_count,
                    sum_log_pred_dens=cv_state.sum_log_pred_dens + cond_pred,
                    hmc_state=hmc_state,
                )
                return updated_state, hmc_state.position

            draw_keys = random.split(rng_key, draws)
            accumulator, positions = lax.scan(
                one_step, cv_initial_accumulator, draw_keys
            )
            return accumulator, positions

    else:

        def do_cv():
            # each step operates vector of states (representing a cross-section
            # across chains) and vector of rng keys, one per draw
            def one_step(
                cv_state: CrossValidationState, rng_subkey: chex.ArrayDevice
            ) -> Tuple[CrossValidationState, CVHMCState]:
                keys = random.split(rng_subkey, chains * cv_folds)
                hmc_state, hmc_info = vmap(kernel)(keys, cv_state.hmc_state)
                cond_pred = vmap(cv_cond_pred)(hmc_state.position, hmc_state.cv_fold)
                div_count = cv_state.divergence_count + jnp.where(
                    hmc_info.is_divergent, 1, 0
                )
                accept_count = cv_state.accepted_count + jnp.where(
                    hmc_info.is_accepted, 1, 0
                )
                updated_state = CrossValidationState(
                    divergence_count=div_count,
                    accepted_count=accept_count,
                    sum_log_pred_dens=cv_state.sum_log_pred_dens + cond_pred,
                    hmc_state=hmc_state,
                )
                return updated_state, None

            draw_keys = random.split(rng_key, draws)
            accumulator, _ = lax.scan(one_step, cv_initial_accumulator, draw_keys)
            return accumulator, None

    j_do_cv = jax.jit(do_cv)
    accumulator, positions = j_do_cv()
    return accumulator, positions


def new_cv_state(position: PyTree, potential_fn: Callable, cv_fold: int) -> CVHMCState:
    """Initial cross-validation state

    Keyword arguments
        position:     starting position
        potential_fn: model potential
    """
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(
        position, cv_fold
    )
    return CVHMCState(position, potential_energy, potential_energy_grad, cv_fold)


def cv_velocity_verlet(
    potential_fn: Callable, kinetic_energy_fn: Callable[[PyTree], float]
) -> Callable:
    """The velocity Verlet (or Verlet-Störmer) integrator, specialized for CV

    This version of the standard blackjax leapfrog integrator keeps track of
    the cross-validation fold.
    """
    # pylint: disable=invalid-name
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    potential_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(
        state: IntegratorState, step_size: float, cv_fold: int
    ) -> IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_multimap(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_multimap(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )

        potential_energy, potential_energy_grad = potential_grad_fn(position, cv_fold)
        momentum = jax.tree_util.tree_multimap(
            lambda momentum, potential_grad: momentum - b1 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
        )

        return IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )

    return one_step


# pylint: disable=too-many-locals
def cv_kernel(
    potential_fn: Callable,
    step_size: float,
    inverse_mass_matrix: Array,
    num_integration_steps: int,
    divergence_threshold: int = 1000,
) -> Callable:
    """Create CV HMC kernel

    The CV HMC kernel takes one MCMC step (comprising a fixed number of integrator
    steps), advancing from one CVHMC state to another.

    Args:
        potential_fn:          potential function for model, takes a parameter as a
                               dict and cross-validation fold number
        step_size:             HMC step size
        inverse_mass_matrix:   HMC inv mass matrix. If diagonal, a 1D array, or a
                               square 2D array.
        num_integration_steps: number of steps to run the integrator for each proposal
        divergence_threshold:  minimum change in energy to declare a divergence

    Returns:
        Callable: HMC kernel as a function

    Raises:
        ValueError: if inverse_mass_matrix is of incorrect shape.
    """

    ndim = jnp.ndim(inverse_mass_matrix)
    shape = jnp.shape(inverse_mass_matrix)[:1]

    if ndim == 1:  # diagonal mass matrix
        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))
        dot, matmul = jnp.multiply, jnp.multiply

    elif ndim == 2:
        tril_inv = jscipy.linalg.cholesky(inverse_mass_matrix)
        identity = jnp.identity(shape[0])
        mass_matrix_sqrt = jscipy.linalg.solve_triangular(
            tril_inv, identity, lower=True
        )
        dot, matmul = jnp.dot, jnp.matmul

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.ndim(inverse_mass_matrix)}."
        )

    def momentum_generator(rng_key: jnp.ndarray, position: PyTree) -> PyTree:
        _, unravel_fn = ravel_pytree(position)
        standard_normal_sample = jax.random.normal(rng_key, shape)
        momentum = dot(mass_matrix_sqrt, standard_normal_sample)
        momentum_unravel = unravel_fn(momentum)
        return momentum_unravel

    def kinetic_energy_fn(momentum: PyTree) -> float:
        momentum, _ = ravel_pytree(momentum)
        momentum = jnp.array(momentum)
        velocity = matmul(inverse_mass_matrix, momentum)
        kinetic_energy_val = 0.5 * jnp.dot(velocity, momentum)
        return kinetic_energy_val

    symplectic_integrator = cv_velocity_verlet(potential_fn, kinetic_energy_fn)

    proposal_generator = cv_hmc_proposal(
        symplectic_integrator,
        kinetic_energy_fn,
        step_size,
        num_integration_steps,
        divergence_threshold,
    )

    # @jax.jit
    # @chex.assert_max_traces(n=1)
    def kernel(
        rng_key: jnp.ndarray, state: CVHMCState
    ) -> Tuple[CVHMCState, NamedTuple]:
        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, potential_energy, potential_energy_grad, cv_fold = state
        momentum = momentum_generator(key_momentum, position)

        augmented_state = IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(key_integrator, augmented_state, cv_fold)
        proposal = CVHMCState(
            proposal.position,
            proposal.potential_energy,
            proposal.potential_energy_grad,
            cv_fold,
        )
        return proposal, info

    return kernel


def cv_hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: float,
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
) -> Callable:
    """Vanilla HMC algorithm.

    :param integrator: symplectic integrator used to build the trajectory step by step.
    :param kinetic_energy: Function that computes the kinetic energy.
    :param step_size: Size of the integration step.
    :param num_integration_steps: Number of times we run the symplectic integrator to
        build the trajectory
    :param divergence_threshold: Threshold above which we say that there is a
        divergent transition.
    :return: A kernel that generates a new chain state and information about the
        transition.
    """

    def build_trajectory(
        initial_state: IntegratorState, cv_fold: int
    ) -> IntegratorState:
        def one_step(state, _):
            state = integrator(state, step_size, cv_fold)
            return state, state

        last_state, _ = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )
        return last_state

    def init_proposal(state: IntegratorState) -> Proposal:
        energy = state.potential_energy + kinetic_energy(state.momentum)
        return Proposal(state, energy, 0.0, -np.inf)

    def generate_proposal(
        initial_energy: float, state: IntegratorState
    ) -> Tuple[Proposal, bool]:
        new_energy = state.potential_energy + kinetic_energy(state.momentum)

        delta_energy = initial_energy - new_energy
        delta_energy = jnp.where(jnp.isnan(delta_energy), -jnp.inf, delta_energy)
        is_transition_divergent = jnp.abs(delta_energy) > divergence_threshold

        # The weight of the new proposal is equal to H0 - H(z_new)
        weight = delta_energy
        # Acceptance statistic min(e^{H0 - H(z_new)}, 1)
        sum_log_p_accept = jnp.minimum(delta_energy, 0.0)

        return (
            Proposal(
                state,
                new_energy,
                weight,
                sum_log_p_accept,
            ),
            is_transition_divergent,
        )

    def flip_momentum(
        state: IntegratorState,
    ) -> IntegratorState:
        """To guarantee time-reversibility (hence detailed balance) we
        need to flip the last state's momentum. If we run the hamiltonian
        dynamics starting from the last state with flipped momentum we
        should indeed retrieve the initial state (with flipped momentum).

        """
        flipped_momentum = jax.tree_util.tree_multimap(
            lambda m: -1.0 * m, state.momentum
        )
        return IntegratorState(
            state.position,
            flipped_momentum,
            state.potential_energy,
            state.potential_energy_grad,
        )

    def generate(
        rng_key, state: IntegratorState, cv_fold: int
    ) -> Tuple[IntegratorState, HMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, cv_fold)
        end_state = flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal, is_diverging = generate_proposal(proposal.energy, end_state)
        p_accept = jnp.clip(jnp.exp(new_proposal.weight), a_max=1)
        do_accept = jax.random.bernoulli(rng_key, p_accept)
        sampled_proposal = jax.lax.cond(
            do_accept, lambda _: new_proposal, lambda _: proposal, None
        )

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
        )

        return sampled_proposal.state, info

    return generate
