"""Implementation of HMC borrowed from blackjax and modified
for use as a driver for cross-validation.
"""


from typing import Callable, Dict, List, NamedTuple, Tuple, Union

import jax
from jax import numpy as jnp, scipy as jscipy
import numpy as np
from jax.flatten_util import ravel_pytree

Array = Union[np.ndarray, jnp.DeviceArray]
PyTree = Union[Dict, List, Tuple]


class IntegratorState(NamedTuple):
    position: PyTree
    momentum: PyTree
    potential_energy: float
    potential_energy_grad: PyTree


class HMCInfo(NamedTuple):
    momentum: PyTree
    acceptance_probability: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: IntegratorState


class CVHMCState(NamedTuple):
    position: PyTree
    potential_energy: float
    potential_energy_grad: PyTree
    cv_fold: int


class Proposal(NamedTuple):
    state: IntegratorState
    energy: float
    weight: float  # log sum canonical densities of eah state e^{-H(z)} along trajectory
    sum_log_p_accept: float  # sum of MH acceptance probs along trajectory


def new_cv_state(position: PyTree, potential_fn: Callable, cv_fold: int) -> CVHMCState:
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(
        position, cv_fold
    )
    return CVHMCState(position, potential_energy, potential_energy_grad, cv_fold)


EuclideanKineticEnergy = Callable[[PyTree], float]


def cv_velocity_verlet(
    potential_fn: Callable, kinetic_energy_fn: EuclideanKineticEnergy
) -> Callable:
    """The velocity Verlet (or Verlet-Störmer) integrator, specialized for CV

    This version of the standard blackjax leapfrog integrator keeps track of
    the cross-validation fold.
    """
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


def cv_hmc(
    momentum_generator: Callable,
    proposal_generator: Callable,
) -> Callable:
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


def cv_kernel(
    potential_fn: Callable,
    step_size: float,
    inverse_mass_matrix: Array,
    num_integration_steps: int,
    divergence_threshold: int = 1000,
):

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
    kernel = cv_hmc(momentum_generator, proposal_generator)
    return kernel


def cv_hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: float,
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
) -> Callable:
    """Vanilla HMC algorithm.

    Parameters
    ----------
    integrator
        Symplectic integrator used to build the trajectory step by step.
    kinetic_energy
        Function that computes the kinetic energy.
    step_size
        Size of the integration step.
    num_integration_steps
        Number of times we run the symplectic integrator to build the trajectory
    divergence_threshold
        Threshold above which we say that there is a divergence.

    Returns
    -------
    A kernel that generates a new chain state and information about the transition.

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
