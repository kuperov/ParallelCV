from typing import Callable, NamedTuple, Tuple

import jax
from jax import numpy as jnp

from blackjax.hmc import Array, HMCInfo, PyTree
from blackjax.inference import integrators, proposal, metrics


class CVHMCState(NamedTuple):
    """State of the HMC algorithm, incorporating CV fold.

    The CV model needs to know which fold should be dropped when
    evaluating the potential and its gradient, so we track that here
    """
    position: PyTree
    potential_energy: float
    potential_energy_grad: PyTree
    cv_fold: int


def new_cv_state(position: PyTree, potential_fn: Callable, cv_fold: int) -> CVHMCState:
    """Create a chain state from a position, identifying the CV fold

    Parameters
    ----------
    position
        The current values of the random variables whose posterior we want to
        sample from. Can be anything from a list, a (named) tuple or a dict of
        arrays. The arrays can either be Numpy arrays or JAX DeviceArrays.
    potential_fn
        A function that returns the value of the potential energy when called
        with a position.
    cv_fold
        CV fold number for identifying likelihood contributions to drop when
        evaluating the potential

    Returns
    -------
    A HMC state that contains the position, the associated potential energy and gradient of the
    potential energy.
    """
    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(
        position, cv_fold
    )
    return CVHMCState(position, potential_energy, potential_energy_grad, cv_fold)


def cv_velocity_verlet(
    potential_fn: Callable, kinetic_energy_fn: metrics.EuclideanKineticEnergy
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
        state: integrators.IntegratorState, step_size: float, cv_fold: int
    ) -> integrators.IntegratorState:
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

        return integrators.IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )

    return one_step


def cv_hmc(
    momentum_generator: Callable,
    proposal_generator: Callable,
) -> Callable:
    """Create a Hamiltonian Monte Carlo transition kernel.

    Hamiltonian Monte Carlo (HMC) is known to yield effective Markov
    transitions and has been a major empirical success, leading to an extensive
    use in probabilistic programming languages and libraries [1,2,3]_.

    HMC works by augmenting the state space in which the chain evolves with an
    auxiliary momentum :math:`p`. At each step of the chain we draw a momentum
    value from the `momentum_generator` function. We then use Hamilton's
    equations [4]_ to push the state forward; we then compute the new state's
    energy using the `kinetic_energy` function and `logpdf` (potential energy).
    While the hamiltonian dynamics is conservative, numerical integration can
    introduce some discrepancy; we perform a Metropolis acceptance test to
    compensate for integration errors after having flipped the new state's
    momentum to make the transition reversible.

    I encourage anyone interested in the theoretical underpinning of the method
    to read Michael Betancourts' excellent introduction [3]_ and his more
    technical paper [5]_ on the geometric foundations of the method.

    This implementation is very general and should accomodate most variations
    on the method.

    Parameters
    ----------
    proposal_generator:
        The function used to propose a new state for the chain. For vanilla HMC this
        function integrates the trajectory over many steps, but gets more involved
        with other algorithms such as empirical and dynamical HMC.
    momentum_generator:
        A function that returns a new value for the momentum when called.
    kinetic_energy:
        A function that computes the current state's kinetic energy.
    divergence_threshold:
        The maximum difference in energy between the initial and final state
        after which we consider the transition to be divergent.

    Returns
    -------
    A kernel that moves the chain by one step when called.

    References
    ----------
    .. [1]: Duane, Simon, et al. "Hybrid monte carlo." Physics letters B
            195.2 (1987): 216-222.
    .. [2]: Neal, Radford M. "An improved acceptance procedure for the
            hybrid Monte Carlo algorithm." Journal of Computational Physics 111.1
            (1994): 194-203.
    .. [3]: Betancourt, Michael. "A conceptual introduction to
            Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2018).
    .. [4]: "Hamiltonian Mechanics", Wikipedia.
            https://en.wikipedia.org/wiki/Hamiltonian_mechanics#Deriving_Hamilton's_equations
    .. [5]: Betancourt, Michael, et al. "The geometric foundations
            of hamiltonian monte carlo." Bernoulli 23.4A (2017): 2257-2298.

    """

    def kernel(rng_key: jnp.ndarray, state: CVHMCState) -> Tuple[CVHMCState, NamedTuple]:
        """Moves the chain by one step using the Hamiltonian dynamics.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the chain: position, log-probability and gradient
            of the log-probability.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, potential_energy, potential_energy_grad, cv_fold = state
        momentum = momentum_generator(key_momentum, position)

        augmented_state = integrators.IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(key_integrator, augmented_state, cv_fold)
        proposal = CVHMCState(
            proposal.position, proposal.potential_energy, proposal.potential_energy_grad, cv_fold
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
    """Build a Parallel CV HMC kernel.

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given position.
    parameters
        A NamedTuple that contains the parameters of the kernel to be built.

    Returns
    -------
    A kernel that takes a rng_key, Pytree with the current state, and CV fold
    and that returns a new state of the chain along with information about the
    transition.
    """
    momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
        inverse_mass_matrix
    )
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


def cv_static_integration(
    integrator: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    directed_step_size = direction * step_size

    def integrate(
        initial_state: integrators.IntegratorState, cv_fold: int
    ) -> integrators.IntegratorState:
        def one_step(state, _):
            state = integrator(state, directed_step_size, cv_fold)
            return state, state

        last_state, _ = jax.lax.scan(
            one_step, initial_state, jnp.arange(num_integration_steps)
        )

        return last_state

    return integrate


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
    build_trajectory = cv_static_integration(
        integrator, step_size, num_integration_steps
    )
    init_proposal, generate_proposal = proposal.proposal_generator(
        kinetic_energy, divergence_threshold
    )
    sample_proposal = proposal.static_binomial_sampling

    def flip_momentum(
        state: integrators.IntegratorState,
    ) -> integrators.IntegratorState:
        """To guarantee time-reversibility (hence detailed balance) we
        need to flip the last state's momentum. If we run the hamiltonian
        dynamics starting from the last state with flipped momentum we
        should indeed retrieve the initial state (with flipped momentum).

        """
        flipped_momentum = jax.tree_util.tree_multimap(
            lambda m: -1.0 * m, state.momentum
        )
        return integrators.IntegratorState(
            state.position,
            flipped_momentum,
            state.potential_energy,
            state.potential_energy_grad,
        )

    def generate(
        rng_key, state: integrators.IntegratorState, cv_fold: int
    ) -> Tuple[integrators.IntegratorState, HMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, cv_fold)
        end_state = flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal, is_diverging = generate_proposal(proposal.energy, end_state)
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate
