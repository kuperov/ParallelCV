from jax import value_and_grad, random
import jax.numpy as jnp
import jax.scipy.stats as st
from tqdm import tqdm


__all__ = ["hmc"]


def hmc(
    n_samples,
    potential,
    initial_position,
    random_key,
    initial_potential=None,
    initial_potential_grad=None,
    tune=500,
    path_len=1,
    initial_step_size=0.1,
):
    """Run Hamiltonian Monte Carlo sampling.

    Args:
        n_samples: Number of samples to return
        negative_log_prob: The negative log probability to sample from
        initial_position: A place to start sampling from.
        tune: Number of iterations to run tuning
        path_len: How long each integration path is.
        initial_step_size: How long each integration step is.

    Returns:
        Array of length `n_samples`.
    """
    pot_vg = value_and_grad(potential)

    def leapfrog(q, p, dVdq, path_len, step_size):
        """Leapfrog integrator for Hamiltonian Monte Carlo.

        Args:
            q: Initial position
            p: Initial momentum
            dVdq: Gradient of the potential at the initial coordinates
            path_len: How long to integrate for
            step_size: How long each integration step should be

        Returns:
            q, p: New position and momentum
        """
        # q, p = jnp.copy(q), jnp.copy(p)
        p -= step_size * dVdq / 2  # half step
        nsteps = max(0, round(path_len / step_size - 1))
        for _ in jnp.arange(nsteps):
            q += step_size * p  # whole step
            V, dVdq = pot_vg(q)
            p -= step_size * dVdq  # whole step
        q += step_size * p  # whole step
        V, dVdq = pot_vg(q)
        p -= step_size * dVdq / 2  # half step
        return q, -p, V, dVdq  # momentum flip at end

    initial_position = jnp.array(initial_position)
    initial_potential, initial_potential_grad = pot_vg(initial_position)

    # collect all our samples in a list
    samples = [initial_position]

    step_size = initial_step_size
    step_size_tuning = DualAveragingStepSize(step_size)
    # If initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws
    # we can do this in one call to jnp.random.normal, and iterate over rows
    size = (n_samples + tune,) + initial_position.shape[:1]
    random_key, subkey = random.split(random_key)
    momentum_draws = random.normal(subkey, shape=size)
    for idx, p0 in tqdm(enumerate(momentum_draws), total=size[0]):
        # random key for jitter
        random_key, subkey = random.split(random_key)
        # Integrate over our path to get a new position and momentum
        jittered_path_len = 2 * random.uniform(subkey) * path_len
        q_new, p_new, final_V, final_dVdq = leapfrog(
            samples[-1],
            p0,
            initial_potential_grad,
            path_len=jittered_path_len,
            step_size=step_size,
        )

        start_log_p = jnp.sum(st.norm.logpdf(p0)) - initial_potential
        new_log_p = jnp.sum(st.norm.logpdf(p_new)) - final_V
        energy_change = new_log_p - start_log_p

        # Check Metropolis acceptance criterion
        p_accept = min(1, jnp.exp(energy_change))
        random_key, subkey = random.split(random_key)
        if random.uniform(subkey) < p_accept:
            samples.append(q_new)
            initial_potential = final_V
            initial_potential_grad = final_dVdq
        else:
            samples.append(samples[-1])

        if idx < tune - 1:
            step_size, _ = step_size_tuning.update(p_accept)
        elif idx == tune - 1:
            _, step_size = step_size_tuning.update(p_accept)

    return jnp.array(samples[1 + tune :])


class DualAveragingStepSize:
    def __init__(
        self,
        initial_step_size,
        target_accept=0.8,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
    ):
        """Tune the step size to achieve a desired target acceptance.

        Uses stochastic approximation of Robbins and Monro (1951), described in
        Hoffman and Gelman (2013), section 3.2.1, and using those default values.

        Parameters
        ----------
        initial_step_size: float > 0
            Used to set a reasonable value for the stochastic step to drift towards
        target_accept: float in (0, 1)
            Will try to find a step size that accepts this percent of proposals
        gamma: float
            How quickly the stochastic step size reverts to a value mu
        t0: float > 0
            Larger values stabilize step size exploration early, while perhaps slowing
            convergence
        kappa: float in (0.5, 1]
            The smaller kappa is, the faster we forget earlier step size iterates
        """
        self.mu = jnp.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        """Propose a new step size.

        This method returns both a stochastic step size and a dual-averaged
        step size. While tuning, the HMC algorithm should use the stochastic
        step size and call `update` every loop. After tuning, HMC should use
        the dual-averaged step size for sampling.

        Args:
            p_accept: The probability of the previous HMC proposal being accepted

        Returns:
            Tuple: A stochastic step size, and a dual-averaged step size
        """
        self.error_sum += self.target_accept - p_accept
        log_step = self.mu - self.error_sum / (jnp.sqrt(self.t) * self.gamma)
        eta = self.t ** (-self.kappa)
        self.log_averaged_step = (
            eta * log_step + (1 - eta) * self.log_averaged_step
        )
        self.t += 1
        return jnp.exp(log_step), jnp.exp(self.log_averaged_step)
