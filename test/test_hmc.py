import jax.numpy as jnp
from jax import random

from ploo import (
    hmc,
    neg_log_normal,
    neg_log_mvnormal,
)


def test_hamiltonian_monte_carlo():
    # This mostly tests consistency. Tolerance chosen by experiment
    # Do statistical tests on your own time.
    neg_log_p = neg_log_normal(2, 0.1)
    key = random.PRNGKey(42)
    samples = hmc(
        100, neg_log_p, jnp.array(0.0), key
    )
    assert samples.shape[0] == 100
    assert jnp.allclose(2.0, jnp.mean(samples), atol=0.1)
    assert jnp.allclose(0.1, jnp.std(samples), atol=0.1)


def test_hamiltonian_monte_carlo_mv():
    mu = jnp.arange(2)
    cov = 0.8 * jnp.ones((2, 2)) + 0.2 * jnp.eye(2)
    neg_log_p = neg_log_mvnormal(mu, cov)

    samples = hmc(
        100, neg_log_p, jnp.zeros(mu.shape), path_len=2.0
    )
    assert samples.shape[0] == 100
    assert jnp.allclose(mu, jnp.mean(samples, axis=0), atol=0.3)
    assert jnp.allclose(cov, jnp.cov(samples.T), atol=0.5)


if __name__ == '__main__':
    test_hamiltonian_monte_carlo()
    test_hamiltonian_monte_carlo_mv()
