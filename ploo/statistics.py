"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

Diagnostic statistics for MCMC chains.
"""

from jax import numpy as jnp


def split_chains(samples: jnp.DeviceArray) -> jnp.DeviceArray:
    """Split chains in half, doubling number of chains and halving draws"""
    M, N = samples.shape
    combined = jnp.stack([samples[:, : (N // 2)], samples[:, (N // 2) :]])
    return combined.reshape((2 * M, N // 2))


def split_rhat(samples: jnp.DeviceArray) -> jnp.DeviceArray:
    r"""Computes split-R̂ per Gelman et al (2013)

    Let the between-chains variance :math:`B` and within-chain variance :math:`W`
    for draws :math:`\theta^{(nm)}` be given by

      .. math::

         B = \frac{N}{M-1}\sum_{m=1}^N\left(\bar{\theta}^{(\cdot m)}
         - \bar{\theta}^{\cdot\cdot}\right)^2

         W = \frac{1}{M}\sum_{m=1}^M \left[\frac{1}{N-1}\sum_{m=1}^N
         \left(\theta^{(nm)}-\bar{\theta}^{(\cdot m)}\right)\right]^2

    Then the split-:math:`\hat{R}` is given by

      .. math::

         \hat{R} = \sqrt{\frac{N-1}{N} + \frac{B}{NW}}

    Keyword arguments:
        samples: 2D array of samples θ⁽ⁿᵐ⁾ (chains m on axis 0, draw n axis 1)

    Returns:
        Estimate of the ratio :math:`\hat{R}`
    """
    assert len(samples.shape) == 2, "Samples should be 2D (scalars only)"
    ssamples = split_chains(samples)
    M, N = ssamples.shape
    theta_m = jnp.mean(ssamples, axis=1)  # chain averages θ⁽⋅ᵐ⁾
    s_m = jnp.var(ssamples, axis=1, ddof=1)  # sample variances sₘ
    B = N * jnp.var(theta_m, ddof=1)  # between-chain variance
    W = jnp.sum(s_m) / M  # within-chain variance
    return jnp.sqrt((N - 1) / N + B / (W * N))


