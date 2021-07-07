"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>

Diagnostic statistics for MCMC chains.
"""

from jax import lax
from jax import numpy as jnp
from scipy.fftpack import next_fast_len


def _split_chains(samples: jnp.DeviceArray) -> jnp.DeviceArray:
    """Split chains in half, doubling number of chains and halving draws"""
    _, N = samples.shape
    return jnp.vstack([samples[:, : (N // 2)], samples[:, (N // 2) :]])


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

    Args:
        samples: 2D array of samples θ⁽ⁿᵐ⁾ (chains m on axis 0, draw n axis 1)

    Returns:
        Estimate of the ratio :math:`\hat{R}`
    """
    assert len(samples.shape) == 2, "Samples should be 2D (scalars only)"
    ssamples = _split_chains(samples)
    _, N = ssamples.shape
    _, within_var, var_plus = chain_variance(ssamples)
    return jnp.sqrt(var_plus / within_var)


def chain_variance(samples):
    r"""Computes split-R̂ per Gelman et al (2013)

    Let the between-chains variance :math:`B` and within-chain variance :math:`W`
    for draws :math:`\theta^{(nm)}` be given by

      .. math::

         B = \frac{N}{M-1}\sum_{m=1}^N\left(\bar{\theta}^{(\cdot m)}
         - \bar{\theta}^{\cdot\cdot}\right)^2

         W = \frac{1}{M}\sum_{m=1}^M \left[\frac{1}{N-1}\sum_{m=1}^N
         \left(\theta^{(nm)}-\bar{\theta}^{(\cdot m)}\right)\right]^2

    Args:
        samples: 2D array of samples θ⁽ⁿᵐ⁾ (chains m on axis 0, draw n axis 1)

    Returns:
        (B, W, var_plus) tuple
    """
    M, N = samples.shape
    theta_m = jnp.mean(samples, axis=1)  # chain averages θ⁽⋅ᵐ⁾
    s_m = jnp.var(samples, axis=1, ddof=1)  # sample variances sₘ
    between_var = N * jnp.var(theta_m, ddof=1)  # between-chain variance
    within_var = jnp.sum(s_m) / M  # within-chain variance
    var_est = within_var * (N - 1) / N + between_var / N
    return between_var, within_var, var_est


def ess(x: jnp.DeviceArray, relative=False):
    r"""Effective sample size as described in Vehtari et al 2021 and Geyer 2011.

    Adapted for JAX from pyro implementation, see:
    https://github.com/pyro-ppl/numpyro/blob/048d2c80d9f4087aa9614225568bb88e1f74d669/numpyro/diagnostics.py#L148
    Some parts also adapted from ArviZ, see:
    https://github.com/arviz-devs/arviz/blob/8115c7a1b8046797229b654c8389b7c26769aa82/arviz/stats/diagnostics.py#L65

    Args:
        samples: 2D array of samples
        relative: if true return relative measure
    """  # noqa: B950
    assert x.ndim >= 2
    assert x.shape[1] >= 2

    # find autocovariance for each chain at lag k
    N = x.shape[1]
    M = next_fast_len(N)
    # transpose axis with -1 for Fourier transform
    acov_x = jnp.swapaxes(x, 1, -1)
    # centering x
    centered_signal = acov_x - acov_x.mean(axis=-1, keepdims=True)
    freqvec = jnp.fft.rfft(centered_signal, n=2 * M, axis=-1)
    autocorr = jnp.fft.irfft(freqvec * jnp.conjugate(freqvec), n=2 * M, axis=-1)
    # truncate and normalize the result, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / jnp.arange(N, 0.0, -1)
    autocorr = autocorr / autocorr[..., :1]
    autocorr = jnp.swapaxes(autocorr, 1, -1)
    gamma_k_c = autocorr * x.var(axis=1, keepdims=True)
    # find autocorrelation at lag k (from Stan reference)
    _, var_within, var_estimator = chain_variance(x)
    rho_k = jnp.concatenate(
        [
            jnp.array([1.0]),
            jnp.array(1.0 - (var_within - gamma_k_c.mean(axis=0)) / var_estimator)[1:],
        ]
    )
    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_k = rho_k[:-1:2, ...] + rho_k[1::2, ...]
    # initial monotone (decreasing) sequence
    Rho_k_pos = jnp.clip(Rho_k[1:, ...], a_min=0, a_max=None)
    _, init_mon_seq = lax.scan(
        lambda c, a: (jnp.minimum(c, a), jnp.minimum(c, a)), jnp.inf, Rho_k_pos
    )
    Rho_k = jnp.concatenate([Rho_k[:1], init_mon_seq])
    tau = -1.0 + 2.0 * jnp.sum(Rho_k, axis=0)
    n_eff = jnp.prod(jnp.array(x.shape[:2])) / tau
    return n_eff
