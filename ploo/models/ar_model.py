from functools import partial

import jax.scipy.stats as st
from jax import jit
from jax import numpy as jnp
from jax import random

import ploo


class AR1(ploo.Model):
    r"""Simple AR(1) model with LOO

    $$ y_t = \mu + \rho y_{t-1} + \sigma \varepsilon_t, \qquad t=1,\dots,(N-1) $$
    $$ y_0 = \frac{\mu}{1-\rho}+\varepsilon $$
    $$ \varepsilon \sim \mathcal{N}\left( 0, 1\right) $$

    We assume we also have priors
    $$ \mu \sim N(0,1/16), \qquad \sigma \sim Gamma(2,2), \qquad\mathrm{and}\qquad
    \rho \sim Beta(2,2).$$
    """

    def __init__(self, y) -> None:
        self.y = y
        self.folds = jnp.arange(0, len(y))

    def log_joint(self, cv_fold, mu, sigma, rho):
        log_prior = (
            st.norm.logpdf(mu, loc=0, scale=1.0 / 4.0)
            + st.gamma.logpdf(sigma, a=2, scale=1.0 / 2.0)
            + st.beta.logpdf(rho, a=2, b=2)
        )
        resid = self.y[1:] - mu - rho * self.y[:-1]
        lik_contribs = st.norm.logpdf(resid, loc=0, scale=sigma)
        log_lik = jnp.sum(lik_contribs[self.folds != cv_fold])
        return log_prior + log_lik

    @partial(jit, static_argnums=(2,))
    def potential(self, param, cv_fold):
        # todo: log transform for sigma, logit for rho
        return -self.log_joint(cv_fold, **param)

    @classmethod
    def generate(cls, N=200, mu=0.5, sigma=2.0, rho=0.9, seed=42):
        key = random.PRNGKey(seed=seed)
        key, subkey = random.split(key)
        epsilon = random.normal(key, shape=(N,))
        y = [mu * sigma / (1 - rho) + epsilon[0]]
        for t in range(1, N):
            y.append(mu + rho * y[t - 1] + epsilon[t])
        y = jnp.array(y)
