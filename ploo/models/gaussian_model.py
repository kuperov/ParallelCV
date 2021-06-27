from jax import random, numpy as jnp, jit
import jax.scipy.stats as st
from functools import partial

import ploo


class GaussianModel(ploo.CVModel):
    """Simple Gaussian experiment for testing inference

    The model is given by
    $$ y_t \sim \mathcal{N}\left(\mu, \sigma\right), \qquad t=0,\dots,(N-1) $$

    We assume we have priors
    $$ \mu \sim N(0,1), \qquad \sigma \sim Gamma(2,2).$$
    """
    name = 'Gaussian model'

    def __init__(self, y) -> None:
        self.y = y
        self.folds = jnp.arange(0, len(y))

    def log_joint(self, cv_fold, mu, sigma):
        log_prior = (
            st.norm.logpdf(mu, loc=0., scale=1.)
            + st.gamma.logpdf(sigma, a=2, scale=1.0 / 2.0)
        )
        lik_contribs = st.norm.logpdf(self.y, loc=mu, scale=sigma)
        log_lik = jnp.where(self.folds != cv_fold, lik_contribs, 0).sum()
        return log_prior + log_lik

    @property
    def initial_value(self):
        return {'mu': 0., 'sigma': 1.}

    @property
    def cv_folds(self):
        return len(self.y)

    @classmethod
    def generate(cls, N=200, mu=0.5, sigma=2.0, seed=42):
        key = random.PRNGKey(seed=seed)
        return mu + sigma*random.normal(key, shape=(N,))
