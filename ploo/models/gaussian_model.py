from ploo.transforms import LogTransform
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

    name = "Gaussian model"
    sigma_transform = LogTransform()

    def __init__(
        self, y, mu_loc=0.0, mu_scale=1.0, sigma_shape=2.0, sigma_rate=2.0
    ) -> None:
        self.y = y
        self.folds = jnp.arange(0, len(y))
        self.mu_loc = mu_loc
        self.mu_scale = mu_scale
        self.sigma_shape = sigma_shape
        self.sigma_rate = sigma_rate

    def log_joint(self, cv_fold, mu, sigma):
        log_prior = st.norm.logpdf(
            mu, loc=self.mu_loc, scale=self.mu_scale
        ) + st.gamma.logpdf(sigma, a=self.sigma_shape, scale=1.0 / self.sigma_rate)
        lik_contribs = st.norm.logpdf(self.y, loc=mu, scale=sigma)
        log_lik = jnp.where(self.folds != cv_fold, lik_contribs, 0).sum()
        return log_prior + log_lik

    def cv_potential(self, param, cv_fold):
        mu = param["mu"]
        sigma_orig = param["sigma"]
        sigma = self.sigma_transform(sigma_orig)
        return -self.log_joint(cv_fold, mu, sigma) - self.sigma_transform.log_det(
            sigma_orig
        )

    @property
    def initial_value(self):
        return {"mu": 0.0, "sigma": 1.0}

    @property
    def cv_folds(self):
        return len(self.y)

    @classmethod
    def generate(cls, N=200, mu=0.5, sigma=2.0, seed=42):
        key = random.PRNGKey(seed=seed)
        return mu + sigma * random.normal(key, shape=(N,))
