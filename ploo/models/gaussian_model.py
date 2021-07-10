"""ploo is a package for parallel cross-validation

Confidential code not for distribution.
Alex Cooper <alex@acooper.org>
"""
from typing import Tuple

import chex
import distrax
import jax.scipy.stats as st
from jax import numpy as jnp
from jax import random

import ploo
from ploo.model import CVFold, InfParams, ModelParams


class GaussianModel(ploo.Model):
    r"""Simple Gaussian experiment for testing inference

    The model is given by
    $$ y_t \sim \mathcal{N}\left(\mu, \sigma\right), \qquad t=0,\dots,(N-1) $$

    We assume we have priors
    $$ \mu \sim N(0,1), \qquad \sigma \sim Gamma(2,2).$$
    """

    name = "Gaussian model"
    sigma_transform = distrax.Lambda(jnp.log)

    def __init__(
        self,
        y: chex.ArrayDevice,
        mu_loc: float = 0.0,
        mu_scale: float = 1.0,
        sigma_shape: float = 2.0,
        sigma_rate: float = 2.0,
    ) -> None:
        self.y = y
        self.folds = jnp.arange(0, len(y))
        self.mu_loc = mu_loc
        self.mu_scale = mu_scale
        self.sigma_shape = sigma_shape
        self.sigma_rate = sigma_rate

    def log_prior_likelihood(
        self, model_params: ModelParams
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        mu_prior = st.norm.logpdf(
            model_params["mu"], loc=self.mu_loc, scale=self.mu_scale
        )
        sigma_prior = st.gamma.logpdf(
            model_params["sigma"], a=self.sigma_shape, scale=1.0 / self.sigma_rate
        )
        lprior = mu_prior + sigma_prior
        log_lik = st.norm.logpdf(
            self.y, loc=model_params["mu"], scale=model_params["sigma"]
        )
        return lprior, log_lik

    def log_cond_pred(self, model_params: ModelParams, coords: CVFold):
        y_tilde = self.y[coords]  # in this example cv_fold just indexes the data
        return st.norm.logpdf(
            y_tilde, loc=model_params["mu"], scale=model_params["sigma"]
        )

    def initial_value(self) -> ModelParams:
        return {"mu": 0.0, "sigma": 1.0}

    def inverse_transform_log_det(
        self, inf_params: InfParams
    ) -> Tuple[ModelParams, chex.ArrayDevice]:
        sigma = inf_params["sigma"]
        sigma_t, sigma_ldet = self.sigma_transform.inverse_and_log_det(sigma)
        model_params = {"mu": inf_params["mu"], "sigma": sigma_t}
        return model_params, sigma_ldet

    def forward_transform(self, model_params: ModelParams) -> InfParams:
        sigma_t = self.sigma_transform.forward(model_params["sigma"])
        inf_params = {"mu": model_params["mu"], "sigma": sigma_t}
        return inf_params

    @classmethod
    def generate(cls, N=200, mu=0.5, sigma=2.0, seed=42):
        key = random.PRNGKey(seed=seed)
        return mu + sigma * random.normal(key, shape=(N,))
